import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils import round_ste


class Squeeze(nn.Module):
    """
    Squeeze operation: trades spatial resolution for channel depth.
    H x W x C -> H/2 x W/2 x 4C (forward) or H/2 x W/2 x 4C -> H x W x C (reverse)
    """
    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        if not reverse:
            x = x.view(B, C, H // 2, 2, W // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(B, C * 4, H // 2, W // 2)
        else:
            x = x.view(B, C // 4, 2, 2, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(B, C // 4, H * 2, W * 2)
        return x


class IntegerAdditiveCoupling(nn.Module):
    """
    Integer Additive Coupling Layer - the core building block of IDF.
    
    Splits input channels into two halves (x1, x2).
    Forward:  x2 <- x2 + round(NN(x1))
    Inverse:  x2 <- x2 - round(NN(x1))
    
    The rounding with STE allows gradient flow during training while
    maintaining perfect invertibility for lossless compression.
    """
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.split_idx = in_channels // 2
        
        # Neural network that predicts the transformation
        # Takes x1 and outputs transformation for x2
        self.nn = nn.Sequential(
            nn.Conv2d(self.split_idx, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels - self.split_idx, 3, padding=1),
        )
        
        # Initialize last layer to zero for stable training start
        nn.init.zeros_(self.nn[-1].weight)
        nn.init.zeros_(self.nn[-1].bias)
    
    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]
        
        # Compute integer shift using straight-through estimator
        shift = round_ste(self.nn(x1))
        
        if not reverse:
            x2 = x2 + shift
        else:
            x2 = x2 - shift
        
        return torch.cat([x1, x2], dim=1)


class Permute(nn.Module):
    """
    Learnable 1x1 convolution permutation for channel mixing.
    Uses LU decomposition for efficient inverse computation.
    """
    def __init__(self, num_channels):
        super().__init__()
        # Initialize with random orthogonal matrix
        W = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        self.W = nn.Parameter(W)
    
    def forward(self, x, reverse=False):
        if not reverse:
            W = self.W
        else:
            W = torch.inverse(self.W)
        
        return F.conv2d(x, W.unsqueeze(-1).unsqueeze(-1))


class ActNorm(nn.Module):
    """
    Activation Normalization - data-dependent initialization.
    Normalizes per-channel with learnable scale and bias.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.initialized = False
    
    def initialize(self, x):
        with torch.no_grad():
            # Compute per-channel mean and std
            # Use float32 to avoid FP16 parameter assignment when inside autocast
            mean = x.float().mean(dim=[0, 2, 3], keepdim=True)
            std = x.float().std(dim=[0, 2, 3], keepdim=True) + 1e-6
            self.bias.data = -mean
            self.scale.data = 1.0 / std
            self.initialized = True
    
    def forward(self, x, reverse=False):
        if not self.initialized and not reverse:
            self.initialize(x)
        
        if not reverse:
            return self.scale * (x + self.bias)
        else:
            return x / self.scale - self.bias


class FlowStep(nn.Module):
    """
    Single flow step combining ActNorm, Permutation, and Coupling.
    """
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.permute = Permute(in_channels)
        self.coupling = IntegerAdditiveCoupling(in_channels, hidden_channels)
    
    def forward(self, x, reverse=False):
        if not reverse:
            x = self.actnorm(x, reverse=False)
            x = self.permute(x, reverse=False)
            x = self.coupling(x, reverse=False)
        else:
            x = self.coupling(x, reverse=True)
            x = self.permute(x, reverse=True)
            x = self.actnorm(x, reverse=True)
        return x


class FlowBlock(nn.Module):
    """
    A block of flow steps followed by optional split for hierarchical factorization.
    """
    def __init__(self, in_channels, hidden_channels=256, num_steps=4, split=True):
        super().__init__()
        self.split = split
        
        self.flows = nn.ModuleList([
            FlowStep(in_channels, hidden_channels) for _ in range(num_steps)
        ])
        
        # If splitting, we need a prior network for the split-off channels
        if split:
            self.split_idx = in_channels // 2
            # Prior network predicts mean/logscale for split-off channels
            self.prior = nn.Sequential(
                nn.Conv2d(self.split_idx, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.split_idx * 2, 3, padding=1),
            )
            nn.init.zeros_(self.prior[-1].weight)
            nn.init.zeros_(self.prior[-1].bias)
    
    def forward(self, x, reverse=False):
        if not reverse:
            # Forward pass through flow steps
            for flow in self.flows:
                x = flow(x, reverse=False)
            
            if self.split:
                # Split channels: keep half, send half to entropy coder
                x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]
                # Compute prior parameters from x1
                prior_params = self.prior(x1)
                mean, log_scale = prior_params.chunk(2, dim=1)
                return x1, (x2, mean, log_scale)
            else:
                return x, None
        else:
            if self.split:
                x1, (x2, _, _) = x
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x[0] if isinstance(x, tuple) else x
            
            # Reverse pass through flow steps
            for flow in reversed(self.flows):
                x = flow(x, reverse=True)
            
            return x

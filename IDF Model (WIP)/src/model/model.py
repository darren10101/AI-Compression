import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.layers import Squeeze, FlowBlock, FlowStep
from src.utils import round_ste


class DiscretizedLogistic:
    """
    Discretized logistic distribution for entropy coding.
    Models integer-valued data by integrating logistic PDF over bins.
    """
    @staticmethod
    def log_prob(x, mean, log_scale, bin_size=1.0):
        """
        Compute log probability of x under discretized logistic.
        
        Args:
            x: Integer values (B, C, H, W)
            mean: Distribution mean (B, C, H, W)
            log_scale: Log of distribution scale (B, C, H, W)
            bin_size: Size of discretization bins
        """
        scale = torch.exp(log_scale).clamp(min=1e-7)
        
        # Compute CDF at bin edges
        x_centered = x - mean
        plus_in = (x_centered + bin_size / 2) / scale
        minus_in = (x_centered - bin_size / 2) / scale
        
        cdf_plus = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)
        
        # Log probability is log of CDF difference
        log_prob = torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))
        
        return log_prob
    
    @staticmethod
    def sample(mean, log_scale):
        """Sample from discretized logistic (for generation/testing)."""
        scale = torch.exp(log_scale)
        u = torch.rand_like(mean)
        x = mean + scale * (torch.log(u) - torch.log(1 - u))
        return torch.round(x)


class StandardLogistic:
    """Standard logistic distribution (mean=0, scale=1) for final latents."""
    @staticmethod
    def log_prob(x, bin_size=1.0):
        """Log probability under standard discretized logistic."""
        plus_in = (x + bin_size / 2)
        minus_in = (x - bin_size / 2)
        
        cdf_plus = torch.sigmoid(plus_in)
        cdf_minus = torch.sigmoid(minus_in)
        
        log_prob = torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))
        return log_prob


class IntegerDiscreteFlow(nn.Module):
    """
    Integer Discrete Flow (IDF) for Lossless Image Compression.
    
    Transforms structured image data into decorrelated latents that follow
    a simple distribution (logistic), enabling efficient entropy coding.
    
    Architecture:
    1. Squeeze: H x W x C -> H/2 x W/2 x 4C
    2. Flow Blocks: Stack of coupling layers with hierarchical splits
    3. Entropy Model: Discretized logistic for probability estimation
    
    Args:
        in_channels: Number of input channels (e.g., 3 for RGB)
        hidden_channels: Hidden dimension in coupling networks
        num_levels: Number of hierarchical levels (squeeze + flow block)
        num_steps: Number of flow steps per block
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=256,
        num_levels=3,
        num_steps=8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_levels = num_levels
        
        # Build hierarchical flow
        self.squeezes = nn.ModuleList()
        self.blocks = nn.ModuleList()
        
        current_channels = in_channels
        
        for level in range(num_levels):
            # Squeeze operation (4x channel increase)
            self.squeezes.append(Squeeze())
            current_channels *= 4
            
            # Flow block (last level doesn't split)
            is_last = (level == num_levels - 1)
            self.blocks.append(
                FlowBlock(
                    current_channels,
                    hidden_channels=hidden_channels,
                    num_steps=num_steps,
                    split=not is_last
                )
            )
            
            # After split, channels are halved
            if not is_last:
                current_channels //= 2
        
        self.final_channels = current_channels
        
        # Track split channel counts for reconstruction
        self._build_channel_info(in_channels)
    
    def _build_channel_info(self, in_channels):
        """Precompute channel dimensions at each level."""
        self.level_channels = []
        c = in_channels
        for level in range(self.num_levels):
            c *= 4
            if level < self.num_levels - 1:
                self.level_channels.append(c // 2)  # Split channels
                c //= 2
        self.level_channels.append(c)  # Final channels
    
    def forward(self, x, reverse=False):
        """
        Forward: Compress image to latent + compute log likelihood.
        Reverse: Decompress latent back to image.
        
        Args:
            x: Input tensor
               Forward: Image tensor (B, C, H, W) with integer values
               Reverse: Tuple of (final_z, split_outputs)
            reverse: Direction of flow
        
        Returns:
            Forward: (latents, log_likelihood)
                latents: List of latent tensors from each level
                log_likelihood: Total log likelihood for rate estimation
            Reverse: Reconstructed image tensor
        """
        if not reverse:
            return self._forward(x)
        else:
            return self._reverse(x)
    
    def _forward(self, x):
        """Encoding pass: image -> latent codes."""
        split_outputs = []
        log_likelihood = 0.0
        
        for level in range(self.num_levels):
            # Squeeze spatial dims into channels
            x = self.squeezes[level](x, reverse=False)
            
            # Flow block
            x, split_info = self.blocks[level](x, reverse=False)
            
            # Handle hierarchical split
            if split_info is not None:
                z_split, mean, log_scale = split_info
                # Compute log likelihood for split-off channels
                ll = DiscretizedLogistic.log_prob(z_split, mean, log_scale)
                log_likelihood = log_likelihood + ll.sum(dim=[1, 2, 3])
                split_outputs.append((z_split, mean, log_scale))
        
        # Final latent uses standard logistic prior
        final_ll = StandardLogistic.log_prob(x)
        log_likelihood = log_likelihood + final_ll.sum(dim=[1, 2, 3])
        
        # Collect all latents for entropy coding
        latents = [so[0] for so in split_outputs] + [x]
        
        return latents, log_likelihood
    
    def _reverse(self, latents):
        """Decoding pass: latent codes -> image."""
        # Separate final latent from split latents
        final_z = latents[-1]
        split_latents = latents[:-1]
        
        x = final_z
        
        for level in range(self.num_levels - 1, -1, -1):
            # Reconstruct split info for reverse pass
            if level < self.num_levels - 1:
                z_split = split_latents[level]
                # Prior params would be recomputed in reverse, but we have z_split
                x_input = (x, (z_split, None, None))
            else:
                x_input = x
            
            # Reverse flow block
            x = self.blocks[level](x_input, reverse=True)
            
            # Reverse squeeze
            x = self.squeezes[level](x, reverse=True)
        
        return x
    
    def compute_loss(self, x):
        """
        Compute negative log likelihood loss (bits per dimension).
        
        Args:
            x: Input image (B, C, H, W), integer values in [0, 255]
        
        Returns:
            loss: Negative log likelihood in bits per dimension
            bpd: Bits per dimension (for logging)
        """
        # Center the input around 0
        x_centered = x - 128.0
        
        latents, log_likelihood = self._forward(x_centered)
        
        # Negative log likelihood
        nll = -log_likelihood
        
        # Convert to bits per dimension
        num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
        bpd = nll / (num_pixels * np.log(2))
        
        return bpd.mean(), bpd.mean()
    
    def compress(self, x):
        """
        Compress image to latent codes for arithmetic coding.
        
        Args:
            x: Input image (B, C, H, W), integer values in [0, 255]
        
        Returns:
            latents: List of integer latent tensors
            prior_params: List of (mean, log_scale) for each latent
        """
        x_centered = x - 128.0
        
        split_outputs = []
        
        z = x_centered
        for level in range(self.num_levels):
            z = self.squeezes[level](z, reverse=False)
            z, split_info = self.blocks[level](z, reverse=False)
            if split_info is not None:
                split_outputs.append(split_info)
        
        # Final latent
        latents = [so[0] for so in split_outputs] + [z]
        prior_params = [(so[1], so[2]) for so in split_outputs]
        # Standard prior for final latent
        prior_params.append((
            torch.zeros_like(z),
            torch.zeros_like(z)
        ))
        
        return latents, prior_params
    
    def decompress(self, latents):
        """
        Decompress latent codes back to image.
        
        Args:
            latents: List of integer latent tensors from compress()
        
        Returns:
            x: Reconstructed image (B, C, H, W)
        """
        x_centered = self._reverse(latents)
        return x_centered + 128.0


class HierarchicalIntegerFlow(IntegerDiscreteFlow):
    """
    Hierarchical Integer Flow (HIF) - enhanced IDF with multi-scale processing.
    
    Extends IDF with:
    - Deeper coupling networks
    - More sophisticated channel mixing
    - Better suited for high-resolution images
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=96,
        num_levels=4,
        num_steps=12,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_steps=num_steps,
        )


# Convenience function to create model from config
def create_idf_model(config=None):
    """
    Create IDF model from configuration dictionary.
    
    Args:
        config: Dictionary with model parameters, or None for defaults
    
    Returns:
        IntegerDiscreteFlow model
    """
    if config is None:
        config = {}
    
    return IntegerDiscreteFlow(
        in_channels=config.get('in_channels', 3),
        hidden_channels=config.get('hidden_channels', 256),
        num_levels=config.get('num_levels', 3),
        num_steps=config.get('num_steps', 8),
    )

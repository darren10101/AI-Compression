"""
Standalone script to evaluate PixelCNN compression performance against traditional codecs.

Usage:
    python evaluate_compression.py --checkpoint checkpoints/pixelcnn_best.pt --num_samples 500

This script:
1. Loads a trained PixelCNN model
2. Compresses validation images using PixelCNN + ANS encoding
3. Compares with PNG, JPEG, and WebP codecs
4. Generates detailed comparison reports and visualizations
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    import constriction
    CONSTRICTION_AVAILABLE = True
except ImportError:
    CONSTRICTION_AVAILABLE = False
    print("Warning: constriction not installed. PixelCNN compression will be estimated from loss.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Plots will be skipped.")


# =============================================================================
# Model Definition (must match training)
# =============================================================================

class MaskedConv2d(nn.Conv2d):
    """Masked Convolution for autoregressive models."""
    
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, **kwargs):
        assert mask_type in ['A', 'B']
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        
        self.mask_type = mask_type
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        _, _, h, w = self.weight.shape
        center_h, center_w = h // 2, w // 2
        
        self.mask[:, :, center_h + 1:, :] = 0
        self.mask[:, :, center_h, center_w + 1:] = 0
        
        if mask_type == 'A':
            self.mask[:, :, center_h, center_w] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlock(nn.Module):
    """Residual block with masked convolutions."""
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = MaskedConv2d('B', channels, channels // 2, 1)
        self.conv2 = MaskedConv2d('B', channels // 2, channels // 2, kernel_size)
        self.conv3 = MaskedConv2d('B', channels // 2, channels, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return self.relu(out + residual)


class PixelCNN(nn.Module):
    """PixelCNN for image compression."""
    
    def __init__(self, in_channels=3, hidden_channels=128, num_residual_blocks=12, 
                 num_classes=256, kernel_size=7):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_conv = MaskedConv2d('A', in_channels, hidden_channels, kernel_size)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, kernel_size=3) 
            for _ in range(num_residual_blocks)
        ])
        
        self.output_conv1 = MaskedConv2d('B', hidden_channels, hidden_channels, 1)
        self.output_conv2 = MaskedConv2d('B', hidden_channels, hidden_channels, 1)
        self.final_conv = MaskedConv2d('B', hidden_channels, in_channels * num_classes, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels
    
    def forward(self, x):
        if x.dtype == torch.long:
            x = x.float() / 127.5 - 1
        
        out = self.relu(self.input_conv(x))
        
        for block in self.residual_blocks:
            out = block(out)
        
        out = self.relu(self.output_conv1(out))
        out = self.relu(self.output_conv2(out))
        out = self.final_conv(out)
        
        B, _, H, W = out.shape
        out = out.view(B, self.in_channels, self.num_classes, H, W)
        
        return out
    
    def loss(self, x, target):
        logits = self.forward(x)
        B, C, num_classes, H, W = logits.shape
        
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        logits = logits.view(-1, num_classes)
        target = target.view(-1)
        
        loss = F.cross_entropy(logits, target, reduction='mean')
        bits_per_subpixel = loss / np.log(2)
        
        return bits_per_subpixel


# =============================================================================
# Compression Classes
# =============================================================================

class PixelCNNCompressor:
    """Compress images using PixelCNN + ANS encoding (using constriction library)."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def compress(self, image):
        """Compress a single image using ANS encoding with constriction."""
        if not CONSTRICTION_AVAILABLE:
            raise RuntimeError("constriction not available")
        
        self.model.eval()
        C, H, W = image.shape
        image = image.to(self.device)
        
        input_img = image.float().unsqueeze(0) / 127.5 - 1
        logits = self.model(input_img)
        probs = F.softmax(logits, dim=2).squeeze(0)
        
        # Shape: (C, 256, H, W) -> (C*H*W, 256)
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        symbols = image.view(-1).to(torch.int32).cpu().numpy()
        
        # Normalize probabilities and convert to numpy
        probs = probs.clamp(min=1e-9)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        probs_np = probs.cpu().numpy().astype(np.float32)
        
        # Use constriction's ANS encoder with Categorical model family
        ans = constriction.stream.stack.AnsCoder()
        model_family = constriction.stream.model.Categorical(perfect=False)
        
        # ANS uses stack semantics (LIFO), so encode_reverse to decode in forward order
        ans.encode_reverse(symbols, model_family, probs_np)
        
        # Get compressed data as uint32 array and convert to bytes
        compressed = ans.get_compressed()
        compressed_bytes = compressed.tobytes()
        
        return compressed_bytes, (C, H, W)
    
    @torch.no_grad()
    def estimate_bpp(self, image):
        """Estimate BPP from model loss (faster than actual compression)."""
        self.model.eval()
        image = image.to(self.device)
        
        input_img = image.float().unsqueeze(0) / 127.5 - 1
        target = image.long().unsqueeze(0)
        
        loss = self.model.loss(input_img, target)
        return loss.item() * 3  # bits per pixel (3 channels)


class TraditionalCodec:
    """Wrapper for traditional image codecs."""
    
    @staticmethod
    def compress_png(image_np):
        img = Image.fromarray(image_np)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        return buffer.getvalue()
    
    @staticmethod
    def compress_jpeg(image_np, quality=95):
        img = Image.fromarray(image_np)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        return buffer.getvalue()
    
    @staticmethod
    def decompress_jpeg(compressed_bytes):
        buffer = io.BytesIO(compressed_bytes)
        img = Image.open(buffer)
        return np.array(img)
    
    @staticmethod
    def compress_webp(image_np, quality=100, lossless=True):
        img = Image.fromarray(image_np)
        buffer = io.BytesIO()
        img.save(buffer, format='WEBP', quality=quality, lossless=lossless)
        return buffer.getvalue()
    
    @staticmethod
    def decompress_webp(compressed_bytes):
        buffer = io.BytesIO(compressed_bytes)
        img = Image.open(buffer)
        return np.array(img)


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_metrics(original, reconstructed):
    """Compute image quality metrics."""
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)
    
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    
    return {'mse': mse, 'psnr': psnr}


def compute_bpp(compressed_size_bytes, image_shape):
    """Compute bits per pixel."""
    _, h, w = image_shape
    total_pixels = h * w
    total_bits = compressed_size_bytes * 8
    return total_bits / total_pixels


def evaluate_compression(model, data_loader, device, num_samples=500, use_actual_compression=True):
    """Evaluate compression performance."""
    model.eval()
    compressor = PixelCNNCompressor(model, device)
    codec = TraditionalCodec()
    
    results = {
        'pixelcnn': {'bpp': [], 'psnr': []},
        'pixelcnn_theoretical': {'bpp': [], 'psnr': []},
        'png': {'bpp': [], 'psnr': []},
        'jpeg_95': {'bpp': [], 'psnr': []},
        'jpeg_75': {'bpp': [], 'psnr': []},
        'jpeg_50': {'bpp': [], 'psnr': []},
        'webp_lossless': {'bpp': [], 'psnr': []},
        'webp_95': {'bpp': [], 'psnr': []},
    }
    
    sample_count = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    for batch in tqdm(data_loader, desc='Evaluating'):
        for img_tensor in batch:
            if sample_count >= num_samples:
                break
            
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            # PixelCNN - theoretical (from loss)
            try:
                theoretical_bpp = compressor.estimate_bpp(img_tensor)
                results['pixelcnn_theoretical']['bpp'].append(theoretical_bpp)
                results['pixelcnn_theoretical']['psnr'].append(float('inf'))
            except Exception as e:
                results['pixelcnn_theoretical']['bpp'].append(float('nan'))
                results['pixelcnn_theoretical']['psnr'].append(float('nan'))
            
            # PixelCNN - actual compression
            if use_actual_compression and CONSTRICTION_AVAILABLE:
                try:
                    compressed_pixelcnn, shape = compressor.compress(img_tensor)
                    pixelcnn_bpp = compute_bpp(len(compressed_pixelcnn), img_tensor.shape)
                    results['pixelcnn']['bpp'].append(pixelcnn_bpp)
                    results['pixelcnn']['psnr'].append(float('inf'))
                except Exception as e:
                    results['pixelcnn']['bpp'].append(float('nan'))
                    results['pixelcnn']['psnr'].append(float('nan'))
            
            # PNG
            compressed_png = codec.compress_png(img_np)
            png_bpp = compute_bpp(len(compressed_png), img_tensor.shape)
            results['png']['bpp'].append(png_bpp)
            results['png']['psnr'].append(float('inf'))
            
            # JPEG at different quality levels
            for quality, key in [(95, 'jpeg_95'), (75, 'jpeg_75'), (50, 'jpeg_50')]:
                compressed = codec.compress_jpeg(img_np, quality=quality)
                bpp = compute_bpp(len(compressed), img_tensor.shape)
                decoded = codec.decompress_jpeg(compressed)
                metrics = compute_metrics(img_np, decoded)
                results[key]['bpp'].append(bpp)
                results[key]['psnr'].append(metrics['psnr'])
            
            # WebP lossless
            compressed_webp = codec.compress_webp(img_np, lossless=True)
            webp_bpp = compute_bpp(len(compressed_webp), img_tensor.shape)
            results['webp_lossless']['bpp'].append(webp_bpp)
            results['webp_lossless']['psnr'].append(float('inf'))
            
            # WebP lossy
            compressed_webp_lossy = codec.compress_webp(img_np, quality=95, lossless=False)
            webp_lossy_bpp = compute_bpp(len(compressed_webp_lossy), img_tensor.shape)
            decoded_webp = codec.decompress_webp(compressed_webp_lossy)
            webp_metrics = compute_metrics(img_np, decoded_webp)
            results['webp_95']['bpp'].append(webp_lossy_bpp)
            results['webp_95']['psnr'].append(webp_metrics['psnr'])
            
            sample_count += 1
        
        if sample_count >= num_samples:
            break
    
    # Compute summary statistics
    summary = {}
    for codec_name, metrics in results.items():
        valid_bpp = [b for b in metrics['bpp'] if not np.isnan(b)]
        valid_psnr = [p for p in metrics['psnr'] if not np.isnan(p) and not np.isinf(p)]
        
        if valid_bpp:
            summary[codec_name] = {
                'avg_bpp': float(np.mean(valid_bpp)),
                'std_bpp': float(np.std(valid_bpp)),
                'min_bpp': float(np.min(valid_bpp)),
                'max_bpp': float(np.max(valid_bpp)),
                'avg_psnr': float(np.mean(valid_psnr)) if valid_psnr else 'lossless',
                'num_samples': len(valid_bpp)
            }
    
    return results, summary


def print_results(summary):
    """Print formatted results table."""
    print("\n" + "=" * 85)
    print("COMPRESSION COMPARISON RESULTS")
    print("=" * 85)
    print(f"{'Codec':<25} {'Avg BPP':<12} {'Std BPP':<12} {'Min BPP':<12} {'PSNR (dB)':<12}")
    print("-" * 85)
    
    # Sort by average BPP (lossless first, then lossy)
    lossless = {k: v for k, v in summary.items() if v.get('avg_psnr') == 'lossless'}
    lossy = {k: v for k, v in summary.items() if v.get('avg_psnr') != 'lossless'}
    
    print("LOSSLESS:")
    for codec, stats in sorted(lossless.items(), key=lambda x: x[1]['avg_bpp']):
        print(f"  {codec:<23} {stats['avg_bpp']:<12.4f} {stats['std_bpp']:<12.4f} "
              f"{stats['min_bpp']:<12.4f} {'inf':<12}")
    
    print("\nLOSSY:")
    for codec, stats in sorted(lossy.items(), key=lambda x: x[1]['avg_bpp']):
        psnr_str = f"{stats['avg_psnr']:.2f}" if isinstance(stats['avg_psnr'], (int, float)) else stats['avg_psnr']
        print(f"  {codec:<23} {stats['avg_bpp']:<12.4f} {stats['std_bpp']:<12.4f} "
              f"{stats['min_bpp']:<12.4f} {psnr_str:<12}")
    
    print("=" * 85)


def plot_results(results, summary, output_dir):
    """Generate comparison plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots (matplotlib not available)")
        return
    
    # BPP comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    codecs = [k for k in summary.keys() if summary[k]['num_samples'] > 0]
    bpps = [summary[c]['avg_bpp'] for c in codecs]
    stds = [summary[c]['std_bpp'] for c in codecs]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(codecs)))
    
    bars = axes[0].bar(range(len(codecs)), bpps, yerr=stds, capsize=5, color=colors)
    axes[0].set_xticks(range(len(codecs)))
    axes[0].set_xticklabels(codecs, rotation=45, ha='right')
    axes[0].set_ylabel('Bits Per Pixel (BPP)')
    axes[0].set_title('Average Compression Rate')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    bpp_data = [results[c]['bpp'] for c in codecs]
    bp = axes[1].boxplot(bpp_data, labels=codecs, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Bits Per Pixel (BPP)')
    axes[1].set_title('BPP Distribution')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Rate-distortion plot (for lossy codecs)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lossy_codecs = [k for k in codecs if summary[k].get('avg_psnr') != 'lossless']
    
    for codec in lossy_codecs:
        ax.scatter(summary[codec]['avg_bpp'], summary[codec]['avg_psnr'], 
                   s=100, label=codec, alpha=0.8)
    
    # Add lossless codecs as vertical lines
    lossless_codecs = [k for k in codecs if summary[k].get('avg_psnr') == 'lossless']
    for i, codec in enumerate(lossless_codecs):
        ax.axvline(x=summary[codec]['avg_bpp'], linestyle='--', alpha=0.5,
                   label=f'{codec} (lossless)')
    
    ax.set_xlabel('Bits Per Pixel (BPP)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Rate-Distortion Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rate_distortion.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PixelCNN compression')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/pixelcnn_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loading')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--skip_actual_compression', action='store_true',
                        help='Skip actual ANS compression (use theoretical BPP only)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load dataset
    print("\nLoading CIFAR-10 dataset...")
    
    class CIFAR10Compression(torch.utils.data.Dataset):
        def __init__(self, root='./data', train=True, download=True):
            self.dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=download,
                transform=transforms.ToTensor()
            )
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, _ = self.dataset[idx]
            return (img * 255).to(torch.uint8)
    
    full_train_dataset = CIFAR10Compression(train=True, download=True)
    
    # Create validation split
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    _, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load model
    print("\nLoading model...")
    model = PixelCNN(in_channels=3, hidden_channels=128, num_residual_blocks=12).to(args.device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f} bpp")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using randomly initialized model (results will be poor)")
    
    # Evaluate
    results, summary = evaluate_compression(
        model, val_loader, args.device,
        num_samples=args.num_samples,
        use_actual_compression=not args.skip_actual_compression
    )
    
    # Print results
    print_results(summary)
    
    # Plot results
    plot_results(results, summary, args.output_dir)
    
    # Save results to JSON
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Print key comparisons
    print("\n" + "=" * 60)
    print("KEY COMPARISONS")
    print("=" * 60)
    
    if 'pixelcnn_theoretical' in summary and 'png' in summary:
        pixelcnn_bpp = summary['pixelcnn_theoretical']['avg_bpp']
        png_bpp = summary['png']['avg_bpp']
        improvement = (png_bpp - pixelcnn_bpp) / png_bpp * 100
        print(f"PixelCNN vs PNG: {improvement:+.1f}% {'better' if improvement > 0 else 'worse'}")
        print(f"  PixelCNN: {pixelcnn_bpp:.4f} bpp")
        print(f"  PNG:      {png_bpp:.4f} bpp")
    
    if 'pixelcnn_theoretical' in summary and 'webp_lossless' in summary:
        pixelcnn_bpp = summary['pixelcnn_theoretical']['avg_bpp']
        webp_bpp = summary['webp_lossless']['avg_bpp']
        improvement = (webp_bpp - pixelcnn_bpp) / webp_bpp * 100
        print(f"\nPixelCNN vs WebP (lossless): {improvement:+.1f}% {'better' if improvement > 0 else 'worse'}")
        print(f"  PixelCNN: {pixelcnn_bpp:.4f} bpp")
        print(f"  WebP:     {webp_bpp:.4f} bpp")
    
    print("=" * 60)


if __name__ == '__main__':
    main()

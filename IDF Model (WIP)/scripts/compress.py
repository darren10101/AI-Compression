"""
Compress an image using the Integer Discrete Flow (IDF) model.

Splits the image into 256x256 patches, compresses each patch,
and saves the compressed representation to a .idf file.

Supports two compression modes:
- Arithmetic coding (default): Uses learned prior distributions for optimal compression
- Legacy zlib: Original zlib-based compression (use --no-arithmetic flag)

Usage:
    python scripts/compress.py input.png output.idf
    python scripts/compress.py input.png output.idf --model checkpoints/best_model.pt
    python scripts/compress.py input.png output.idf --no-arithmetic  # Use legacy zlib mode
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import struct
import pickle
import zlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_idf_model
from src.arithmetic_coding import fast_encode_latents


PATCH_SIZE = 256
MAGIC_HEADER = b'IDF2'  # Version 2 - includes ActNorm state (zlib)
MAGIC_HEADER_V3 = b'IDF3'  # Version 3 - arithmetic coding with learned priors (lossy)
MAGIC_HEADER_V4 = b'IDF4'  # Version 4 - arithmetic coding + residuals (lossless)


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load the IDF model from checkpoint.
    
    Note: ActNorm layers are left uninitialized (initialized=False).
    They will initialize on the first forward pass, and we save that state.
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = create_idf_model()
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).float()
    model.eval()
    
    # NOTE: We intentionally do NOT set initialized=True here.
    # ActNorm will initialize on first compress, and we save that state.
    
    return model


def get_actnorm_state(model) -> dict:
    """Extract ActNorm scale/bias parameters from model."""
    state = {}
    for block_idx, block in enumerate(model.blocks):
        for flow_idx, flow in enumerate(block.flows):
            key = f"block{block_idx}_flow{flow_idx}"
            state[key] = {
                'scale': flow.actnorm.scale.data.cpu().numpy(),
                'bias': flow.actnorm.bias.data.cpu().numpy(),
            }
    return state


def load_image(image_path: str) -> np.ndarray:
    """Load image and convert to numpy array."""
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img, dtype=np.uint8)


def pad_image(image: np.ndarray, patch_size: int) -> tuple:
    """
    Pad image so dimensions are divisible by patch_size.
    
    Returns:
        padded_image: Padded numpy array
        original_shape: Original (H, W) before padding
        pad_h, pad_w: Amount of padding added
    """
    h, w, c = image.shape
    original_shape = (h, w)
    
    # Calculate padding needed
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        # Pad with edge values (reflection would also work)
        padded = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='edge'
        )
    else:
        padded = image
    
    return padded, original_shape, pad_h, pad_w


def split_into_patches(image: np.ndarray, patch_size: int) -> list:
    """Split image into patch_size x patch_size patches."""
    h, w, c = image.shape
    patches = []
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    
    return patches


def compress_patch(model, patch: np.ndarray, device: torch.device) -> tuple:
    """
    Compress a single patch using the IDF model.
    
    Returns:
        latent_arrays: List of latent tensors (as numpy arrays, float32)
        prior_param_arrays: List of (mean, log_scale) tuples as numpy arrays
    """
    # Convert to tensor: (1, 3, H, W), float32, values in [0, 255]
    x = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
    x = x.to(device)
    
    with torch.no_grad():
        latents, prior_params = model.compress(x)
    
    # Convert latents to float32 numpy arrays for storage
    # float32 required for exact lossless reconstruction
    latent_arrays = []
    for lat in latents:
        lat_np = lat.cpu().numpy().astype(np.float32)
        latent_arrays.append(lat_np)
    
    # Convert prior parameters to numpy arrays
    prior_param_arrays = []
    for mean, log_scale in prior_params:
        mean_np = mean.cpu().numpy().astype(np.float32)
        log_scale_np = log_scale.cpu().numpy().astype(np.float32)
        prior_param_arrays.append((mean_np, log_scale_np))
    
    return latent_arrays, prior_param_arrays


def save_compressed(
    output_path: str,
    original_shape: tuple,
    pad_h: int,
    pad_w: int,
    num_patches_h: int,
    num_patches_w: int,
    all_latents: list,
    actnorm_state: dict
):
    """
    Save compressed data to a .idf file (legacy zlib format).
    
    File format (version 2):
    - Magic header (4 bytes): 'IDF2'
    - Original height (4 bytes, uint32)
    - Original width (4 bytes, uint32)
    - Padding height (2 bytes, uint16)
    - Padding width (2 bytes, uint16)
    - Num patches height (2 bytes, uint16)
    - Num patches width (2 bytes, uint16)
    - Num latent levels (1 byte, uint8)
    - ActNorm state size (4 bytes, uint32)
    - ActNorm state (pickled dict, zlib compressed)
    - For each patch:
        - For each latent level:
            - Shape info (4 values: B, C, H, W as uint16)
            - Compressed latent data size (4 bytes, uint32)
            - Compressed latent data (float32 array, zlib compressed)
    """
    with open(output_path, 'wb') as f:
        # Write header
        f.write(MAGIC_HEADER)
        f.write(struct.pack('<I', original_shape[0]))  # Original height
        f.write(struct.pack('<I', original_shape[1]))  # Original width
        f.write(struct.pack('<H', pad_h))  # Padding height
        f.write(struct.pack('<H', pad_w))  # Padding width
        f.write(struct.pack('<H', num_patches_h))  # Num patches height
        f.write(struct.pack('<H', num_patches_w))  # Num patches width
        
        # Number of latent levels (from first patch)
        num_levels = len(all_latents[0])
        f.write(struct.pack('<B', num_levels))
        
        # Write ActNorm state (compressed)
        actnorm_bytes = pickle.dumps(actnorm_state)
        actnorm_compressed = zlib.compress(actnorm_bytes, level=9)
        f.write(struct.pack('<I', len(actnorm_compressed)))
        f.write(actnorm_compressed)
        
        # Write each patch's latents
        for patch_latents in all_latents:
            for lat in patch_latents:
                # Shape info
                shape = lat.shape
                for dim in shape:
                    f.write(struct.pack('<H', dim))
                
                # Compress latent data with zlib (float32 data)
                lat_bytes = lat.tobytes()
                compressed = zlib.compress(lat_bytes, level=9)
                
                # Write compressed size and data
                f.write(struct.pack('<I', len(compressed)))
                f.write(compressed)


def save_compressed_arithmetic(
    output_path: str,
    original_shape: tuple,
    pad_h: int,
    pad_w: int,
    num_patches_h: int,
    num_patches_w: int,
    all_latents: list,
    all_prior_params: list,
    actnorm_state: dict,
    symbol_range: int = 1024
):
    """
    Save compressed data using arithmetic coding with learned priors.
    This path is now lossless by storing the quantization residuals.
    
    File format (version 4):
    - Magic header (4 bytes): 'IDF4'
    - Original height (4 bytes, uint32)
    - Original width (4 bytes, uint32)
    - Padding height (2 bytes, uint16)
    - Padding width (2 bytes, uint16)
    - Num patches height (2 bytes, uint16)
    - Num patches width (2 bytes, uint16)
    - Num latent levels (1 byte, uint8)
    - Symbol range (2 bytes, uint16)
    - ActNorm state size (4 bytes, uint32)
    - ActNorm state (pickled dict, zlib compressed)
    - For each patch:
        - For each latent level:
            - Shape info (4 values: B, C, H, W as uint16)
            - Prior params size (4 bytes, uint32)
            - Prior params (mean + log_scale as float32, zlib compressed)
            - Residual size (4 bytes, uint32)
            - Residual (float32 array, zlib compressed)
            - Arithmetic coded data size (4 bytes, uint32)
            - Arithmetic coded latent data
    """
    with open(output_path, 'wb') as f:
        # Write header
        f.write(MAGIC_HEADER_V4)
        f.write(struct.pack('<I', original_shape[0]))  # Original height
        f.write(struct.pack('<I', original_shape[1]))  # Original width
        f.write(struct.pack('<H', pad_h))  # Padding height
        f.write(struct.pack('<H', pad_w))  # Padding width
        f.write(struct.pack('<H', num_patches_h))  # Num patches height
        f.write(struct.pack('<H', num_patches_w))  # Num patches width
        
        # Number of latent levels (from first patch)
        num_levels = len(all_latents[0])
        f.write(struct.pack('<B', num_levels))
        
        # Symbol range for arithmetic coding
        f.write(struct.pack('<H', symbol_range))
        
        # Write ActNorm state (compressed)
        actnorm_bytes = pickle.dumps(actnorm_state)
        actnorm_compressed = zlib.compress(actnorm_bytes, level=9)
        f.write(struct.pack('<I', len(actnorm_compressed)))
        f.write(actnorm_compressed)
        
        # For each patch, encode latents with arithmetic coding
        for patch_idx, (patch_latents, patch_priors) in enumerate(zip(all_latents, all_prior_params)):
            # Encode all latent levels for this patch using arithmetic coding
            compressed_data, _shapes, residuals = fast_encode_latents(
                patch_latents,
                patch_priors,
                symbol_range=symbol_range,
                retain_residuals=True
            )
            
            for level_idx, (lat, (mean, log_scale), arith_data) in enumerate(
                zip(patch_latents, patch_priors, compressed_data)
            ):
                # Shape info
                shape = lat.shape
                for dim in shape:
                    f.write(struct.pack('<H', dim))
                
                # Prior parameters (needed for decoding)
                # Compress mean and log_scale together
                prior_bytes = mean.tobytes() + log_scale.tobytes()
                prior_compressed = zlib.compress(prior_bytes, level=9)
                f.write(struct.pack('<I', len(prior_compressed)))
                f.write(prior_compressed)
                
                # Residual between true latent and rounded latent (for exact reconstruction)
                residual = residuals[level_idx]
                residual_bytes = residual.tobytes()
                residual_compressed = zlib.compress(residual_bytes, level=9)
                f.write(struct.pack('<I', len(residual_compressed)))
                f.write(residual_compressed)
                
                # Arithmetic coded data
                f.write(struct.pack('<I', len(arith_data)))
                f.write(arith_data)


def compress_image(
    input_path: str,
    output_path: str,
    model_path: str,
    device: torch.device,
    use_arithmetic: bool = True,
    symbol_range: int = 1024
):
    """
    Main compression function.
    
    Args:
        input_path: Path to input image
        output_path: Path for compressed output
        model_path: Path to model checkpoint
        device: Torch device
        use_arithmetic: Use arithmetic coding (True) or legacy zlib (False)
        symbol_range: Symbol range for arithmetic coding
    """
    # Load model
    model = load_model(model_path, device)
    
    # Load and prepare image
    print(f"Loading image: {input_path}")
    image = load_image(input_path)
    print(f"  Original size: {image.shape[1]}x{image.shape[0]} (WxH)")
    
    # Pad image
    padded, original_shape, pad_h, pad_w = pad_image(image, PATCH_SIZE)
    print(f"  Padded size: {padded.shape[1]}x{padded.shape[0]} (WxH)")
    
    # Split into patches
    patches = split_into_patches(padded, PATCH_SIZE)
    num_patches_h = padded.shape[0] // PATCH_SIZE
    num_patches_w = padded.shape[1] // PATCH_SIZE
    print(f"  Patches: {num_patches_h}x{num_patches_w} = {len(patches)} total")
    
    # Compress each patch
    # Note: First patch will trigger ActNorm initialization
    compression_mode = "arithmetic coding" if use_arithmetic else "zlib"
    print(f"Compressing patches using {compression_mode}...")
    all_latents = []
    all_prior_params = []
    for i, patch in enumerate(patches):
        latents, prior_params = compress_patch(model, patch, device)
        all_latents.append(latents)
        all_prior_params.append(prior_params)
        
        if (i + 1) % 10 == 0 or i == len(patches) - 1:
            print(f"  Processed {i + 1}/{len(patches)} patches")
    
    # Get ActNorm state after compression (initialized by first patch)
    actnorm_state = get_actnorm_state(model)
    
    # Save compressed file
    print(f"Saving compressed file: {output_path}")
    if use_arithmetic:
        save_compressed_arithmetic(
            output_path,
            original_shape,
            pad_h,
            pad_w,
            num_patches_h,
            num_patches_w,
            all_latents,
            all_prior_params,
            actnorm_state,
            symbol_range=symbol_range
        )
    else:
        save_compressed(
            output_path,
            original_shape,
            pad_h,
            pad_w,
            num_patches_h,
            num_patches_w,
            all_latents,
            actnorm_state
        )
    
    # Report compression stats
    original_size = image.shape[0] * image.shape[1] * image.shape[2]
    compressed_size = Path(output_path).stat().st_size
    ratio = original_size / compressed_size
    bpp = (compressed_size * 8) / (image.shape[0] * image.shape[1])
    
    print("\nCompression Statistics:")
    print(f"  Mode: {compression_mode}")
    print(f"  Original size: {original_size:,} bytes ({original_size / 1024:.1f} KB)")
    print(f"  Compressed size: {compressed_size:,} bytes ({compressed_size / 1024:.1f} KB)")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Bits per pixel: {bpp:.3f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compress an image using IDF model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/compress.py photo.png photo.idf
    python scripts/compress.py photo.jpg photo.idf --model best_model.pt
    python scripts/compress.py photo.png photo.idf --no-arithmetic  # Use legacy zlib mode
        """
    )
    
    parser.add_argument("input", type=str, help="Input image path (PNG, JPG, etc.)")
    parser.add_argument("output", type=str, help="Output compressed file path (.idf)")
    parser.add_argument(
        "--model", type=str, default="best_model.pt",
        help="Path to model checkpoint (default: best_model.pt)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--no-arithmetic", action="store_true",
        help="Use legacy zlib compression instead of arithmetic coding"
    )
    parser.add_argument(
        "--symbol-range", type=int, default=1024,
        help="Symbol range for arithmetic coding (default: 1024)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run compression
    compress_image(
        str(input_path), 
        str(output_path), 
        str(model_path), 
        device,
        use_arithmetic=not args.no_arithmetic,
        symbol_range=args.symbol_range
    )
    print("\nDone!")


if __name__ == "__main__":
    main()

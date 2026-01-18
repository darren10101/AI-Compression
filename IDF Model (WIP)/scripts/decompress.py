"""
Decompress a .idf file back to an image using the IDF model.

Reads the compressed latent representation, decompresses each patch,
and reassembles the original image.

Supports three file formats:
- IDF3: Arithmetic coding with learned priors (best compression)
- IDF2: Legacy zlib with ActNorm state
- IDF1: Legacy zlib without ActNorm state

Usage:
    python scripts/decompress.py input.idf output.png
    python scripts/decompress.py input.idf output.png --model checkpoints/best_model.pt
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import struct
import zlib
import pickle

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_idf_model
from src.arithmetic_coding import fast_decode_latents


PATCH_SIZE = 256
MAGIC_HEADER_V1 = b'IDF1'
MAGIC_HEADER_V2 = b'IDF2'
MAGIC_HEADER_V3 = b'IDF3'
MAGIC_HEADER_V4 = b'IDF4'


def load_model(checkpoint_path: str, device: torch.device):
    """Load the IDF model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = create_idf_model()
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).float()
    model.eval()
    
    return model


def set_actnorm_state(model, actnorm_state: dict, device: torch.device):
    """Load ActNorm scale/bias parameters into model."""
    for block_idx, block in enumerate(model.blocks):
        for flow_idx, flow in enumerate(block.flows):
            key = f"block{block_idx}_flow{flow_idx}"
            if key in actnorm_state:
                flow.actnorm.scale.data = torch.from_numpy(
                    actnorm_state[key]['scale']
                ).to(device)
                flow.actnorm.bias.data = torch.from_numpy(
                    actnorm_state[key]['bias']
                ).to(device)
                flow.actnorm.initialized = True


def load_compressed(input_path: str) -> dict:
    """
    Load compressed data from a .idf file.
    
    Returns:
        Dictionary with:
        - version: File format version (1, 2, 3, or 4)
        - original_shape: (H, W)
        - pad_h, pad_w: Padding amounts
        - num_patches_h, num_patches_w: Grid dimensions
        - all_latents: List of latent arrays for each patch
        - all_prior_params: List of prior params for each patch (v3/v4)
        - actnorm_state: ActNorm parameters (for v2/v3/v4 format)
        - symbol_range: Symbol range for arithmetic coding (v3/v4)
    """
    with open(input_path, 'rb') as f:
        # Read and verify header
        magic = f.read(4)
        if magic == MAGIC_HEADER_V4:
            version = 4
        elif magic == MAGIC_HEADER_V3:
            version = 3
        elif magic == MAGIC_HEADER_V2:
            version = 2
        elif magic == MAGIC_HEADER_V1:
            version = 1
        else:
            raise ValueError(f"Invalid file format. Expected IDF1, IDF2, IDF3, or IDF4, got {magic}")
        
        # Read metadata
        original_h = struct.unpack('<I', f.read(4))[0]
        original_w = struct.unpack('<I', f.read(4))[0]
        pad_h = struct.unpack('<H', f.read(2))[0]
        pad_w = struct.unpack('<H', f.read(2))[0]
        num_patches_h = struct.unpack('<H', f.read(2))[0]
        num_patches_w = struct.unpack('<H', f.read(2))[0]
        num_levels = struct.unpack('<B', f.read(1))[0]
        
        # Read symbol range (v3/v4 only)
        symbol_range = 1024  # default
        if version >= 3:
            symbol_range = struct.unpack('<H', f.read(2))[0]
        
        # Read ActNorm state (v2 and v3)
        actnorm_state = None
        if version >= 2:
            actnorm_size = struct.unpack('<I', f.read(4))[0]
            actnorm_compressed = f.read(actnorm_size)
            actnorm_bytes = zlib.decompress(actnorm_compressed)
            actnorm_state = pickle.loads(actnorm_bytes)
        
        total_patches = num_patches_h * num_patches_w
        
        # Read each patch's latents
        all_latents = []
        all_prior_params = []
        
        if version in (3, 4):
            # Arithmetic coded format
            for _ in range(total_patches):
                patch_latents = []
                patch_priors = []
                patch_compressed = []
                patch_shapes = []
                patch_residuals = [] if version == 4 else None
                
                for _ in range(num_levels):
                    # Read shape (4 dimensions)
                    shape = tuple(struct.unpack('<H', f.read(2))[0] for _ in range(4))
                    patch_shapes.append(shape)
                    
                    # Read prior parameters
                    prior_size = struct.unpack('<I', f.read(4))[0]
                    prior_compressed = f.read(prior_size)
                    prior_bytes = zlib.decompress(prior_compressed)
                    
                    # Split into mean and log_scale
                    num_elements = int(np.prod(shape))
                    mean = np.frombuffer(
                        prior_bytes[:num_elements * 4], dtype=np.float32
                    ).reshape(shape).copy()
                    log_scale = np.frombuffer(
                        prior_bytes[num_elements * 4:], dtype=np.float32
                    ).reshape(shape).copy()
                    patch_priors.append((mean, log_scale))
                    
                    # Read residuals for lossless reconstruction (v4)
                    if version == 4:
                        residual_size = struct.unpack('<I', f.read(4))[0]
                        residual_bytes = f.read(residual_size)
                        residual = np.frombuffer(
                            zlib.decompress(residual_bytes), dtype=np.float32
                        ).reshape(shape).copy()
                        patch_residuals.append(residual)
                    
                    # Read arithmetic coded data
                    arith_size = struct.unpack('<I', f.read(4))[0]
                    arith_data = f.read(arith_size)
                    patch_compressed.append(arith_data)
                
                # Decode latents using arithmetic decoding
                decoded_latents = fast_decode_latents(
                    patch_compressed, patch_shapes, patch_priors, 
                    symbol_range=symbol_range
                )
                
                # Restore exact latents using stored residuals (v4)
                if version == 4 and patch_residuals is not None:
                    decoded_latents = [
                        lat + res for lat, res in zip(decoded_latents, patch_residuals)
                    ]
                
                all_latents.append(decoded_latents)
                all_prior_params.append(patch_priors)
        else:
            # Legacy zlib format (v1, v2)
            for _ in range(total_patches):
                patch_latents = []
                for _ in range(num_levels):
                    # Read shape (4 dimensions)
                    shape = tuple(struct.unpack('<H', f.read(2))[0] for _ in range(4))
                    
                    # Read compressed data size
                    compressed_size = struct.unpack('<I', f.read(4))[0]
                    
                    # Read and decompress data
                    compressed_data = f.read(compressed_size)
                    lat_bytes = zlib.decompress(compressed_data)
                    
                    # Reconstruct numpy array (float32 data)
                    lat = np.frombuffer(lat_bytes, dtype=np.float32).reshape(shape).copy()
                    patch_latents.append(lat)
                
                all_latents.append(patch_latents)
    
    return {
        'version': version,
        'original_shape': (original_h, original_w),
        'pad_h': pad_h,
        'pad_w': pad_w,
        'num_patches_h': num_patches_h,
        'num_patches_w': num_patches_w,
        'all_latents': all_latents,
        'all_prior_params': all_prior_params if version >= 3 else None,
        'actnorm_state': actnorm_state,
        'symbol_range': symbol_range
    }


def decompress_patch(model, latent_arrays: list, device: torch.device) -> np.ndarray:
    """
    Decompress a single patch using the IDF model.
    
    Args:
        model: IDF model
        latent_arrays: List of numpy float32 arrays (latents)
        device: Torch device
    
    Returns:
        Decompressed patch as numpy array (H, W, 3), uint8
    """
    # Convert latents to tensors
    latents = []
    for lat_np in latent_arrays:
        lat = torch.from_numpy(lat_np).to(device)
        latents.append(lat)
    
    with torch.no_grad():
        x = model.decompress(latents)
    
    # Convert back to numpy: (1, 3, H, W) -> (H, W, 3)
    x = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Round, clip to valid range, and convert to uint8
    x = np.clip(np.round(x), 0, 255).astype(np.uint8)
    
    return x


def reassemble_patches(
    patches: list,
    num_patches_h: int,
    num_patches_w: int,
    patch_size: int
) -> np.ndarray:
    """Reassemble patches into a full image."""
    h = num_patches_h * patch_size
    w = num_patches_w * patch_size
    
    image = np.zeros((h, w, 3), dtype=np.uint8)
    
    idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start = i * patch_size
            x_start = j * patch_size
            image[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches[idx]
            idx += 1
    
    return image


def remove_padding(image: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Remove padding to restore original dimensions."""
    h, w = original_shape
    return image[:h, :w, :]


def save_image(image: np.ndarray, output_path: str):
    """Save numpy array as image."""
    img = Image.fromarray(image)
    img.save(output_path)


def decompress_image(
    input_path: str,
    output_path: str,
    model_path: str,
    device: torch.device
):
    """Main decompression function."""
    # Load model
    model = load_model(model_path, device)
    
    # Load compressed file
    print(f"Loading compressed file: {input_path}")
    data = load_compressed(input_path)
    
    original_shape = data['original_shape']
    num_patches_h = data['num_patches_h']
    num_patches_w = data['num_patches_w']
    all_latents = data['all_latents']
    actnorm_state = data['actnorm_state']
    
    version = data['version']
    version_names = {
        1: "IDF1 (zlib)",
        2: "IDF2 (zlib + ActNorm)",
        3: "IDF3 (arithmetic coding, lossy latents)",
        4: "IDF4 (arithmetic coding + residuals, lossless)",
    }
    print(f"  File version: {version_names.get(version, f'v{version}')}")
    print(f"  Original size: {original_shape[1]}x{original_shape[0]} (WxH)")
    print(f"  Patches: {num_patches_h}x{num_patches_w} = {len(all_latents)} total")
    
    # Load ActNorm state if available (v2 format)
    if actnorm_state is not None:
        print("  Loading ActNorm state from file...")
        set_actnorm_state(model, actnorm_state, device)
    else:
        # For v1 files, mark ActNorm as initialized (use checkpoint params)
        print("  Using ActNorm params from model checkpoint (v1 format)...")
        for block in model.blocks:
            for flow in block.flows:
                flow.actnorm.initialized = True
    
    # Decompress each patch
    print("Decompressing patches...")
    patches = []
    for i, patch_latents in enumerate(all_latents):
        patch = decompress_patch(model, patch_latents, device)
        patches.append(patch)
        
        if (i + 1) % 10 == 0 or i == len(all_latents) - 1:
            print(f"  Processed {i + 1}/{len(all_latents)} patches")
    
    # Reassemble image
    print("Reassembling image...")
    full_image = reassemble_patches(patches, num_patches_h, num_patches_w, PATCH_SIZE)
    
    # Remove padding
    final_image = remove_padding(full_image, original_shape)
    print(f"  Final size: {final_image.shape[1]}x{final_image.shape[0]} (WxH)")
    
    # Save output
    print(f"Saving image: {output_path}")
    save_image(final_image, output_path)
    
    # Report stats
    compressed_size = Path(input_path).stat().st_size
    output_size = Path(output_path).stat().st_size
    
    print("\nDecompression Statistics:")
    print(f"  Compressed size: {compressed_size:,} bytes ({compressed_size / 1024:.1f} KB)")
    print(f"  Output image size: {output_size:,} bytes ({output_size / 1024:.1f} KB)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Decompress an IDF file back to an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/decompress.py photo.idf photo_restored.png
    python scripts/decompress.py photo.idf photo_restored.png --model best_model.pt
        """
    )
    
    parser.add_argument("input", type=str, help="Input compressed file path (.idf)")
    parser.add_argument("output", type=str, help="Output image path (PNG recommended for lossless)")
    parser.add_argument(
        "--model", type=str, default="best_model.pt",
        help="Path to model checkpoint (default: best_model.pt)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
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
    
    # Run decompression
    decompress_image(str(input_path), str(output_path), str(model_path), device)
    print("\nDone!")


if __name__ == "__main__":
    main()

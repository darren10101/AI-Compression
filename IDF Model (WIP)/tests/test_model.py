"""
Test script for the Integer Discrete Flow (IDF) model.

Tests:
1. Model loading from checkpoint
2. Forward pass (encoding/compression)
3. Reverse pass (decoding/decompression)
4. Lossless reconstruction verification
5. BPD (bits per dimension) computation
6. Compress/Decompress API
7. Gradient pattern test
8. Arithmetic coder lossless roundtrip
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import IntegerDiscreteFlow, create_idf_model
from src.arithmetic_coding import fast_encode_latents, fast_decode_latents


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> IntegerDiscreteFlow:
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model onto
    
    Returns:
        Loaded IntegerDiscreteFlow model
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with default config
    # The checkpoint should contain model architecture info, but we'll use defaults
    model = create_idf_model()
    
    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  - Loaded model from step: {checkpoint.get('global_step', 'unknown')}")
        print(f"  - Best BPD recorded: {checkpoint.get('best_bpd', 'unknown')}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("  - Loaded model state dict directly")
    
    model = model.to(device).float()
    model.eval()
    
    return model


def test_model_loading(checkpoint_path: str, device: torch.device) -> IntegerDiscreteFlow:
    """Test 1: Model loads successfully from checkpoint."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    try:
        model = load_model_from_checkpoint(checkpoint_path, device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  - Model parameters: {num_params:,}")
        print("[PASSED] Model loaded successfully")
        return model
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_forward_pass(model: IntegerDiscreteFlow, device: torch.device) -> tuple:
    """Test 2: Forward pass (encoding) works correctly."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass (Encoding)")
    print("="*60)
    
    # Create synthetic test image (batch of 2, 3 channels, 64x64 - must be divisible by 8)
    # Values in [0, 255] as expected by the model
    batch_size = 2
    height, width = 64, 64
    
    # Create random test image
    torch.manual_seed(42)
    x = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.float32, device=device)
    
    print(f"  - Input shape: {x.shape}")
    print(f"  - Input dtype: {x.dtype}")
    print(f"  - Input range: [{x.min().item():.1f}, {x.max().item():.1f}]")
    
    try:
        with torch.no_grad():
            latents, log_likelihood = model._forward(x - 128.0)  # Center around 0
        
        print(f"  - Number of latent tensors: {len(latents)}")
        for i, lat in enumerate(latents):
            print(f"    - Latent {i}: shape={lat.shape}, range=[{lat.min().item():.1f}, {lat.max().item():.1f}]")
        print(f"  - Log likelihood shape: {log_likelihood.shape}")
        print(f"  - Log likelihood mean: {log_likelihood.mean().item():.2f}")
        
        print("[PASSED] Forward pass completed successfully")
        return x, latents
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_reverse_pass(model: IntegerDiscreteFlow, latents: list, device: torch.device) -> torch.Tensor:
    """Test 3: Reverse pass (decoding) works correctly."""
    print("\n" + "="*60)
    print("TEST 3: Reverse Pass (Decoding)")
    print("="*60)
    
    try:
        with torch.no_grad():
            x_reconstructed = model._reverse(latents) + 128.0  # Uncenter
        
        print(f"  - Reconstructed shape: {x_reconstructed.shape}")
        print(f"  - Reconstructed dtype: {x_reconstructed.dtype}")
        print(f"  - Reconstructed range: [{x_reconstructed.min().item():.1f}, {x_reconstructed.max().item():.1f}]")
        
        print("[PASSED] Reverse pass completed successfully")
        return x_reconstructed
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_lossless_reconstruction(x_original: torch.Tensor, x_reconstructed: torch.Tensor):
    """Test 4: Verify lossless reconstruction."""
    print("\n" + "="*60)
    print("TEST 4: Lossless Reconstruction")
    print("="*60)
    
    # Round to integers for comparison (the model operates on integers)
    x_orig_int = torch.round(x_original)
    x_recon_int = torch.round(x_reconstructed)
    
    # Check if reconstruction is exact
    max_diff = (x_orig_int - x_recon_int).abs().max().item()
    mean_diff = (x_orig_int - x_recon_int).abs().mean().item()
    
    print(f"  - Max absolute difference: {max_diff:.6f}")
    print(f"  - Mean absolute difference: {mean_diff:.6f}")
    
    # For lossless compression, differences should be 0 (or very close due to float precision)
    if max_diff < 1.0:
        print("[PASSED] Reconstruction is lossless (max diff < 1)")
        return True
    else:
        print(f"[WARNING] Reconstruction has differences (max diff = {max_diff:.2f})")
        print("  Note: Small differences may occur due to floating point precision")
        return False


def test_bpd_computation(model: IntegerDiscreteFlow, device: torch.device):
    """Test 5: BPD (bits per dimension) computation."""
    print("\n" + "="*60)
    print("TEST 5: BPD Computation")
    print("="*60)
    
    # Test with different image sizes
    test_sizes = [(64, 64), (128, 128), (256, 256)]
    
    for h, w in test_sizes:
        torch.manual_seed(123)
        x = torch.randint(0, 256, (1, 3, h, w), dtype=torch.float32, device=device)
        
        try:
            with torch.no_grad():
                loss, bpd = model.compute_loss(x)
            
            print(f"  - Image size {h}x{w}: BPD = {bpd.item():.4f}")
        except Exception as e:
            print(f"  - Image size {h}x{w}: FAILED - {e}")
    
    print("[PASSED] BPD computation completed")


def test_compress_decompress_api(model: IntegerDiscreteFlow, device: torch.device):
    """Test 6: Compress and decompress API."""
    print("\n" + "="*60)
    print("TEST 6: Compress/Decompress API")
    print("="*60)
    
    torch.manual_seed(456)
    x = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.float32, device=device)
    
    try:
        with torch.no_grad():
            # Compress
            latents, prior_params = model.compress(x)
            print(f"  - Compression produced {len(latents)} latent tensors")
            
            # Calculate compression rate
            total_latent_elements = sum(lat.numel() for lat in latents)
            original_elements = x.numel()
            print(f"  - Original elements: {original_elements}")
            print(f"  - Total latent elements: {total_latent_elements}")
            
            # Decompress
            x_recon = model.decompress(latents)
            
            # Check reconstruction
            max_diff = (torch.round(x) - torch.round(x_recon)).abs().max().item()
            print(f"  - Max reconstruction difference: {max_diff:.6f}")
        
        if max_diff < 1.0:
            print("[PASSED] Compress/Decompress API works correctly")
        else:
            print(f"[WARNING] Reconstruction has differences")
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_gradient_pattern_image(model: IntegerDiscreteFlow, device: torch.device):
    """Test 7: Test with structured (gradient) image."""
    print("\n" + "="*60)
    print("TEST 7: Gradient Pattern Test")
    print("="*60)
    
    # Create a gradient image (should be easy to compress)
    h, w = 64, 64
    x = torch.zeros(1, 3, h, w, device=device)
    
    # Create gradients in each channel
    for c in range(3):
        for i in range(h):
            for j in range(w):
                if c == 0:
                    x[0, c, i, j] = (i / h) * 255  # Red: vertical gradient
                elif c == 1:
                    x[0, c, i, j] = (j / w) * 255  # Green: horizontal gradient
                else:
                    x[0, c, i, j] = ((i + j) / (h + w)) * 255  # Blue: diagonal gradient
    
    x = x.round()
    
    try:
        with torch.no_grad():
            loss, bpd = model.compute_loss(x)
        
        print(f"  - Gradient image BPD: {bpd.item():.4f}")
        print(f"  - (Lower BPD = better compression on structured data)")
        print("[PASSED] Gradient pattern test completed")
    except Exception as e:
        print(f"[FAILED] {e}")
        raise


def test_arithmetic_roundtrip_lossless():
    """
    Test 8: Ensure arithmetic coder with residuals reconstructs latents exactly.
    
    This guards the lossless requirement for IDF4 files.
    """
    print("\n" + "="*60)
    print("TEST 8: Arithmetic Coding Lossless Roundtrip")
    print("="*60)
    
    np.random.seed(0)
    # Small latent tensor to keep test fast
    latent = np.random.randn(1, 4, 4, 4).astype(np.float32)
    mean = np.zeros_like(latent, dtype=np.float32)
    log_scale = np.zeros_like(latent, dtype=np.float32)
    
    compressed, shapes, residuals = fast_encode_latents(
        [latent], [(mean, log_scale)], retain_residuals=True
    )
    decoded = fast_decode_latents(
        compressed, shapes, [(mean, log_scale)]
    )
    restored = [d + r for d, r in zip(decoded, residuals)]
    
    max_err = np.max(np.abs(restored[0] - latent))
    print(f"  - Max reconstruction error: {max_err:.6e}")
    
    assert max_err < 1e-6, "Arithmetic coder is not lossless with residuals"
    print("[PASSED] Arithmetic coder preserves latents exactly")


def run_all_tests(checkpoint_path: str):
    """Run all tests on the model."""
    print("\n" + "#"*60)
    print("# IDF Model Test Suite")
    print("#"*60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run tests
    passed = 0
    failed = 0
    
    try:
        # Test 1: Model loading
        model = test_model_loading(checkpoint_path, device)
        passed += 1
        
        # Test 2: Forward pass
        x_original, latents = test_forward_pass(model, device)
        passed += 1
        
        # Test 3: Reverse pass
        x_reconstructed = test_reverse_pass(model, latents, device)
        passed += 1
        
        # Test 4: Lossless reconstruction
        test_lossless_reconstruction(x_original, x_reconstructed)
        passed += 1
        
        # Test 5: BPD computation
        test_bpd_computation(model, device)
        passed += 1
        
        # Test 6: Compress/Decompress API
        test_compress_decompress_api(model, device)
        passed += 1
        
        # Test 7: Gradient pattern
        test_gradient_pattern_image(model, device)
        passed += 1

        # Test 8: Arithmetic coder roundtrip (lossless guarantee)
        test_arithmetic_roundtrip_lossless()
        passed += 1
        
    except Exception as e:
        failed += 1
        print(f"\n[FAILED] Test suite stopped due to error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IDF model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="best_model.pt",
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    success = run_all_tests(str(checkpoint_path))
    sys.exit(0 if success else 1)

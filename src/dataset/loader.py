import torch
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from pathlib import Path


def get_hf_cache_dir():
    """
    Get the HuggingFace cache directory location.
    
    Returns:
        Path to HuggingFace cache directory
    """
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_dir:
        return Path(cache_dir) / "datasets"
    
    # Default locations
    if os.name == "nt":  # Windows
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    else:  # Linux/Mac
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
    
    return cache_dir


def load_imagenet_stream(split="train"):
    """
    Load ImageNet-1k dataset from HuggingFace and create a stream.
    
    Args:
        split: Dataset split to load ('train', 'validation', 'test')
    
    Returns:
        Dataset stream (iterable dataset)
    
    Note:
        Streaming mode downloads files on-the-fly and doesn't store the full dataset.
        Cache location: {get_hf_cache_dir()}
    """
    cache_dir = get_hf_cache_dir()
    print(f"Loading ILSVRC/imagenet-1k {split} split...")
    print(f"HuggingFace cache directory: {cache_dir}")
    
    dataset = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
    print("✓ Dataset loaded in streaming mode (files downloaded on-demand)")
    return dataset


def display_image(image, title="Image"):
    """
    Display an image using matplotlib.
    
    Args:
        image: PIL Image or numpy array
        title: Title for the plot
    """
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        if image.dim() == 4:  # Batch dimension
            image = image[0]
        if image.dim() == 3 and image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0)  # Convert to HWC
        image = image.cpu().numpy()
        # Denormalize if needed (assuming ImageNet normalization)
        if image.min() < 0:
            image = (image + 1) / 2
        image = np.clip(image, 0, 1)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def image_to_tensor(image, normalize=True):
    """
    Convert PIL Image to PyTorch tensor.
    
    Args:
        image: PIL Image
        normalize: If True, normalize to [0, 1] range
    
    Returns:
        PyTorch tensor in CHW format
    """
    # Convert PIL Image to tensor
    tensor = torch.from_numpy(np.array(image)).float()
    
    # Handle different image formats
    if tensor.dim() == 2:  # Grayscale
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    elif tensor.dim() == 3:  # RGB
        tensor = tensor.permute(2, 0, 1)  # Convert HWC to CHW
    
    # Normalize to [0, 1] if requested
    if normalize:
        tensor = tensor / 255.0
    
    return tensor


if __name__ == "__main__":
    print("Loading ILSVRC/imagenet-1k dataset in streaming mode...")
    dataset = load_imagenet_stream(split="train")
    print("\nGetting first sample from stream...")
    first_sample = next(iter(dataset))
    print(f"First sample keys: {first_sample.keys()}")
    
    if 'image' in first_sample:
        img_data = first_sample['image']
        if isinstance(img_data, Image.Image):
            image = img_data
        elif isinstance(img_data, bytes):
            image = Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            image = Image.open(io.BytesIO(img_data['bytes']))
        else:
            image = img_data
    else:
        raise ValueError(f"Could not find 'image' key in sample. Available keys: {first_sample.keys()}")
    
    print(f"\nImage type: {type(image)}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Display the first image
    print("\nDisplaying first image...")
    display_image(image, title="First ImageNet-1k Image")
    
    # Convert to tensor
    print("\nConverting image to tensor...")
    tensor = image_to_tensor(image, normalize=True)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")
    print(f"Tensor mean: {tensor.mean().item():.4f}")
    
    # Show label if available
    if 'label' in first_sample:
        print(f"\nLabel: {first_sample['label']}")
    elif 'labels' in first_sample:
        print(f"\nLabels: {first_sample['labels']}")

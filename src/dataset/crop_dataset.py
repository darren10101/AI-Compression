import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from huggingface_hub import HfFolder
from PIL import Image
import numpy as np
import io
from typing import Optional, Tuple


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from stored credentials."""
    try:
        return HfFolder.get_token()
    except Exception:
        return None


class RandomCropDataset(IterableDataset):
    """
    Iterable dataset that streams ImageNet images and extracts random 64x64 crops.
    
    For images smaller than 64x64, pads with zeros on the right/bottom.
    For images larger than 64x64, randomly crops a 64x64 region.
    
    Args:
        split: Dataset split ('train', 'validation')
        crop_size: Size of random crop (default 64)
        buffer_size: Number of samples to shuffle in buffer (for streaming randomness)
    """
    def __init__(
        self,
        split: str = "train",
        crop_size: int = 64,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.split = split
        self.crop_size = crop_size
        self.buffer_size = buffer_size
        self._dataset = None  # Cache the dataset connection
        
    def _load_dataset(self):
        """Load the streaming dataset (cached to avoid reconnecting each iteration)."""
        if self._dataset is None:
            # Get token for gated dataset access (ImageNet-1k requires authentication)
            token = _get_hf_token()
            self._dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split=self.split,
                streaming=True,
                trust_remote_code=True,
                token=token,
            )
        # Return a shuffled view - this creates a new iterator but reuses the connection
        return self._dataset.shuffle(buffer_size=self.buffer_size, seed=None)
    
    def _extract_image(self, sample) -> Optional[Image.Image]:
        """Extract PIL Image from sample dict."""
        if 'image' not in sample:
            return None
            
        img_data = sample['image']
        
        if isinstance(img_data, Image.Image):
            return img_data
        elif isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        elif isinstance(img_data, dict) and 'bytes' in img_data:
            return Image.open(io.BytesIO(img_data['bytes']))
        else:
            return img_data
    
    def _random_crop_with_padding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract a random 64x64 crop from image.
        If image is smaller than 64x64, pad with zeros on right/bottom.
        
        Args:
            image: PIL Image (any size)
            
        Returns:
            Tensor of shape (3, crop_size, crop_size) with values in [0, 255]
        """
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        crop_size = self.crop_size
        
        # Convert to numpy array (H, W, C)
        img_array = np.array(image, dtype=np.float32)
        
        # Case 1: Image is smaller than crop_size in one or both dimensions
        # Pad first, then we can treat it uniformly
        if height < crop_size or width < crop_size:
            # Pad to at least crop_size in both dimensions
            pad_h = max(0, crop_size - height)
            pad_w = max(0, crop_size - width)
            
            # Pad on bottom and right
            img_array = np.pad(
                img_array,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            height, width = img_array.shape[:2]
        
        # Case 2: Now image is >= crop_size, do random crop
        if height == crop_size and width == crop_size:
            # Already exact size
            top, left = 0, 0
        else:
            # Random crop position
            top = np.random.randint(0, height - crop_size + 1)
            left = np.random.randint(0, width - crop_size + 1)
        
        # Extract crop
        crop = img_array[top:top + crop_size, left:left + crop_size, :]
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(crop).permute(2, 0, 1)
        
        return tensor
    
    def __iter__(self):
        """Iterate over dataset, yielding random 64x64 crops."""
        dataset = self._load_dataset()
        
        for sample in dataset:
            image = self._extract_image(sample)
            if image is None:
                continue
                
            try:
                tensor = self._random_crop_with_padding(image)
                yield tensor
            except Exception as e:
                # Skip problematic images
                continue


def create_dataloader(
    split: str = "train",
    crop_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    buffer_size: int = 1000,
) -> DataLoader:
    """
    Create a DataLoader for training the IDF model.
    
    Args:
        split: Dataset split ('train', 'validation')
        crop_size: Size of random crop (default 64)
        batch_size: Batch size
        num_workers: Number of worker processes
        buffer_size: Shuffle buffer size for streaming
    
    Returns:
        DataLoader yielding batches of shape (B, 3, crop_size, crop_size)
    """
    dataset = RandomCropDataset(
        split=split,
        crop_size=crop_size,
        buffer_size=buffer_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test the dataset
    print("Testing RandomCropDataset...")
    
    dataset = RandomCropDataset(split="train", crop_size=64)
    
    # Get a few samples
    count = 0
    for tensor in dataset:
        print(f"Sample {count + 1}: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  min={tensor.min().item():.1f}, max={tensor.max().item():.1f}")
        count += 1
        if count >= 5:
            break
    
    print("\nTesting DataLoader...")
    loader = create_dataloader(split="train", batch_size=8, num_workers=0)
    
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Batch min: {batch.min().item():.1f}, max: {batch.max().item():.1f}")

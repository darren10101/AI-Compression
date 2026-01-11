"""
Random crop dataset for streaming ImageNet data.

Produces 64×64 (or configurable) square cutouts from streamed images,
converted to PyTorch tensors for training the IDF compression model.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
import random
from typing import Iterator, Optional
from datasets import load_dataset


class RandomCropDataset(IterableDataset):
    """
    Iterable dataset that streams images from HuggingFace datasets
    and yields random square crops as tensors.
    
    Uses streaming mode to avoid downloading the entire dataset,
    making it suitable for large datasets like ImageNet-1k.
    
    Args:
        dataset_name: HuggingFace dataset identifier (default: ImageNet-1k)
        split: Dataset split ('train' or 'validation')
        crop_size: Size of square crops to extract (default: 64)
        buffer_size: Size of shuffle buffer for randomization (default: 1000)
        min_image_size: Minimum image dimension to accept (default: crop_size)
        image_key: Key for image column in dataset (default: 'image')
    """
    
    def __init__(
        self,
        dataset_name: str = "ILSVRC/imagenet-1k",
        split: str = "train",
        crop_size: int = 64,
        buffer_size: int = 1000,
        min_image_size: Optional[int] = None,
        image_key: str = "image",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.crop_size = crop_size
        self.buffer_size = buffer_size
        self.min_image_size = min_image_size or crop_size
        self.image_key = image_key
        
        # Dataset will be loaded lazily in __iter__
        self._dataset = None
    
    def _load_dataset(self):
        """Load the streaming dataset."""
        return load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
    
    def _random_crop(self, image: Image.Image) -> torch.Tensor:
        """
        Extract a random square crop from an image.
        
        Args:
            image: PIL Image to crop
            
        Returns:
            Tensor of shape (3, crop_size, crop_size) with values in [0, 255]
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        width, height = image.size
        
        # Random crop position
        max_x = width - self.crop_size
        max_y = height - self.crop_size
        
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))
        
        # Crop the image
        crop = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        
        # Convert to tensor (C, H, W) with float values in [0, 255]
        tensor = torch.from_numpy(
            __import__('numpy').array(crop, dtype='float32')
        ).permute(2, 0, 1)
        
        return tensor
    
    def _is_valid_image(self, image: Image.Image) -> bool:
        """Check if image is large enough for cropping."""
        if image is None:
            return False
        width, height = image.size
        return width >= self.min_image_size and height >= self.min_image_size
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Iterate over the dataset, yielding random crops as tensors.
        
        Uses a shuffle buffer for better randomization of the stream.
        """
        # Load dataset fresh for each iterator (supports multiple workers)
        dataset = self._load_dataset()
        
        # Apply shuffling if buffer size > 1
        if self.buffer_size > 1:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        
        for sample in dataset:
            try:
                image = sample[self.image_key]
                
                # Skip images that are too small
                if not self._is_valid_image(image):
                    continue
                
                # Yield random crop as tensor
                yield self._random_crop(image)
                
            except Exception as e:
                # Skip problematic images
                continue
    
    def take(self, n: int) -> Iterator[torch.Tensor]:
        """Take first n samples from the dataset."""
        count = 0
        for sample in self:
            if count >= n:
                break
            yield sample
            count += 1


def create_dataloader(
    split: str = "train",
    dataset_name: str = "ILSVRC/imagenet-1k",
    crop_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 0,
    buffer_size: int = 1000,
    pin_memory: bool = True,
    image_key: str = "image",
) -> DataLoader:
    """
    Create a DataLoader for streaming random crops.
    
    Args:
        split: Dataset split ('train' or 'validation')
        dataset_name: HuggingFace dataset identifier
        crop_size: Size of square crops (default: 64)
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes (0 for streaming recommended)
        buffer_size: Shuffle buffer size (larger = better shuffle, slower start)
        pin_memory: Whether to pin memory for faster GPU transfer
        image_key: Key for image column in dataset
    
    Returns:
        DataLoader yielding batches of shape (B, 3, crop_size, crop_size)
    
    Note:
        For streaming datasets, num_workers=0 is often required to avoid
        issues with multiprocessing and IterableDataset.
    """
    dataset = RandomCropDataset(
        dataset_name=dataset_name,
        split=split,
        crop_size=crop_size,
        buffer_size=buffer_size,
        image_key=image_key,
    )
    
    # Determine if we should use pin_memory
    use_pin_memory = pin_memory and torch.cuda.is_available()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,  # Ensure consistent batch sizes
    )


# Utility functions that may be imported from this module
def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a tensor.
    
    Args:
        image: PIL Image (any mode, will be converted to RGB)
    
    Returns:
        Tensor of shape (3, H, W) with float values in [0, 255]
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    import numpy as np
    return torch.from_numpy(
        np.array(image, dtype='float32')
    ).permute(2, 0, 1)


def display_image(tensor: torch.Tensor):
    """
    Display a tensor as an image using matplotlib.
    
    Args:
        tensor: Tensor of shape (3, H, W) or (H, W, 3) with values in [0, 255]
    """
    import matplotlib.pyplot as plt
    
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        # (C, H, W) -> (H, W, C)
        tensor = tensor.permute(1, 2, 0)
    
    img = tensor.cpu().numpy() / 255.0
    plt.imshow(img.clip(0, 1))
    plt.axis('off')
    plt.show()

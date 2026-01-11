import torch
from typing import Tuple


def split_image_into_blocks(image_tensor: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Split an image tensor into n×n blocks with padding on the right and bottom if needed.
    
    Args:
        image_tensor: Image tensor in CHW format (channels, height, width)
                      Expected from loader.image_to_tensor()
        block_size: Size of each block (n in n×n blocks)
    
    Returns:
        blocks: Tensor of shape (num_blocks_h, num_blocks_w, channels, block_size, block_size)
        padding_info: Tuple (pad_h, pad_w) indicating how much padding was added to height and width
    
    Example:
        >>> from src.dataset.loader import image_to_tensor
        >>> image = Image.open("image.jpg")
        >>> tensor = image_to_tensor(image)
        >>> blocks, (pad_h, pad_w) = split_image_into_blocks(tensor, block_size=32)
        >>> print(f"Original shape: {tensor.shape}")
        >>> print(f"Blocks shape: {blocks.shape}")
        >>> print(f"Padding added: height={pad_h}, width={pad_w}")
    """
    if image_tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (CHW format), got {image_tensor.dim()}D tensor")
    
    channels, height, width = image_tensor.shape
    
    # Calculate padding needed
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    
    # Pad the tensor on the right and bottom
    if pad_h > 0 or pad_w > 0:
        # Pad format: (pad_left, pad_right, pad_top, pad_bottom) for 2D padding
        # We only pad right and bottom, so left=0, right=pad_w, top=0, bottom=pad_h
        padded_tensor = torch.nn.functional.pad(
            image_tensor,
            (0, pad_w, 0, pad_h),  # (left, right, top, bottom)
            mode='constant',
            value=0
        )
    else:
        padded_tensor = image_tensor
    
    padded_height, padded_width = padded_tensor.shape[1], padded_tensor.shape[2]
    
    # Calculate number of blocks
    num_blocks_h = padded_height // block_size
    num_blocks_w = padded_width // block_size
    
    # Reshape into blocks
    # First, unfold the tensor to extract blocks
    # For CHW format, we need to unfold height and width dimensions
    blocks = padded_tensor.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    # blocks shape: (channels, num_blocks_h, num_blocks_w, block_size, block_size)
    
    # Permute to get (num_blocks_h, num_blocks_w, channels, block_size, block_size)
    blocks = blocks.permute(1, 2, 0, 3, 4)
    
    return blocks, (pad_h, pad_w)

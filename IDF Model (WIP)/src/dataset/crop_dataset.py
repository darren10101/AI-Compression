import torch
import torch.nn.functional as TF
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from torchvision.transforms import functional as F
import random

class CropDataStream(IterableDataset):
  def __init__(self, split, crop_size=256, dataset_name="ILSVRC/imagenet-1k", image_key='image', buffer_size=1000):
    self.dataset_name = dataset_name
    self.split = split
    self.crop_size = crop_size
    self.image_key = image_key
    self.buffer_size = buffer_size
    self.hf_stream = load_dataset(
      self.dataset_name,
      split=self.split,
      streaming=True
    )
  def _process_image(self, image):
    # Ensure RGB format (handles grayscale, palette, RGBA images)
    if image.mode != 'RGB':
      image = image.convert('RGB')
    img_tensor = F.to_tensor(image)
    c, h, w = img_tensor.shape
    top = random.randint(0, h-1)
    left = random.randint(0, w-1)
    crop = F.crop(img_tensor, top, left, min(self.crop_size, h - top), min(self.crop_size, w - left))
    if h < self.crop_size + top or w < self.crop_size + left:
      # torch.nn.functional.pad format for 3D tensor (C,H,W): (left, right, top, bottom)
      pad_right = max(self.crop_size + left - w, 0)
      pad_bottom = max(self.crop_size + top - h, 0)
      crop = TF.pad(crop, (0, pad_right, 0, pad_bottom))
    return crop
  def __iter__(self):
    for sample in self.hf_stream:
      if self.image_key not in sample:
        continue
      yield self._process_image(sample[self.image_key])

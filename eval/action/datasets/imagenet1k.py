import logging
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)

class ImageNet1K(Dataset):
    """Dataset wrapper for ImageNet-1K dataset using Hugging Face datasets"""

    def __init__(
            self,
            split: str = "train",
            transform: Optional[Callable] = None,
    ):
        """
        Args:
            split (str): Which split to use ('train', 'validation')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.split = 'validation' if split == 'val' else split
        self.transform = transform

        # Load dataset from Hugging Face
        logger.info(f"Loading ImageNet-1K {split} split from Hugging Face...")
        self.dataset = load_dataset("imagenet-1k", split=self.split)
        self.num_classes = 1000  # ImageNet-1K has 1000 classes
        
        logger.info(f"Loaded ImageNet-1K {split} split with {len(self.dataset)} images")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, label) where image is a transformed PIL Image
            and label is the class index
        """
        sample = self.dataset[idx]
        img = sample['image']  # Already a PIL Image
        label = sample['label']
        
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label)

    def get_class_name(self, idx: int) -> str:
        """Get the class name for a given index"""
        return self.dataset.features['label'].names[idx]

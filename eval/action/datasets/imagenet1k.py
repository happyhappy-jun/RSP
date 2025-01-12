import os
import json
import logging
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

class ImageNet1K(Dataset):
    """Dataset wrapper for ImageNet-1K dataset"""

    def __init__(
            self,
            data_root: str = '/data/imagenet',
            split: str = "train",
            transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_root (str): Path to ImageNet root directory
            split (str): Which split to use ('train', 'val')
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform

        # Set up split-specific path
        split_path = os.path.join(data_root, 'train' if split == 'train' else 'val')
        
        # Use ImageFolder to handle class subdirectories
        self.dataset = ImageFolder(split_path)
        self.num_classes = len(self.dataset.classes)
        
        logger.info(f"Loaded ImageNet-1K {split} split with {len(self.dataset)} images and {self.num_classes} classes")

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
        img, label = self.dataset[idx]
        
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label)

    def get_class_name(self, idx: int) -> str:
        """Get the class name for a given index"""
        return self.dataset.classes[idx]

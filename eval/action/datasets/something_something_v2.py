import os
import json
import logging
import random
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import av
from PIL import Image

logger = logging.getLogger(__name__)


class SomethingSomethingV2(Dataset):
    """Dataset wrapper for Something-Something V2 dataset using raw data"""

    def __init__(
            self,
            data_root: str = '/data/something-something-v2',
            split: str = "train",
            transform: Optional[Callable] = None,
            frames_per_video: int = 1,
    ):
        """
        Args:
            data_root (str): Path to dataset root directory
            split (str): Which split to use ('train', 'validation', 'test')
            transform (callable, optional): Optional transform to be applied on frames
            frames_per_video (int): Number of frames to sample from each video
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.raw_data_path = os.path.join(data_root, 'raw_data')

        # Load label mapping
        logger.info("Loading label mapping...")
        with open(os.path.join(data_root, 'labels', 'labels.json'), 'r') as f:
            self.label_mapping = json.load(f)
        self.num_classes = len(self.label_mapping)
        logger.info(f"Loaded {self.num_classes} classes")

        # Load annotations based on split
        logger.info(f"Loading Something-Something V2 {split} split...")
        self.processed_data = []

        if split in ['train', 'validation']:
            # Handle train and validation splits
            with open(os.path.join(data_root, 'labels', f'{split}.json'), 'r') as f:
                annotations = json.load(f)

            for anno in annotations:
                video_id = anno['id']
                template = anno['template'].replace('[', '').replace(']', '')

                self.processed_data.append({
                    'video_id': video_id,
                    'video': os.path.join(self.raw_data_path, f'{video_id}.webm'),
                    'label': self.label_mapping.get(template, -1),  # Use -1 if label not found
                    'text': template,
                    'placeholders': anno.get('placeholders', [])
                })

        elif split == 'test':
            # Handle test split (no labels available during testing)
            with open(os.path.join(data_root, 'labels', 'test.json'), 'r') as f:
                annotations = json.load(f)

            for anno in annotations:
                video_id = anno['id']
                self.processed_data.append({
                    'video_id': video_id,
                    'video': os.path.join(self.raw_data_path, f'{video_id}.webm'),
                    'label': -1,  # No labels for test set
                    'text': '',
                    'placeholders': []
                })

        logger.info(f"Loaded {len(self.processed_data)} samples for {split} split")
        
    def __len__(self) -> int:
        return len(self.processed_data)

    @staticmethod
    def _load_video_frames(video_path):
        """Load video frames with optimized decoding"""
        with av.open(video_path) as container:
            container.streams.video[0].thread_type = "AUTO"
            container.streams.video[0].thread_count = 8
            
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame)
            
            return frames, len(frames)
            
    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (frames, label) where frames is a tensor of shape (T, C, H, W)
            and label is the class index
        """
        # Get preprocessed data
        sample = self.processed_data[idx]
        video_path = sample['video']
        label = torch.tensor(int(sample['label']))

        # Load frames from disk
        video_frames, total_frames = self._load_video_frames(video_path)

        # Handle frame sampling
        frames = []
        if total_frames >= self.frames_per_video:
            # Randomly sample frames if we have enough
            frame_indices = random.sample(range(total_frames), self.frames_per_video)
            for idx in frame_indices:
                pil_img = video_frames[idx].to_image()
                pil_img = self.transform(pil_img)
                frames.append(pil_img)
        else:
            # If video is too short, use all frames and pad with zeros
            for frame in video_frames:
                pil_img = frame.to_image()
                pil_img = self.transform(pil_img)
                frames.append(pil_img)

            # Pad remaining frames with zeros
            zero_frame = torch.zeros_like(frames[0])
            while len(frames) < self.frames_per_video:
                frames.append(zero_frame)

        # Stack frames
        if self.frames_per_video == 1:
            frames = frames[0]  # Return single frame tensor
        else:
            frames = torch.stack(frames)

        return frames, label

    def get_text_description(self, idx: int) -> str:
        """Get the text description for a given index with placeholders filled in"""
        sample = self.processed_data[idx]
        text = sample['text']
        placeholders = sample['placeholders']

        # Replace placeholders in template
        for placeholder in placeholders:
            text = text.replace('[something]', placeholder, 1)

        return text

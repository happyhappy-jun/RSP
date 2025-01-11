import os
import json
import logging
import random
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import av
import io

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
            split (str): Which split to use ('train', 'validation', or 'test')
            transform (callable, optional): Optional transform to be applied on frames
            frames_per_video (int): Number of frames to sample from each video
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.raw_data_path = os.path.join(data_root, 'raw_data')
        
        # Load annotations
        logger.info(f"Loading Something-Something V2 {split} split...")
        with open(os.path.join(data_root, 'labels', f'{split}.json'), 'r') as f:
            self.annotations = json.load(f)
        logger.info(f"Loaded {len(self.annotations)} videos")
        
        # Load class labels and create mappings
        self.classes = []
        self.class_to_idx = {}
        self.processed_data = []
        
        # Process all annotations and create label mappings
        unique_labels = set()
        for anno in self.annotations:
            label_text = anno['template'].replace('[', '').replace(']', '').strip()
            unique_labels.add(label_text)
        
        # Create sorted class list and mapping
        self.classes = sorted(list(unique_labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Preprocess all data
        logger.info("Preprocessing dataset...")
        for anno in self.annotations:
            video_id = anno['id']
            label_text = anno['template'].replace('[', '').replace(']', '').strip()
            label_idx = self.class_to_idx[label_text]
            
            self.processed_data.append({
                'video_id': video_id,
                'video': os.path.join(self.raw_data_path, f'{video_id}.webm'),
                'label': label_idx,
                'text': label_text
            })
        logger.info(f"Preprocessed {len(self.processed_data)} samples with {self.num_classes} unique classes")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
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
        label = sample['label']
        
        # Open video file
        container = av.open(video_path)
        video = container.decode(video=0)
        
        # Get total frames
        total_frames = container.streams.video[0].frames
        if total_frames == 0:  # Some videos don't report frames correctly
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)  # Reset to beginning
        
        # Convert video frames to list for direct indexing
        video_frames = list(video)
        
        # Randomly sample frames
        frame_indices = random.sample(range(total_frames), min(self.frames_per_video, total_frames))
        frames = []
        for idx in frame_indices:
            pil_img = video_frames[idx].to_image()
            if self.transform is not None:
                pil_img = self.transform(pil_img)
            frames.append(pil_img)
        
        container.close()
        
        # Sample frames if needed
        if len(frames) > self.frames_per_video:
            indices = np.linspace(0, len(frames)-1, self.frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]
        
        if self.frames_per_video == 1:
            frames = frames[0]  # Return single frame tensor
        else:
            frames = torch.stack(frames)  # Stack multiple frames

        return frames, label

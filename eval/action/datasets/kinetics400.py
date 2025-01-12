import os
import logging
import random
import numpy as np
import pandas as pd
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from PIL import Image

logger = logging.getLogger(__name__)

class Kinetics400(Dataset):
    """Dataset wrapper for Kinetics-400 dataset"""

    def __init__(
            self,
            data_root: str = '/data/kinetics400-test',
            split: str = "train",
            transform: Optional[Callable] = None,
            frames_per_video: int = 1,
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            random_seed: int = 42,
            timeout: int = 30,  # Timeout in seconds for video loading
    ):
        """
        Args:
            data_root (str): Path to dataset root directory
            split (str): Which split to use ('train', 'val', 'test')
            transform (callable, optional): Optional transform to be applied on frames
            frames_per_video (int): Number of frames to sample from each video
            train_ratio (float): Ratio of data to use for training (default: 0.7)
            val_ratio (float): Ratio of data to use for validation (default: 0.15)
            random_seed (int): Random seed for reproducible splits (default: 42)
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.videos_dir = os.path.join(data_root, 'videos')

        # Load annotations
        logger.info("Loading Kinetics-400 annotations...")
        annotations_file = os.path.join(data_root, "test.csv")
        all_annotations = pd.read_csv(annotations_file)

        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Filter out non-existent and invalid videos
        valid_videos = []
        for idx, row in all_annotations.iterrows():
            video_id = row['youtube_id']
            start_time = row['time_start']
            end_time = row['time_end']
            video_path = os.path.join(self.videos_dir, f"{video_id}_{start_time:0>6}_{end_time:0>6}.mp4")
            if os.path.exists(video_path):
                try:
                    # Try to open video with Decord to verify it's valid
                    vr = VideoReader(video_path)
                    if len(vr) > 0:  # Check if video has frames
                        valid_videos.append(idx)
                    else:
                        logger.warning(f"Video has no frames: {video_path}")
                except Exception as e:
                    logger.warning(f"Invalid video file {video_path}: {str(e)}")
            
        filtered_annotations = all_annotations.iloc[valid_videos].reset_index(drop=True)
        total_size = len(filtered_annotations)
        
        if total_size == 0:
            raise RuntimeError(f"No valid videos found in {data_root}")
            
        indices = np.random.permutation(total_size)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Select the appropriate split
        if split == 'train':
            self.annotations = filtered_annotations.iloc[train_indices].reset_index(drop=True)
        elif split == 'val':
            self.annotations = filtered_annotations.iloc[val_indices].reset_index(drop=True)
        else:  # test
            self.annotations = filtered_annotations.iloc[test_indices].reset_index(drop=True)
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.annotations['label'].unique()))}
        self.num_classes = len(self.label_to_idx)
        
        logger.info(f"Loaded {len(self.annotations)} videos with {self.num_classes} classes")

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_video_frames(self, video_path: str, start_time: float, end_time: float):
        """Load frames from video between start and end times
        
        Args:
            video_path (str): Path to video file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
        """
        try:
            # Load video with Decord
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # If video is too short, return None
            if total_frames < 1:
                logger.warning(f"Video {video_path} has no frames")
                return None
            
            # Randomly sample frames
            if self.frames_per_video == 1:
                # Random single frame
                frame_idx = random.randint(0, total_frames - 1)
                frame = vr[frame_idx].asnumpy()
                return [Image.fromarray(frame)]
            else:
                # Sample multiple frames
                frame_indices = sorted(random.sample(range(total_frames), self.frames_per_video))
                frames = vr.get_batch(frame_indices).asnumpy()
                return [Image.fromarray(frame) for frame in frames]
        except Exception as e:
            logger.warning(f"Error loading video {video_path}: {str(e)}")
            return None

        return frames

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (frames, label) where frames is a tensor of shape (T, C, H, W) for multiple
                  frames or (C, H, W) for single frame, and label is the class index
        """
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Get video metadata
                row = self.annotations.iloc[idx]
                video_id = row['youtube_id']
                label = self.label_to_idx[row['label']]
                start_time = row['time_start']
                end_time = row['time_end']
                
                # Load video frames
                video_path = os.path.join(self.videos_dir, f"{video_id}_{start_time:0>6}_{end_time:0>6}.mp4")
                frames = self._load_video_frames(video_path, start_time, end_time)
                
                if frames is None:
                    raise RuntimeError(f"Failed to load video {video_path}")
                
                # Apply transforms
                if self.transform is not None:
                    frames = [self.transform(frame) for frame in frames]
                    
                # Stack frames if multiple
                if self.frames_per_video == 1:
                    frames = frames[0]  # Return single frame tensor
                else:
                    frames = torch.stack(frames)  # Return stacked tensor (T, C, H, W)

                return frames, torch.tensor(label)
                
            except Exception as e:
                if retry == max_retries - 1:
                    logger.error(f"Failed to load video after {max_retries} retries: {str(e)}")
                    # Return a random valid sample as fallback
                    return self.__getitem__((idx + 1) % len(self))
                else:
                    logger.warning(f"Retry {retry + 1}/{max_retries} for video load")

    def get_class_name(self, idx: int) -> str:
        """Get the class name for a given index"""
        for label, label_idx in self.label_to_idx.items():
            if label_idx == idx:
                return label
        return "unknown"

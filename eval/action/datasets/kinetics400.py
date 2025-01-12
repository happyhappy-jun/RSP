import os
import logging
import pandas as pd
from typing import Optional, Callable
import torch
from torch.utils.data import Dataset
import av
from PIL import Image

logger = logging.getLogger(__name__)

class Kinetics400(Dataset):
    """Dataset wrapper for Kinetics-400 dataset"""

    def __init__(
            self,
            data_root: str = '/data/kinetics400-test',
            split: str = "test",
            transform: Optional[Callable] = None,
            frames_per_video: int = 1,
    ):
        """
        Args:
            data_root (str): Path to dataset root directory
            split (str): Which split to use (currently only 'test' supported)
            transform (callable, optional): Optional transform to be applied on frames
            frames_per_video (int): Number of frames to sample from each video
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.videos_dir = os.path.join(data_root, 'videos')

        # Load annotations
        logger.info(f"Loading Kinetics-400 {split} split annotations...")
        annotations_file = os.path.join(data_root, f"{split}.csv")
        self.annotations = pd.read_csv(annotations_file)
        
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
        frames = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.thread_count = 8

            # Calculate target frame indices
            fps = float(stream.average_rate)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            total_frames = end_frame - start_frame

            if self.frames_per_video == 1:
                # Take middle frame
                target_frame = start_frame + total_frames // 2
                container.seek(int(target_frame * stream.time_base * 1000000))
                for frame in container.decode(video=0):
                    frames.append(frame.to_image())
                    break
            else:
                # Sample evenly spaced frames
                frame_indices = torch.linspace(start_frame, end_frame-1, self.frames_per_video).long().tolist()
                container.seek(int(start_frame * stream.time_base * 1000000))

                for i, frame in enumerate(container.decode(video=0)):
                    if i + start_frame in frame_indices:
                        frames.append(frame.to_image())
                    if len(frames) == self.frames_per_video:
                        break

        return frames

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (frames, label) where frames is a tensor of shape (T, C, H, W) for multiple
                  frames or (C, H, W) for single frame, and label is the class index
        """
        # Get video metadata
        row = self.annotations.iloc[idx]
        video_id = row['youtube_id']
        label = self.label_to_idx[row['label']]
        start_time = row['time_start']
        end_time = row['time_end']
        
        # Load video frames
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
        frames = self._load_video_frames(video_path, start_time, end_time)
        
        # Apply transforms
        if self.transform is not None:
            frames = [self.transform(frame) for frame in frames]
            
        # Stack frames if multiple
        if self.frames_per_video == 1:
            frames = frames[0]  # Return single frame tensor
        else:
            frames = torch.stack(frames)  # Return stacked tensor (T, C, H, W)

        return frames, torch.tensor(label)

    def get_class_name(self, idx: int) -> str:
        """Get the class name for a given index"""
        for label, label_idx in self.label_to_idx.items():
            if label_idx == idx:
                return label
        return "unknown"

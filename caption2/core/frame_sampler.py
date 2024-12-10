from abc import ABC, abstractmethod
import random
import numpy as np
from decord import VideoReader, cpu

class BaseFrameSampler(ABC):
    """Abstract base class for frame sampling strategies"""
    
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    @abstractmethod
    def sample_frames(self, video_path, num_frames=4):
        """Sample frames from a video file
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to sample
            
        Returns:
            list: List of frame indices
        """
        pass

class UniformFrameSampler(BaseFrameSampler):
    """Sample evenly spaced frames from video"""
    
    def sample_frames(self, video_path, num_frames=4):
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames >= num_frames:
            # Divide video into num_frames segments and sample one frame from each
            segment_size = total_frames // num_frames
            frame_indices = [
                random.randint(i * segment_size, (i + 1) * segment_size - 1)
                for i in range(num_frames)
            ]
        else:
            # If video is too short, sample with replacement
            frame_indices = random.choices(range(total_frames), k=num_frames)
            
        frame_indices.sort()  # Keep temporal order
        return frame_indices

class PairedFrameSampler(BaseFrameSampler):
    """Sample pairs of frames with configurable gap"""
    
    def __init__(self, min_gap=5, max_gap=30, seed=None):
        super().__init__(seed)
        self.min_gap = min_gap
        self.max_gap = max_gap
    
    def sample_frames(self, video_path, num_pairs=2):
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)
        
        frame_indices = []
        for _ in range(num_pairs):
            # Select start frame
            if frame_indices:
                # Ensure some gap between pairs
                start = frame_indices[-1] + self.min_gap
                if start >= total_frames:
                    break
            else:
                start = 0
                
            # Select end frame with gap
            max_end = min(start + self.max_gap, total_frames - 1)
            if max_end <= start + self.min_gap:
                break
                
            end = random.randint(start + self.min_gap, max_end)
            frame_indices.extend([start, end])
            
        # Pad with duplicates if needed
        while len(frame_indices) < num_pairs * 2:
            frame_indices.extend(frame_indices[-2:])
            
        return frame_indices[:num_pairs * 2]

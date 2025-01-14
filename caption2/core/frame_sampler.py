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
    """Sample pairs of frames with configurable gap between them.
    
    Based on Kinetics dataset sampling strategy:
    - For videos longer than max_distance + 1 frames:
      - Randomly select start frame
      - Select second frame with random interval between min_gap and max_distance
    - For shorter videos:
      - Randomly sample any two frames in order
    """
    
    def __init__(self, min_gap=4, max_distance=48, num_pairs=6, seed=None):
        super().__init__(seed)
        self.min_gap = min_gap
        self.max_distance = max_distance
        self.num_pairs = num_pairs
        
    def sample_frames(self, video_path):
        """Sample num_pairs pairs of frames from video
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            list: List of frame indices, arranged as [start1, end1, start2, end2, ...]
        """
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        total_frames = len(vr)
        least_frames = self.max_distance + 1
        
        frame_indices = []
        for _ in range(self.num_pairs):
            if total_frames >= least_frames:
                # For longer videos, maintain minimum gap
                start = random.randint(0, total_frames - least_frames)
                interval = random.randint(self.min_gap, self.max_distance)
                end = start + interval
            else:
                # For shorter videos, just sample any two frames
                indices = random.sample(range(total_frames), 2)
                indices.sort()  # Keep temporal order
                start, end = indices
                
            frame_indices.extend([start, end])
            
        return frame_indices

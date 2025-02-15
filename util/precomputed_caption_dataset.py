"""
Dataset class for precomputed caption annotations.

The dataset file is a JSON file containing a list of video entries.
Each entry should have:
  - "video_path": the relative path (from data_root) to the video file.
  - "video_name": the video file name.
  - "frame_pairs": a list of dictionaries, where each dictionary has:
         "frame_indices": a list of two indices [src_idx, tgt_idx],
         "caption": the precomputed caption text.
         
For each video, if there are 64 pairs available, the dataset will sample
a number of pairs equal to the 'repeated_sampling' parameter (e.g. 2).
For each sample, the video file (located at data_root/video_path) is read using decord,
and frames at the specified indices are extracted. Then, optional paired transformations
and basic transforms (to tensor and normalize) are applied.
"""

import os
import json
import random
from torch.utils.data import Dataset
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import torch
from torchvision import transforms

# Import a paired random resized crop transform if available
from util.transform import PairedRandomResizedCrop


class PrecomputedCaptionDataset(Dataset):
    def __init__(self, dataset_file: str, data_root: str, repeated_sampling: int = 2, 
                 paired_transform=None, basic_transform=None):
        """
        Args:
            dataset_file (str): Path to the JSON file containing precomputed caption annotations.
            data_root (str): Root directory where video files are stored.
            repeated_sampling (int): Number of frame pairs to sample per video.
            paired_transform (callable, optional): Callable that takes two PIL images and returns transformed images.
                                                   Defaults to PairedRandomResizedCrop().
            basic_transform (callable, optional): Callable to convert a PIL image to tensor and normalize.
                                                   Defaults to standard normalization.
        """
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        self.data_root = data_root
        self.repeated_sampling = repeated_sampling
        self.samples = []
        
        self.paired_transform = paired_transform if paired_transform is not None else PairedRandomResizedCrop()
        self.basic_transform = basic_transform if basic_transform is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        for entry in data:
            video_rel_path = entry.get("video_path")
            video_path = os.path.join(data_root, video_rel_path)
            frame_pairs = entry.get("frame_pairs", [])
            if len(frame_pairs) == 0:
                continue
            # Sample 'repeated_sampling' pairs if available, else use all
            if len(frame_pairs) > repeated_sampling:
                sampled_pairs = random.sample(frame_pairs, repeated_sampling)
            else:
                sampled_pairs = frame_pairs
            for pair in sampled_pairs:
                indices = pair.get("frame_indices")
                caption = pair.get("caption", "")
                if not indices or len(indices) < 2:
                    continue
                self.samples.append({
                    "video_path": video_path,
                    "frame_indices": indices,
                    "caption": caption
                })
    
    def __len__(self):
        return len(self.samples)
    
    def read_frame(self, video_path: str, index: int) -> np.ndarray:
        """Read a single frame from a video file using decord."""
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        frame = vr[index].asnumpy()
        return frame
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        indices = sample["frame_indices"]
        
        # Read source and target frames
        frame1_np = self.read_frame(video_path, indices[0])
        frame2_np = self.read_frame(video_path, indices[1])
        
        # Convert numpy arrays to PIL Images
        img1 = Image.fromarray(frame1_np)
        img2 = Image.fromarray(frame2_np)
        
        # Apply paired transformation if available
        if self.paired_transform:
            img1, img2 = self.paired_transform(img1, img2)
        
        # Apply basic transform (conversion to tensor and normalization)
        if self.basic_transform:
            img1 = self.basic_transform(img1)
            img2 = self.basic_transform(img2)
        
        return {
            "src_image": img1,
            "tgt_image": img2,
            "caption": sample["caption"]
        }

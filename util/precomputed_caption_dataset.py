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

from pptx.media import Video
from torch.utils.data import Dataset
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import torch
from torchvision import transforms

# Import a paired random resized crop transform if available
from util.transform import PairedRandomResizedCrop


class PrecomputedCaptionDataset(Dataset):
    def __init__(self, dataset_file: str, data_root: str, repeated_sampling: int = 2, max_pair_pool: int = 64, seed=42):
        """
        Args:
            dataset_file (str): Path to the JSON file containing precomputed caption annotations.
            data_root (str): Root directory where video files are stored.
            repeated_sampling (int): Number of frame pairs to sample per video.
            paired_transform (callable, optional): Callable that takes two PIL images and returns transformed images.
                                                   Defaults to PairedRandomResizedCrop().
            basic_transform (callable, optional): Callable to convert a PIL image to tensor and normalize.
                                                   Defaults to standard normalization.
            max_pair_pool (int): Maximum number of frame pairs to consider per video. Defaults to 64.
        """
        random.seed(seed)
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        self.data_root = data_root
        self.repeated_sampling = repeated_sampling
        self.max_pair_pool = max_pair_pool
        self.samples = []
        self._video_reader_cache = {}
        
        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        
        for entry in data:
            video_rel_path = entry.get("video_path")
            video_path = os.path.join(data_root, video_rel_path)
            frame_pairs = entry.get("frame_pairs", [])
            if len(frame_pairs) == 0:
                continue
            if len(frame_pairs) > self.max_pair_pool:
                frame_pairs = random.sample(frame_pairs, self.max_pair_pool)
            self.samples.append({
                "video_path": video_path,
                "frame_pairs": frame_pairs
            })
    
    def __len__(self):
        return len(self.samples)
    
    def read_frame(self, vr: VideoReader, index: int) -> np.ndarray:
        frame = vr[index].asnumpy()
        return frame

    def transform(self, src_image, tgt_image):
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=16)
        pool = sample["frame_pairs"]
        if len(pool) > self.repeated_sampling:
            selected_pairs = random.sample(pool, self.repeated_sampling)
        else:
            selected_pairs = pool
        src_images = []
        tgt_images = []
        captions = []
        for pair in selected_pairs:
            indices = pair.get("frame_indices")
            caption = pair.get("caption", "")
            if caption is "":
                print("Warning: Empty caption found.")
            if not indices or len(indices) < 2:
                continue
            frame1_np = self.read_frame(vr, indices[0])
            frame2_np = self.read_frame(vr, indices[1])
            src_image, tgt_image = self.transform(frame1_np, frame2_np)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            captions.append(caption)
        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return {
            "src_images": src_images,
            "tgt_images": tgt_images,
            "captions": captions
        }

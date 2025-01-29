import os
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from decord import VideoReader, cpu
from torchvision import transforms

from util.transform import PairedRandomResizedCrop


class PairedKineticsWithGlobalCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON and JSONL embeddings"""

    def __init__(
            self,
            video_root,  # Root directory containing videos
            frame_info_path,  # Path to global_frame.json
            embeddings_path,  # Path to global_embedding.jsonl
            repeated_sampling=2,
            max_distance=48,
            seed=42
    ):
        super().__init__()
        self.video_root = video_root
        self.repeated_sampling = repeated_sampling
        self.max_distance = max_distance

        # Load frame info
        print("Loading frame info...")
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            self.video_data = {
                f"video_{video['video_idx']}": {
                    'video_idx': video['video_idx'],
                    'video_path': os.path.join(self.video_root, video['video_path']),
                    'frame_indices': video['frame_indices'],
                    'class_label': video['class_label']
                }
                for video in frame_info['videos']
            }

        # Load embeddings
        print("Loading embeddings...")
        self.embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                custom_id = record[-1]["custom_id"]
                embedding = record[1]["data"][0]["embedding"]
                self.embeddings[custom_id] = embedding

        # Match video data with embeddings
        self.valid_videos = []
        for custom_id, video_info in self.video_data.items():
            if custom_id in self.embeddings:
                video_info['embedding'] = self.embeddings[custom_id]
                self.valid_videos.append(video_info)

        # Setup transforms
        self.transforms = PairedRandomResizedCrop(seed=seed)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

        print(f"\nDataset Statistics:")
        print(f"Total videos found: {len(self.valid_videos)}")
        print(f"Total frames per video: {len(self.valid_videos[0]['frame_paths']) if self.valid_videos else 0}")

    def load_frames(self, video_path):
        """Load and sample frame pairs from video"""
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        
        # handle temporal segments
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1
        
        if seg_len >= least_frames_num:
            idx_cur = random.randint(0, seg_len - least_frames_num)
            interval = random.randint(4, self.max_distance)
            idx_fut = idx_cur + interval
        else:
            indices = random.sample(range(seg_len), 2)
            indices.sort()
            idx_cur, idx_fut = indices
            
        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()

        return frame_cur, frame_fut

    def transform(self, src_image, tgt_image):
        """Apply transforms to image pair"""
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

    def __len__(self):
        return len(self.valid_videos)

    def __getitem__(self, index):
        video_data = self.valid_videos[index]
        video_path = video_data['video_path']
        
        src_images = []
        tgt_images = []
        
        # Sample multiple pairs from the same video
        for _ in range(self.repeated_sampling):
            frame_cur, frame_fut = self.load_frames(video_path)
            
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)

        # Get embedding and repeat for each sample
        embedding = torch.tensor(video_data['embedding'])
        embedding = embedding.repeat(self.repeated_sampling, 1)

        return {
            "video_idx": video_data['video_idx'],
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": embedding,
        }


def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "embeddings": torch.stack([x['embeddings'] for x in batch], dim=0),
    }


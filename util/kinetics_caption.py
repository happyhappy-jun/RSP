import os
import random
from typing import Dict, List, Union
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from collections import defaultdict
from PIL import Image

from DeBERTa import deberta

class PairedRandomResizedCrop:
    def __init__(
        self,
        hflip_p=0.5,
        size=(224, 224),
        scale=(0.5, 1.0),
        ratio=(3./4., 4./3.),
        interpolation=F.InterpolationMode.BICUBIC
    ):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, np_RGB_img_1, np_RGB_img_2):
        # Convert numpy images to PIL Images
        pil_RGB_img_1 = F.to_pil_image(np_RGB_img_1)
        pil_RGB_img_2 = F.to_pil_image(np_RGB_img_2)

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            pil_RGB_img_1, scale=self.scale, ratio=self.ratio
        )
        # Apply the crop on both images
        cropped_img_1 = F.resized_crop(pil_RGB_img_1,
                                       i, j, h, w,
                                       size=self.size,
                                       interpolation=self.interpolation)
        cropped_img_2 = F.resized_crop(pil_RGB_img_2,
                                       i, j, h, w,
                                       size=self.size,
                                       interpolation=self.interpolation)

        if random.random() < self.hflip_p:
            cropped_img_1 = F.hflip(cropped_img_1)
            cropped_img_2 = F.hflip(cropped_img_2)

        return cropped_img_1, cropped_img_2

class PairedKineticsWithCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON"""
    def __init__(
        self,
        data_path,           # Path to preprocessed JSON file
        embeddings_path,     # Path to precomputed embeddings
        repeated_sampling=2  # Number of augmented samples per pair
    ):
        super().__init__()
        # Load preprocessed data
        with open(data_path, 'r') as f:
            data = json.load(f)
    
        # Sort results first by video_idx
        sorted_results = sorted(data['results'], key=lambda x: x['video_idx'])
        
        self.videos = defaultdict(list)
        for i, pair in enumerate(sorted_results):
            # Within each video, sort by pair_idx if it exists
            self.videos[pair["video_idx"]].append(pair)
        
        # Sort pairs within each video
        for video_idx in self.videos:
            self.videos[video_idx].sort(key=lambda x: x.get('pair_idx', 0))
            
        self.video_indices = sorted(self.videos.keys())
        
        # Load precomputed embeddings
        print(f"Loading precomputed embeddings from {embeddings_path}")
        self.embeddings = torch.load(embeddings_path)
        print(f"Loaded {len(self.embeddings)} embeddings")
        
        self.repeated_sampling = repeated_sampling
        
        # Setup transforms
        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.video_indices)} videos")
        print(f"Total samples with repeated sampling: {len(self.video_indices) * repeated_sampling}")

    def __len__(self):
        return len(self.video_indices)

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def __getitem__(self, index):
        video_idx = self.video_indices[index]
        pair_infos = self.videos[video_idx]

        src_images = []
        tgt_images = []
        embeddings = []
        
        for pair_idx, pair in enumerate(pair_infos):
            frame_cur = self.load_frame(pair['frame_cur_path'])
            frame_fut = self.load_frame(pair['frame_fut_path'])
                
            # Apply transforms
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(self.embeddings[pair['video_idx']* len(pair_infos) + pair_idx])
            

        # Get precomputed embedding and repeat for each sample
        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "input_ids": torch.stack(embeddings, dim=0),
            "video_idx": video_idx
        }
            
def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "input_ids": torch.stack([x['input_ids'] for x in batch], dim=0),
        "video_idx": [x['video_idx'] for x in batch]
    }


if __name__ == "__main__":
    dataset = PairedKineticsWithCaption(
        data_path="/home/junyoon/rsp-llm/artifacts/results/frame_analysis_results_complete.json",
        embeddings_path="/home/junyoon/rsp-llm/artifacts/deberta_embeddings.pt"
    )
    print(dataset[0])

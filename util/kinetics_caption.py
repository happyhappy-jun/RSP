import os
import random
import numpy as np
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class PairedKineticsWithCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON"""
    def __init__(
        self,
        data_path,           # Path to preprocessed JSON file
        embeddings_path,     # Path to precomputed embeddings
        repeated_sampling=2, # Number of augmented samples per pair
        seed=42             # Random seed for reproducibility
    ):
        super().__init__()
        set_seed(seed)
        # Load preprocessed data from JSONL
        results = []
        with open(data_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                # Extract embedding from the OpenAI API response
                embedding = record['response'].get('body', {}).get('data', [{}])[0].get('embedding', [])
                if embedding:
                    results.append({
                        'video_idx': int(record['custom_id'].split('-')[0]),
                        'pair_idx': int(record['custom_id'].split('-')[1]),
                        'frame_cur_path': record.get('frame_cur_path', ''),
                        'frame_fut_path': record.get('frame_fut_path', ''),
                        'embedding': embedding
                    })
        
        # Sort results first by video_idx
        sorted_results = sorted(results, key=lambda x: (x['video_idx'], x['pair_idx']))
        if sorted_results:
            print("First result:", sorted_results[0])
        
        self.videos = defaultdict(list)
        for i, pair in enumerate(sorted_results):
            # Within each video, sort by pair_idx if it exists
            self.videos[pair["video_idx"]].append(pair)
        
        # Sort pairs within each video
        for video_idx in self.videos:
            self.videos[video_idx].sort(key=lambda x: x.get('pair_idx', 0))
            
        self.video_indices = sorted(self.videos.keys())
        
        # Convert embeddings to tensors
        self.embeddings = {(result['video_idx'], result['pair_idx']): torch.tensor(result['embedding']) 
                          for result in sorted_results}
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
            print(pair['video_idx'], pair_idx)
            embeddings.append(self.embeddings[(pair['video_idx'], pair_idx)])
            

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
        embeddings_path="/home/junyoon/rsp-llm/artifacts/deberta_embeddings.pt",
        seed=42
    )
    a = dataset[0]['input_ids'][0]
    b = dataset[0]['input_ids'][1]
    c = dataset[1]['input_ids'][0]
    d = dataset[1]['input_ids'][1]
    
    print(a.shape, b.shape)
    
    
    print(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(torch.nn.functional.cosine_similarity(c, d, dim=0))
    print(torch.nn.functional.cosine_similarity(a, c, dim=0))

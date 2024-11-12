import os
import random
import json
from collections import defaultdict
from pathlib import Path

from decord import VideoReader, cpu
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
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

class PairedKineticsWithGlobalCaption(Dataset):
    def __init__(
        self,
        video_root,  # Root directory containing videos
        json_path,   # Path to caption JSON file
        embeddings_path=None,  # Path to precomputed embeddings
        max_distance=48,
        repeated_sampling=2
    ):
        super().__init__()
        self.video_root = video_root
        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling

        # Setup transforms
        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load caption data and organize video info
        print("Loading caption data...")
        with open(json_path, 'r') as f:
            caption_data = json.load(f)

        self.video_info = defaultdict(list)
        for result in caption_data['results']:
            video_path = os.path.join(result['label'], result['video_name'] + '.mp4')
            self.video_info[result['video_idx']] = {
                'video_path': video_path,
                'label': result['label']
            }

        self.video_indices = sorted(self.video_info.keys())

        # Load or compute embeddings
        if embeddings_path and os.path.exists(embeddings_path):
            print(f"Loading precomputed embeddings from {embeddings_path}")
            self.embeddings = torch.load(embeddings_path)
        else:
            print("Computing DeBERTa embeddings...")
            if embeddings_path:
                os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            self.embeddings = precompute_deberta_embeddings(json_path, embeddings_path)

        print(f"Loaded {len(self.video_indices)} videos")

    def load_frames(self, vr):
        """Sample two frames with temporal constraint"""
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

        return frame_cur, frame_fut, (idx_cur, idx_fut)

    def transform(self, src_image, tgt_image):
        """Apply transforms to image pair"""
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

    def __len__(self):
        return len(self.video_indices)

    def __getitem__(self, index):
        video_idx = self.video_indices[index]
        info = self.video_info[video_idx]
        video_path = os.path.join(self.video_root, info['video_path'])

        # Load video
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

        src_images = []
        tgt_images = []
        frame_indices = []

        # Sample frames multiple times
        for _ in range(self.repeated_sampling):
            src_image, tgt_image, indices = self.load_frames(vr)
            src_image, tgt_image = self.transform(src_image, tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            frame_indices.append(indices)

        # Get precomputed embedding and repeat for each sample
        embedding = self.embeddings[video_idx]
        embedding = embedding.repeat(self.repeated_sampling, 1)

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "input_ids": embedding,
            "video_idx": video_idx,
            "label": info['label'],
            "frame_indices": frame_indices
        }

def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "input_ids": torch.stack([x['input_ids'] for x in batch], dim=0),
        "video_idx": [x['video_idx'] for x in batch],
        "label": [x['label'] for x in batch],
        "frame_indices": [x['frame_indices'] for x in batch]
    }

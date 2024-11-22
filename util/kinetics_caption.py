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
        repeated_sampling=2  # Number of augmented samples per pair
    ):
        super().__init__()
        # Load preprocessed data
        with open(data_path, 'r') as f:
            data = json.load(f)
    
        self.videos = defaultdict(list)
        for i, pair in enumerate(data['results']):
            self.videos[pair["video_idx"]].append(pair)
        self.videos = list(self.videos.values())
        print(self.videos[0])
        
        self.repeated_sampling = repeated_sampling
        
        # Setup transforms
        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base", use_fast=True)
        vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
        self.tokenizer = deberta.tokenizers[vocab_type](vocab_path)

        
        print(f"Loaded {len(self.videos)} video")
        print(f"Total samples with repeated sampling: {len(self.videos) * repeated_sampling}")

    def __len__(self):
        return len(self.videos)

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def tokenize_caption(self, caption):
        max_seq_len = 512
        tokens = self.tokenizer.tokenize(caption)
        tokens = tokens[:max_seq_len - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        input_masks = [1] * len(input_ids)
        paddings = max_seq_len - len(input_ids)
        input_ids = input_ids + [0] * paddings
        input_masks = input_masks + [0] * paddings
        return torch.tensor(input_ids, dtype=torch.int)

    def __getitem__(self, index):
        pair_infos = self.videos[index]
        assert len(pair_infos) == self.repeated_sampling

        src_images = []
        tgt_images = []
        captions = []
        
        for pair_idx, pair in enumerate(pair_infos):
            frame_cur = self.load_frame(pair['frame_cur_path'])
            frame_fut = self.load_frame(pair['frame_fut_path'])
                
                # Apply transforms
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
                
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            
            caption = self.tokenize_caption(pair['analysis'])
            captions.append(caption)
            
        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "input_ids": torch.stack(captions, dim=0)
        }
            
def collate_fn(batch):
    src_images = torch.stack([x['src_images'] for x in batch], dim=0)
    tgt_images = torch.stack([x['tgt_images'] for x in batch], dim=0)
    input_ids = torch.stack([x['input_ids'] for x in batch], dim=0)
    # token_type_ids = torch.stack([x['token_type_ids'] for x in batch], dim=0)
    
    return {
        "src_images": src_images,
        "tgt_images": tgt_images,
        "input_ids": input_ids,
        # "token_type_ids": token_type_ids
    }
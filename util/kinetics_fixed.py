import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from util.transform import PairedRandomResizedCrop

class PairedKineticsFixed(Dataset):
    def __init__(
            self,
            frame_root,
            frame_info_path,
            repeated_sampling=2,
            seed=42
    ):
        super().__init__()
        self.frame_root = frame_root
        
        # Load frame info data
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            self.frames = frame_info['videos']

        self.transforms = PairedRandomResizedCrop(seed=seed)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.repeated_sampling = repeated_sampling

    def __len__(self):
        return len(self.frames)

    def _process_path(self, frame_path):
        """add frame_root to frame_path"""
        return f"{self.frame_root}/{frame_path}"

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        frame_data = self.frames[index]
        frame_paths = frame_data['frame_paths']
        
        src_images = []
        tgt_images = []
        
        for _ in range(self.repeated_sampling):
            frame_cur = self.load_frame(self._process_path(frame_paths[0]))
            frame_fut = self.load_frame(self._process_path(frame_paths[1]))
            
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)

        return torch.stack(src_images, dim=0), torch.stack(tgt_images, dim=0), 0

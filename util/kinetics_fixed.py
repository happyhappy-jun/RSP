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
            frame_info_additional_path=None,
            seed=42
    ):
        super().__init__()
        self.frame_root = frame_root
        
        # Load main frame info data
        print("Loading main frame info data...")
        self.frames = []
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            self._process_frame_info(frame_info['videos'], prefix="frames", pair_idx_offset=0)

        # Load additional frame info if provided
        if frame_info_additional_path:
            print("Loading additional frame info data...")
            with open(frame_info_additional_path, 'r') as f:
                frame_info_additional = json.load(f)
                self._process_frame_info(
                    frame_info_additional['videos'], 
                    prefix="frames_additional",
                    pair_idx_offset=2  # Add 2 to pair_idx for additional dataset
                )

        self.transforms = PairedRandomResizedCrop(seed=seed)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.repeated_sampling = repeated_sampling

    def __len__(self):
        return len(self.frames)

    def _process_frame_info(self, videos, prefix="frames", pair_idx_offset=0):
        """Process frame info with prefix and optional pair_idx offset"""
        for frame in videos:
            video_idx = frame['video_idx']
            # Add offset to pair_idx for additional dataset
            pair_idx = frame.get('pair_idx', 0) + pair_idx_offset
            
            processed_paths = [
                f"{self.frame_root}/{prefix}/{path}" 
                for path in frame['frame_paths']
            ]
            self.frames.append({
                'video_idx': video_idx,
                'pair_idx': pair_idx,
                'frame_paths': processed_paths
            })

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
            frame_cur = self.load_frame(frame_paths[0])
            frame_fut = self.load_frame(frame_paths[1])
            
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)

        return torch.stack(src_images, dim=0), torch.stack(tgt_images, dim=0), 0


if __name__ == "__main__":
    dataset = PairedKineticsFixed(
        frame_root="data/kinetics/frames",
        frame_info_path="data/kinetics/frames_info.json",
        repeated_sampling=2
    )

    print(f"Dataset length: {len(dataset)}")

    src_images, tgt_images, _ = dataset[0]
    print(f"Source images shape: {src_images.shape}")
    print(f"Target images shape: {tgt_images.shape}")

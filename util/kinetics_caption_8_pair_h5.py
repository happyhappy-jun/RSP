import h5py
import torch
import random
import cv2
import ast
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from util.transform import PairedRandomResizedCrop


class PairedKineticsWithCaption8PairH5(Dataset):
    def __init__(
            self,
            h5_path,
            data_root,
            repeated_sampling=2
    ):
        super().__init__()
        self.h5_path = h5_path
        self.data_root = data_root
        self.repeated_sampling = repeated_sampling

        # Open H5 file in read mode
        self.h5_file = h5py.File(h5_path, 'r')

        # Get all video-pair combinations
        self.valid_pairs = []
        for key in self.h5_file['frame_paths'].keys():
            video_idx, pair_idx = key.split('_')
            self.valid_pairs.append((video_idx, pair_idx))

        # Group pairs by video_idx
        self.video_pairs = {}
        for video_idx, pair_idx in self.valid_pairs:
            if video_idx not in self.video_pairs:
                self.video_pairs[video_idx] = []
            self.video_pairs[video_idx].append(pair_idx)

        # Convert to list for indexing
        self.videos = list(self.video_pairs.items())

        print(f"\nDataset Statistics:")
        print(f"Total videos: {len(self.videos)}")
        print(f"Total pairs: {len(self.valid_pairs)}")

        # Setup transforms
        self.transforms = PairedRandomResizedCrop(seed=42)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.videos)

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_pair(self, video_idx, pair_idx):
        """Load a single pair of frames and its embedding"""
        # Get frame paths
        paths_data = self.h5_file['frame_paths'][f"{video_idx}_{pair_idx}"][()]
        paths_dict = ast.literal_eval(paths_data)

        # Construct full paths based on whether it's from additional dataset
        frame_cur_path = os.path.join(self.data_root, paths_dict['frame_cur'])
        frame_fut_path = os.path.join(self.data_root, paths_dict['frame_fut'])

        # Load frames
        frame_cur = self.load_frame(frame_cur_path)
        frame_fut = self.load_frame(frame_fut_path)

        # Apply transforms
        src_image, tgt_image = self.transforms(frame_cur, frame_fut)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)

        # Get embedding
        embedding = self.h5_file['embeddings'][f"{video_idx}_{pair_idx}"][:]
        embedding = torch.from_numpy(embedding)

        return src_image, tgt_image, embedding

    def __getitem__(self, index):
        # Get video and its pairs
        video_idx, pair_indices = self.videos[index]

        src_images = []
        tgt_images = []
        embeddings = []

        # Randomly sample pairs for this video
        sampled_pairs = random.sample(pair_indices,
                                      min(self.repeated_sampling, len(pair_indices)))

        # Process each sampled pair
        for pair_idx in sampled_pairs:
            src_image, tgt_image, embedding = self._load_pair(video_idx, pair_idx)
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(embedding)

        # If we need more samples, repeat the last pair
        while len(src_images) < self.repeated_sampling:
            src_images.append(src_images[-1])
            tgt_images.append(tgt_images[-1])
            embeddings.append(embeddings[-1])

        return {
            "video_idx": int(video_idx),
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0)
        }

    def __del__(self):
        """Ensure H5 file is closed when dataset is deleted"""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "embeddings": torch.stack([x['embeddings'] for x in batch], dim=0),
    }


if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    import tqdm
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()

    print("\nInitializing dataset...")
    dataset = PairedKineticsWithCaption8PairH5(
        h5_path=args.h5_path,
        data_root=args.data_root
    )

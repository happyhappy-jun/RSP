import os
import cv2
import json
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from decord import VideoReader, cpu
from torchvision import transforms

from util.misc import seed_everything
from util.transform import PairedRandomResizedCrop


class PairedKineticsWithGlobalCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON and JSONL embeddings"""

    def __init__(
            self,
            frame_root,  # Root directory containing frames
            frame_info_path,  # Path to global_frame.json
            embeddings_path,  # Path to global_embedding.jsonl
            repeated_sampling=2,
            seed=42
    ):
        super().__init__()
        seed_everything(seed)

        self.frame_root = frame_root
        self.repeated_sampling = repeated_sampling

        # Load frame info
        print("Loading frame info...")
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            self.frame_data = {
                f"video_{video['video_idx']}": {
                    'video_idx': video['video_idx'],
                    'frame_paths': [os.path.join(self.frame_root, path) for path in video['frame_paths']],
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

        # Match frame data with embeddings
        self.valid_videos = []
        for custom_id, frame_info in self.frame_data.items():
            if custom_id in self.embeddings:
                frame_info['embedding'] = self.embeddings[custom_id]
                self.valid_videos.append(frame_info)

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

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def sample_frame_pairs(self, frame_paths):
        """Sample pairs of frames from available frames"""
        num_frames = len(frame_paths)
        pairs = []
        for _ in range(self.repeated_sampling):
            # Randomly select two different indices
            idx1, idx2 = random.sample(range(num_frames), 2)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            pairs.append((idx1, idx2))
        return pairs

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
        frame_paths = video_data['frame_paths']
        
        src_images = []
        tgt_images = []
        
        # Sample frame pairs
        frame_pairs = self.sample_frame_pairs(frame_paths)
        
        # Load and process each pair
        for idx1, idx2 in frame_pairs:
            frame_cur = self.load_frame(frame_paths[idx1])
            frame_fut = self.load_frame(frame_paths[idx2])
            
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


if __name__ == "__main__":
    print("\nInitializing dataset...")
    dataset = PairedKineticsWithGlobalCaption(
        root="/home/junyoon/kinetics400",
        caption_path="/home/junyoon/RSP/artifacts/global/results/frame_analysis_results_complete.json",
        embeddings_path="/home/junyoon/RSP/artifacts/global/embedding_results.jsonl",
    )

    print(f"\nTotal number of videos: {len(dataset)}")

    # Test loading a few samples
    samples = [dataset[i] for i in [0, 1, 500, 501]]

    # Print cosine similarities between embeddings
    print("\nComputing cosine similarities between embeddings:")
    embeddings = [s['embeddings'][0] for s in samples]  # Take first sample from each
    names = ['0', '1', '500', '501']

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            )
            print(f"Similarity between video {names[i]} and {names[j]}: {sim.item():.4f}")

    # Test dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print("\nTesting dataloader...")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"Batch shapes:")
            print(f"src_images: {batch['src_images'].shape}")
            print(f"tgt_images: {batch['tgt_images'].shape}")
            print(f"embeddings: {batch['embeddings'].shape}")
            break

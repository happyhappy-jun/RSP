import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
import cv2
import h5pickle as h5py
from torchvision import transforms
import random
from util.transform import PairedRandomResizedCrop


class MemmapPairedKineticsDataset(Dataset):
    def __init__(self, memmap_dir, repeated_sampling=2):
        """
        Args:
            memmap_dir (str): Directory containing the preprocessed memory-mapped files
            repeated_sampling (int): Number of pairs to sample per video
        """
        super().__init__()

        # Load metadata
        metadata = np.load(
            os.path.join(memmap_dir, "metadata.npy"), allow_pickle=True
        ).item()
        self.dataset_size = metadata["dataset_size"]
        self.video_indices = metadata["video_indices"]
        self.pair_counts = metadata["pair_counts"]

        # Load frame paths
        with open(os.path.join(memmap_dir, "frame_paths.json"), "r") as f:
            self.frame_paths = json.load(f)

        # Open HDF5 file for embeddings
        self.embeddings_file = h5py.File(os.path.join(memmap_dir, "embeddings.h5"), "r")

        self.repeated_sampling = repeated_sampling

        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total number of videos: {len(self.frame_paths)}")
        print(f"Total number of samples: {self.dataset_size}")
        print(f"Repeated sampling per video: {self.repeated_sampling}")
        
        # Calculate average pairs per video
        total_pairs = sum(len(self.frame_paths[str(idx)]) for idx in self.video_indices)
        avg_pairs = total_pairs / len(self.frame_paths)
        print(f"Average pairs per video: {avg_pairs:.2f}")
        
        # Find min and max pairs
        min_pairs = min(len(self.frame_paths[str(idx)]) for idx in self.video_indices)
        max_pairs = max(len(self.frame_paths[str(idx)]) for idx in self.video_indices)
        print(f"Min pairs per video: {min_pairs}")
        print(f"Max pairs per video: {max_pairs}")

        # Setup transforms
        self.transforms = PairedRandomResizedCrop(seed=42)
        self.basic_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        video_idx = self.video_indices[index]
        video_data = self.frame_paths[str(video_idx)]
        available_pairs = len(video_data)

        # Randomly sample pairs
        pair_indices = random.sample(
            range(available_pairs), min(self.repeated_sampling, available_pairs)
        )

        src_images = []
        tgt_images = []
        embeddings = []
        future_embeddings = []

        video_group = self.embeddings_file[str(video_idx)]

        # Process each sampled pair
        for pair_idx in pair_indices:
            # Load frames
            pair_data = video_data[str(pair_idx)]
            frame_cur = self.load_frame(pair_data["cur"])
            frame_fut = self.load_frame(pair_data["fut"])

            # Apply transforms
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)

            # Load embeddings
            embedding = torch.from_numpy(video_group[f"embedding_{pair_idx}"][:])
            future_embedding = torch.from_numpy(
                video_group[f"future_embedding_{pair_idx}"][:]
            )

            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(embedding)
            future_embeddings.append(future_embedding)

        # If we need more samples, repeat the last pair
        while len(src_images) < self.repeated_sampling:
            src_images.append(src_images[-1])
            tgt_images.append(tgt_images[-1])
            embeddings.append(embeddings[-1])
            future_embeddings.append(future_embeddings[-1])

        return {
            "video_idx": video_idx,
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0),
            "future_embeddings": torch.stack(future_embeddings, dim=0),
        }

    def __del__(self):
        """Close the HDF5 file when the dataset is deleted"""
        if hasattr(self, "embeddings_file"):
            self.embeddings_file.close()


def collate_fn(batch):
    return {
        "src_images": torch.stack([x["src_images"] for x in batch], dim=0),
        "tgt_images": torch.stack([x["tgt_images"] for x in batch], dim=0),
        "embeddings": torch.stack([x["embeddings"] for x in batch], dim=0),
        "future_embeddings": torch.stack(
            [x["future_embeddings"] for x in batch], dim=0
        ),
    }


if __name__ == "__main__":
    # Example usage
    dataset = MemmapPairedKineticsDataset("memmap_data", repeated_sampling=8)
    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print("\nSample shapes:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")

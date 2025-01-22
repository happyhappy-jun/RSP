import os
from itertools import combinations, islice
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import transforms
from joblib import Parallel, delayed
import tqdm
import ujson as json  # Using ujson instead of standard json
from util.transform import PairedRandomResizedCrop

import torch
from torch.utils.data import Dataset


class MockPairedKineticsWithCaption8PairM3AE(Dataset):
    def __init__(self, dataset_size=240000, repeated_sampling=8, **kwargs):
        """
        Args:
            dataset_size (int): Number of video samples in the dataset (default: 240000)
            repeated_sampling (int): Number of pairs per video (default: 8)
        """
        super().__init__()
        self.dataset_size = dataset_size
        self.repeated_sampling = repeated_sampling

        # Pre-generate all data in memory
        print(
            f"Generating {dataset_size} samples with {repeated_sampling} pairs each..."
        )

        # Generate all tensors at once for better memory allocation
        self.src_images = torch.rand(dataset_size, repeated_sampling, 3, 224, 224)
        self.tgt_images = torch.rand(dataset_size, repeated_sampling, 3, 224, 224)
        self.embeddings = torch.rand(dataset_size, repeated_sampling, 512)
        self.future_embeddings = torch.rand(dataset_size, repeated_sampling, 3072)

        # Create mock valid_videos structure
        self.valid_videos = [(i, None) for i in range(dataset_size)]

        print("Data generation complete")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return {
            "video_idx": index,
            "src_images": self.src_images[index],
            "tgt_images": self.tgt_images[index],
            "embeddings": self.embeddings[index],
            "future_embeddings": self.future_embeddings[index],
        }


def process_lines(lines, is_future=True):
    """Process a batch of lines from jsonl file."""
    results = {}
    for line in lines:
        try:
            record = json.loads(line)
            parts = record[-1]["custom_id"].split("_")
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            embedding = record[1]["data"][0]["embedding"]
            if is_future:
                results[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)
            else:
                results[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)[
                    :512
                ]
        except Exception as e:
            print(f"Error processing line: {e}")
    return results


class PairedKineticsWithCaption8PairM3AE(Dataset):
    def _load_embeddings(self, embeddings_path, is_future=False):
        """Load embeddings from jsonl file using joblib parallel processing"""
        print(f"\nLoading embeddings from {embeddings_path}")
        if is_future:
            print(f"Loading future embeddings from {embeddings_path}")

        # Count total lines and calculate chunks
        with open(embeddings_path, "r") as f:
            total_lines = sum(1 for _ in f)

        # Optimized for 500GB RAM, 192 cores, and 200GB dataset
        chunk_size = 10000  # ~1GB per chunk (assuming avg 20KB per record)
        n_jobs = 30  # Slightly less than total cores to leave room for system processes

        print(f"Total lines to process: {total_lines:,}")
        print(f"Using {n_jobs} workers with chunk size of {chunk_size:,}")

        # Read file in chunks
        chunks = []
        with open(embeddings_path, "r") as f:
            while True:
                chunk = list(islice(f, chunk_size))
                if not chunk:
                    break
                chunks.append(chunk)

        # Process chunks in parallel using joblib
        print("\nProcessing chunks in parallel...")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(lambda chunk: process_lines(chunk, is_future))(chunk)
            for chunk in chunks
        )

        # Merge results
        print("\nMerging results...")
        if is_future:
            for chunk_result in tqdm.tqdm(results, desc="Merging future embeddings"):
                self.future_embeddings.update(chunk_result)
            print(
                f"\nFinished loading {len(self.future_embeddings):,} future embeddings"
            )
        else:
            for chunk_result in tqdm.tqdm(results, desc="Merging embeddings"):
                self.embeddings.update(chunk_result)
            print(f"\nFinished loading {len(self.embeddings):,} embeddings")

    def __init__(
        self,
        frame_root,
        frame_info_path,  # Path to frame_info.json
        embeddings_path,  # Path to combined_output.jsonl
        future_embeddings_path,  # Path to future embeddings jsonl
        frame_info_additional_path=None,  # Optional path to additional frame info
        embeddings_additional_path=None,  # Optional path to additional embeddings
        future_embeddings_additional_path=None,  # Optional path to additional future embeddings
        repeated_sampling=2,  # Number of augmented samples per pair
    ):
        super().__init__()
        self.frame_root = frame_root

        print("Loading main frame info data...")
        self.results = defaultdict(list)
        with open(frame_info_path, "r") as f:
            frame_info = json.load(f)
            videos = frame_info["videos"]
            self._process_frame_info(videos, prefix="frames")

        # Load main embeddings and future embeddings data
        print("Loading main embeddings data...")
        self.embeddings = {}
        self.future_embeddings = {}
        self._load_embeddings(embeddings_path, False)
        self._load_embeddings(future_embeddings_path, True)

        # Load additional data if provided
        if frame_info_additional_path and embeddings_additional_path:
            with open(frame_info_additional_path, "r") as f:
                frame_info_additional = json.load(f)
                self._process_frame_info(
                    frame_info_additional["videos"],
                    prefix="frames_additional",
                    pair_idx_offset=2,  # Add 2 to pair_idx for additional dataset
                )
            self._load_embeddings(embeddings_additional_path, False)

        # Filter and group valid pairs by video_idx
        self.video_pairs = defaultdict(list)
        missing_embeddings = defaultdict(list)

        for video_idx, frame_data in self.results.items():
            for pair in frame_data:
                key = (video_idx, pair["pair_idx"])
                if key in self.embeddings and key in self.future_embeddings:
                    pair["embedding"] = self.embeddings[key]
                    pair["future_embedding"] = self.future_embeddings[key]
                    self.video_pairs[video_idx].append(pair)
                else:
                    missing_embeddings[video_idx].append(pair["pair_idx"])

        # Convert to list of (video_idx, pairs) for indexing
        self.valid_videos = [
            (video_idx, pairs) for video_idx, pairs in self.video_pairs.items()
        ]

        print(f"\nDataset Statistics:")
        print(f"Total videos: {len(self.valid_videos)}")
        print(f"Total pairs found: {sum(len(pairs) for _, pairs in self.valid_videos)}")
        print(f"Total embeddings found: {len(self.embeddings)}")
        print(f"Videos with missing embeddings: {len(missing_embeddings)}")

        # Print some example pair counts
        print("\nExample video pair counts:")
        for video_idx, pairs in list(self.video_pairs.items())[:5]:
            print(
                f"Video {video_idx}: {len(pairs)} pairs - Pair indices: {[p['pair_idx'] for p in pairs]}"
            )

        del self.video_pairs  # Keep only the flattened valid_videos list
        del self.embeddings  # We can safely delete embeddings now

        self.repeated_sampling = repeated_sampling

        # Setup transforms with seed
        self.transforms = PairedRandomResizedCrop(seed=42)
        self.basic_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _process_frame_info(self, frames, prefix="frames", pair_idx_offset=0):
        """Process frame info with prefix and optional pair_idx offset"""
        for frame in tqdm.tqdm(frames, desc=f"Processing {prefix} frame info"):
            video_idx = frame["video_idx"]
            frame_paths = frame["frame_paths"]
            # Apply offset to pair_idx for additional dataset
            pair_idx = frame["pair_idx"] + pair_idx_offset

            # Add prefix to frame paths
            processed_paths = [
                self._process_path(f"{prefix}/{path}") for path in frame_paths
            ]

            self.results[video_idx].extend(
                [
                    {
                        "video_idx": video_idx,
                        "pair_idx": pair_idx,
                        "frame_cur_path": processed_paths[0],
                        "frame_fut_path": processed_paths[1],
                    }
                ]
            )

    def _process_path(self, frame_path):
        """add frame_root to frame_path"""
        return f"{self.frame_root}/{frame_path}"

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {frame_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.valid_videos)

    def __getitem__(self, index):
        # Get all pairs for the video at index
        video_idx, video_pairs = self.valid_videos[index]

        src_images = []
        tgt_images = []
        embeddings = []
        future_embeddings = []

        # Randomly sample pairs for this video
        sampled_pairs = random.sample(
            video_pairs, min(self.repeated_sampling, len(video_pairs))
        )

        # Process each sampled pair
        for pair in sampled_pairs:
            frame_cur = self.load_frame(pair["frame_cur_path"])
            frame_fut = self.load_frame(pair["frame_fut_path"])
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)

            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(torch.from_numpy(pair["embedding"]))
            future_embeddings.append(torch.from_numpy(pair["future_embedding"]))

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
    import tqdm
    import sys
    from torch.utils.data import DataLoader
    import random

    print("\nInitializing dataset...")
    dataset = PairedKineticsWithCaption8PairM3AE(
        frame_root="/data/kinetics400caption",
        frame_info_path="/data/kinetics400caption/frame_info.json",
        embeddings_path="/data/kinetics400caption/embedding_large_512.jsonl",
        future_embeddings_path="/data/kinetics400caption/future_embedding_8.jsonl",
        frame_info_additional_path="/data/kinetics400caption/frame_info_additional.json",
        embeddings_additional_path="/data/kinetics400caption/embedding_6_pair_512.jsonl",
    )

    print(f"\nTotal number of valid pairs: {len(dataset)}")

    # Print some random samples from results
    print("\nRandom samples from dataset:")
    if dataset.results:
        video_indices = list(dataset.results.keys())
        sample_videos = random.sample(video_indices, min(5, len(video_indices)))
        for video_idx in sample_videos:
            pairs = dataset.results[video_idx]
            if pairs:  # Check if there are any pairs for this video
                sample_pair = random.choice(pairs)
                print(f"\nVideo {video_idx}:")
                print(f"Frame current path: {sample_pair['frame_cur_path']}")
                print(f"Frame future path: {sample_pair['frame_fut_path']}")
                print(f"Pair idx: {sample_pair['pair_idx']}")

    # Create dataloader with small batch size for validation
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    print("\nValidating dataset by attempting to load all samples...")
    total_samples = len(dataloader)
    failed_indices = []

    try:
        for idx, batch in tqdm.tqdm(enumerate(dataloader), total=total_samples):
            # Verify batch contents
            assert (
                batch["src_images"].shape[0] == batch["tgt_images"].shape[0]
            ), f"Batch {idx}: Mismatched batch sizes between source and target images"
            assert (
                batch["embeddings"].shape[0] == batch["src_images"].shape[0]
            ), f"Batch {idx}: Mismatched batch sizes between images and embeddings"

            # Check for NaN values
            if torch.isnan(batch["src_images"]).any():
                failed_indices.append((idx, "NaN in source images"))
            if torch.isnan(batch["tgt_images"]).any():
                failed_indices.append((idx, "NaN in target images"))
            if torch.isnan(batch["embeddings"]).any():
                failed_indices.append((idx, "NaN in embeddings"))
            if torch.isnan(batch["future_embeddings"]).any():
                failed_indices.append((idx, "NaN in future embeddings"))

    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        sys.exit(1)

    print("\nDataset validation complete!")
    print(f"Successfully processed {total_samples} batches")

    if failed_indices:
        print("\nWarning: Found issues in the following batches:")
        for idx, issue in failed_indices:
            print(f"Batch {idx}: {issue}")
    else:
        print("No issues found. All samples are valid and accessible.")

    # Print sample batch shapes
    sample_batch = next(iter(dataloader))
    print("\nSample batch shapes:")
    print(f"Source images: {sample_batch['src_images'].shape}")
    print(f"Target images: {sample_batch['tgt_images'].shape}")
    print(f"Embeddings: {sample_batch['embeddings'].shape}")
    print(f"Future Embeddings: {sample_batch['future_embeddings'].shape}")

    # Print shapes of individual samples within the batch
    print("\nIndividual sample shapes in batch:")
    print(f"Source image shape (per sample): {sample_batch['src_images'][0].shape}")
    print(f"Target image shape (per sample): {sample_batch['tgt_images'][0].shape}")
    print(f"Embedding shape (per sample): {sample_batch['embeddings'][0].shape}")
    print(
        f"Future embedding shape (per sample): {sample_batch['future_embeddings'][0].shape}"
    )

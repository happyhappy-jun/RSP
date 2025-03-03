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


def process_lines(lines):
    """Process a batch of lines from jsonl file."""
    results = {}
    for line in lines:
        try:
            record = json.loads(line)
            parts = record[-1]["custom_id"].split("_")
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            embedding = record[1]["data"][0]["embedding"]
            results[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)[:384]
        except Exception as e:
            print(f"Error processing line: {e}")
    return results


class PairedKineticsWithCaption8Pair(Dataset):
    def _load_embeddings(self, embeddings_path):
        """Load embeddings from jsonl file using joblib parallel processing"""
        print(f"\nLoading embeddings from {embeddings_path}")

        # Count total lines and calculate chunks
        with open(embeddings_path, "r") as f:
            total_lines = sum(1 for _ in f)

        # Optimized for 500GB RAM, 192 cores, and 200GB dataset
        chunk_size = 10000  # ~1GB per chunk (assuming avg 20KB per record)
        n_jobs = 60  # Slightly less than total cores to leave room for system processes

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
            delayed(process_lines)(chunk) for chunk in chunks
        )

        # Merge results
        print("\nMerging results...")
        for chunk_result in tqdm.tqdm(results, desc="Merging embeddings"):
            self.embeddings.update(chunk_result)

        print(f"\nFinished loading {len(self.embeddings):,} embeddings")


    def shuffle_embeddings_dict(self, embeddings_dict, seed=42):
        """
        Shuffle the values in the embeddings dictionary while keeping the keys intact.
        
        Args:
            embeddings_dict (dict): Dictionary with (video_idx, pair_idx) as keys and embeddings as values
            seed (int): Random seed for reproducibility
        
        Returns:
            dict: New dictionary with shuffled embeddings
        """
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Get all keys and values
        keys = list(embeddings_dict.keys())
        values = list(embeddings_dict.values())
        
        # Shuffle the values
        random.shuffle(values)
        
        # Create new dictionary with original keys but shuffled values
        shuffled_dict = {key: value for key, value in zip(keys, values)}
        
        return shuffled_dict


    def __init__(
        self,
        frame_root,
        frame_info_path,  # Path to frame_info.json
        embeddings_path,  # Path to combined_output.jsonl
        frame_info_additional_path=None,  # Optional path to additional frame info
        embeddings_additional_path=None,  # Optional path to additional embeddings,
        shuffle=False,
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

        # Load main embeddings data
        print("Loading main embeddings data...")
        self.embeddings = {}
        self._load_embeddings(embeddings_path)

        # Load additional data if provided
        if frame_info_additional_path and embeddings_additional_path:
            with open(frame_info_additional_path, "r") as f:
                frame_info_additional = json.load(f)
                self._process_frame_info(
                    frame_info_additional["videos"],
                    prefix="frames_additional",
                    pair_idx_offset=2,  # Add 2 to pair_idx for additional dataset
                )
            self._load_embeddings(embeddings_additional_path)

        # Filter and group valid pairs by video_idx
        self.video_pairs = defaultdict(list)
        missing_embeddings = defaultdict(list)

        if shuffle:
            print("Shuffling embedding for random embedding")
            self.embeddings = self.shuffle_embeddings_dict(self.embeddings)

        for video_idx, frame_data in self.results.items():
            for pair in frame_data:
                key = (video_idx, pair["pair_idx"])
                if key in self.embeddings:
                    pair["embedding"] = self.embeddings[key]
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
        self.transforms = PairedRandomResizedCrop(seed=42, hflip_p=0)
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
                (
                    self._process_path(os.path.join(prefix, path))
                    if prefix
                    else self._process_path(path)
                )
                for path in frame_paths
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

        # If we need more samples, repeat the last pair
        while len(src_images) < self.repeated_sampling:
            src_images.append(src_images[-1])
            tgt_images.append(tgt_images[-1])
            embeddings.append(embeddings[-1])

        return {
            "video_idx": video_idx,
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0),
        }


def collate_fn(batch):
    return {
        "src_images": torch.stack([x["src_images"] for x in batch], dim=0),
        "tgt_images": torch.stack([x["tgt_images"] for x in batch], dim=0),
        "embeddings": torch.stack([x["embeddings"] for x in batch], dim=0),
    }


if __name__ == "__main__":
    import tqdm
    import sys
    from torch.utils.data import DataLoader
    import random

    print("\nInitializing dataset...")
    dataset = PairedKineticsWithCaption8Pair(
        frame_root="/data/rlwrld-common/junyoon/kinetics400caption",
        frame_info_path="/data/rlwrld-common/junyoon/kinetics400caption/frame_info.json",
        embeddings_path="/data/rlwrld-common/junyoon/kinetics400caption/embedding_large_512.jsonl",
        frame_info_additional_path="/data/rlwrld-common/junyoon/kinetics400caption/frame_info_additional.json",
        embeddings_additional_path="/data/rlwrld-common/junyoon/kinetics400caption/embedding_6_pair_512.jsonl",
    )
    
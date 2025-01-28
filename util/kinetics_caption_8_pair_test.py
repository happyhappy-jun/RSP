import h5py
import torch
import random
import cv2
import numpy as np
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from util.transform import PairedRandomResizedCrop
import mmap
import os
import json
from typing import Dict, List, Tuple
import gc


class MemoryEfficientJsonlReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.line_offsets = []
        self._build_index()

    def _build_index(self):
        """Build an index of line positions"""
        with open(self.file_path, 'rb') as f:
            # Memory-map the file
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Record the starting position of each line
            pos = 0
            while pos < self.file_size:
                self.line_offsets.append(pos)
                pos = mm.find(b'\n', pos) + 1
                if pos == 0:  # No more newlines found
                    break

            mm.close()

    def get_embedding(self, index: int) -> Tuple[int, int, np.ndarray]:
        """Get embedding at specific index"""
        with open(self.file_path, 'rb') as f:
            f.seek(self.line_offsets[index])
            line = f.readline().decode('utf-8')
            record = json.loads(line)
            parts = record[-1]['custom_id'].split('_')
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            embedding = record[1]['data'][0]['embedding']
            return video_idx, pair_idx, np.array(embedding, dtype=np.float32)[:512]


class EfficientDataset(Dataset):
    def __init__(
            self,
            frame_root: str,
            frame_info_path: str,
            embeddings_path: str,
            frame_info_additional_path: str = None,
            embeddings_additional_path: str = None,
            repeated_sampling: int = 2,
            cache_size: int = 1000  # Number of embeddings to cache per worker
    ):
        super().__init__()
        self.frame_root = frame_root
        self.repeated_sampling = repeated_sampling
        self.cache_size = cache_size

        # Initialize transforms
        self.transforms = PairedRandomResizedCrop(seed=42)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Process frame info
        self.video_pairs: Dict[int, List[Dict]] = {}
        self._process_frame_info(frame_info_path, "frames")
        if frame_info_additional_path:
            self._process_frame_info(frame_info_additional_path, "frames_additional", 2)

        # Setup embeddings readers
        self.embedding_readers = [MemoryEfficientJsonlReader(embeddings_path)]
        if embeddings_additional_path:
            self.embedding_readers.append(MemoryEfficientJsonlReader(embeddings_additional_path))

        # Convert video_pairs to list for indexing
        self.videos = list(self.video_pairs.items())

        # Initialize per-worker cache
        self.embedding_cache = {}

    def _process_frame_info(self, info_path: str, prefix: str, pair_idx_offset: int = 0):
        """Process frame info file"""
        with open(info_path, 'r') as f:
            frame_info = json.load(f)
            for video in frame_info['videos']:
                video_idx = video['video_idx']
                if video_idx not in self.video_pairs:
                    self.video_pairs[video_idx] = []

                frame_paths = [
                    os.path.join(self.frame_root, prefix, path)
                    for path in video['frame_paths']
                ]

                self.video_pairs[video_idx].append({
                    'pair_idx': video['pair_idx'] + pair_idx_offset,
                    'frame_cur_path': frame_paths[0],
                    'frame_fut_path': frame_paths[1]
                })

    def _get_worker_cache(self):
        """Get or create worker-specific embedding cache"""
        worker = get_worker_info()
        worker_id = worker.id if worker else 0

        if worker_id not in self.embedding_cache:
            self.embedding_cache[worker_id] = {}

        return self.embedding_cache[worker_id]

    def _get_embedding(self, video_idx: int, pair_idx: int) -> np.ndarray:
        """Get embedding from cache or load it"""
        cache = self._get_worker_cache()
        key = (video_idx, pair_idx)

        if key not in cache:
            # If cache is full, remove oldest items
            if len(cache) >= self.cache_size:
                # Remove 10% of oldest items
                remove_count = self.cache_size // 10
                for _ in range(remove_count):
                    cache.popitem(last=False)

            # Try to find embedding in any reader
            for reader in self.embedding_readers:
                try:
                    found_video_idx, found_pair_idx, embedding = reader.get_embedding(pair_idx)
                    if found_video_idx == video_idx and found_pair_idx == pair_idx:
                        cache[key] = embedding
                        break
                except:
                    continue

            if key not in cache:
                raise KeyError(f"Embedding not found for video {video_idx}, pair {pair_idx}")

        return cache[key]

    def load_frame(self, frame_path: str) -> np.ndarray:
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {frame_path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Get video and its pairs
        video_idx, pairs = self.videos[index]

        # Randomly sample pairs
        sampled_pairs = random.sample(pairs, min(self.repeated_sampling, len(pairs)))

        src_images = []
        tgt_images = []
        embeddings = []

        for pair in sampled_pairs:
            # Load frames
            frame_cur = self.load_frame(pair['frame_cur_path'])
            frame_fut = self.load_frame(pair['frame_fut_path'])

            # Apply transforms
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)

            # Get embedding
            embedding = torch.from_numpy(self._get_embedding(video_idx, pair['pair_idx']))

            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(embedding)

            # Clear some memory
            del frame_cur, frame_fut

        # Handle case when we need more samples
        while len(src_images) < self.repeated_sampling:
            src_images.append(src_images[-1])
            tgt_images.append(tgt_images[-1])
            embeddings.append(embeddings[-1])

        # Stack tensors
        return {
            "video_idx": video_idx,
            "src_images": torch.stack(src_images),
            "tgt_images": torch.stack(tgt_images),
            "embeddings": torch.stack(embeddings)
        }


def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch]),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch]),
        "embeddings": torch.stack([x['embeddings'] for x in batch]),
    }
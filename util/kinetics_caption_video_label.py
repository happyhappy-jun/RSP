import os
from itertools import combinations
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torchvision import transforms
from util.transform import PairedRandomResizedCrop

class PairedKineticsCaptionVideoLabel(Dataset):
    def __init__(
        self,
        frame_root,
        frame_info_path,
        class_embeddings_path,  # New parameter for class embeddings
        frame_info_additional_path=None,
        shuffle=False,
        repeated_sampling=2,
    ):
        super().__init__()
        self.frame_root = frame_root

        # Load class embeddings
        print("Loading class embeddings...")
        with open(class_embeddings_path, 'r') as f:
            self.class_embeddings = json.load(f)

        print("Loading main frame info data...")
        self.results = defaultdict(list)
        with open(frame_info_path, "r") as f:
            frame_info = json.load(f)
            videos = frame_info["videos"]
            self._process_frame_info(videos, prefix="frames")

        # Load additional data if provided
        if frame_info_additional_path:
            with open(frame_info_additional_path, "r") as f:
                frame_info_additional = json.load(f)
                self._process_frame_info(
                    frame_info_additional["videos"],
                    prefix="frames_additional",
                    pair_idx_offset=2,
                )

        # Filter and group valid pairs by video_idx
        self.video_pairs = defaultdict(list)
        missing_embeddings = defaultdict(list)

        if shuffle:
            print("Shuffling embeddings for random embedding")
            shuffled_classes = list(self.class_embeddings.keys())
            random.shuffle(shuffled_classes)
            self.class_embeddings = {
                orig: self.class_embeddings[shuffled]
                for orig, shuffled in zip(self.class_embeddings.keys(), shuffled_classes)
            }

        for video_idx, frame_data in self.results.items():
            for pair in frame_data:
                if pair["class_label"] in self.class_embeddings:
                    pair["embedding"] = np.array(self.class_embeddings[pair["class_label"]], dtype=np.float32)[:512]
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
        print(f"Total class embeddings: {len(self.class_embeddings)}")
        print(f"Videos with missing embeddings: {len(missing_embeddings)}")

        # Print some example pair counts
        print("\nExample video pair counts:")
        for video_idx, pairs in list(self.video_pairs.items())[:5]:
            print(
                f"Video {video_idx}: {len(pairs)} pairs - Class: {pairs[0]['class_label']}"
            )

        del self.video_pairs  # Keep only the flattened valid_videos list
        del self.class_embeddings  # We can safely delete embeddings now

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
        for frame in frames:
            video_idx = frame["video_idx"]
            frame_paths = frame["frame_paths"]
            class_label = frame["class_label"]  # New: get class name from frame info
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
                        "class_label": class_label,  # New: store class name
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
        class_labels = []  # New: track class names for debugging

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
            class_labels.append(pair["class_label"])

        # If we need more samples, repeat the last pair
        while len(src_images) < self.repeated_sampling:
            src_images.append(src_images[-1])
            tgt_images.append(tgt_images[-1])
            embeddings.append(embeddings[-1])
            class_labels.append(class_labels[-1])

        return {
            "video_idx": video_idx,
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0),
        }


def collate_fn(batch):
    return {
        "video_idx": [x["video_idx"] for x in batch],
        "src_images": torch.stack([x["src_images"] for x in batch], dim=0),
        "tgt_images": torch.stack([x["tgt_images"] for x in batch], dim=0),
        "embeddings": torch.stack([x["embeddings"] for x in batch], dim=0),
    }

if __name__ == "__main__":
    # Dataset paths
    frame_root = "/data/kinetics400caption"
    frame_info_path = "/data/kinetics400caption/frame_info.json"
    class_embeddings_path = "/data/kinetics400caption/class_embeddings.json"
    frame_info_additional_path = "/data/kinetics400caption/frame_info_additional.json"

    # Create dataset
    print("Initializing dataset...")
    dataset = PairedKineticsCaptionVideoLabel(
        frame_root=frame_root,
        frame_info_path=frame_info_path,
        class_embeddings_path=class_embeddings_path,
        frame_info_additional_path=frame_info_additional_path,
        repeated_sampling=2
    )

    # Create dataloader
    batch_size = 4
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("\nDataset size:", len(dataset))
    print("Number of batches:", len(dataloader))

    # Test a few batches
    print("\nTesting batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # Test first 2 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"Batch size: {batch['src_images'].size()}")
        print(f"Source images shape: {batch['src_images'].size()}")
        print(f"Target images shape: {batch['tgt_images'].size()}")
        print(f"Embeddings shape: {batch['embeddings'].size()}")
        print(f"Video indices: {batch['video_idx']}")
        print(f"Class names: {batch['class_labels'][0]}")  # Print first item's classes

        # Check embedding values
        print(f"Embedding sample (first 5 values): {batch['embeddings'][0][0][:5]}")

        # Memory check
        if torch.cuda.is_available():
            print("\nGPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


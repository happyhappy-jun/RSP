from itertools import combinations
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from models import BatchOutput
from torchvision import transforms
from util.transform import PairedRandomResizedCrop

class PairedKineticsWithCaption8Pair(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON with support for additional data"""
    def __init__(
        self,
        frame_root,
        frame_info_path,      # Path to frame_info.json
        embeddings_path,      # Path to combined_output.jsonl
        frame_info_additional_path=None,  # Optional path to additional frame info
        embeddings_additional_path=None,  # Optional path to additional embeddings
        repeated_sampling=2   # Number of augmented samples per pair
    ):
        super().__init__()
        self.frame_root = frame_root
        
        # Load main frame info data
        self.results = defaultdict(list)
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            self._process_frame_info(frame_info['videos'], prefix="frames")

        # Load main embeddings data
        self.embeddings = {}
        self._load_embeddings(embeddings_path)

        # Load additional data if provided
        if frame_info_additional_path and embeddings_additional_path:
            with open(frame_info_additional_path, 'r') as f:
                frame_info_additional = json.load(f)
                self._process_frame_info(
                    frame_info_additional['videos'], 
                    prefix="frames_additional",
                    pair_idx_offset=2  # Add 2 to pair_idx for additional dataset
                )
            self._load_embeddings(embeddings_additional_path)

        # Filter and flatten pairs that have embeddings
        self.valid_pairs = []
        missing_embeddings = defaultdict(list)

        for video_idx, frame_data in self.results.items():
            for pair in frame_data:
                if (video_idx, pair["pair_idx"]) in self.embeddings:
                    self.valid_pairs.extend([pair])
                else:
                    missing_embeddings[video_idx].append(pair["pair_idx"])

        self.results = self.valid_pairs

        print(f"\nDataset Statistics:")
        print(f"Total pairs found: {len(self.valid_pairs)}")
        print(f"Total embeddings found: {len(self.embeddings)}")
        print(f"Videos with missing embeddings: {len(missing_embeddings)}")

        # Print some example missing embeddings
        if missing_embeddings:
            print("\nExample videos with missing embeddings:")
            for video_idx, pairs in list(missing_embeddings.items())[:5]:  # Show first 5 examples
                print(f"Video {video_idx}: Missing {len(pairs)} pairs - {pairs}")

        self.repeated_sampling = repeated_sampling
        
        # Setup transforms with seed
        self.transforms = PairedRandomResizedCrop(seed=42)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

        del self.embeddings

    def _process_frame_info(self, frames, prefix="frames", pair_idx_offset=0):
        """Process frame info with prefix and optional pair_idx offset"""
        for frame in frames:
            video_idx = frame['video_idx']
            frame_paths = frame['frame_paths']
            # Apply offset to pair_idx for additional dataset
            pair_idx = frame['pair_idx'] + pair_idx_offset
            
            # Add prefix to frame paths
            processed_paths = [
                self._process_path(f"{prefix}/{path}") 
                for path in frame_paths
            ]
            
            self.results[video_idx].extend([{
                'pair_idx': pair_idx,
                'frame_cur_path': processed_paths[0],
                'frame_fut_path': processed_paths[1],
            }])

    def _load_embeddings(self, embeddings_path):
        """Load embeddings from jsonl file"""
        with open(embeddings_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                # Parse video_idx and pair_idx from custom_id (format: video_X_pair_Y)
                parts = record[-1]['custom_id'].split('_')
                video_idx = int(parts[1])
                pair_idx = int(parts[-1])
                embedding = record[1]['data'][0]['embedding']
                self.embeddings[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)

    def __len__(self):
        return len(self.results)

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
        sample = self.results[index]
        
        src_images = []
        tgt_images = []
        embeddings = []

        # Create repeated samples with different augmentations
        for _ in range(self.repeated_sampling):
            frame_cur = self.load_frame(sample["frame_cur_path"])
            frame_fut = self.load_frame(sample["frame_fut_path"])
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            embeddings.append(torch.from_numpy(
                self.embeddings[(sample["video_idx"], sample["pair_idx"])]
            ))

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0)
        }

def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "embeddings": torch.stack([x['embeddings'] for x in batch], dim=0),
    }

if __name__ == "__main__":
    import tqdm
    import sys
    from torch.utils.data import DataLoader
    import random

    print("\nInitializing dataset...")
    dataset = PairedKineticsWithCaption8Pair(
        frame_root="/data/kinetics400caption",
        frame_info_path="/data/kinetics400caption/frame_info.json",
        embeddings_path="/data/kinetics400caption/embeddings.jsonl",
        frame_info_additional_path="/data/kinetics400caption8/frame_info_additional.json",
        embeddings_additional_path="/data/kinetics400caption8/embeddings_additional.jsonl"
    )
    
    print(f"\nTotal number of valid pairs: {len(dataset)}")
    
    # Print some random samples from results
    print("\nRandom samples from dataset:")
    sample_indices = random.sample(range(len(dataset.results)), min(5, len(dataset.results)))
    for idx in sample_indices:
        sample = dataset.results[idx]
        print(f"\nSample {idx}:")
        print(f"Frame current path: {sample['frame_cur_path']}")
        print(f"Frame future path: {sample['frame_fut_path']}")
        print(f"Pair idx: {sample['pair_idx']}")

    # Create dataloader with small batch size for validation
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    print("\nValidating dataset by attempting to load all samples...")
    total_samples = len(dataloader)
    failed_indices = []

    try:
        for idx, batch in tqdm.tqdm(enumerate(dataloader), total=total_samples):
            # Verify batch contents
            assert batch['src_images'].shape[0] == batch['tgt_images'].shape[0], \
                f"Batch {idx}: Mismatched batch sizes between source and target images"
            assert batch['embeddings'].shape[0] == batch['src_images'].shape[0], \
                f"Batch {idx}: Mismatched batch sizes between images and embeddings"

            # Check for NaN values
            if torch.isnan(batch['src_images']).any():
                failed_indices.append((idx, "NaN in source images"))
            if torch.isnan(batch['tgt_images']).any():
                failed_indices.append((idx, "NaN in target images"))
            if torch.isnan(batch['embeddings']).any():
                failed_indices.append((idx, "NaN in embeddings"))

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
    
    # Print shapes of individual samples within the batch
    print("\nIndividual sample shapes in batch:")
    print(f"Source image shape (per sample): {sample_batch['src_images'][0].shape}")
    print(f"Target image shape (per sample): {sample_batch['tgt_images'][0].shape}")
    print(f"Embedding shape (per sample): {sample_batch['embeddings'][0].shape}")
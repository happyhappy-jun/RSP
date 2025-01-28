import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from util.transform import PairedRandomResizedCrop


class PairedKineticsWithCaption8PairH5(Dataset):
    def __init__(
            self,
            h5_path,
            repeated_sampling=2
    ):
        super().__init__()
        self.h5_path = h5_path
        self.repeated_sampling = repeated_sampling

        # Open H5 file in read mode
        self.h5_file = h5py.File(h5_path, 'r')

        # Get all valid video-pair combinations
        self.valid_pairs = []
        for video_idx in self.h5_file['images'].keys():
            video_group = self.h5_file['images'][video_idx]
            pairs = [(video_idx, pair_idx) for pair_idx in video_group.keys()]
            if pairs:
                self.valid_pairs.append((video_idx, pairs))

        print(f"\nDataset Statistics:")
        print(f"Total videos: {len(self.valid_pairs)}")
        print(f"Total pairs: {sum(len(pairs) for _, pairs in self.valid_pairs)}")

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
        return len(self.valid_pairs)

    def _load_pair(self, video_idx, pair_idx):
        """Load a single pair of frames and its embedding"""
        # Get frame data
        pair_group = self.h5_file['images'][video_idx][pair_idx]
        frame_cur = pair_group['frame_cur'][:]
        frame_fut = pair_group['frame_fut'][:]

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
        video_idx, video_pairs = self.valid_pairs[index]

        src_images = []
        tgt_images = []
        embeddings = []

        # Randomly sample pairs for this video
        sampled_pairs = random.sample(video_pairs,
                                      min(self.repeated_sampling, len(video_pairs)))

        # Process each sampled pair
        for _, pair_idx in sampled_pairs:
            src_image, tgt_image, embedding = self._load_pair(
                str(video_idx), str(pair_idx)
            )
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
    args = parser.parse_args()

    print("\nInitializing dataset...")
    dataset = PairedKineticsWithCaption8PairH5(
        h5_path=args.h5_path
    )

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


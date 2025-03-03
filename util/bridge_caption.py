import io
import os
import random
import logging
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer
from util.transform import PairedRandomResizedCrop

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler("traj_reader_iter.log")],
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_image_from_bytes(byte_data):
    """Convert raw bytes into a NumPy array (H, W, C)."""
    with io.BytesIO(byte_data) as buffer:
        with Image.open(buffer) as img:
            return np.array(img.convert("RGB"))
        
def npy_to_numpy_array(file_path):
    """Load numpy array from file, allowing for object arrays."""
    return np.load(file_path, allow_pickle=True)
class BridgeCaption(Dataset):
    def __init__(
        self,
        data_dir: str,
        repeated_sampling: int = 2,
        interval: int = 4,
        tokenizer_name: str = "Alibaba-NLP/gte-large-en-v1.5",
        max_length: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.repeated_sampling = repeated_sampling
        self.interval = interval
        self.max_length = max_length
        self.seed = seed

        self.pair_transform = PairedRandomResizedCrop(seed=seed, hflip_p=0)

        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Discover all trajectory files, excluding those insufficient for repeated_sampling
        self.traj_files = self._get_trajectory_files()
        print(f"Loaded {len(self.traj_files)} eligible trajectories")

    def _get_trajectory_files(self):
        traj_files = []
        for traj_dir in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, traj_dir)
            if os.path.isdir(full_path):
                images_files = sorted([f for f in os.listdir(full_path) if f.startswith('images')])
                moves_files = sorted([f for f in os.listdir(full_path) if f.startswith('moves')])
                
                image_map = {f.split('_traj')[1].split('.')[0]: f for f in images_files}
                move_map = {f.split('_traj')[1].split('.')[0]: f for f in moves_files}

                common_indices = set(image_map.keys()).intersection(set(move_map.keys()))

                for index in common_indices:
                    image_path = os.path.join(full_path, image_map[index])
                    move_path = os.path.join(full_path, move_map[index])
                    
                    # Load npy files to verify at least repeated_sampling pairs
                    images_list = npy_to_numpy_array(image_path)
                    if (len(images_list) - self.interval) >= self.repeated_sampling:
                        traj_files.append((image_path, move_path))
                
        return traj_files

    def __len__(self):
        return len(self.traj_files)

    def __getitem__(self, idx):
        obs_path, move_path = self.traj_files[idx]
        return self.process_trajectory(obs_path, move_path)
    
    def tokenize_text(self, text):
        """Tokenize a text string and return the token dictionary."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        

    def process_trajectory(self, obs_path, move_path):
        images_list = npy_to_numpy_array(obs_path)
        moves_list = npy_to_numpy_array(move_path)

        pair_indices = [(i, i + self.interval) for i in range(len(images_list) - self.interval)]
        sampled_pairs = random.sample(pair_indices, min(self.repeated_sampling, len(pair_indices)))

        src_images, tgt_images, embeddings = [], [], []

        for src_idx, tgt_idx in sampled_pairs:
            src_arr = load_image_from_bytes(images_list[src_idx])
            tgt_arr = load_image_from_bytes(images_list[tgt_idx])

            src_cropped, tgt_cropped = self.pair_transform(src_arr, tgt_arr)

            src_tensor = self.basic_transform(src_cropped)
            tgt_tensor = self.basic_transform(tgt_cropped)

            caption_text = moves_list[src_idx]
            token_dict = self.tokenize_text(caption_text)
            embedding = token_dict['input_ids']

            src_images.append(src_tensor)
            tgt_images.append(tgt_tensor)
            embeddings.append(embedding)
        
        assert len(src_images) == 2

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "captions": torch.stack(embeddings, dim=0)
        }

if __name__ == "__main__":
    # Example usage
    dataset = BridgeCaption(
        data_dir="/root/RSP/demo/npy_dataset",
        repeated_sampling=2,
        interval=4,
        seed=42,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=1,
    )
    
    # Test the length estimation
    print(f"Estimated length: {len(loader.dataset)}")
    
    # Count total samples to verify
    print(len(dataset))
    sample_count = 0
    for i, batch in enumerate(loader):
        print(
            f"Batch {i}: src_images={batch['src_images'].shape}, "
            f"tgt_images={batch['tgt_images'].shape}, "
            f"captions shape={batch['captions'].shape}"
        )
        sample_count += batch['src_images'].shape[0]
        if i >= 5:
            break
    
    print(f"Loaded {sample_count} samples successfully.")
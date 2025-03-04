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
import openai
import json
import asyncio
from util.transform import PairedRandomResizedCrop
from tqdm import tqdm

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
        seed: int = 42,
        embedding_json_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.repeated_sampling = repeated_sampling
        self.interval = interval
        self.seed = seed

        self.pair_transform = PairedRandomResizedCrop(seed=seed, hflip_p=0)

        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if embedding_json_path:
            with open(embedding_json_path, 'r') as f:
                self.embedding_map = json.load(f)
        else:
            self.embedding_map = {}

        # Discover all trajectory files, excluding those insufficient for repeated_sampling
        self.traj_files = self._get_trajectory_files()
        print(f"Loaded {len(self.traj_files)} eligible trajectories")

    def _get_trajectory_files(self):
        traj_files = []
        for traj_dir in tqdm(os.listdir(self.data_dir)):
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

            caption_text = str(moves_list[src_idx]).strip()
            if caption_text not in self.embedding_map:
                print(f"Caption not found in embedding map: {caption_text}")
                embedding = []
            else:
                embedding = self.embedding_map[caption_text]

            embedding = torch.tensor(embedding)
            print(embedding.shape)

            src_images.append(src_tensor)
            tgt_images.append(tgt_tensor)
            embeddings.append(embedding)
        
        assert len(src_images) == 2

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "captions": torch.stack(embeddings, dim=0)
        }

def get_all_unique_captions(data_dir: str) -> List[str]:
    """Collect unique movement captions from all trajectory files in the dataset."""
    dataset = BridgeCaption(data_dir)
    unique_captions = set()
    for _, move_path in dataset.traj_files:
        moves_list = npy_to_numpy_array(move_path)
        for caption in moves_list:
            unique_captions.add(str(caption).strip())
    return list(unique_captions)

async def precompute_embeddings(data_dir: str, output_json: str, openai_api_key: str):
    """Precompute embeddings for unique captions using OpenAI async API and save them to a JSON file."""
    openai.api_key = openai_api_key
    client = openai.OpenAI(api_key=openai_api_key)
    unique_captions = get_all_unique_captions(data_dir)
    embedding_map = {}
    semaphore = asyncio.Semaphore(50)  # Limit concurrent requests to ~50 (~3000 per minute)

    async def process_caption(caption):
        norm_caption = str(caption).strip()
        if not norm_caption:
            return
        async with semaphore:
            try:
                response = await asyncio.to_thread(client.embeddings.create, input=norm_caption, model="text-embedding-3-large")
                embedding = response.data[0].embedding
                embedding_map[norm_caption] = embedding
            except Exception as e:
                logger.error(f"Error computing embedding for caption {norm_caption}: {e}")

    tasks = [process_caption(caption) for caption in unique_captions + [""]]
    await asyncio.gather(*tasks)
    # Check for missing embeddings
    missing = [caption for caption in unique_captions if caption.strip() and caption not in embedding_map]
    if missing:
        logger.warning(f"{len(missing)} embeddings were not computed for captions: {missing}")
    with open(output_json, 'w') as f:
        json.dump(embedding_map, f)

if __name__ == "__main__":
    import argparse
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument('--precompute', action='store_true', help='Precompute embeddings for unique captions')
    parser.add_argument('--output_json', type=str, default='embedding_map.json', help='Output JSON file for embeddings')
    parser.add_argument('--data_dir', type=str, default='/root/RSP/demo/npy_dataset', help='Dataset directory')
    parser.add_argument('--openai_api_key', type=str, default='', help='OpenAI API Key')
    args = parser.parse_args()

    if args.precompute:
        if not args.openai_api_key:
            logger.error("OpenAI API Key is required for precomputation.")
        else:
            asyncio.run(precompute_embeddings(args.data_dir, args.output_json, args.openai_api_key))
            print(f"Precomputed embeddings saved to {args.output_json}")
    else:
        dataset = BridgeCaption(
            data_dir=args.data_dir,
            repeated_sampling=2,
            interval=4,
            seed=42,
            embedding_json_path='embedding_map.json'
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=1,
        )

        print(f"Loaded {len(dataset.traj_files)} eligible trajectories")
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

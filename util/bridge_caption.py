import os
import random
import logging
import pickle
import io
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import webdataset as wds
from transformers import AutoTokenizer
from util.transform import PairedRandomResizedCrop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("traj_reader_map.log")],
)
logger = logging.getLogger(__name__)

def bytes_to_numpy(byte_data):
    """
    Convert raw bytes (JPEG/PNG) into a NumPy array (H, W, C).
    """
    with io.BytesIO(byte_data) as buffer:
        with Image.open(buffer) as img:
            img = img.convert("RGB")  # ensure 3-channel
            arr = np.array(img)
    return arr

class BridgeCaption(Dataset):
    """
    A "map-style" Dataset version of the previous IterableDataset logic.
    This code:
      1) Expands all shard files and loads every trajectory sample into memory.
      2) For each trajectory, we store a list of possible (src_idx, tgt_idx) pairs
         or "items" in self.samples, giving a stable overall dataset length.
      3) At __getitem__, we decode, transform, and tokenize the text.

    WARNING:
      - This approach can be very memory-intensive if your dataset is large.
      - It's only suitable if you can afford to load all shards in memory
        or your dataset is small.
      - Because it's map-style, each worker in DDP will replicate the entire dataset
        unless you do custom splitting. Typically you'd rely on DistributedSampler
        or something similar to handle the split among ranks.

    Usage:
      dataset = BridgeCaptionMapDataset(
          wds_pattern="/root/RSP/demo/webdataset_shards_repeated/bridge_arm_move-*.tar.gz",
          interval=4, repeated_sampling=2, ...
      )
      # Then pass to DataLoader with distributed sampler in DDP context
    """

    def __init__(
        self,
        wds_pattern: str,
        repeated_sampling: int = 2,
        interval: int = 4,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        self.wds_pattern = wds_pattern
        self.shard_files = glob.glob(wds_pattern)
        if not self.shard_files:
            raise FileNotFoundError(f"No files match pattern {wds_pattern}")

        self.repeated_sampling = repeated_sampling
        self.interval = interval
        self.max_length = max_length
        random.seed(seed)

        # Pairwise transform
        self.pair_transform = PairedRandomResizedCrop(seed=seed, hflip_p=0)

        # Basic transform: PIL -> Tensor, normalize
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        logger.info(f"MapDataset: scanning all shards to build sample list.")
        logger.info(f"Found {len(self.shard_files)} shard files: {self.shard_files}")

        # self.samples will hold tuples of:
        #   (images_list, moves_list, src_idx, tgt_idx)
        # OR we can store raw bytes in memory to decode later
        self.samples: List[Dict[str, Any]] = []

        # Expand all shard files, parse each trajectory
        for shard_file in tqdm(self.shard_files):
            # Open each shard individually
            dataset_iter = wds.WebDataset([shard_file])

            for sample in dataset_iter:
                # Each sample has: "__key__", "images.pkl", "moves.pkl"
                images_pkl = sample["images.pkl"]
                moves_pkl = sample["moves.pkl"]

                images_list = pickle.loads(images_pkl)  # list of raw bytes
                moves_list = pickle.loads(moves_pkl)

                length = len(images_list)
                if length == 0:
                    continue

                # Repeated sampling approach:
                # We'll add 'repeated_sampling' items, each with random src_idx/tgt_idx
                # If you prefer enumerating all pairs, do a for-loops approach. 
                # But that might create an enormous dataset.
                for _ in range(self.repeated_sampling):
                    src_idx = random.randint(0, length-1)
                    tgt_idx = min(src_idx + self.interval, length-1)
                    self.samples.append({
                        "images_list": images_list,
                        "moves_list": moves_list,
                        "src_idx": src_idx,
                        "tgt_idx": tgt_idx,
                    })

        logger.info(f"Done building sample list, total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize one string, returns dict of Tensors with shape (seq_len,).
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return one training example with:
          "src_images": shape (3, H, W),
          "tgt_images": shape (3, H, W),
          "captions": { 'input_ids': (seq_len,), ... }
        """
        example = self.samples[idx]
        images_list = example["images_list"]
        moves_list = example["moves_list"]
        src_idx = example["src_idx"]
        tgt_idx = example["tgt_idx"]

        # Decode from raw bytes -> Numpy
        src_arr = bytes_to_numpy(images_list[src_idx])
        tgt_arr = bytes_to_numpy(images_list[tgt_idx])

        # Pairwise transform
        src_cropped, tgt_cropped = self.pair_transform(src_arr, tgt_arr)

        # Basic transform (PIL -> Tensor -> Normalized)
        src_tensor = self.basic_transform(src_cropped)
        tgt_tensor = self.basic_transform(tgt_cropped)

        # Tokenize the text from moves_list
        caption_text = moves_list[src_idx]
        token_dict = self.tokenize_text(caption_text)

        # Return a dict that will be collated by DataLoader
        return {
            "src_images": src_tensor,      # shape (3, H, W)
            "tgt_images": tgt_tensor,      # shape (3, H, W)
            "captions": token_dict,        # dict of { 'input_ids': (seq_len,), ... }
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    wds_url = "/root/RSP/demo/webdataset_shards_repeated/bridge_arm_move-*.tar.gz"
    dataset = BridgeCaption(
        wds_pattern=wds_url,
        repeated_sampling=2,
        interval=4,
        tokenizer_name="bert-base-uncased",
        max_length=128,
        seed=42,
    )

    logger.info(f"MapDataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, num_workers=2, shuffle=True)

    for i, batch in enumerate(loader):
        # batch["src_images"] => (B, 3, H, W)
        # batch["tgt_images"] => (B, 3, H, W)
        # batch["captions"] => dict of Tensors, each (B, seq_len)
        logger.info(f"Batch {i}: src_images={batch['src_images'].shape}, "
                    f"tgt_images={batch['tgt_images'].shape}, "
                    f"captions keys={list(batch['captions'].keys())}")
        if i >= 2:
            break
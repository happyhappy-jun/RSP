import random
import logging
import pickle
import io
import glob
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("traj_reader.log")]
)
logger = logging.getLogger(__name__)

class BridgeCaption(IterableDataset):
    """
    A WebDataset-based iterable dataset that streams trajectory data without loading all
    trajectories into memory. Each shard sample is expected to contain:
      - "__key__": a unique trajectory key
      - "images.pkl": pickled list of raw JPEG bytes
      - "moves.pkl": pickled list of move texts

    For repeated sampling, we pick random pairs from each trajectory sample in memory.
    We do NOT rely on a 'bufsize' parameter, as older WebDataset versions may not recognize it.
    """

    def __init__(
        self,
        wds_pattern: str,
        repeated_sampling: int = 2,
        interval: int = 4,
        transform: Optional[transforms.Compose] = None,
        shardshuffle: bool = True,
        shardshuffle_size: int = 10,
        sample_shuffle: int = 0
    ):
        """
        Args:
            wds_pattern: Glob or list of shard paths, e.g. "/root/shards/bridge_arm_move-*.tar.gz"
            repeated_sampling: How many pairs to sample from each trajectory per iteration.
            interval: The offset between frames (idx_cur, idx_cur + interval).
            transform: Optional transform to apply to each decoded image.
            shardshuffle: If True, randomize the order of shards.
            shardshuffle_size: If shardshuffle=True, how often to reshuffle shard order. (some versions accept this)
            sample_shuffle: If >0, apply an in-sample shuffle with that buffer size after loading.
        """
        super().__init__()
        self.wds_pattern = wds_pattern
        self.shard_files = glob.glob(wds_pattern)
        self.repeated_sampling = repeated_sampling
        self.interval = interval
        self.transform = transform
        self.shardshuffle = shardshuffle
        self.shardshuffle_size = shardshuffle_size
        self.sample_shuffle = sample_shuffle

    def process_trajectory(self, sample: Dict[str, Any]):
        """
        Given a single WebDataset sample that represents a full trajectory:
          - "images.pkl": pickled list of raw JPEG bytes
          - "moves.pkl": pickled list of strings
        We decode them, pick repeated_sampling random pairs, and yield them.
        """
        key = sample["__key__"]
        images_pkl = sample["images.pkl"]
        moves_pkl = sample["moves.pkl"]

        # Unpickle
        images_list = pickle.loads(images_pkl)  # A list of raw JPEG bytes
        moves_list = pickle.loads(moves_pkl)    # A list of strings
        length = len(images_list)

        for _ in range(self.repeated_sampling):
            # Randomly pick a src index
            idx_cur = random.randint(0, length - 1)
            idx_fut = idx_cur + self.interval
            if idx_fut >= length:
                idx_fut = min(idx_fut, length - 1)

            # Decode images
            src_jpeg_bytes = images_list[idx_cur]
            tgt_jpeg_bytes = images_list[idx_fut]

            src_pil = Image.open(
                io.BytesIO(src_jpeg_bytes)
            ).convert("RGB")
            tgt_pil = Image.open(
                io.BytesIO(tgt_jpeg_bytes)
            ).convert("RGB")

            # Transform
            if self.transform is not None:
                src_tensor = self.transform(src_pil)
                tgt_tensor = self.transform(tgt_pil)
            else:
                src_tensor = transforms.ToTensor()(src_pil)
                tgt_tensor = transforms.ToTensor()(tgt_pil)

            caption = moves_list[idx_cur]
            yield {
                "traj_key": key,
                "src_image": src_tensor,
                "tgt_image": tgt_tensor,
                "caption": caption
            }

    def __iter__(self):
        """
        Main iteration. Yields repeated_samples from each trajectory sample.
        """
        # Construct base WebDataset pipeline
        dataset_iter = wds.WebDataset(self.shard_files)

        # If we want shard-level shuffle:
        if self.shardshuffle:
            dataset_iter = dataset_iter.shuffle(self.shardshuffle_size)

        # Optionally shuffle individual samples
        if self.sample_shuffle > 0:
            dataset_iter = dataset_iter.shuffle(self.sample_shuffle)

        # For each sample (trajectory) in the dataset, produce repeated pairs
        for sample in dataset_iter:
            yield from self.process_trajectory(sample)

# Example usage:
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    wds_url = "/root/RSP/demo/webdataset_shards_repeated/bridge_arm_move-*.tar.gz"
    dataset = BridgeCaption(
        wds_pattern=wds_url,
        repeated_sampling=2,
        interval=4,
        transform=None,
        shardshuffle=True,
        shardshuffle_size=10,
        sample_shuffle=0
    )

    loader = DataLoader(dataset, batch_size=4, num_workers=2)

    for i, batch in enumerate(loader):
        # 'batch' will be collated dict: { "traj_key": [...], "src_image": (B, 3, H, W), ... }
        logger.info(f"Batch {i}: src_image shape={batch['src_image'].shape}, caption len={len(batch['caption'])}")
        logger.info(batch['caption'])
        if i >= 2:
            break
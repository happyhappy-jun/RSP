import os
import cv2
import json
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from decord import VideoReader, cpu
from torchvision import transforms

from util.misc import seed_everything
from util.transform import PairedRandomResizedCrop


class PairedKineticsWithGlobalCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON and JSONL embeddings"""

    def __init__(
            self,
            root,  # Root directory containing videos
            caption_path,
            embeddings_path,  # Path to embeddings.jsonl
            max_distance=48,
            repeated_sampling=2,
            seed=42
    ):
        super().__init__()
        seed_everything(seed)

        self.root = root
        self.video_root = os.path.join(self.root, "train2")
        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling

        # Load video samples from pickle
        with open(os.path.join(self.root, "labels", f"label_full_1.0.pickle"), "rb") as f:
            self.samples = pickle.load(f)

        self.captions = json.load(open(caption_path))["results"]
        self.data = dict()
        for item in self.captions:
            self.data[item["custom_id"]] = {
                "label": item["label"],
                "video_name": item["video_name"],
            }

        with open(embeddings_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                custom_id = record[-1]["custom_id"]
                embedding = record[1]["data"][0]["embedding"]
                self.data[custom_id]["embedding"] = embedding

        # Filter samples to only those with embeddings
        self.valid_samples = [video_path.split(".")[0] for _, video_path in self.samples]
        self.data = [v for _, v in self.data.items() if v["video_name"] not in self.valid_samples]

        # Setup transforms
        self.transforms = PairedRandomResizedCrop(seed=seed)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"\nDataset Statistics:")
        print(f"Total videos found: {len(self.data)}")

    def load_frames(self, vr):
        """Sample two frames with temporal constraint"""
        seg_len = len(vr)
        least_frames_num = self.max_distance + 1

        if seg_len >= least_frames_num:
            idx_cur = random.randint(0, seg_len - least_frames_num)
            interval = random.randint(4, self.max_distance)
            idx_fut = idx_cur + interval
        else:
            indices = random.sample(range(seg_len), 2)
            indices.sort()
            idx_cur, idx_fut = indices

        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()

        return frame_cur, frame_fut, (idx_cur, idx_fut)

    def transform(self, src_image, tgt_image):
        """Apply transforms to image pair"""
        src_image, tgt_image = self.transforms(src_image, tgt_image)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        return src_image, tgt_image

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, index):
        try:
            video_path = os.path.join(self.video_root, self.data[index]["label"], self.data[index]["video_name"] + ".mp4")
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))

            src_images = []
            tgt_images = []
            frame_indices = []

            # Sample frames multiple times
            for _ in range(self.repeated_sampling):
                src_image, tgt_image, indices = self.load_frames(vr)
                src_image, tgt_image = self.transform(src_image, tgt_image)

                src_images.append(src_image)
                tgt_images.append(tgt_image)
                frame_indices.append(indices)

        except Exception as e:
            print(f"Error loading index {index}: {str(e)}")
            raise

        # Get embedding and repeat for each sample
        embedding = torch.Tensor(self.data[index]["embedding"])
        embedding = embedding.repeat(self.repeated_sampling, 1)

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": embedding,
        }


def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "embeddings": torch.stack([x['embeddings'] for x in batch], dim=0),
    }


if __name__ == "__main__":
    print("\nInitializing dataset...")
    dataset = PairedKineticsWithGlobalCaption(
        root="/home/junyoon/kinetics400",
        caption_path="/home/junyoon/RSP/artifacts/global/results/frame_analysis_results_complete.json",
        embeddings_path="/home/junyoon/RSP/artifacts/global/embedding_results.jsonl",
    )

    print(f"\nTotal number of videos: {len(dataset)}")

    # Test loading a few samples
    samples = [dataset[i] for i in [0, 1, 500, 501]]

    # Print cosine similarities between embeddings
    print("\nComputing cosine similarities between embeddings:")
    embeddings = [s['embeddings'][0] for s in samples]  # Take first sample from each
    names = ['0', '1', '500', '501']

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            )
            print(f"Similarity between video {names[i]} and {names[j]}: {sim.item():.4f}")

    # Test dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print("\nTesting dataloader...")
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"Batch shapes:")
            print(f"src_images: {batch['src_images'].shape}")
            print(f"tgt_images: {batch['tgt_images'].shape}")
            print(f"embeddings: {batch['embeddings'].shape}")
            break

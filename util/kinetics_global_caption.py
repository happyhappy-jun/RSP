import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from decord import VideoReader, cpu
from torchvision import transforms
from util.transform import PairedRandomResizedCrop


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairedKineticsWithGlobalCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON and JSONL embeddings"""
    def __init__(
        self,
        video_root,      # Root directory containing videos
        frame_info_path, # Path to frame_info.json
        embeddings_path, # Path to embeddings.jsonl
        max_distance=48,
        repeated_sampling=2,
        seed=42
    ):
        super().__init__()
        seed_everything(seed)
        
        self.video_root = video_root
        self.max_distance = max_distance
        self.repeated_sampling = repeated_sampling

        # Load frame info data
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            videos = frame_info['videos']

        # Load embeddings data
        self.embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                video_idx = int(record['video_idx'])
                embedding = record['embedding']
                self.embeddings[video_idx] = np.array(embedding, dtype=np.float32)
        
        # Process videos
        self.video_info = {}
        for video in videos:
            video_idx = video['video_idx']
            if video_idx in self.embeddings:  # Only keep videos with embeddings
                self.video_info[video_idx] = {
                    'video_path': video['video_path'],
                    'label': video['label']
                }

        self.video_indices = sorted(self.video_info.keys())
        
        # Setup transforms
        self.transforms = PairedRandomResizedCrop(seed=seed)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

        print(f"\nDataset Statistics:")
        print(f"Total videos found: {len(videos)}")
        print(f"Total embeddings found: {len(self.embeddings)}")
        print(f"Valid videos after filtering: {len(self.video_indices)}")
        
        # Print missing embeddings info
        missing = set(v['video_idx'] for v in videos) - set(self.embeddings.keys())
        if missing:
            print(f"\nVideos missing embeddings: {len(missing)}")
            print(f"Example missing video_idx: {list(missing)[:5]}")

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
        return len(self.video_indices)

    def __getitem__(self, index):
        try:
            video_idx = self.video_indices[index]
            info = self.video_info[video_idx]
            
            # Verify embedding exists
            if video_idx not in self.embeddings:
                raise KeyError(f"Missing embedding for video_{video_idx}")
            
            # Load video and sample frames
            video_path = info['video_path']
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
        embedding = torch.from_numpy(self.embeddings[video_idx])
        embedding = embedding.repeat(self.repeated_sampling, 1)

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": embedding,
            "video_idx": video_idx,
            "label": info['label'],
            "frame_indices": frame_indices
        }

def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "embeddings": torch.stack([x['embeddings'] for x in batch], dim=0),
        "video_idx": [x['video_idx'] for x in batch],
        "label": [x['label'] for x in batch],
        "frame_indices": [x['frame_indices'] for x in batch]
    }


if __name__ == "__main__":
    print("\nInitializing dataset...")
    dataset = PairedKineticsWithGlobalCaption(
        video_root="/path/to/videos",
        frame_info_path="/path/to/frame_info.json",
        embeddings_path="/path/to/embeddings.jsonl",
    )
    
    print(f"\nTotal number of videos: {len(dataset)}")
    
    # Test loading a few samples
    samples = [dataset[i] for i in [0, 1, 500, 501]]
    
    # Print cosine similarities between embeddings
    print("\nComputing cosine similarities between embeddings:")
    embeddings = [s['embeddings'][0] for s in samples]  # Take first sample from each
    names = ['0', '1', '500', '501']
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
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

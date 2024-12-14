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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairedKineticsWithCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON"""
    def __init__(
        self,
        frame_info_path,     # Path to frame_info.json
        embeddings_path,     # Path to combined_output.jsonl
        repeated_sampling=2  # Number of augmented samples per pair
    ):
        super().__init__()
        # Load frame info data
        with open(frame_info_path, 'r') as f:
            frame_info = json.load(f)
            videos = frame_info['videos']

        # Load embeddings data
        self.embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                # Parse video_idx and pair_idx from custom_id (format: video_X_pair_Y)
                parts = record['custom_id'].split('_')
                video_idx = int(parts[1])
                pair_idx = int(parts[3])
                embedding = record['embedding']
                self.embeddings[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)
        
        # Process videos and create pairs
        results = []
        for video in videos:
            video_idx = video['video_idx']
            frame_paths = video['frame_paths']
            # Process frames in pairs
            for i in range(0, len(frame_paths), 2):
                if i + 1 < len(frame_paths):  # Ensure we have a complete pair
                    pair = {
                        'video_idx': video_idx,
                        'pair_idx': i // 2,
                        'frame_cur_path': frame_paths[i],
                        'frame_fut_path': frame_paths[i + 1]
                    }
                    results.append(pair)

        # Filter and flatten pairs that have embeddings
        self.valid_pairs = []
        missing_embeddings = defaultdict(list)
        
        for pair in results:
            video_idx, pair_idx = pair['video_idx'], pair['pair_idx']
            if (video_idx, pair_idx) in self.embeddings:
                self.valid_pairs.append(pair)
            else:
                missing_embeddings[video_idx].append(pair_idx)
        
        print(f"\nDataset Statistics:")
        print(f"Total pairs found: {len(results)}")
        print(f"Total embeddings found: {len(self.embeddings)}")
        print(f"Valid pairs after filtering: {len(self.valid_pairs)}")
        print(f"Videos with missing embeddings: {len(missing_embeddings)}")
        
        # Print some example missing embeddings
        if missing_embeddings:
            print("\nExample videos with missing embeddings:")
            for video_idx, pairs in list(missing_embeddings.items())[:5]:
                print(f"Video {video_idx}: Missing {len(pairs)} pairs - {pairs[:5]}")
            
        self.repeated_sampling = repeated_sampling
        
        # Setup transforms with seed
        self.transforms = PairedRandomResizedCrop(seed=42)
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.valid_pairs)

    def _process_path(self, frame_path):
        """Remove environment-specific prefix from path"""
        if '/RSP/' in frame_path:
            return frame_path.split('/RSP/', 1)[1]
        return frame_path

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        processed_path = self._process_path(frame_path)
        frame = cv2.imread(processed_path)
        if frame is None:
            raise ValueError(f"Failed to load frame from path: {processed_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def __getitem__(self, index):
        try:
            pair = self.valid_pairs[index]
            video_idx = pair['video_idx']
            pair_idx = pair['pair_idx']
            
            # Verify embedding exists
            if (video_idx, pair_idx) not in self.embeddings:
                raise KeyError(f"Missing embedding for video_{video_idx}_pair_{pair_idx}")
            
            # Load and process frames
            frame_cur = self.load_frame(pair['frame_cur_path'])
            frame_fut = self.load_frame(pair['frame_fut_path'])
        except Exception as e:
            print(f"Error loading index {index}: {str(e)}")
            # Re-raise to maintain DataLoader's error handling
            raise
            
        # Apply transforms
        src_image, tgt_image = self.transforms(frame_cur, frame_fut)
        src_image = self.basic_transform(src_image)
        tgt_image = self.basic_transform(tgt_image)
        
        # Get embedding
        embedding = torch.from_numpy(self.embeddings[(video_idx, pair_idx)])

        return {
            "src_images": src_image.unsqueeze(0),  # Add batch dimension
            "tgt_images": tgt_image.unsqueeze(0),
            "embeddings": embedding.unsqueeze(0),
        }
            
def collate_fn(batch):
    return {
        "src_images": torch.stack([x['src_images'] for x in batch], dim=0),
        "tgt_images": torch.stack([x['tgt_images'] for x in batch], dim=0),
        "embeddings": torch.stack([x['embeddings'] for x in batch], dim=0),
    }


if __name__ == "__main__":
    print("\nInitializing dataset...")
    dataset = PairedKineticsWithCaption(
        frame_info_path="/home/junyoon/RSP/artifacts/frames/frame_info.json",
        embeddings_path="/home/junyoon/RSP/artifacts/embeddings/embedding_results.jsonl",
    )
    
    print(f"Total number of videos: {len(dataset)}")
    a = dataset[0]['embeddings']
    b = dataset[1]['embeddings']
    c = dataset[500]['embeddings']
    d = dataset[501]['embeddings']
    e = dataset[999]['embeddings']
    f = dataset[1000]['embeddings']

    # Print cosine similarities between embeddings
    print("\nComputing cosine similarities between embeddings:")
    
    # List of embeddings to compare
    embeddings = [a, b, c, d, e, f]
    embedding_names = ['0-cur', '0-fut', '500-cur', '500-fut', '999-cur', '999-fut']
    
    # Compute cosine similarity for all pairs
    for (e1, n1), (e2, n2) in combinations(zip(embeddings, embedding_names), 2):
        sim = torch.nn.functional.cosine_similarity(e1, e2)
        print(f"Cosine similarity between {n1} and {n2}: {sim.item():.4f}")

    print(f"Embedding snippet {a[:30]}")
    print(f"Embedding snippet {b[:30]}")
    print(f"Embedding snippet {c[:30]}")
    print(f"Embedding snippet {d[:30]}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    for i, batch in enumerate(dataloader):
        if i == 10:
            break
        print(batch['src_images'].shape)
        print(batch['tgt_images'].shape)
        print(batch['embeddings'].shape)

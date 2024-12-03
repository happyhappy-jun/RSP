from itertools import combinations
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from models import BatchOutput
from torchvision import transforms
from util.transform import PairedRandomResizedCrop


class PairedKineticsWithCaption(Dataset):
    """PairedKinetics dataset that loads from preprocessed JSON"""
    def __init__(
        self,
        data_path,           # Path to preprocessed JSON file
        embeddings_path,     # Path to precomputed embeddings
        repeated_sampling=2, # Number of augmented samples per pair
    ):
        super().__init__()
        # Load video frame data
        with open(data_path, 'r') as f:
            results = json.load(f)["results"]

        # Load embeddings data more efficiently using numpy memmap
        self.embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                id = int(record['custom_id'].split('-')[-1]) - 1
                video_idx, pair_idx = id // 2, id % 2
                embedding = BatchOutput(**record).response.body.data[0].embedding
                # Store as numpy array instead of tensor
                self.embeddings[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)
        
        # Sort results by video_idx and pair_idx
        sorted_results = sorted(results, key=lambda x: (x['video_idx'], x['pair_idx']))
        print(f"Loaded {len(sorted_results)} pairs")
        print(f"Loaded {len(self.embeddings)} embeddings")
        assert len(self.embeddings.keys()) == len(sorted_results)
        
        self.videos = defaultdict(list)
        for i, pair in enumerate(sorted_results):
            self.videos[pair["video_idx"]].append(pair)
            self.videos[pair["video_idx"]].sort(key=lambda x: x.get('pair_idx', 0))
        
        # Get original indices and find gaps
        orig_indices = sorted(self.videos.keys())
        min_idx = min(orig_indices)
        max_idx = max(orig_indices)
        
        # Create mapping to fill gaps
        new_idx = 0
        self.idx_mapping = {}
        for idx in range(min_idx, max_idx + 1):
            if idx in self.videos:
                self.idx_mapping[idx] = new_idx
                new_idx += 1
                
        # Remap videos to new continuous indices
        remapped_videos = defaultdict(list)
        for old_idx, pairs in self.videos.items():
            new_idx = self.idx_mapping[old_idx]
            remapped_videos[new_idx] = pairs
            
        self.videos = remapped_videos
        self.video_indices = sorted(self.videos.keys())
        
        # Verify no gaps after remapping
        expected_indices = set(range(len(self.video_indices)))
        actual_indices = set(self.video_indices)
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            raise ValueError(f"Index continuity error after remapping. Missing: {missing}, Extra: {extra}")
            
        self.repeated_sampling = repeated_sampling
        
        # Setup transforms
        self.transforms = PairedRandomResizedCrop()
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_indices)

    def load_frame(self, frame_path):
        """Load and convert frame to RGB"""
        frame = cv2.imread(frame_path)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def __getitem__(self, index):
        video_idx = self.video_indices[index]
        pair_infos = self.videos[video_idx]

        src_images = []
        tgt_images = []
        embeddings = []
        
        for pair_idx, pair in enumerate(pair_infos):
            frame_cur = self.load_frame(pair['frame_cur_path'])
            frame_fut = self.load_frame(pair['frame_fut_path'])
                
            # Apply transforms
            src_image, tgt_image = self.transforms(frame_cur, frame_fut)
            src_image = self.basic_transform(src_image)
            tgt_image = self.basic_transform(tgt_image)
            
            src_images.append(src_image)
            tgt_images.append(tgt_image)
            # Convert numpy array to tensor here
            embeddings.append(torch.from_numpy(self.embeddings[(video_idx, pair_idx)]))

        return {
            "src_images": torch.stack(src_images, dim=0),
            "tgt_images": torch.stack(tgt_images, dim=0),
            "embeddings": torch.stack(embeddings, dim=0),
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
        data_path="/home/junyoon/RSP/artifacts/frame_analysis_results_complete.json",
        embeddings_path="/home/junyoon/RSP/artifacts/combined_output.jsonl",
    )
    
    print(f"Total number of videos: {len(dataset)}")
    a = dataset[0]['embeddings'][0]
    b = dataset[0]['embeddings'][1]
    c = dataset[1]['embeddings'][0]
    d = dataset[1]['embeddings'][1]
    e = dataset[3]['embeddings'][0]
    f = dataset[3]['embeddings'][1]
    
    # Print cosine similarities between embeddings
    print("\nComputing cosine similarities between embeddings:")
    
    # List of embeddings to compare
    embeddings = [a, b, c, d, e, f]
    embedding_names = ['0-0', '0-1', '1-0', '1-1', '3-0', '3-1']
    
    # Compute cosine similarity for all pairs
    for (e1, n1), (e2, n2) in combinations(zip(embeddings, embedding_names), 2):
        sim = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
        print(f"Cosine similarity between {n1} and {n2}: {sim.item():.4f}")
        
    print(dataset[0])

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

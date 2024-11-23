import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

class CaptionDataset(Dataset):
    def __init__(self, caption_data):
        # Sort results by video_idx and pair_idx
        self.results = sorted(caption_data['results'], 
                            key=lambda x: (x['video_idx'], x['pair_idx']))
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        result = self.results[idx]
        tokens = self.tokenizer(result['analysis'], return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        
        sample_idx = result['video_idx']*2 + result['pair_idx']
        return {'input_ids': input_ids, 'attention_mask': attention_mask}, sample_idx
    
def collate_fn(batch):
    input_ids = torch.stack([x[0]['input_ids'] for x in batch], dim=0)
    attention_mask = torch.stack([x[0]['attention_mask'] for x in batch], dim=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, [x[1] for x in batch]

def precompute_embeddings(json_path, output_dir, batch_size=32, seed=42):
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    lm_model = AutoModel.from_pretrained('microsoft/deberta-v3-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = CaptionDataset(caption_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    embeddings = []
    
    for batch_input_ids, _ in tqdm(dataloader):
        with torch.no_grad():
            batch_input = batch_input_ids['input_ids'].to(device), batch_input_ids['attention_mask'].to(device)
            
            batch_output = lm_model(input_ids=batch_input[0], attention_mask=batch_input[1])['last_hidden_state']
            batch_embeddings = batch_output[:, 0, :] 
            embeddings.extend(batch_embeddings.cpu())
    
    print(embeddings[0][:10])
    output_path = output_dir / 'deberta_embeddings.pt'
    torch.save(embeddings, output_path)
    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    precompute_embeddings(args.json_path, args.output_dir, args.batch_size, args.seed)

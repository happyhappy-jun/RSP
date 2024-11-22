import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from DeBERTa import deberta

class CaptionDataset(Dataset):
    def __init__(self, caption_data, tokenizer, max_seq_len=512):
        # Sort results by video_idx and pair_idx
        self.results = sorted(caption_data['results'][:100], 
                            key=lambda x: (x['video_idx'], x['pair_idx']))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        result = self.results[idx]
        tokens = self.tokenizer.tokenize(result['analysis'])
        tokens = tokens[:self.max_seq_len - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        input_masks = [1] * len(input_ids)
        paddings = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [0] * paddings
        input_masks = input_masks + [0] * paddings
        
        sample_idx = result['video_idx']*2 + result['pair_idx']
        return torch.tensor(input_ids), torch.tensor(input_masks), sample_idx

def precompute_embeddings(json_path, output_dir, batch_size=32):
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    lm_model = deberta.DeBERTa(pre_trained='base')
    lm_model.apply_state()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = CaptionDataset(caption_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    
    for batch_input_ids, batch_masks, sample_indices in tqdm(dataloader):
        with torch.no_grad():
            print(batch_input_ids[0][:30])
            batch_input = batch_input_ids.to(device)
            batch_output = lm_model(batch_input)['hidden_states'][-1] # last hidden state
            batch_embeddings = batch_output[:, 0, :] 
            embeddings.extend(batch_embeddings.cpu())
    print(embeddings[0][:30])
    
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
    
    args = parser.parse_args()
    precompute_embeddings(args.json_path, args.output_dir, args.batch_size)

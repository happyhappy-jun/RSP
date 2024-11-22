import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from pathlib import Path
from DeBERTa import deberta

class CaptionDataset(Dataset):
    def __init__(self, caption_data, tokenizer, max_seq_len=512):
        self.results = caption_data['results']
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

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        return rank, world_size
    return 0, 1

def precompute_embeddings(json_path, output_dir, batch_size=32):
    """Precompute DeBERTa embeddings for all captions"""
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    is_main_process = rank == 0
    
    # Load caption data
    if is_main_process:
        print(f"Loading captions from {json_path}")
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    # Initialize DeBERTa model
    if is_main_process:
        print("Initializing DeBERTa model...")
    lm_model = deberta.DeBERTa(pre_trained='base')
    lm_model.apply_state()
    device = torch.device('cuda')
    lm_model = lm_model.to(device)
    if world_size > 1:
        lm_model = DDP(lm_model, device_ids=[rank])
    lm_model.eval()
    
    # Setup tokenizer
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)
    
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = CaptionDataset(caption_data, tokenizer)
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Process captions in batches
    embeddings = {}
    
    if is_main_process:
        print("Computing embeddings...")
    
    for batch_input_ids, batch_masks, sample_indices in tqdm(dataloader, disable=not is_main_process):
        with torch.no_grad():
            batch_input = batch_input_ids.to(device)
            batch_output = lm_model(batch_input)['embeddings']
            batch_embeddings = batch_output[:, -1, :]  # Get last token embeddings
            
            # Store embeddings
            for idx, sample_idx in enumerate(sample_indices):
                embeddings[sample_idx.item()] = batch_embeddings[idx].cpu()
    
    # Gather embeddings from all processes
    if world_size > 1:
        all_embeddings = [None for _ in range(world_size)]
        dist.all_gather_object(all_embeddings, embeddings)
        
        if is_main_process:
            combined_embeddings = {}
            for proc_embeddings in all_embeddings:
                combined_embeddings.update(proc_embeddings)
            embeddings = combined_embeddings

    # Save embeddings (only in main process)
    if is_main_process:
        output_path = output_dir / 'deberta_embeddings.pt'
        torch.save(embeddings, output_path)
        print(f"Saved embeddings to {output_path}")
        print(f"Total embeddings computed: {len(embeddings)}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to caption JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding computation')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    
    precompute_embeddings(args.json_path, args.output_dir, args.batch_size)
    
    if args.local_rank != -1:
        dist.destroy_process_group()

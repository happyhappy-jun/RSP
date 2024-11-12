import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from DeBERTa import deberta

def precompute_embeddings(json_path, output_dir, batch_size=32):
    """Precompute DeBERTa embeddings for all captions"""
    
    # Load caption data
    print(f"Loading captions from {json_path}")
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    # Initialize DeBERTa model
    print("Initializing DeBERTa model...")
    lm_model = deberta.DeBERTa(pre_trained='base')
    lm_model.apply_state()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = lm_model.to(device)
    lm_model.eval()
    
    # Setup tokenizer
    vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
    tokenizer = deberta.tokenizers[vocab_type](vocab_path)
    
    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process captions in batches
    embeddings = {}
    captions = []
    video_indices = []
    
    print("Tokenizing and computing embeddings...")
    for result in tqdm(caption_data['results']):
        # Tokenize caption
        max_seq_len = 512
        tokens = tokenizer.tokenize(result['analysis'])
        tokens = tokens[:max_seq_len - 2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        input_masks = [1] * len(input_ids)
        paddings = max_seq_len - len(input_ids)
        input_ids = input_ids + [0] * paddings
        input_masks = input_masks + [0] * paddings
        
        captions.append(torch.tensor(input_ids))
        video_indices.append(result['video_idx'])
        
        # Process in batches
        if len(captions) == batch_size:
            with torch.no_grad():
                batch_input = torch.stack(captions).to(device)
                batch_output = lm_model(batch_input)['embeddings']
                batch_embeddings = batch_output[:, -1, :]  # Get last token embeddings
                
                # Store embeddings
                for idx, video_idx in enumerate(video_indices):
                    embeddings[video_idx] = batch_embeddings[idx].cpu()
            
            captions = []
            video_indices = []
    
    # Process remaining captions
    if captions:
        with torch.no_grad():
            batch_input = torch.stack(captions).to(device)
            batch_output = lm_model(batch_input)['embeddings']
            batch_embeddings = batch_output[:, -1, :]
            
            for idx, video_idx in enumerate(video_indices):
                embeddings[video_idx] = batch_embeddings[idx].cpu()
    
    # Save embeddings
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
    
    args = parser.parse_args()
    precompute_embeddings(args.json_path, args.output_dir, args.batch_size)
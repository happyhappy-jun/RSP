import ujson as json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
from typing import Dict, Any

def process_line(line: str, target_size: int = 512) -> Dict[str, Any]:
    """Process a single line from the embeddings file and resize the embedding.
    
    Args:
        line: JSON line from the embeddings file
        target_size: Size to truncate the embedding to
        
    Returns:
        Processed record with resized embedding
    """
    record = json.loads(line)
    embedding = record[1]['data'][0]['embedding']
    # Truncate embedding to target size
    record[1]['data'][0]['embedding'] = embedding[:target_size]
    return record

def resize_embeddings(input_path: str, output_path: str, target_size: int = 512):
    """Resize embeddings in a JSONL file to specified size.
    
    Args:
        input_path: Path to input embeddings JSONL file
        output_path: Path to save processed embeddings
        target_size: Size to truncate embeddings to
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count total lines for progress bar
    print("Counting total lines...")
    with input_path.open('r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"\nProcessing {total_lines:,} embeddings...")
    processed = 0
    errors = 0
    
    # Process file line by line
    with input_path.open('r') as fin, output_path.open('w') as fout:
        for line in tqdm(fin, total=total_lines, desc="Resizing embeddings"):
            try:
                processed_record = process_line(line.strip(), target_size)
                json.dump(processed_record, fout)
                fout.write('\n')
                processed += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                errors += 1
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {processed:,}")
    print(f"Total errors: {errors:,}")
    print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Resize embeddings in JSONL file')
    parser.add_argument('input_path', type=str, help='Path to input embeddings JSONL file')
    parser.add_argument('output_path', type=str, help='Path to save processed embeddings')
    parser.add_argument('--target-size', type=int, default=512, 
                        help='Size to truncate embeddings to (default: 512)')
    
    args = parser.parse_args()
    resize_embeddings(args.input_path, args.output_path, args.target_size)

if __name__ == "__main__":
    main()
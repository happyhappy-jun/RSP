import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from caption2.core.embedding_creator import EmbeddingCreator

async def main():
    parser = argparse.ArgumentParser(description='Create embeddings from caption results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to caption results file (.json or .jsonl)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save embedding results')
    parser.add_argument('--batch-response', type=str,
                       help='Path to batch response file (.jsonl)')
    args = parser.parse_args()

    creator = EmbeddingCreator()
    caption_results_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load caption results
    caption_results = []

    print(f"\nLoading caption results from: {caption_results_path}")
    # Load main caption results
    with open(caption_results_path) as f:
        if str(caption_results_path).endswith('.jsonl'):
            raw_results = [json.loads(line) for line in f]
        else:
            raw_results = json.load(f)
        
        caption_results = [{"custom_id": result["custom_id"], "response": result["response"]} for result in raw_results]
    print(len(caption_results))
 
    await creator.process_caption_results(caption_results, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

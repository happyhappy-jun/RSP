import asyncio
import argparse
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from caption2.core.embedding_creator import EmbeddingCreator
from models import Response


class BatchOutput(BaseModel):
    """
    BatchOutput class is used to store the output of the batch processing
    """
    id: str
    custom_id: str  
    response: Response
    error: Optional[str] = None

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
    raw_results = []
    with open(caption_results_path) as f:
        for line in f:
            try:
                result = json.loads(line)
                raw_results.append(result)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(raw_results)} raw results")
    caption_results = []
    for result in raw_results:
        try:
            batch_output = BatchOutput(**result)
            caption_results.append(batch_output)
        except Exception as e:
            print(f"Error processing result: {e}")
            continue
    
    print(f"Successfully processed {len(caption_results)} results")


    print("\nLoaded caption results")
    # Load and merge batch responses if provided
    if args.batch_response:
        batch_path = Path(args.batch_response)
        if batch_path.exists():
            with open(batch_path) as f:
                raw_batch = [json.loads(line) for line in f]
                batch_results = [BatchOutput(**result) for result in raw_batch]
                # Merge based on custom_id
                caption_results.extend(batch_results)
 
    print(f"{len(caption_results)}")
    await creator.process_caption_results(caption_results, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

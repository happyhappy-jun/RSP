import asyncio
import argparse
import json
from pathlib import Path
from caption2.core.embedding_creator import EmbeddingCreator

async def main():
    parser = argparse.ArgumentParser(description='Create embeddings from caption results')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to caption results file (.json or .jsonl)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save embedding results')
    args = parser.parse_args()

    creator = EmbeddingCreator()
    caption_results_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load caption results
    with open(caption_results_path) as f:
        if str(caption_results_path).endswith('.jsonl'):
            caption_results = [json.loads(line) for line in f]
        else:
            caption_results = json.load(f)
    
    await creator.process_caption_results(caption_results, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

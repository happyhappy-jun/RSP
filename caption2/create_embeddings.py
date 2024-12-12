import asyncio
import argparse
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
    
    await creator.process_caption_results(caption_results_path, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

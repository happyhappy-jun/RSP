import asyncio
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from caption2.core.embedding_creator import EmbeddingCreator

async def main():
    parser = argparse.ArgumentParser(description='Step 6: Create embeddings from captions')
    parser.add_argument('--caption_results', type=str, required=True,
                       help='Path to combined_results.json from step 5')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--model', type=str, default="text-embedding-3-small", help="OpenAI embedding model")
    args = parser.parse_args()

    creator = EmbeddingCreator()
    caption_results_path = Path(args.caption_results)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load caption results
    print(f"\nLoading caption results from: {caption_results_path}")
    with open(caption_results_path) as f:
        results = json.load(f)
    
    # Process results
    caption_results = []
    skipped = 0
    
    for result in tqdm(results):
        # Skip entries with errors
        if result.get('error'):
            skipped += 1
            continue
            
        caption = result["response"]["body"]["choices"][0]["message"]["content"]
            
        caption_results.append({
            "custom_id": result["custom_id"],
            "caption": caption
        })

    print(f"Found {len(caption_results)} valid captions")
    if skipped:
        print(f"Skipped {skipped} invalid/error results")
        
    print("\nProcessing captions...")
    await creator.process_caption_results(caption_results[:100], output_dir)

if __name__ == "__main__":
    asyncio.run(main())

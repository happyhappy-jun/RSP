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
    parser.add_argument('--output_file', type=str, default="embeddings.jsonl",)
    parser.add_argument('--model', type=str, default="text-embedding-3-small", help="OpenAI embedding model")
    args = parser.parse_args()

    creator = EmbeddingCreator(model=args.model)
    caption_results_path = Path(args.caption_results)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load caption results
    print(f"\nLoading caption results from: {caption_results_path}")
    results = []
    with open(caption_results_path) as f:
        # Skip empty lines and parse valid JSON lines
        results = []
        for line in f:
            line = line.strip()
            if line:  # This will skip empty lines
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:100]}...")  # Print first 100 chars of problematic line
                    continue

    # Process results
    caption_results = []
    skipped = 0
    
    for result in tqdm(results):
        # Skip entries with errors
        if result.get('error'):
            skipped += 1
            continue

        try:
            caption = result["response"]["body"]["choices"][0]["message"]["content"]
        except:
            caption = ""
            print(f"Error parsing caption for custom_id: {result['custom_id']}")

        caption_results.append({
            "custom_id": result["custom_id"],
            "caption": caption
        })

    print(f"Found {len(caption_results)} valid captions")
    if skipped:
        print(f"Skipped {skipped} invalid/error results")
        
    print("\nProcessing captions...")
    await creator.process_caption_results(caption_results, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

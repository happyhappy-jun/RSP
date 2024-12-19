import asyncio
import argparse
import json
from pathlib import Path
from caption2.core.embedding_creator import EmbeddingCreator

async def main():
    parser = argparse.ArgumentParser(description='Step 4: Create embeddings from captions')
    parser.add_argument('--caption_results', type=str, required=True,
                       help='Path to caption_results.json from step 3')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for embeddings')
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
        caption_results = []
        for result in results:
            if isinstance(result, dict) and 'custom_id' in result:
                caption_results.append({
                    "custom_id": result["custom_id"],
                    "response": result.get("response", {}).get("body", {})
                                     .get("choices", [{}])[0]
                                     .get("message", {}).get("content", "")
                })

    print(f"Processing {len(caption_results)} captions")
    await creator.process_caption_results(caption_results, output_dir)

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import argparse
import json
from pathlib import Path
from caption2.core.embedding_creator import EmbeddingCreator

def extract_caption(result: dict) -> str:
    """Safely extract caption from API response"""
    try:
        return (result.get("response", {})
                     .get("body", {})
                     .get("choices", [{}])[0]
                     .get("message", {})
                     .get("content", ""))
    except (AttributeError, IndexError):
        return ""

async def main():
    parser = argparse.ArgumentParser(description='Step 6: Create embeddings from captions')
    parser.add_argument('--caption_results', type=str, required=True,
                       help='Path to combined_results.json from step 5')
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
    
    # Process results
    caption_results = []
    skipped = 0
    
    for result in results:
        if not isinstance(result, dict) or 'custom_id' not in result:
            skipped += 1
            continue
            
        # Skip entries with errors
        if result.get('error'):
            skipped += 1
            continue
            
        caption = extract_caption(result)
        if not caption:
            skipped += 1
            continue
            
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

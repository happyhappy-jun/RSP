import argparse
import json
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

from caption2.core.batch_api import BatchProcessor

def main():
    parser = argparse.ArgumentParser(description='Step 6-1: Create embeddings using batch API')
    parser.add_argument('--caption_results', type=str, required=True,
                       help='Path to combined_results.json from step 5')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--model', type=str, default="text-embedding-3-small",
                       help="OpenAI embedding model")
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of captions to process in each batch')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run sanity check with single request only')
    args = parser.parse_args()

    # Setup paths
    caption_results_path = Path(args.caption_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load caption results
    print(f"\nLoading caption results from: {caption_results_path}")
    with open(caption_results_path) as f:
        results = [json.loads(line) for line in f.readlines()]

    # Process results into embedding requests
    print("\nPreparing embedding requests...")
    embedding_requests = []
    skipped = 0
    
    for result in tqdm(results):
        # Skip entries with errors
        if result.get('error'):
            skipped += 1
            continue
            
        caption = result["response"]["body"]["choices"][0]["message"]["content"]
        
        # Create embedding request
        request = {
            "custom_id": result["custom_id"],
            "body": {
                "model": args.model,
                "input": caption
            }
        }
        embedding_requests.append(request)

    print(f"Created {len(embedding_requests)} embedding requests")
    if skipped:
        print(f"Skipped {skipped} invalid/error results")

    # Setup batch processor
    client = OpenAI()
    processor = BatchProcessor(
        client=client,
        output_dir=output_dir
    )

    # Process in batches
    total_processed = 0
    current_batch = []
    current_shard = 0

    if args.sanity_check:
        # For sanity check, just process first request
        results = processor.submit_requests(
            [embedding_requests[0]],
            description="Embedding sanity check",
            sanity_check=True
        )
        print("\nSanity check results:")
        print(json.dumps(results, indent=2))
        return

    print("\nProcessing embedding requests in batches...")
    for i in range(0, len(embedding_requests), args.batch_size):
        batch = embedding_requests[i:i + args.batch_size]
        results_file = output_dir / f"embedding_results_{current_shard:04d}.json"
        
        if results_file.exists():
            print(f"Skipping existing results file: {results_file}")
            current_shard += 1
            continue

        try:
            # Submit batch
            batch_ids = processor.submit_requests(
                batch,
                description=f"Embeddings batch {current_shard}",
                shard_idx=current_shard
            )

            # Monitor and save results
            results = processor.monitor_batches(batch_ids)
            
            # Save results
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            total_processed += len(results)
            current_shard += 1
            
            # Clear memory
            del results
            
        except Exception as e:
            print(f"Error processing batch {current_shard}: {str(e)}")

    print(f"\nProcessed {total_processed} embedding requests")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()

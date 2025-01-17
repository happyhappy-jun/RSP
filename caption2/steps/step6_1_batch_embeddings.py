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
    parser.add_argument('--num_shards', type=int, required=True,
                       help='Number of shards to split requests into')
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
        
        # Create embedding request in expected batch format
        request = {
            "custom_id": result["custom_id"],
            "method": "POST", 
            "url": "/v1/embeddings",
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

    # Calculate shard size
    shard_size = len(embedding_requests) // args.num_shards
    if len(embedding_requests) % args.num_shards:
        shard_size += 1

    # Process in shards
    total_processed = 0
    all_results = []

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

    print("\nSubmitting all embedding request shards...")
    all_batch_ids = []
    
    # First submit all shards
    for shard_idx in range(args.num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(embedding_requests))
        shard = embedding_requests[start_idx:end_idx]
        
        if not shard:  # Skip empty shards
            continue

        try:
            # Submit shard with proper request format
            batch_ids = processor.submit_requests(
                shard,
                description=f"Embeddings shard {shard_idx}",
                shard_idx=shard_idx
            )
            all_batch_ids.extend(batch_ids)
            
        except Exception as e:
            if "batch_expired" in str(e):
                print(f"Shard {shard_idx} expired: {str(e)}")
            else:
                print(f"Error submitting shard {shard_idx}: {str(e)}")
            
    # Then monitor all batches together
    print("\nMonitoring all batches...")
    try:
        results = processor.monitor_batches(all_batch_ids)
        
        # Transform results to match step6 schema
        for result in results:
            embedding_result = {
                "custom_id": result["custom_id"],
                "embedding": result["response"]["data"][0]["embedding"]
            }
            all_results.append(embedding_result)
        
        total_processed = len(results)
        
        # Clear memory
        del results
        
    except Exception as e:
        print(f"Error monitoring batches: {str(e)}")

    # Sort results by custom_id
    all_results.sort(key=lambda x: x["custom_id"])

    # Save combined results
    output_file = output_dir / "embeddings.json"
    with open(output_file, 'w') as f:
        for result in all_results:
            json.dump(result, f)
            f.write('\n')

    print(f"\nProcessed {total_processed} embedding requests")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()

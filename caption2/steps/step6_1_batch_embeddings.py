import argparse
import json
import time
import logging
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

from caption2.core.batch_api import BatchProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_embeddings.log'),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Step 6-1: Create embeddings using batch API')
    parser.add_argument('--caption_results', type=str, required=True,
                       help='Path to combined_results.json from step 5')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--model', type=str, default="text-embedding-3-small",
                       help="OpenAI embedding model")
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

    # Process in batches of 50,000 requests
    BATCH_SIZE = 50000
    MAX_ACTIVE_BATCHES = 20  # To stay under 1M request limit
    
    total_processed = 0
    all_results = []
    active_batches = []

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

    print(f"\nProcessing {len(embedding_requests)} requests in batches of {BATCH_SIZE}...")
    
    # Process all requests
    current_idx = 0
    batch_num = 0
    
    while current_idx < len(embedding_requests):
        # Wait if we have too many active batches
        while len(active_batches) >= MAX_ACTIVE_BATCHES:
            print(f"\nWaiting for batches to complete ({len(active_batches)} active)...")
            completed_batches = []
            for batch_ids in active_batches:
                try:
                    results = processor.monitor_batches(batch_ids)
                    if results:
                        all_results.extend(results)
                        total_processed += len(results)
                        completed_batches.append(batch_ids)
                        logging.info(f"Successfully processed batch with {len(results)} results")
                except Exception as e:
                    if "still processing" not in str(e).lower():
                        logging.error(f"Error monitoring batch {batch_ids}: {str(e)}", exc_info=True)
            
            # Remove completed batches
            for batch in completed_batches:
                active_batches.remove(batch)
                
            if len(active_batches) >= MAX_ACTIVE_BATCHES:
                time.sleep(60)  # Wait before checking again
        
        # Submit next batch
        end_idx = min(current_idx + BATCH_SIZE, len(embedding_requests))
        current_batch = embedding_requests[current_idx:end_idx]
        
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                batch_ids = processor.submit_requests(
                    current_batch,
                    description=f"Embeddings batch {batch_num}",
                    shard_idx=batch_num
                )
                active_batches.append(batch_ids)
                logging.info(f"Successfully submitted batch {batch_num} with {len(current_batch)} requests")
                break
                    
            except Exception as e:
                retry_count += 1
                logging.error(f"Error submitting batch {batch_num} (requests {current_idx}-{end_idx}): {str(e)}", exc_info=True)
                # Log failed request IDs
                failed_ids = [req["custom_id"] for req in current_batch]
                logging.warning(f"Failed request IDs: {', '.join(failed_ids)}")
                    
                if retry_count < max_retries:
                    wait_time = 60 * retry_count  # Exponential backoff
                    logging.info(f"Retrying batch {batch_num} in {wait_time} seconds (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Failed to submit batch {batch_num} after {max_retries} attempts")
                    continue
            
        current_idx = end_idx
        batch_num += 1
        
    # Process remaining active batches
    print("\nProcessing remaining batches...")
    while active_batches:
        completed_batches = []
        for batch_ids in active_batches:
            try:
                results = processor.monitor_batches(batch_ids)
                all_results.extend(results)
                total_processed += len(results)
                completed_batches.append(batch_ids)
            except Exception as e:
                logging.error(f"Error monitoring final batch {batch_ids}: {str(e)}", exc_info=True)
                
        # Remove completed batches
        for batch in completed_batches:
            active_batches.remove(batch)
            
        if active_batches:
            time.sleep(60)
        
    # Transform results to match step6 schema
    transformed_results = []
    try:
        for result in all_results:
            try:
                embedding_result = {
                    "custom_id": result["custom_id"],
                    "embedding": result["response"]["data"][0]["embedding"]
                }
                transformed_results.append(embedding_result)
            except Exception as e:
                logging.error(f"Error transforming result for {result.get('custom_id', 'unknown')}: {str(e)}")
                logging.debug(f"Problematic result: {json.dumps(result, indent=2)}")
                continue
                
        total_processed = len(transformed_results)
        logging.info(f"Successfully transformed {total_processed} results")
        
    except Exception as e:
        logging.error(f"Error in results transformation: {str(e)}", exc_info=True)

    # Sort results by custom_id
    all_results.sort(key=lambda x: x["custom_id"])

    # Save combined results
    output_file = output_dir / "embeddings.json"
    try:
        with open(output_file, 'w') as f:
            for result in transformed_results:
                json.dump(result, f)
                f.write('\n')
        logging.info(f"Successfully saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results to {output_file}: {str(e)}", exc_info=True)

    print(f"\nProcessed {total_processed} embedding requests")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()

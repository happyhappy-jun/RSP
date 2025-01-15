import argparse
import json
from pathlib import Path

from tqdm import tqdm

from caption2.core.batch_api import BatchProcessor
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='Step 3: Process caption requests')
    parser.add_argument('--requests_dir', type=str, required=True,
                       help='Directory containing sharded JSONL request files from step 2')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run sanity check with single request only')
    args = parser.parse_args()

    # Get list of all shard files
    requests_dir = Path(args.requests_dir)
    shard_files = sorted(list(requests_dir.glob("shard_*.jsonl")))
    if not shard_files:
        raise ValueError(f"No shard files found in {args.requests_dir}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process shards
    client = OpenAI()
    processor = BatchProcessor(
        client=client,
        output_dir=output_dir
    )
    
    total_processed = 0
    
    if args.sanity_check:
        # For sanity check, just process first request from first shard
        with open(shard_files[0]) as f:
            first_request = json.loads(f.readline())
            results = processor.process_requests(
                [first_request],
                description="Sanity check",
                sanity_check=True
            )
            print("\nSanity check results:")
            print(json.dumps(results, indent=2))
            total_processed = 1
    else:
        print("\nProcessing shards...")
        for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="Processing shards")):
            # Process and save results for current shard
            results_file = output_dir / f"caption_results_{shard_idx:04d}.json"
            
            # Skip if already processed
            if results_file.exists():
                print(f"Skipping existing results file: {results_file}")
                continue
                
            # Load and process current shard
            shard_requests = []
            with open(shard_file) as f:
                for line in f:
                    shard_requests.append(json.loads(line))
            
            # Process current shard
            results = processor.process_requests(
                shard_requests,
                description=f"Processing {shard_file.name}",
                sanity_check=False
            )
            
            # Save results immediately
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            total_processed += len(results)
            
            # Clear memory
            del results
            del shard_requests
            
    if not args.sanity_check:
        print(f"\nProcessed {total_processed} caption requests")
        print(f"Results saved as individual files in: {output_dir}")

if __name__ == "__main__":
    main()

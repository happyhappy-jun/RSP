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
    
    # Process each shard
    all_results = []
    
    if args.sanity_check:
        # For sanity check, just process first request from first shard
        with open(shard_files[0]) as f:
            first_request = json.loads(f.readline())
            results = processor.process_requests(
                [first_request],
                description="Sanity check",
                sanity_check=True
            )
            all_results.extend(results)
    else:
        print("\nProcessing shards...")
        for shard_file in tqdm(shard_files, desc="Processing shards"):
            # Load requests from current shard
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
            all_results.extend(results)
    
    if args.sanity_check:
        print("\nSanity check results:")
        print(json.dumps(all_results, indent=2))

    if not args.sanity_check:
        # Save raw results
        for i, result in enumerate(all_results):
            shard_results_file = output_dir / f"caption_results_{i:04d}.json"
            with open(shard_results_file, 'w') as f:
                json.dump(result, f, indent=2)

        print(f"\nProcessed {len(all_results)} caption requests")
        print(f"Results saved as individual files in: {output_dir}")

if __name__ == "__main__":
    main()

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
    
    all_results = []
    for shard_file in tqdm(shard_files):
        # Load requests from shard
        requests = []
        with open(shard_file) as f:
            for line in f:
                requests.append(json.loads(line))
                
        if args.sanity_check:
            requests = requests[:1]  # Only process first request for sanity check
            
        # Process this shard
        results = processor.process_requests(
            requests,
            description=f"Processing {shard_file.name}",
            sanity_check=args.sanity_check
        )
        all_results.extend(results)
        
        if args.sanity_check:
            print("\nSanity check results:")
            print(json.dumps(results, indent=2))
            break

    if not args.sanity_check:
        # Save all results
        results_file = output_dir / "caption_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nProcessed {len(all_results)} caption requests")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

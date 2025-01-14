import argparse
import json
from pathlib import Path
from caption2.core.batch_api import BatchProcessor
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='Step 3: Process caption requests')
    parser.add_argument('--requests_dir', type=str, required=True,
                       help='Directory containing sharded request files from step 2')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run sanity check with single request only')
    args = parser.parse_args()

    # Get list of all shard files
    requests_dir = Path(args.requests_dir)
    shard_files = sorted(list(requests_dir.glob("shard_*.json")))
    if not shard_files:
        raise ValueError(f"No shard files found in {args.requests_dir}")

    # For sanity check, only process first shard
    if args.sanity_check:
        with open(shard_files[0]) as f:
            requests = json.load(f)
    else:
        # Load all requests from shards
        requests = []
        for shard_file in shard_files:
            with open(shard_file) as f:
                requests.extend(json.load(f))

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process requests
    client = OpenAI()
    processor = BatchProcessor(
        client=client,
        output_dir=output_dir
    )
    
    caption_results = processor.process_requests(
        requests,
        description="Frame caption generation",
        sanity_check=args.sanity_check
    )

    if args.sanity_check:
        print("\nSanity check results:")
        print(json.dumps(caption_results, indent=2))
        return

    # Save results
    results_file = output_dir / "caption_results.json"
    with open(results_file, 'w') as f:
        json.dump(caption_results, f, indent=2)

    print(f"\nProcessed {len(caption_results)} caption requests")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

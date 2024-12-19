import argparse
import json
from pathlib import Path
from caption2.core.batch_api import BatchProcessor
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='Step 3: Process caption requests')
    parser.add_argument('--requests_file', type=str, required=True,
                       help='Path to caption_requests.json from step 2')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run sanity check with single request only')
    args = parser.parse_args()

    # Load requests
    with open(args.requests_file) as f:
        requests = json.load(f)

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

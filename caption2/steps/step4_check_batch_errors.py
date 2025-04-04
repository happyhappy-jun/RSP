import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

def load_error_file(batch_id: str, output_dir: Path, client: OpenAI) -> List[Dict]:
    """Load and parse error.jsonl file for a batch"""
    error_file = output_dir / f"output_{batch_id}_error.jsonl"
    errors = []
    
    # Get batch status and error file
    status = client.batches.retrieve(batch_id)
    if status.error_file_id:
        error_content = client.files.content(status.error_file_id)
        with open(error_file, "wb") as f:
            f.write(error_content.read())
            
        # Parse errors from file
        with open(error_file) as f:
            for line in f:
                try:
                    errors.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
                    
    return errors

def create_retry_requests(errors: List[Dict], original_requests: List[Dict]) -> List[Dict]:
    """Create new requests for failed items"""
    retry_requests = []
    
    # Create mapping of custom_id to original request
    request_map = {req['custom_id']: req for req in original_requests}
    
    for error in errors:
        custom_id = error.get('custom_id')
        if custom_id and custom_id in request_map:
            # Copy original request for retry
            retry_request = request_map[custom_id].copy()
            retry_requests.append(retry_request)
            
    return retry_requests

def main():
    parser = argparse.ArgumentParser(description='Step 4: Check batch errors and prepare retries')
    parser.add_argument('--start_batch', type=str, required=True,
                       help='Starting batch ID to check')
    parser.add_argument('--end_batch', type=str, required=True,
                       help='Ending batch ID to check')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing batch outputs')
    parser.add_argument('--requests_file', type=str, required=True,
                       help='Original requests file for retry preparation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    client = OpenAI()

    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
    
    # Load original requests for retry preparation
    with open(args.requests_file) as f:
        original_requests = json.load(f)

    # List all batches
    print("\nListing batch requests...")
    all_batches = list(client.batches.list())
    ids = [batch.id for batch in all_batches]
    
    # Convert start/end batch indices to integers
    start_idx = ids.index(args.start_batch)
    end_idx = ids.index(args.end_batch)
    
    relevant_batch_ids = all_batches[end_idx:start_idx+1]
    
    print(f"Processing batches {start_idx} to {end_idx} " 
          f"(IDs: {args.start_batch} to {args.end_batch})")

    all_errors = []
    print("\nChecking batches for errors...")
    for batch in tqdm(relevant_batch_ids):
        status = client.batches.retrieve(batch.id)
        if status.status == "failed":
            print(f"\nBatch {batch.id} failed completely")
            continue
            
        errors = load_error_file(batch.id, output_dir, client)
        if errors:
            print(f"\nFound {len(errors)} errors in batch {batch.id}")
            all_errors.extend(errors)

    if all_errors:
        print(f"\nTotal errors found: {len(all_errors)}")
        
        # Save all errors
        error_file = output_dir / "combined_errors.json"
        with open(error_file, 'w') as f:
            json.dump(all_errors, f, indent=2)
        print(f"Combined errors saved to: {error_file}")
        
        # Create retry requests
        retry_requests = create_retry_requests(all_errors, original_requests)
        if retry_requests:
            retry_file = output_dir / "retry_requests.json"
            with open(retry_file, 'w') as f:
                json.dump(retry_requests, f, indent=2)
            print(f"\nCreated {len(retry_requests)} retry requests")
            print(f"Retry requests saved to: {retry_file}")
    else:
        print("\nNo errors found in the specified batch range")

if __name__ == "__main__":
    main()

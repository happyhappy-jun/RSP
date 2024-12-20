import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

def load_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load results from a JSON file"""
    if not file_path.exists():
        return []
    with open(file_path) as f:
        return json.load(f)

def extract_custom_ids(requests: List[Dict[str, Any]]) -> set:
    """Extract all custom IDs from original requests"""
    return {req['custom_id'] for req in requests}

def combine_results(original_results: List[Dict], 
                   retry_results: List[Dict]) -> List[Dict]:
    """Combine original and retry results, preferring retry results for duplicates"""
    # Create mapping of custom_id to result
    combined = {}
    
    # Add original results
    for result in original_results:
        combined[result['custom_id']] = result
        
    # Override with retry results where available
    for result in retry_results:
        combined[result['custom_id']] = result
        
    # Convert back to sorted list
    return sorted(combined.values(), key=lambda x: x['custom_id'])

def find_missing_ids(all_ids: set, results: List[Dict]) -> set:
    """Find custom IDs that are missing from results"""
    result_ids = {r['custom_id'] for r in results}
    return all_ids - result_ids

def main():
    parser = argparse.ArgumentParser(description='Step 5: Combine original and retry results')
    parser.add_argument('--original_results', type=str, required=True,
                       help='Path to original caption_results.json')
    parser.add_argument('--retry_results', type=str,
                       help='Path to retry results JSON (optional)')
    parser.add_argument('--requests_file', type=str, required=True,
                       help='Original requests file to check for missing IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for combined results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original requests to get all expected custom IDs
    print("\nLoading original requests...")
    with open(args.requests_file) as f:
        original_requests = json.load(f)
    all_ids = extract_custom_ids(original_requests)
    print(f"Found {len(all_ids)} total requests")

    # Load results
    print("\nLoading results...")
    original_results = load_results(Path(args.original_results))
    print(f"Original results: {len(original_results)}")
    
    retry_results = []
    if args.retry_results:
        retry_results = load_results(Path(args.retry_results))
        print(f"Retry results: {len(retry_results)}")

    # Combine results
    print("\nCombining results...")
    combined_results = combine_results(original_results, retry_results)
    
    # Check for missing IDs
    missing_ids = find_missing_ids(all_ids, combined_results)
    if missing_ids:
        print(f"\nWARNING: Missing results for {len(missing_ids)} requests:")
        for missing_id in sorted(missing_ids):
            print(f"  - {missing_id}")
            
        # Save missing IDs
        missing_file = output_dir / "missing_ids.json"
        with open(missing_file, 'w') as f:
            json.dump(sorted(list(missing_ids)), f, indent=2)
        print(f"\nMissing IDs saved to: {missing_file}")

    # Save combined results
    combined_file = output_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"\nSaved {len(combined_results)} combined results to: {combined_file}")

if __name__ == "__main__":
    main()

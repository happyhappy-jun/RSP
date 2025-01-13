import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

def load_frame_info(file_path: Path) -> Dict[str, Any]:
    """Load frame info from JSON file"""
    if not file_path.exists():
        raise FileNotFoundError(f"Frame info file not found: {file_path}")
    with open(file_path) as f:
        return json.load(f)

def create_label_results(frame_info: Dict[str, Any], 
                        requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create results using class labels from frame_info"""
    results = []
    
    for req in tqdm(requests, desc="Creating label results"):
        custom_id = req['custom_id']
        
        # Extract video_id and frame_idx from custom_id (assuming format: video_id_frame_idx)
        video_id, frame_idx = custom_id.rsplit('_', 1)
        
        # Get label from frame_info
        if video_id not in frame_info:
            print(f"Warning: Video ID {video_id} not found in frame info")
            continue
            
        label = frame_info[video_id]['label']
        
        # Create result entry matching the schema expected by step6
        result = {
            'custom_id': custom_id,
            'response': {
                'body': {
                    'choices': [{
                        'message': {
                            'content': label
                        }
                    }]
                }
            }
        }
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Step 5-1: Create results using class labels')
    parser.add_argument('--frame_info', type=str, required=True,
                       help='Path to frame_info.json containing class labels')
    parser.add_argument('--requests_file', type=str, required=True,
                       help='Original requests file to get sample IDs')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frame info
    print("\nLoading frame info...")
    frame_info = load_frame_info(Path(args.frame_info))
    print(f"Loaded info for {len(frame_info)} videos")

    # Load original requests
    print("\nLoading requests...")
    with open(args.requests_file) as f:
        requests = json.load(f)
    print(f"Found {len(requests)} requests")

    # Create results using labels
    print("\nCreating results from class labels...")
    results = create_label_results(frame_info, requests)

    # Save results
    results_file = output_dir / "combined_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to: {results_file}")
    
    # Report any skipped entries
    skipped = len(requests) - len(results)
    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries due to missing video IDs")

if __name__ == "__main__":
    main()

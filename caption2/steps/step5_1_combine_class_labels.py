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
        return json.load(f)['videos']

def create_label_results(frame_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create results using class labels from frame_info"""
    results = []
    
    for frame in tqdm(frame_info, desc="Creating label results"):
        label = frame['class_label']
        
        # Create a result for each pair (assuming one pair per video for now)
        custom_id = f'video_{frame["video_idx"]}_pair_{frame["pair_idx"]}'
        
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
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load frame info
    print("\nLoading frame info...")
    frame_info = load_frame_info(Path(args.frame_info))
    print(f"Loaded info for {len(frame_info)} videos")

    # Create results using labels
    print("\nCreating results from class labels...")
    results = create_label_results(frame_info)

    # Save results
    results_file = output_dir / "combined_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to: {results_file}")
    
    print(f"\nCreated {len(results)} results")

if __name__ == "__main__":
    main()

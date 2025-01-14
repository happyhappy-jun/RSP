import argparse
import json
from pathlib import Path

from tqdm import tqdm

from caption2.core.request_builder import RequestBuilder
from caption2.core.config import Config

def main():
    parser = argparse.ArgumentParser(description='Step 2: Create caption requests')
    parser.add_argument('--frame_info', type=str, required=True,
                       help='Path to frame_info.json from step 1')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for requests')
    parser.add_argument('--config_path', type=str,
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load frame info
    with open(args.frame_info) as f:
        frame_info = json.load(f)

    # Setup request builder
    config = Config(args.config_path, frame_output_dir=str(Path(args.frame_info).parent)) if args.config_path else Config(frame_output_dir=str(Path(args.frame_info).parent))
    builder = RequestBuilder(config=config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create requests
    requests = []
    for video in tqdm(frame_info['videos']):
        try:
            metadata = {
                'video_name': video['video_name'],
                'class_label': video['class_label'],
                'frame_indices': json.dumps(video['frame_indices']),
            }
            
            # Create custom ID based on video_idx and optional pair_idx
            custom_id = (f"video_{video['video_idx']}_pair_{video['pair_idx']+2}"
                        if video['pair_idx'] is not None 
                        else f"video_{video['video_idx']}")
            
            request = builder.build_caption_request(
                frame_paths=video['frame_paths'],
                custom_id=custom_id,
                metadata=metadata
            )
            requests.append(request)
        except Exception as e:
            print(f"Error creating request for video {video['video_idx']}: {str(e)}")
            continue

    # Save requests
    requests_file = output_dir / "caption_requests.json"
    with open(requests_file, 'w') as f:
        json.dump(requests, f, indent=2)

    print(f"\nCreated {len(requests)} caption requests")
    print(f"Requests saved to: {requests_file}")

if __name__ == "__main__":
    main()

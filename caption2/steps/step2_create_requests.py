import argparse
import json
from pathlib import Path
import gc

from tqdm import tqdm

from caption2.core.request_builder import RequestBuilder
from caption2.core.config import Config

def process_batch(videos, builder, batch_num):
    requests = []
    for video in tqdm(videos, desc=f'Processing batch {batch_num}'):
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
    return requests

def main():
    parser = argparse.ArgumentParser(description='Step 2: Create caption requests')
    parser.add_argument('--frame_info', type=str, required=True,
                       help='Path to frame_info.json from step 1')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for requests')
    parser.add_argument('--config_path', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of videos to process in each batch')
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

    # Process videos in batches
    total_requests = 0
    videos = frame_info['videos']
    num_batches = (len(videos) + args.batch_size - 1) // args.batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * args.batch_size
        end_idx = min((batch_num + 1) * args.batch_size, len(videos))
        
        # Process current batch
        batch_videos = videos[start_idx:end_idx]
        batch_requests = process_batch(batch_videos, builder, batch_num + 1)
        
        # Save batch requests
        batch_file = output_dir / f"caption_requests_batch_{batch_num + 1}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_requests, f, indent=2)
        
        total_requests += len(batch_requests)
        
        # Force garbage collection
        del batch_requests
        gc.collect()
        
        print(f"Batch {batch_num + 1}/{num_batches} completed. Saved to: {batch_file}")

    # Combine all batches into one file
    print("\nCombining all batches into one file...")
    all_requests = []
    for batch_num in range(num_batches):
        batch_file = output_dir / f"caption_requests_batch_{batch_num + 1}.json"
        with open(batch_file) as f:
            batch_requests = json.load(f)
            all_requests.extend(batch_requests)
        # Remove individual batch file
        batch_file.unlink()

    # Save combined requests
    combined_file = output_dir / "caption_requests.json"
    with open(combined_file, 'w') as f:
        json.dump(all_requests, f, indent=2)

    print(f"\nCreated {total_requests} caption requests")
    print(f"Combined requests saved to: {combined_file}")

if __name__ == "__main__":
    main()

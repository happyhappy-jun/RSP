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

    # Process all videos
    print("\nProcessing all videos...")
    all_requests = process_batch(frame_info['videos'], builder, 1)
    total_requests = len(all_requests)

    # Estimate size and create shards
    from caption2.core.batch_api import BatchProcessor
    processor = BatchProcessor(client=None, output_dir=str(output_dir))
    
    shards = []
    current_shard = []
    current_size = 0
    max_shard_size = 100 * 1024 * 1024  # 100MB per shard
    
    print("\nCreating size-based shards...")
    for request in tqdm(all_requests, desc="Sharding requests"):
        request_size = processor._estimate_request_size(request)
        
        if current_size + request_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_size = 0
            
        current_shard.append(request)
        current_size += request_size
        
    if current_shard:
        shards.append(current_shard)

    # Save shards
    print(f"\nSaving {len(shards)} shards...")
    for i, shard in enumerate(tqdm(shards, desc="Saving shards")):
        shard_file = output_dir / f"shard_{i:04d}.json"
        with open(shard_file, 'w') as f:
            json.dump(shard, f, indent=2)

    print(f"\nCreated {total_requests} caption requests")
    print(f"Split into {len(shards)} shards in: {output_dir}")

if __name__ == "__main__":
    main()

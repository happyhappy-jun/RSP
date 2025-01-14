import argparse
import json
from pathlib import Path
import gc

from tqdm import tqdm

from caption2.core.request_builder import RequestBuilder
from caption2.core.config import Config

def save_shard(shard, output_dir, shard_num):
    """Save a shard of requests to disk as JSONL"""
    shard_file = output_dir / f"shard_{shard_num:04d}.jsonl"
    with open(shard_file, 'w') as f:
        for request in shard:
            json.dump(request, f)
            f.write('\n')
    return shard_file

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

    # Setup for processing
    from caption2.core.batch_api import BatchProcessor
    processor = BatchProcessor(client=None, output_dir=str(output_dir))
    
    current_shard = []
    current_size = 0
    max_shard_size = 150 * 1024 * 1024  # 150MB per shard
    shard_count = 0
    total_requests = 0
    
    # Process videos in smaller batches
    print("\nProcessing videos and creating shards...")
    batch_size = 100  # Process 100 videos at a time
    
    for i, video in enumerate(tqdm(frame_info['videos'], desc="Processing videos")):
        try:
            metadata = {
                'video_name': video['video_name'],
                'class_label': video['class_label'],
                'frame_indices': json.dumps(video['frame_indices']),
            }
            
            # Create custom ID
            custom_id = (f"video_{video['video_idx']}_pair_{video['pair_idx']+2}"
                        if video['pair_idx'] is not None 
                        else f"video_{video['video_idx']}")
            
            request = builder.build_caption_request(
                frame_paths=video['frame_paths'],
                custom_id=custom_id,
                metadata=metadata
            )
            
            # Calculate size of this single request
            request_size = processor._estimate_request_size(request)
            
            # If adding this request would exceed max size, save current shard
            if current_size + request_size >= max_shard_size and current_shard:
                save_shard(current_shard, output_dir, shard_count)
                shard_count += 1
                current_shard = []
                current_size = 0
                gc.collect()
            
            # Add request to current shard
            current_shard.append(request)
            current_size += request_size
            total_requests += 1
            
            # Optional batch-based garbage collection
            if i > 0 and i % batch_size == 0:
                gc.collect()

        except Exception as e:
            print(f"Error creating request for video {video['video_idx']}: {str(e)}")
            continue
    
    # Save final shard if it contains any requests
    if current_shard:
        save_shard(current_shard, output_dir, shard_count)
        shard_count += 1

    print(f"\nCreated {total_requests} caption requests")
    print(f"Split into {shard_count} shards in: {output_dir}")

if __name__ == "__main__":
    main()

import argparse
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from caption2.core.request_builder import RequestBuilder
from caption2.core.batch_api import BatchProcessor
from caption2.core.config import Config
from openai import OpenAI

async def main():
    parser = argparse.ArgumentParser(description='Step 7: Create future frame captions')
    parser.add_argument('--frame_info', type=str, required=True,
                       help='Path to frame_info.json from step 1')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for future captions')
    parser.add_argument('--config_path', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run sanity check with single request only')
    parser.add_argument("--frame_dir", type=str,)
    args = parser.parse_args()

    # Setup paths and config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config(args.config_path, frame_output_dir=args.frame_dir) if args.config_path else Config(frame_output_dir=args.frame_dir)
    
    # Load frame info
    print(f"\nLoading frame info from: {args.frame_info}")
    with open(args.frame_info) as f:
        frame_info = json.load(f)

    # Create requests for future frame prediction
    print("\nCreating future frame requests...")
    builder = RequestBuilder(config=config)
    future_requests = []
    
    for video in tqdm(frame_info['videos']):
        try:
            # For each pair, predict what comes after the second frame
            if len(video['frame_paths']) >= 2:  # Ensure we have at least 2 frames
                metadata = {
                    'video_name': video['video_name'],
                    'class_label': video['class_label'],
                    'frame_indices': json.dumps(video['frame_indices']),
                    'pair_idx': str(video['pair_idx']),
                    'prediction_type': 'future_frame',
                    'sampling_seed': video.get('sampling_seed', 42)
                }
                
                # Create custom ID matching existing caption structure
                custom_id = f"video_{video['video_idx']}_pair_{video['pair_idx']}"
                
                request = builder.build_caption_request(
                    frame_paths=[video['frame_paths'][-1]],  # Use last two frames
                    custom_id=custom_id,
                    metadata=metadata,
                    system_prompt=config.prompt_config['caption']['prompts']['future']
                )
                future_requests.append(request)
                
        except Exception as e:
            print(f"Error creating request for video {video['video_idx']}: {str(e)}")
            continue

    print(f"Created {len(future_requests)} future prediction requests")

    # Process requests
    client = OpenAI()
    processor = BatchProcessor(
        client=client,
        output_dir=output_dir
    )
    
    if args.sanity_check:
        print("\nRunning sanity check...")
        if future_requests:
            results = processor.submit_requests(
                [future_requests[0]],
                description="Future caption sanity check",
                sanity_check=True
            )
            print("\nSanity check results:")
            print(json.dumps(results, indent=2))
    else:
        print("\nProcessing future caption requests...")
        batch_ids = processor.submit_requests(
            future_requests,
            description="Future frame caption generation"
        )

        del future_requests
        
        print("\nMonitoring batch processing...")
        results = processor.monitor_batches(batch_ids)
        
        # Save results
        results_file = output_dir / "future_captions.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nProcessed {len(results)} future caption requests")
        print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())

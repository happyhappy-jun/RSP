import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

from caption2.core.logging_config import setup_logging
from caption2.core.pipeline_state import PipelineState, PipelineStage

from caption2.core.frame_extractor import extract_frames
from caption2.core.request_builder import RequestBuilder
from caption2.core.batch_api import BatchProcessor
from caption2.core.config import Config

def setup_directories(output_root: str) -> Dict[str, Path]:
    """Setup directory structure for pipeline"""
    paths = {
        'frames': Path(output_root) / "frames",
        'requests': Path(output_root) / "requests",
        'results': Path(output_root) / "results",
        'embeddings': Path(output_root) / "embeddings"
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return paths


def create_requests(
    frame_info: Dict[str, Any],
    config: Config = None
) -> List[Dict[str, Any]]:
    """Create API requests for frame analysis"""
    if config is None:
        config = Config()
        
    builder = RequestBuilder(config=config)
    requests = []
    
    for video in frame_info['videos']:
        try:
            # Determine if this is paired sampling based on frame count
            is_paired = len(video['frame_paths']) % 2 == 0 and frame_info['config'].get('num_pairs', 0) > 0
            
            if is_paired:
                # Process frames in pairs
                for pair_idx in range(0, len(video['frame_paths']), 2):
                    pair_frames = video['frame_paths'][pair_idx:pair_idx + 2]
                    pair_indices = video['frame_indices'][pair_idx:pair_idx + 2]
                    
                    metadata = {
                        'video_name': video['video_name'],
                        'class_label': video['class_label'],
                        'frame_indices': json.dumps(pair_indices),
                        'pair_index': str(pair_idx // 2),
                        'sampling_seed': str(video['sampling_seed'])
                    }
                    
                    request = builder.build_caption_request(
                        frame_paths=pair_frames,
                        custom_id=f"video_{video['video_idx']}_pair_{pair_idx//2}",
                        metadata=metadata,
                        system_prompt=config.prompt_config['caption']['prompts'][config.prompt_config['caption']['default_prompt']]
                    )
                    requests.append(request)
            else:
                # Process all frames in single request for uniform sampling
                metadata = {
                    'video_name': video['video_name'],
                    'class_label': video['class_label'],
                    'frame_indices': json.dumps(video['frame_indices']),
                    'sampling_seed': str(video['sampling_seed'])
                }
                
                request = builder.build_caption_request(
                    frame_paths=video['frame_paths'],
                    custom_id=f"video_{video['video_idx']}",
                    metadata=metadata,
                    system_prompt=config.prompt_config['caption']['prompts'][config.prompt_config['caption']['default_prompt']]
                )
                requests.append(request)
        except Exception as e:
            print(f"Error creating request for video {video['video_idx']}: {str(e)}")
            continue
            
    return requests


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing videos')
    parser.add_argument('--output_root', type=str, required=True,
                       help='Root directory for output artifacts')
    parser.add_argument('--config_path', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--sampler', type=str, default='uniform',
                       choices=['uniform', 'paired'],
                       help='Frame sampling strategy')
    parser.add_argument('--sanity_check', action='store_true',
                       help='Run sanity check with single request only')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_file=Path(args.log_file) if args.log_file else None
    )
    logger = logging.getLogger(__name__)
    
    # Setup pipeline
    paths = setup_directories(args.output_root)
    config = Config(args.config_path) if args.config_path else Config()
    state = PipelineState(Path(args.output_root))
    
    # Get video paths
    video_paths = []
    for root, _, files in os.walk(args.data_root):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
                
    print(f"Found {len(video_paths)} videos")
    
    # Check if frame extraction was already done
    frame_info_path = paths['frames'] / "frame_info.json"
    if frame_info_path.exists():
        print("\nLoading existing frame info...")
        with open(frame_info_path) as f:
            frame_info = json.load(f)
    else:
        # Extract frames with seed
        print("\nExtracting frames...")
        frame_info = extract_frames(
            video_paths,
            paths['frames'],
            args.sampler,
            config,
            seed=args.seed
        )
        
        # Save frame info
        with open(frame_info_path, 'w') as f:
            json.dump(frame_info, f, indent=2)
        
    # Create caption requests
    print("\nCreating caption requests...")
    caption_requests = create_requests(frame_info, config)
    
    # Process caption requests
    print("\nProcessing caption requests...")
    client = OpenAI()
    processor = BatchProcessor(
        client=client,
        output_dir=paths['requests']
    )
    
    caption_results = processor.process_requests(
        caption_requests,
        description="Frame caption generation",
        sanity_check=args.sanity_check
    )
    
    if args.sanity_check:
        print("\nSanity check results:")
        print(json.dumps(caption_results, indent=2))
        return
    
    # Save caption results
    with open(paths['results'] / "caption_results.json", 'w') as f:
        json.dump(caption_results, f, indent=2)

    # Create embeddings from the caption results
    print("\nCreating embeddings...")
    from caption2.core.embedding_creator import EmbeddingCreator
    creator = EmbeddingCreator()
    await creator.process_caption_results(caption_results, paths['embeddings'])
        
    print("\nPipeline complete!")
    print(f"Results saved to: {args.output_root}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

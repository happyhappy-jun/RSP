import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm

from caption2.core.frame_sampler import UniformFrameSampler, PairedFrameSampler
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

def extract_frames(
    video_paths: List[str],
    output_dir: Path,
    sampler_type: str = "uniform",
    config: Config = None,
    seed: int = 42
) -> Dict[str, Any]:
    """Extract frames using configured sampler"""
    if config is None:
        config = Config()
        
    # Create sampler with seed
    if sampler_type == "uniform":
        sampler = UniformFrameSampler(seed=seed)
        frame_config = config.frame_config['uniform']
    else:
        sampler = PairedFrameSampler(seed=seed, **config.frame_config['paired'])
        frame_config = config.frame_config['paired']
    
    frame_config['seed'] = seed  # Track seed in config
    
    frame_info = {
        'config': frame_config,
        'videos': []
    }
    
    for video_idx, video_path in enumerate(tqdm(video_paths)):
        try:
            # Get video metadata
            video_path = Path(video_path)
            video_name = video_path.stem
            class_label = video_path.parent.name
            # Create class directory first, then video directory
            class_dir = output_dir / class_label
            class_dir.mkdir(exist_ok=True, parents=True)
            video_dir = class_dir / video_name
            video_dir.mkdir(exist_ok=True)
            
            # Sample and save frames
            frame_indices = sampler.sample_frames(str(video_path))
            frame_paths = []
            
            for frame_idx, video_frame_idx in enumerate(frame_indices):
                frame_path = video_dir / f"frame_{frame_idx}.jpg"
                frame_paths.append(str(frame_path))
                
            frame_info['videos'].append({
                'video_idx': video_idx,
                'video_path': str(video_path),
                'video_name': video_name,
                'class_label': class_label,
                'frame_indices': frame_indices,
                'frame_paths': frame_paths,
                'sampling_seed': seed
            })
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            continue
            
    return frame_info

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
            # Include metadata in request
            metadata = {
                'video_name': video['video_name'],
                'class_label': video['class_label'],
                'frame_indices': video['frame_indices'],
                'sampling_seed': video['sampling_seed']
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

def create_embedding_requests(
    caption_results: List[Dict[str, Any]],
    config: Config = None
) -> List[Dict[str, Any]]:
    """Create embedding requests for captions"""
    if config is None:
        config = Config()
        
    builder = RequestBuilder()
    requests = []
    
    for result in caption_results:
        try:
            analysis = result['response']['body']['choices'][0]['message']['content']
            request = builder.build_embedding_request(
                text=analysis,
                custom_id=result['custom_id'],
                model=config.prompt_config['embedding']['model']
            )
            requests.append(request)
        except Exception as e:
            print(f"Error creating embedding request for {result['custom_id']}: {str(e)}")
            continue
            
    return requests

def main():
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
    args = parser.parse_args()
    
    # Setup
    paths = setup_directories(args.output_root)
    config = Config(args.config_path) if args.config_path else Config()
    
    # Get video paths
    video_paths = []
    for root, _, files in os.walk(args.data_root):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
                
    print(f"Found {len(video_paths)} videos")
    
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
    with open(paths['frames'] / "frame_info.json", 'w') as f:
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
        description="Frame caption generation"
    )
    
    # Save caption results
    with open(paths['results'] / "caption_results.json", 'w') as f:
        json.dump(caption_results, f, indent=2)
        
    # Create and process embedding requests
    print("\nCreating embedding requests...")
    embedding_requests = create_embedding_requests(caption_results, config)
    
    print("\nProcessing embedding requests...")
    embedding_results = processor.process_requests(
        embedding_requests,
        description="Caption embedding generation"
    )
    
    # Save embedding results
    with open(paths['embeddings'] / "embedding_results.json", 'w') as f:
        json.dump(embedding_results, f, indent=2)
        
    print("\nPipeline complete!")
    print(f"Results saved to: {args.output_root}")

if __name__ == "__main__":
    main()

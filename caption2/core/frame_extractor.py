import cv2
import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from caption2.core.frame_sampler import UniformFrameSampler, PairedFrameSampler
from caption2.core.config import Config

async def extract_frames(
    video_paths: List[str],
    output_dir: Path,
    sampler_type: str = "uniform",
    config: Config = None,
    seed: int = 42,
    max_workers: int = 4
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
    
    async def process_video(video_idx: int, video_path: str) -> List[Dict[str, Any]]:
        try:
            # Get video metadata and make path relative to data root
            video_path = Path(video_path)
            rel_video_path = video_path.relative_to(Path(config.data_root))
            video_name = video_path.stem
            class_label = video_path.parent.name
            # Create class directory first, then video directory
            class_dir = output_dir / class_label
            class_dir.mkdir(exist_ok=True, parents=True)
            video_dir = class_dir / video_name
            video_dir.mkdir(exist_ok=True)
            
            # Sample frames
            frames = sampler.sample_frames(str(video_path))
            frame_paths = []
            
            # Save frames using OpenCV
            cap = cv2.VideoCapture(str(video_path))
            
            for frame_idx, video_frame_idx in enumerate(frames):
                # For paired sampling, use pair_X_frameY format
                if sampler_type == "paired":
                    pair_idx = frame_idx // 2
                    frame_in_pair = frame_idx % 2
                    frame_path = video_dir / f"pair_{pair_idx}_frame{frame_in_pair}.jpg"
                else:
                    frame_path = video_dir / f"frame_{frame_idx}.jpg"
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    cv2.imwrite(str(frame_path), frame)
                    if frame_path.exists():
                        # Store path relative to output_dir
                        rel_path = frame_path.relative_to(output_dir)
                        frame_paths.append(str(rel_path))
                    else:
                        raise FileNotFoundError(f"Failed to save frame to {frame_path}")
                else:
                    raise ValueError(f"Could not read frame {video_frame_idx} from video")
            
            cap.release()
            
            # For paired sampling, create separate entries for each pair
            video_entries = []
            if sampler_type == "paired":
                for pair_idx in range(0, len(frames), 2):
                    pair_indices = frames[pair_idx:pair_idx + 2]
                    pair_paths = frame_paths[pair_idx:pair_idx + 2]
                    
                    video_entries.append({
                        'video_idx': video_idx,
                        'pair_idx': pair_idx // 2,
                        'video_path': str(rel_video_path),
                        'video_name': video_name,
                        'class_label': class_label,
                        'frame_indices': pair_indices,
                        'frame_paths': pair_paths,
                    })
            else:
                video_entries.append({
                    'video_idx': video_idx,
                    'pair_idx': None,
                    'video_path': str(rel_video_path),
                    'video_name': video_name,
                    'class_label': class_label,
                    'frame_indices': frames,
                    'frame_paths': frame_paths,
                })
                
            return video_entries
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return []

    # Process videos in parallel using ThreadPoolExecutor
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for video_idx, video_path in enumerate(video_paths):
            task = loop.run_in_executor(
                executor,
                lambda idx=video_idx, path=video_path: asyncio.run(process_video(idx, path))
            )
            tasks.append(task)
            
        # Use tqdm to show progress
        for entries in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            frame_info['videos'].extend(await entries)
            
    return frame_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos using uniform or paired sampling')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for extracted frames')
    parser.add_argument('--sampler', type=str, default='uniform',
                       choices=['uniform', 'paired'],
                       help='Frame sampling strategy')
    parser.add_argument('--config_path', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Get video paths
    video_paths = []
    for root, _, files in os.walk(args.data_root):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
                
    print(f"Found {len(video_paths)} videos")

    # Extract frames
    config = Config(args.config_path) if args.config_path else None
    frame_info = extract_frames(
        video_paths,
        Path(args.output_dir),
        args.sampler,
        config,
        seed=args.seed
    )

    # Save frame info
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "frame_info.json", 'w') as f:
        json.dump(frame_info, f, indent=2)

    print(f"\nExtracted frames saved to: {args.output_dir}")

"""
Script to presample frames from videos and generate captions using API endpoints.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import torch
import numpy as np
from decord import VideoReader
from tqdm import tqdm


def load_frames(video_path: str, indices: List[int]) -> List[np.ndarray]:
    """Load specific frames from a video file."""
    vr = VideoReader(video_path)
    frames = vr.get_batch(indices).asnumpy()
    return [frame for frame in frames]


def sample_frame_pairs(video_path: str, num_pairs: int = 64, max_distance: int = 48) -> List[Tuple[int, int]]:
    """Sample pairs of frame indices from a video."""
    vr = VideoReader(video_path)
    video_length = len(vr)
    
    pairs = []
    for _ in range(num_pairs):
        # Sample first frame index
        first_idx = random.randint(0, video_length - max_distance - 1)
        # Sample second frame within max_distance
        second_idx = random.randint(first_idx + 1, min(first_idx + max_distance, video_length - 1))
        pairs.append((first_idx, second_idx))
    
    return pairs


def process_video(
    video_path: str,
    data_root: str,
    caption_endpoint_1: str,
    caption_endpoint_2: str,
    num_pairs: int = 64,
    max_distance: int = 48
) -> Dict:
    """Process a single video, sampling frames and getting captions."""
    # Get relative path by removing data_root
    rel_path = os.path.relpath(video_path, data_root)
    video_name = os.path.basename(video_path)
    
    # Sample frame pairs
    frame_pairs = sample_frame_pairs(video_path, num_pairs, max_distance)
    
    # Load and process frames
    results = []
    for first_idx, second_idx in frame_pairs:
        frames = load_frames(video_path, [first_idx, second_idx])
        
        # TODO: Implement API calls to caption endpoints
        caption_1 = "Placeholder caption 1"  # Replace with actual API call
        caption_2 = "Placeholder caption 2"  # Replace with actual API call
        
        pair_result = {
            "frame_indices": [first_idx, second_idx],
            "caption_1": caption_1,
            "caption_2": caption_2
        }
        results.append(pair_result)
    
    return {
        "video_path": rel_path,
        "video_name": video_name,
        "frame_pairs": results
    }


def main():
    parser = argparse.ArgumentParser(description="Presample frames and generate captions")
    parser.add_argument("--data-root", required=True, help="Root directory containing videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for caption files")
    parser.add_argument("--caption-endpoint-1", required=True, help="First caption API endpoint")
    parser.add_argument("--caption-endpoint-2", required=True, help="Second caption API endpoint")
    parser.add_argument("--num-pairs", type=int, default=64, help="Number of frame pairs to sample per video")
    parser.add_argument("--max-distance", type=int, default=48, help="Maximum frame distance in pairs")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov']:
        video_files.extend(Path(args.data_root).rglob(f'*{ext}'))
    
    # Process each video
    results = []
    for video_path in tqdm(video_files):
        try:
            result = process_video(
                str(video_path),
                args.data_root,
                args.caption_endpoint_1,
                args.caption_endpoint_2,
                args.num_pairs,
                args.max_distance
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
        
        # Save results periodically
        if len(results) % 100 == 0:
            output_path = os.path.join(args.output_dir, f"captions_batch_{len(results)}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results
    final_output = os.path.join(args.output_dir, "captions_final.json")
    with open(final_output, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

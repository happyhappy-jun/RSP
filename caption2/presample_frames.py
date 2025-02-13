"""
Script to presample frames from videos and generate captions using API endpoints.
"""
import glob
import os
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import torch
import numpy as np
from decord import VideoReader
from tqdm import tqdm
from PIL import Image
import io
import base64
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CaptionGenerator:
    def __init__(self, model: str, host: str = "0.0.0.0", port: int = 23333):
        self.model = model
        self.url = f"http://{host}:{port}/v1/chat/completions"

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy array frame to base64 string."""
        try:
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(frame)

            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')

            # Convert to base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to convert frame to base64: {e}")
            raise

    def get_caption(self, frame1: np.ndarray, frame2: np.ndarray, max_retries: int = 3) -> str:
        """Generate caption comparing two frames using LLM"""
        # Convert frames to base64
        img1_b64 = self.frame_to_base64(frame1)
        img2_b64 = self.frame_to_base64(frame2)
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the main differences between these two frames from a video."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
                ]
            }],
            "temperature": 1.0,
        }
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Send request to local server and measure time
                start_time = time.time()
                response = requests.post(self.url, json=payload)
                response.raise_for_status()
                response_json = response.json()
                caption = response_json['choices'][0]['message']['content']
                request_time = time.time() - start_time
                return caption
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise Exception(f"Failed to get caption after {max_retries} retries: {str(e)}")
                logging.warning(f"Request failed (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying


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
        least_frames_num = max_distance + 1
        if video_length >= least_frames_num:
            idx_cur = random.randint(0, video_length - least_frames_num)
            interval = random.randint(4, max_distance)
            idx_fut = idx_cur + interval
        else:
            indices = random.sample(range(video_length), 2)
            indices.sort()
            idx_cur, idx_fut = indices
        pairs.append((idx_cur, idx_fut))
    return pairs


def process_video(
    video_path: str,
    data_root: str,
    caption_generator: CaptionGenerator,
    num_pairs: int = 64,
    max_distance: int = 48,
    max_retries: int = 3
) -> Dict:
    """Process a single video, sampling frames and getting captions."""
    # Get relative path by removing data_root
    rel_path = os.path.relpath(video_path, data_root)
    video_name = os.path.basename(video_path)
    
    # Sample frame pairs
    frame_pairs = sample_frame_pairs(video_path, num_pairs, max_distance)
    
    # Load and process frames
    results = []
    retry_queue = []  # Store failed pairs for retry
    
    for first_idx, second_idx in frame_pairs:
        try:
            frames = load_frames(video_path, [first_idx, second_idx])
            caption = caption_dataset.get_caption(frames[0], frames[1])
            
            pair_result = {
                "frame_indices": [first_idx, second_idx],
                "caption": caption
            }
            results.append(pair_result)
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)
            
        except Exception as e:
            logging.warning(f"Failed to process pair ({first_idx}, {second_idx}): {str(e)}")
            retry_queue.append((first_idx, second_idx))
            time.sleep(60)  # Wait 1 minute before continuing
    
    # Process retry queue
    retry_count = 0
    while retry_queue and retry_count < max_retries:
        retry_count += 1
        current_queue = retry_queue[:]
        retry_queue = []
        
        logging.info(f"Attempting retry {retry_count}/{max_retries} for {len(current_queue)} pairs")
        
        for first_idx, second_idx in current_queue:
            try:
                frames = load_frames(video_path, [first_idx, second_idx])
                caption = caption_dataset.get_caption(frames[0], frames[1])
                
                pair_result = {
                    "frame_indices": [first_idx, second_idx],
                    "caption": caption
                }
                results.append(pair_result)
                
                time.sleep(0.1)
                
            except Exception as e:
                logging.warning(f"Retry {retry_count} failed for pair ({first_idx}, {second_idx}): {str(e)}")
                retry_queue.append((first_idx, second_idx))
                time.sleep(60)
    
    if retry_queue:
        logging.error(f"Failed to process {len(retry_queue)} pairs after {max_retries} retries")
    
    return {
        "video_path": rel_path,
        "video_name": video_name,
        "frame_pairs": results,
        "failed_pairs": retry_queue
    }


def main():
    parser = argparse.ArgumentParser(description="Presample frames and generate captions")
    parser.add_argument("--data-root", required=True, help="Root directory containing videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for caption files")
    parser.add_argument("--llm-model", required=True, help="LLM model name")
    parser.add_argument("--llm-host", default="0.0.0.0", help="LLM API host")
    parser.add_argument("--llm-port", type=int, default=23333, help="LLM API port")
    parser.add_argument("--num-pairs", type=int, default=64, help="Number of frame pairs to sample per video")
    parser.add_argument("--max-distance", type=int, default=48, help="Maximum frame distance in pairs")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of all video files
    video_files = (glob.glob(os.path.join(args.data_root, "*_front.mp4")) +
                            glob.glob(os.path.join(args.data_root, "*_overhead.mp4")))
    
    # Initialize caption generator
    caption_generator = CaptionGenerator(
        model=args.llm_model,
        host=args.llm_host,
        port=args.llm_port
    )

    # Process each video
    results = []
    for video_path in tqdm(video_files):
        try:
            result = process_video(
                str(video_path),
                args.data_root,
                caption_generator,
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

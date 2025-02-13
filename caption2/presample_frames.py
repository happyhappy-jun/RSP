"""
Script to presample frames from videos and generate captions using API endpoints.
"""
import glob
import os
import json
import random
import argparse
import logging
from typing import List, Dict, Tuple

import numpy as np
from decord import VideoReader
from tqdm import tqdm
from PIL import Image
import io
import base64
import asyncio
import aiohttp
import time
from itertools import chain
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('presample_frames.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FramePair:
    video_path: str
    first_idx: int
    second_idx: int
    frames: List[np.ndarray] = None
    caption: str = None
    retries: int = 0


class CaptionGenerator:
    def __init__(self, model: str, endpoints: List[Dict], max_concurrent: int = 5):
        """
        Initialize with multiple endpoints
        endpoints: List of dicts with host and port for each endpoint
        """
        self.model = model
        self.urls = [f"http://{ep['host']}:{ep['port']}/v1/chat/completions" for ep in endpoints]
        self.max_concurrent_per_endpoint = max_concurrent // len(self.urls)
        self.semaphores = [asyncio.Semaphore(self.max_concurrent_per_endpoint) for _ in self.urls]

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
            logger.error(f"Failed to convert frame to base64: {e}", exc_info=True)
            raise

    async def get_caption(self, frame1: np.ndarray, frame2: np.ndarray, endpoint_idx: int, max_retries: int = 3) -> str:
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
                    {"type": "text",
                     "text": "Describe followings. Keep the answer concise. 1. differences between these two frames from a video\n2. temporal change\n3. motion and dynamics\n4. Task of robot\n5. Environment of robot"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}}
                ]
            }],
            "temperature": 1.0,
        }
        retry_count = 0
        url = self.urls[endpoint_idx]
        semaphore = self.semaphores[endpoint_idx]
        
        async with semaphore:  # Limit concurrent requests per endpoint
            async with aiohttp.ClientSession() as session:
                while retry_count < max_retries:
                    try:
                        # Send request to server and measure time
                        start_time = time.time()
                        async with session.post(url, json=payload) as response:
                            response.raise_for_status()
                            response_json = await response.json()
                            caption = response_json['choices'][0]['message']['content']
                            request_time = time.time() - start_time
                            return caption
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            raise Exception(f"Failed to get caption after {max_retries} retries: {str(e)}")
                        logger.warning(f"Request failed (attempt {retry_count}/{max_retries})", exc_info=True)
                        # Exponential backoff with jitter
                        wait_time = min(60 * (2 ** retry_count) + random.uniform(0, 10), 300)  # Cap at 5 minutes
                        logger.info(f"Waiting {wait_time:.1f} seconds before retry {retry_count + 1}")
                        await asyncio.sleep(wait_time)

    async def process_frame_pair(self, pair: FramePair, endpoint_idx: int) -> None:
        """Process a single frame pair using specified endpoint"""
        try:
            if pair.frames is None:
                pair.frames = load_frames(pair.video_path, [pair.first_idx, pair.second_idx])
            pair.caption = await self.get_caption(pair.frames[0], pair.frames[1], endpoint_idx)
        except Exception as e:
            pair.retries += 1
            raise e


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


async def process_videos(
        video_paths: List[str],
        data_root: str,
        caption_generator: CaptionGenerator,
        num_pairs: int = 64,
        max_distance: int = 48,
        max_retries: int = 20,
        batch_size: int = 100
) -> Dict:
    """Process multiple videos concurrently"""
    # Create frame pairs for all videos
    all_pairs = []
    logger.info("Sampling frame pairs from videos...")
    for video_path in tqdm(video_paths):
        try:
            pairs = sample_frame_pairs(video_path, num_pairs, max_distance)
            all_pairs.extend([
                FramePair(video_path=video_path, first_idx=first, second_idx=second)
                for first, second in pairs
            ])
        except Exception as e:
            logger.error(f"Error sampling frames from {video_path}: {str(e)}", exc_info=True)
            continue

    # Process all pairs
    results_by_video = defaultdict(lambda: {"frame_pairs": [], "failed_pairs": []})
    retry_queue = []
    total_pairs = len(all_pairs)
    processed_pairs = 0

    async def process_pair(pair: FramePair, endpoint_idx: int):
        nonlocal processed_pairs
        try:
            await caption_generator.process_frame_pair(pair, endpoint_idx)
            video_result = results_by_video[pair.video_path]
            video_result["frame_pairs"].append({
                "frame_indices": [pair.first_idx, pair.second_idx],
                "caption": pair.caption
            })
            processed_pairs += 1
            if processed_pairs % 100 == 0:
                logger.info(f"Processed {processed_pairs}/{total_pairs} pairs")
        except Exception as e:
            if pair.retries < max_retries:
                retry_queue.append(pair)
            else:
                video_result = results_by_video[pair.video_path]
                video_result["failed_pairs"].append([pair.first_idx, pair.second_idx])
                logger.error(f"Failed to process pair after {max_retries} retries: {str(e)}")

    # Process in batches
    logger.info(f"Processing {total_pairs} pairs in batches of {batch_size}...")
    for i in range(0, len(all_pairs), batch_size):
        batch = all_pairs[i:i + batch_size]
        # Distribute pairs evenly between endpoints
        tasks = []
        for idx, pair in enumerate(batch):
            endpoint_idx = idx % len(caption_generator.urls)
            tasks.append(process_pair(pair, endpoint_idx))
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Save intermediate results every 1000 pairs
        if i % 1000 == 0:
            intermediate_results = []
            for video_path in video_paths:
                if video_path in results_by_video:
                    rel_path = os.path.relpath(video_path, data_root)
                    video_name = os.path.basename(video_path)
                    result = results_by_video[video_path]
                    intermediate_results.append({
                        "video_path": rel_path,
                        "video_name": video_name,
                        "frame_pairs": result["frame_pairs"],
                        "failed_pairs": result["failed_pairs"]
                    })
            
            output_path = os.path.join(os.path.dirname(data_root), f"captions_intermediate_{i}.json")
            with open(output_path, 'w') as f:
                json.dump(intermediate_results, f, indent=2)
            logger.info(f"Saved intermediate results to {output_path}")

    # Process retry queue with exponential backoff
    retry_count = 0
    while retry_queue and retry_count < max_retries:
        retry_count += 1
        current_queue = retry_queue[:]
        retry_queue = []
        
        wait_time = min(60 * (2 ** retry_count) + random.uniform(0, 10), 300)
        logger.info(f"Waiting {wait_time:.1f} seconds before retrying {len(current_queue)} pairs (attempt {retry_count})")
        await asyncio.sleep(wait_time)

        for i in range(0, len(current_queue), batch_size):
            batch = current_queue[i:i + batch_size]
            tasks = []
            for idx, pair in enumerate(batch):
                endpoint_idx = idx % len(caption_generator.urls)
                tasks.append(process_pair(pair, endpoint_idx))
            await asyncio.gather(*tasks, return_exceptions=True)

    # Format results
    final_results = []
    for video_path in video_paths:
        rel_path = os.path.relpath(video_path, data_root)
        video_name = os.path.basename(video_path)
        result = results_by_video[video_path]
        final_results.append({
            "video_path": rel_path,
            "video_name": video_name,
            "frame_pairs": result["frame_pairs"],
            "failed_pairs": result["failed_pairs"]
        })

    return final_results


async def main():
    parser = argparse.ArgumentParser(description="Presample frames and generate captions")
    parser.add_argument("--data-root", required=True, help="Root directory containing videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for caption files")
    parser.add_argument("--llm-model", required=True, help="LLM model name")
    parser.add_argument("--llm-hosts", nargs="+", default=["0.0.0.0", "0.0.0.0"], help="LLM API hosts")
    parser.add_argument("--llm-ports", nargs="+", type=int, default=[23333, 23334], help="LLM API ports")
    parser.add_argument("--num-pairs", type=int, default=64, help="Number of frame pairs to sample per video")
    parser.add_argument("--max-distance", type=int, default=48, help="Maximum frame distance in pairs")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent requests")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of pairs to process in each batch")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of all video files
    video_files = glob.glob(os.path.join(args.data_root, "*_front.mp4"))

    logger.info(f"Found {len(video_files)} video files")
    # Initialize caption generator
    # Create list of endpoints from hosts and ports
    endpoints = [
        {"host": host, "port": port} 
        for host, port in zip(args.llm_hosts, args.llm_ports)
    ]
    
    caption_generator = CaptionGenerator(
        model=args.llm_model,
        endpoints=endpoints,
        max_concurrent=args.max_concurrent
    )

    try:
        # Process all videos concurrently
        results = await process_videos(
            video_files,
            args.data_root,
            caption_generator,
            args.num_pairs,
            args.max_distance,
            batch_size=args.batch_size
        )

        # Save results periodically (every 100 processed pairs)
        processed_pairs = sum(len(r["frame_pairs"]) for r in results)
        if processed_pairs % 100 == 0:
            output_path = os.path.join(args.output_dir, f"captions_batch_{processed_pairs}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        # Save final results
        final_output = os.path.join(args.output_dir, "captions_final.json")
        with open(final_output, 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        logger.error("Error during processing", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

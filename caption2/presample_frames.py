"""
Script to presample frames from videos and generate captions using API endpoints.
"""
import glob
import os
import json
import random
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Set

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
import datetime
import signal
from contextlib import asynccontextmanager

class ProgressTracker:
    def __init__(self, total: int, name: str = ""):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.name = name
        self.last_update_count = 0
        self.update_interval = 100  # Update every 100 items

    def update(self, n: int = 1) -> Optional[str]:
        self.current += n

        # Only update message every N items
        if self.current - self.last_update_count < self.update_interval:
            return None

        self.last_update_count = self.current

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current == 0:
            eta = "unknown"
        else:
            items_per_sec = self.current / elapsed
            remaining_items = self.total - self.current
            eta_seconds = remaining_items / items_per_sec
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

        return (f"{self.name + ' ' if self.name else ''}Progress: {self.current}/{self.total} "
                f"({self.current/self.total*100:.1f}%) [ETA: {eta}]")

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

class GracefulExit(SystemExit):
    pass

def handle_sigterm(signum, frame):
    raise GracefulExit()

class CaptionGenerator:
    def __init__(self, model: str, endpoints: List[Dict], max_concurrent: int = 5, timeout: int = 300):
        """
        Initialize with multiple endpoints
        endpoints: List of dicts with host and port for each endpoint
        timeout: Timeout in seconds for each request
        """
        self.model = model
        self.urls = [f"http://{ep['host']}:{ep['port']}/v1/chat/completions" for ep in endpoints]
        self.max_concurrent_per_endpoint = max_concurrent // len(self.urls)
        self.semaphores = [asyncio.Semaphore(self.max_concurrent_per_endpoint) for _ in self.urls]
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
        self.active_tasks: Set[asyncio.Task] = set()

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy array frame to base64 string."""
        try:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to convert frame to base64: {e}", exc_info=True)
            raise

    @asynccontextmanager
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        try:
            yield self.session
        except Exception:
            if self.session:
                await self.session.close()
                self.session = None
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

        # Cancel all active tasks
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()

    async def get_caption(self, frame1: np.ndarray, frame2: np.ndarray, endpoint_idx: int, max_retries: int = 3) -> str:
        """Generate caption comparing two frames using LLM"""
        img1_b64 = self.frame_to_base64(frame1)
        img2_b64 = self.frame_to_base64(frame2)

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

        url = self.urls[endpoint_idx]
        semaphore = self.semaphores[endpoint_idx]
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                async with semaphore:
                    async with self.get_session() as session:
                        async with session.post(url, json=payload) as response:
                            response.raise_for_status()
                            response_json = await response.json()
                            return response_json['choices'][0]['message']['content']
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_error = e
                logger.warning(f"Request failed for endpoint {url} (attempt {retry_count + 1}/{max_retries}): {str(e)}")
            except Exception as e:
                last_error = e
                logger.warning(f"Unexpected error for endpoint {url} (attempt {retry_count + 1}/{max_retries}): {str(e)}")

            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(60 * (2 ** retry_count) + random.uniform(0, 10), 300)
                logger.info(f"Waiting {wait_time:.1f} seconds before retry for endpoint {url}")
                await asyncio.sleep(wait_time)

        raise Exception(f"Failed to get caption from endpoint {url} after {max_retries} retries. Last error: {str(last_error)}")

    async def process_frame_pair(self, pair: FramePair, endpoint_idx: int) -> None:
        """Process a single frame pair using specified endpoint"""
        try:
            if pair.frames is None:
                pair.frames = load_frames(pair.video_path, [pair.first_idx, pair.second_idx])
            pair.caption = await self.get_caption(pair.frames[0], pair.frames[1], endpoint_idx)
        except Exception as e:
            pair.retries += 1
            logger.error(f"Failed to process pair from video {pair.video_path} using endpoint {self.urls[endpoint_idx]}: {str(e)}")
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

async def save_results(results: List[Dict], output_path: str):
    """Save results to file with error handling"""
    try:
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(results, f, indent=2)
        os.replace(temp_path, output_path)
        logger.info(f"Saved results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {str(e)}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

async def process_videos(
        video_paths: List[str],
        data_root: str,
        caption_generator: CaptionGenerator,
        output_dir: str,
        num_pairs: int = 64,
        max_distance: int = 48,
        max_retries: int = 20,
        batch_size: int = 100
) -> Dict:
    """Process multiple videos concurrently"""
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

    results_by_video = defaultdict(lambda: {"frame_pairs": [], "failed_pairs": []})
    retry_queue = []
    total_pairs = len(all_pairs)
    progress = ProgressTracker(total_pairs, "Processing pairs")

    async def process_pair(pair: FramePair, endpoint_idx: int):
        task = asyncio.current_task()
        caption_generator.active_tasks.add(task)
        try:
            await caption_generator.process_frame_pair(pair, endpoint_idx)
            video_result = results_by_video[pair.video_path]
            video_result["frame_pairs"].append({
                "frame_indices": [pair.first_idx, pair.second_idx],
                "caption": pair.caption
            })
            progress_msg = progress.update()
            if progress_msg:
                logger.info(f"{progress_msg} (Active tasks: {len(caption_generator.active_tasks)})")
        except Exception as e:
            if pair.retries < max_retries:
                retry_queue.append(pair)
            else:
                video_result = results_by_video[pair.video_path]
                video_result["failed_pairs"].append([pair.first_idx, pair.second_idx])
                logger.error(f"Failed to process pair after {max_retries} retries: {str(e)}")
        finally:
            caption_generator.active_tasks.remove(task)

    try:
        # Process in batches
        logger.info(f"Processing {total_pairs} pairs in batches of {batch_size}...")
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            tasks = []
            for idx, pair in enumerate(batch):
                endpoint_idx = idx % len(caption_generator.urls)
                task = asyncio.create_task(process_pair(pair, endpoint_idx))
                tasks.append(task)

            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                     timeout=caption_generator.timeout.total * 2)
            except asyncio.TimeoutError:
                logger.error(f"Batch timeout occurred. Cancelling {len(caption_generator.active_tasks)} active tasks.")
                await caption_generator.cleanup()

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

                output_path = os.path.join(output_dir, f"captions_intermediate_{i}.json")
                await save_results(intermediate_results, output_path)

        # Process retry queue
        if retry_queue:
            retry_count = 0
            while retry_queue and retry_count < max_retries:
                retry_count += 1
                current_queue = retry_queue[:]
                retry_queue = []

                wait_time = min(60 * (2 ** retry_count) + random.uniform(0, 10), 300)
                retry_progress = ProgressTracker(len(current_queue), f"Retry attempt {retry_count}")
                logger.info(f"Waiting {wait_time:.1f} seconds before retrying {len(current_queue)} pairs (attempt {retry_count})")
                await asyncio.sleep(wait_time)

                for i in range(0, len(current_queue), batch_size):
                    batch = current_queue[i:i + batch_size]
                    tasks = []
                    for idx, pair in enumerate(batch):
                        endpoint_idx = idx % len(caption_generator.urls)
                        task = asyncio.create_task(process_pair(pair, endpoint_idx))
                        tasks.append(task)

                    try:
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                             timeout=caption_generator.timeout.total * 2)
                    except asyncio.TimeoutError:
                        logger.error(f"Retry batch timeout occurred. Cancelling {len(caption_generator.active_tasks)} active tasks.")
                        await caption_generator.cleanup()

                    progress_msg = retry_progress.update(len(batch))
                    if progress_msg:
                        logger.info(progress_msg)

        # Format final results
        final_results = []
        for video_path in video_paths:
            if video_path in results_by_video:
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

    except Exception as e:
        logger.error("Error during video processing", exc_info=True)
        raise
    finally:
        await caption_generator.cleanup()

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
    parser.add_argument("--request-timeout", type=int, default=300, help="Timeout for each request in seconds")
    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of all video files, excluding specific tasks
    excluded_tasks = [
        "push_button",
        "take_lid_off_saucepan",
        "phone_on_base",
        "take_umbrella_out_of_umbrella_stand",
        "stack_wine",
        "put_rubbish_in_bin"
    ]

    all_video_files = glob.glob(os.path.join(args.data_root, "*_front.mp4"))
    video_files = [v for v in all_video_files if not any(task in v for task in excluded_tasks)]

    # Log information about excluded videos
    total_videos = len(all_video_files)
    excluded_videos = total_videos - len(video_files)
    logger.info(f"Found total of {total_videos} video files")
    logger.info(f"Excluded {excluded_videos} videos containing excluded tasks")
    logger.info(f"Processing {len(video_files)} videos")

    # Log breakdown of excluded videos by task
    for task in excluded_tasks:
        task_count = sum(1 for v in all_video_files if task in v)
        if task_count > 0:
            logger.info(f"- Excluded {task_count} videos containing '{task}'")

    # Create endpoints configuration
    endpoints = [
        {"host": host, "port": port}
        for host, port in zip(args.llm_hosts, args.llm_ports)
    ]

    caption_generator = CaptionGenerator(
        model=args.llm_model,
        endpoints=endpoints,
        max_concurrent=args.max_concurrent,
        timeout=args.request_timeout
    )

    try:
        # Process all videos
        results = await process_videos(
            video_files,
            args.data_root,
            caption_generator,
            args.output_dir,
            args.num_pairs,
            args.max_distance,
            batch_size=args.batch_size
        )

        # Save final results
        final_output = os.path.join(args.output_dir, "captions_final.json")
        await save_results(results, final_output)

    except GracefulExit:
        logger.info("Received termination signal, cleaning up...")
        await caption_generator.cleanup()
        sys.exit(0)
    except Exception as e:
        logger.error("Error during processing", exc_info=True)
        raise
    finally:
        await caption_generator.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, GracefulExit):
        pass
"""
Script to presample frames from videos and generate captions using API endpoints with rate limiting.
"""
import glob
import os
import json
import random
import argparse
import logging
import queue
import threading
import time
from typing import List, Dict, Tuple, Optional
import requests
import numpy as np
from decord import VideoReader
from tqdm import tqdm
from PIL import Image
import io
import base64
from dataclasses import dataclass
from collections import defaultdict
import datetime

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
        if self.current - self.last_update_count < self.update_interval:
            return None
        self.last_update_count = self.current
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
class CaptionTask:
    video_path: str
    first_idx: int
    second_idx: int
    frames: List[np.ndarray] = None
    caption: str = None
    retries: int = 0
    last_attempt: float = 0
    endpoint_idx: int = 0

class CaptionGenerator:
    def __init__(self, model: str, endpoints: List[Dict], rpm: int = 5):
        """
        Initialize with multiple endpoints
        endpoints: List of dicts with host and port for each endpoint
        rpm: Requests per minute per endpoint
        """
        self.model = model
        self.urls = [f"http://{ep['host']}:{ep['port']}/v1/chat/completions" for ep in endpoints]
        self.rpm = rpm
        self.min_interval = 60.0 / rpm  # Minimum time between requests
        self.task_queue = queue.Queue()
        self.results_by_video = defaultdict(lambda: {"frame_pairs": [], "failed_pairs": []})
        self.endpoint_last_request = [0.0 for _ in self.urls]
        logger.info(f"Initialized {len(endpoints)} endpoints with {rpm} RPM limit")
        for i, url in enumerate(self.urls):
            logger.info(f"Endpoint {i}: {url}")

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

    def get_next_endpoint(self) -> int:
        """Get the endpoint with the longest idle time"""
        current_time = time.time()
        idle_times = [current_time - last_req for last_req in self.endpoint_last_request]
        return idle_times.index(max(idle_times))

    def process_task(self, task: CaptionTask) -> bool:
        """Process a single caption task. Returns True if successful."""
        if task.frames is None:
            try:
                task.frames = load_frames(task.video_path, [task.first_idx, task.second_idx])
            except Exception as e:
                logger.error(f"Failed to load frames from {task.video_path}: {e}")
                return False

        try:
            # Convert frames to base64
            img1_b64 = self.frame_to_base64(task.frames[0])
            img2_b64 = self.frame_to_base64(task.frames[1])

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

            # Select endpoint and enforce rate limit
            endpoint_idx = task.endpoint_idx
            current_time = time.time()
            time_since_last = current_time - self.endpoint_last_request[endpoint_idx]
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)

            # Send request
            url = self.urls[endpoint_idx]
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            self.endpoint_last_request[endpoint_idx] = time.time()

            # Process response
            response_json = response.json()
            task.caption = response_json['choices'][0]['message']['content']

            # Store result
            video_result = self.results_by_video[task.video_path]
            video_result["frame_pairs"].append({
                "frame_indices": [task.first_idx, task.second_idx],
                "caption": task.caption
            })

            logger.info(f"Successfully processed task from {task.video_path} using endpoint {endpoint_idx}")
            return True

        except Exception as e:
            task.retries += 1
            wait_time = min(60 * (2 ** task.retries) + random.uniform(0, 10), 300)
            logger.warning(f"Failed to process task (attempt {task.retries}): {str(e)}")
            logger.info(f"Will retry in {wait_time:.1f} seconds")
            task.last_attempt = time.time() + wait_time
            task.endpoint_idx = (task.endpoint_idx + 1) % len(self.urls)  # Try next endpoint
            return False

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

def save_results(results: List[Dict], output_path: str):
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

def process_videos(
        video_paths: List[str],
        data_root: str,
        caption_generator: CaptionGenerator,
        output_dir: str,
        num_pairs: int = 64,
        max_distance: int = 48,
        max_retries: int = 20
) -> Dict:
    """Process videos using queue-based approach"""
    # Create tasks for all videos
    logger.info("Creating tasks from videos...")
    total_tasks = 0
    for video_path in tqdm(video_paths):
        try:
            pairs = sample_frame_pairs(video_path, num_pairs, max_distance)
            for first, second in pairs:
                task = CaptionTask(video_path=video_path, first_idx=first, second_idx=second)
                caption_generator.task_queue.put(task)
                total_tasks += 1
        except Exception as e:
            logger.error(f"Error sampling frames from {video_path}: {str(e)}")
            continue

    # Process tasks
    logger.info(f"Processing {total_tasks} tasks...")
    progress = ProgressTracker(total_tasks, "Processing")
    completed_tasks = 0
    retry_tasks = []

    while completed_tasks < total_tasks:
        # Process main queue
        while not caption_generator.task_queue.empty():
            task = caption_generator.task_queue.get()
            current_time = time.time()

            # Check if we need to wait before retrying
            if task.last_attempt > current_time:
                retry_tasks.append(task)
                continue

            if caption_generator.process_task(task):
                completed_tasks += 1
                progress_msg = progress.update()
                if progress_msg:
                    logger.info(progress_msg)
            elif task.retries < max_retries:
                retry_tasks.append(task)
            else:
                logger.error(f"Task failed after {max_retries} retries")
                video_result = caption_generator.results_by_video[task.video_path]
                video_result["failed_pairs"].append([task.first_idx, task.second_idx])
                completed_tasks += 1

        # Move retry tasks back to main queue
        for task in retry_tasks:
            if time.time() >= task.last_attempt:
                caption_generator.task_queue.put(task)
        retry_tasks = []

        # Save intermediate results every 1000 tasks
        if completed_tasks % 1000 == 0:
            intermediate_results = []
            for video_path in video_paths:
                if video_path in caption_generator.results_by_video:
                    rel_path = os.path.relpath(video_path, data_root)
                    video_name = os.path.basename(video_path)
                    result = caption_generator.results_by_video[video_path]
                    intermediate_results.append({
                        "video_path": rel_path,
                        "video_name": video_name,
                        "frame_pairs": result["frame_pairs"],
                        "failed_pairs": result["failed_pairs"]
                    })

            output_path = os.path.join(output_dir, f"captions_intermediate_{completed_tasks}.json")
            save_results(intermediate_results, output_path)

        time.sleep(1)  # Prevent busy waiting

    # Format final results
    final_results = []
    for video_path in video_paths:
        if video_path in caption_generator.results_by_video:
            rel_path = os.path.relpath(video_path, data_root)
            video_name = os.path.basename(video_path)
            result = caption_generator.results_by_video[video_path]
            final_results.append({
                "video_path": rel_path,
                "video_name": video_name,
                "frame_pairs": result["frame_pairs"],
                "failed_pairs": result["failed_pairs"]
            })

    return final_results

def main():
    parser = argparse.ArgumentParser(description="Presample frames and generate captions")
    parser.add_argument("--data-root", required=True, help="Root directory containing videos")
    parser.add_argument("--output-dir", required=True, help="Output directory for caption files")
    parser.add_argument("--llm-model", required=True, help="LLM model name")
    parser.add_argument("--llm-hosts", nargs="+", default=["0.0.0.0", "0.0.0.0"], help="LLM API hosts")
    parser.add_argument("--llm-ports", nargs="+", type=int, default=[23333, 23334], help="LLM API ports")
    parser.add_argument("--num-pairs", type=int, default=64, help="Number of frame pairs to sample per video")
    parser.add_argument("--max-distance", type=int, default=48, help="Maximum frame distance in pairs")
    parser.add_argument("--rpm", type=int, default=300, help="Requests per minute per endpoint")
    args = parser.parse_args()

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
        rpm=args.rpm
    )

    try:
        # Process all videos
        results = process_videos(
            video_files,
            args.data_root,
            caption_generator,
            args.output_dir,
            args.num_pairs,
            args.max_distance
        )

        # Save final results
        final_output = os.path.join(args.output_dir, "captions_final.json")
        save_results(results, final_output)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, saving current progress...")
        # Save whatever results we have
        final_output = os.path.join(args.output_dir, "captions_interrupted.json")
        save_results(results, final_output)
    except Exception as e:
        logger.error("Error during processing", exc_info=True)
        raise

if __name__ == "__main__":
    main()
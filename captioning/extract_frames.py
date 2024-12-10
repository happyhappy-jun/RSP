import os
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu
from tqdm import tqdm
import argparse
import multiprocessing as mp
import json
import pickle
from caption2.core.frame_sampler import UniformFrameSampler
from caption2.utils.image_utils import resize_image

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def resize_image(image, max_size=512):
    """Resize image to fit within max_size while maintaining aspect ratio"""
    height, width = image.shape[:2]
    if height > width:
        if height > max_size:
            ratio = max_size / height
            new_height = max_size
            new_width = int(width * ratio)
        else:
            return image
    else:
        if width > max_size:
            ratio = max_size / width
            new_width = max_size
            new_height = int(height * ratio)
        else:
            return image
            
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def get_video_files(data_root):
    """Get all video files from data_root/class_label/*.mp4 structure"""
    videos = []
    # List all directories (class labels) in data_root
    for class_label in sorted(os.listdir(data_root)):
        class_dir = os.path.join(data_root, class_label)
        if not os.path.isdir(class_dir):
            continue
            
        # List all mp4 files in the class directory
        for video_file in os.listdir(class_dir):
            if video_file.endswith('.mp4'):
                rel_path = os.path.join(class_label, video_file)
                videos.append((class_label, rel_path))
    return videos

def extract_frames(args):
    """Extract frames from a video file using configured sampler"""
    video_path, output_dir, video_idx, seed = args
    try:
        # Create frame sampler
        sampler = UniformFrameSampler(seed + video_idx)
        
        # Get label from video path
        label = os.path.dirname(os.path.relpath(video_path, os.path.dirname(os.path.dirname(video_path))))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Directory for this video
        video_dir = os.path.join(output_dir, label, video_name)
        os.makedirs(video_dir, exist_ok=True)
        
        frames_info = {
            'video_path': video_path,
            'label': label,
            'video_idx': video_idx,
            'video_name': video_name,
            'frames': []
        }
        
        # Sample frames using configured strategy
        frame_indices = sampler.sample_frames(video_path)
        
        # Extract and save frames
        for frame_idx, video_frame_idx in enumerate(frame_indices):
            # Extract and resize frame
            frame = vr[video_frame_idx].asnumpy()
            frame = resize_image(frame)
            
            # Save frame
            frame_path = os.path.join(video_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Store frame information
            frame_info = {
                'frame_idx': frame_idx,
                'video_frame_idx': video_frame_idx,
                'path': os.path.relpath(frame_path, output_dir),
                'shape': frame.shape[:2]  # Store height, width
            }
            frames_info['frames'].append(frame_info)
        
        return video_path, True, frames_info, None
    except Exception as e:
        return video_path, False, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, 
                       help='Root directory containing class_label folders with videos')
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--workers', type=int, default=mp.cpu_count())
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set global seed
    set_seed(args.seed)
    
    # Get video files
    print("Collecting video files...")
    samples = get_video_files(args.data_root)
    
    print(f"Found {len(samples)} videos in {len(set(label for label, _ in samples))} classes")
    
    # Prepare output directory
    frames_dir = os.path.join(args.output_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Prepare arguments for processing
    process_args = [
        (
            os.path.join(args.data_root, video_path),
            frames_dir,
            idx,
            args.seed
        )
        for idx, (_, video_path) in enumerate(samples)
    ]
    
    # Process videos
    successful = []
    failed = []
    video_frame_info = []
    
    print(f"Processing {len(samples)} videos using {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        with tqdm(total=len(samples), desc="Extracting frames") as pbar:
            for video_path, success, frames_info, error in pool.imap_unordered(extract_frames, process_args):
                if success:
                    successful.append(video_path)
                    video_frame_info.append(frames_info)
                else:
                    failed.append((video_path, error))
                pbar.update()
    
    # Save all results
    output_dict = {
        'metadata': {
            'seed': args.seed,
            'total_videos': len(samples),
            'successful': len(successful),
            'failed': len(failed),
            'preprocessing_args': vars(args)
        },
        'videos': video_frame_info
    }
    
    # Save as JSON for human readability
    with open(os.path.join(args.output_root, "frame_info.json"), 'w') as f:
        json.dump(output_dict, f, indent=2)
        
    # Save as pickle for efficient loading during training
    with open(os.path.join(args.output_root, "frame_info.pkl"), 'wb') as f:
        pickle.dump(output_dict, f)
    
    if failed:
        with open(os.path.join(args.output_root, "failed_videos.txt"), 'w') as f:
            for video_path, error in failed:
                f.write(f"{video_path}\t{error}\n")
    
    print("\nFrame Extraction Complete!")
    print(f"Successfully processed: {len(successful)} videos")
    print(f"Failed to process: {len(failed)} videos")
    print(f"\nResults saved to:")
    print(f"- {os.path.join(args.output_root, 'frame_info.json')}")
    print(f"- {os.path.join(args.output_root, 'frame_info.pkl')}")

if __name__ == "__main__":
    main()

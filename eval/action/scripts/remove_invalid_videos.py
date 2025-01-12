import os
import logging
import pandas as pd
from decord import VideoReader
from tqdm import tqdm
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_video(video_path):
    """Validate video file using Decord"""
    try:
        vr = VideoReader(video_path)
        if len(vr) == 0:
            return False, "Video has no frames"
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Remove invalid videos from Kinetics-400 dataset')
    parser.add_argument('--data-root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--dry-run', action='store_true', help='Only print actions without executing them')
    args = parser.parse_args()

    videos_dir = os.path.join(args.data_root, 'videos')
    annotations_file = os.path.join(args.data_root, "test.csv")

    if not os.path.exists(annotations_file):
        logger.error(f"Annotations file not found: {annotations_file}")
        return

    # Load annotations
    logger.info("Loading annotations...")
    annotations = pd.read_csv(annotations_file)
    
    invalid_videos = []
    removed_count = 0
    total_size = 0
    
    logger.info("Validating videos...")
    for idx, row in tqdm(annotations.iterrows(), total=len(annotations)):
        video_id = row['youtube_id']
        start_time = row['time_start']
        end_time = row['time_end']
        video_path = os.path.join(videos_dir, f"{video_id}_{start_time:0>6}_{end_time:0>6}.mp4")
        
        if os.path.exists(video_path):
            total_size += os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
            is_valid, error = validate_video(video_path)
            if not is_valid:
                invalid_videos.append((video_path, error))
                if not args.dry_run:
                    try:
                        os.remove(video_path)
                        removed_count += 1
                        logger.info(f"Removed invalid video: {video_path} (Error: {error})")
                    except Exception as e:
                        logger.error(f"Failed to remove {video_path}: {str(e)}")

    # Summary
    logger.info("\nValidation Summary:")
    logger.info(f"Total videos processed: {len(annotations)}")
    logger.info(f"Total dataset size: {total_size:.2f} MB")
    logger.info(f"Invalid videos found: {len(invalid_videos)}")
    if not args.dry_run:
        logger.info(f"Videos removed: {removed_count}")
    else:
        logger.info("Dry run - no files were actually removed")

    # Save list of invalid videos
    if invalid_videos:
        output_file = os.path.join(args.data_root, "invalid_videos.txt")
        with open(output_file, 'w') as f:
            for video_path, error in invalid_videos:
                f.write(f"{video_path}\t{error}\n")
        logger.info(f"\nList of invalid videos saved to: {output_file}")

if __name__ == "__main__":
    main()

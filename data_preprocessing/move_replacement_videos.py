import argparse
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
import requests
import tempfile
import atexit
import os
from typing import Literal

# Kinetics-400 annotation URLs
ANNOTATION_URLS = {
    'train': 'https://s3.amazonaws.com/kinetics/400/annotations/train.csv',
    'val': 'https://s3.amazonaws.com/kinetics/400/annotations/val.csv',
    'test': 'https://s3.amazonaws.com/kinetics/400/annotations/test.csv'
}

def download_csv(url, temp_dir):
    """Download CSV file to temporary directory"""
    response = requests.get(url)
    response.raise_for_status()
    
    temp_path = Path(temp_dir) / "annotations.csv"
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    return temp_path

def load_annotations(csv_path):
    """Load annotations CSV and create video_id to label mapping"""
    df = pd.read_csv(csv_path)
    # Remove quotes from labels if present
    df['label'] = df['label'].str.replace('"', '')
    return dict(zip(df['youtube_id'], df['label']))

def cleanup_temp_dir(temp_dir):
    """Clean up temporary directory"""
    shutil.rmtree(temp_dir)

def process_split(split: str, replacement_dir: Path, output_base: Path, temp_dir: str):
    """Process videos for a specific split"""
    print(f"\nProcessing {split} split...")
    
    # Download CSV and load video ID to label mapping
    csv_path = download_csv(ANNOTATION_URLS[split], temp_dir)
    id_to_label = load_annotations(csv_path)
    
    # Create split-specific output directory

    # Process each replacement video
    for video_path in tqdm(list(replacement_dir.glob('*.mp4'))):
        # Extract youtube ID (first 11 characters of filename)
        video_id = video_path.stem[:11]
        
        if video_id in id_to_label:
            label = id_to_label[video_id]
            
            # Create class directory if it doesn't exist

            class_dir = output_base / label
            if not class_dir.exists():
                raise ValueError(f"Output directory {class_dir} does not exist")

            # Move video to appropriate class directory
            dest_path = class_dir / video_path.name
            shutil.copy2(video_path, dest_path)
            print(f"Moved {video_path.name} to {split}/{label}/")
        else:
            print(f"Warning: No label found for video {video_path.name} in {split} split")

def main():
    parser = argparse.ArgumentParser(description='Move replacement videos to class directories')
    parser.add_argument('--replacement_dir', type=str, required=True, 
                       help='Path to replacement videos directory')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base directory for class-organized videos')
    args = parser.parse_args()

    # Create temporary directory and register cleanup
    temp_dir = tempfile.mkdtemp()
    atexit.register(cleanup_temp_dir, temp_dir)
    
    # Setup paths
    replacement_dir = Path(args.replacement_dir)
    output_base = Path(args.output_base)
    
    # Process all splits
    for split in ['train', 'val', 'test']:
        process_split(split, replacement_dir, output_base, temp_dir)

if __name__ == '__main__':
    main()

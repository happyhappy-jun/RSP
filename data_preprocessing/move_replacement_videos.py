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

def get_combined_annotations(temp_dir: str) -> dict:
    """Download and combine all annotation files"""
    combined_mapping = {}
    
    for split, url in ANNOTATION_URLS.items():
        print(f"Downloading {split} annotations...")
        csv_path = download_csv(url, temp_dir)
        split_mapping = load_annotations(csv_path)
        combined_mapping.update(split_mapping)
    
    return combined_mapping

def process_videos(replacement_dir: Path, output_base: Path, id_to_label: dict):
    """Process all videos using combined annotations"""
    print("\nProcessing videos...")
    
    for video_path in tqdm(list(replacement_dir.glob('*.mp4'))):
        # Extract youtube ID (first 11 characters of filename)
        video_id = video_path.stem[:11]
        
        if video_id in id_to_label:
            label = id_to_label[video_id]
            if " " in label:
                label = label.replace(" ", "_").replace("(" , "").replace(")", "")
            
            class_dir = output_base / label
            if not class_dir.exists():
                raise ValueError(f"Output directory {class_dir} does not exist")

            dest_path = class_dir / video_path.name
            try:
                shutil.copy(video_path, dest_path)
            except PermissionError:
                print(f"Error: Permission denied when copying {video_path.name}")
                print(f"Please ensure you have write permissions for: {dest_path.parent}")
            except OSError as e:
                print(f"Error copying {video_path.name}: {e}")
        else:
            print(f"Warning: No label found for video {video_path.name}")

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
    
    # Get combined annotations from all splits
    id_to_label = get_combined_annotations(temp_dir)
    
    # Process all videos using combined annotations
    process_videos(replacement_dir, output_base, id_to_label)

if __name__ == '__main__':
    main()

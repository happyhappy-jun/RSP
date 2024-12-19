import argparse
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm

def load_annotations(csv_path):
    """Load annotations CSV and create video_id to label mapping"""
    df = pd.read_csv(csv_path)
    # Remove quotes from labels if present
    df['label'] = df['label'].str.replace('"', '')
    return dict(zip(df['youtube_id'], df['label']))

def main():
    parser = argparse.ArgumentParser(description='Move replacement videos to class directories')
    parser.add_argument('--csv', type=str, required=True, help='Path to annotations CSV')
    parser.add_argument('--replacement_dir', type=str, required=True, 
                       help='Path to replacement videos directory')
    parser.add_argument('--output_base', type=str, required=True,
                       help='Base directory for class-organized videos')
    args = parser.parse_args()

    # Load video ID to label mapping
    id_to_label = load_annotations(args.csv)
    
    # Setup paths
    replacement_dir = Path(args.replacement_dir)
    output_base = Path(args.output_base)
    
    # Process each replacement video
    for video_path in tqdm(list(replacement_dir.glob('*.mp4'))):
        # Extract youtube ID (first 11 characters of filename)
        video_id = video_path.stem[:11]
        
        if video_id in id_to_label:
            label = id_to_label[video_id]
            
            # Create class directory if it doesn't exist
            class_dir = output_base / label
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Move video to appropriate class directory
            dest_path = class_dir / video_path.name
            shutil.copy2(video_path, dest_path)
            print(f"Moved {video_path.name} to {label}/")
        else:
            print(f"Warning: No label found for video {video_path.name}")

if __name__ == '__main__':
    main()
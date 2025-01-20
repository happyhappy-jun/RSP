import argparse
import json
from pathlib import Path
import math

def split_frame_info(input_file: Path, output_dir: Path):
    """Split frame_info_additional.json into two balanced parts"""
    
    # Load original file
    print(f"\nLoading frame info from: {input_file}")
    with open(input_file) as f:
        frame_info = json.load(f)
    
    total_videos = len(frame_info['videos'])
    split_size = math.ceil(total_videos / 3)
    
    print(f"Total videos: {total_videos}")
    print(f"Split size: {split_size}")
    
    # Create three parts
    splits = []
    for i in range(3):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, total_videos)
        part = {
            'config': frame_info['config'],
            'videos': frame_info['videos'][start_idx:end_idx]
        }
        splits.append(part)
    
    # Save split files
    print("\nSaving splits:")
    for i, split in enumerate(splits, 1):
        output_file = output_dir / f"frame_info_additional_part{i}.json"
        print(f"Part {i} ({len(split['videos'])} videos): {output_file}")
        with open(output_file, 'w') as f:
            json.dump(split, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Split frame_info_additional.json into two parts')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to frame_info_additional.json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for split files')
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_frame_info(input_file, output_dir)
    print("\nSplit complete!")

if __name__ == "__main__":
    main()

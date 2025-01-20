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
    split_point = math.ceil(total_videos / 2)
    
    print(f"Total videos: {total_videos}")
    print(f"Splitting at: {split_point}")
    
    # Create first half
    first_half = {
        'config': frame_info['config'],
        'videos': frame_info['videos'][:split_point]
    }
    
    # Create second half
    second_half = {
        'config': frame_info['config'],
        'videos': frame_info['videos'][split_point:]
    }
    
    # Save split files
    first_output = output_dir / "frame_info_additional_part1.json"
    second_output = output_dir / "frame_info_additional_part2.json"
    
    print(f"\nSaving splits:")
    print(f"Part 1 ({len(first_half['videos'])} videos): {first_output}")
    with open(first_output, 'w') as f:
        json.dump(first_half, f, indent=2)
        
    print(f"Part 2 ({len(second_half['videos'])} videos): {second_output}")
    with open(second_output, 'w') as f:
        json.dump(second_half, f, indent=2)

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

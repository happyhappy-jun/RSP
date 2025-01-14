import argparse
import json
from pathlib import Path
import os
import asyncio
from caption2.core.frame_extractor import extract_frames
from caption2.core.config import Config

async def main():
    parser = argparse.ArgumentParser(description='Step 1: Extract frames from videos')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for extracted frames')
    parser.add_argument('--sampler', type=str, default='paired',
                       choices=['uniform', 'paired'],
                       help='Frame sampling strategy')
    parser.add_argument('--config_path', type=str,
                       help='Path to configuration YAML file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Get video paths
    video_paths = []
    for root, _, files in os.walk(args.data_root):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
                
    print(f"Found {len(video_paths)} videos")
    
    # Limit to first 100 videos for testing
    print(f"Processing first {len(video_paths)} videos for test run")

    # Extract frames
    config = Config(args.config_path, data_root=args.data_root) if args.config_path else Config(data_root=args.data_root)
    frame_info = await extract_frames(
        video_paths,
        Path(args.output_dir),
        args.sampler,
        config,
        seed=args.seed,
        max_workers=os.cpu_count()
    )

    # Save frame info
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_info_path = output_dir / "frame_info_additional.json"
    with open(frame_info_path, 'w') as f:
        json.dump(frame_info, f, indent=2)

    print(f"\nExtracted frames saved to: {args.output_dir}")
    print(f"Frame info saved to: {frame_info_path}")

if __name__ == "__main__":
    asyncio.run(main())

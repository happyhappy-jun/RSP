import json
import random
import argparse
from pathlib import Path

def create_subset_dataset(frame_info_path, embeddings_path, output_dir, sample_ratio=0.2):
    """
    Create a subset of the dataset by sampling pairs and their corresponding embeddings.
    
    Args:
        frame_info_path: Path to original frame_info.json
        embeddings_path: Path to original embeddings.jsonl
        output_dir: Directory to save new files
        sample_ratio: Ratio of data to sample (default: 0.2 for 20%)
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original frame info
    with open(frame_info_path, 'r') as f:
        frame_info = json.load(f)
    
    # Get all video pairs
    all_pairs = []
    for video in frame_info['videos']:
        all_pairs.append((video['video_idx'], video['pair_idx']))
    
    # Sample pairs
    num_samples = int(len(all_pairs) * sample_ratio)
    sampled_pairs = set(random.sample(all_pairs, num_samples))
    
    # Create new frame info with only sampled pairs
    new_frame_info = {'videos': []}
    for video in frame_info['videos']:
        if (video['video_idx'], video['pair_idx']) in sampled_pairs:
            new_frame_info['videos'].append(video)
    
    # Save new frame info
    output_frame_info = output_dir / f"frame_info_{int(sample_ratio*100)}.json"
    with open(output_frame_info, 'w') as f:
        json.dump(new_frame_info, f, indent=2)
    
    # Process embeddings
    output_embeddings = output_dir / f"embeddings_{int(sample_ratio*100)}.jsonl"
    with open(embeddings_path, 'r') as f_in, open(output_embeddings, 'w') as f_out:
        for line in f_in:
            record = json.loads(line)
            # Parse video_idx and pair_idx from custom_id (format: video_X_pair_Y)
            parts = record[-1]['custom_id'].split('_')
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            
            if (video_idx, pair_idx) in sampled_pairs:
                f_out.write(line)
    
    print(f"Created subset dataset with {len(new_frame_info['videos'])} pairs")
    print(f"Saved to:\n{output_frame_info}\n{output_embeddings}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of the dataset")
    parser.add_argument("--frame-info", required=True, help="Path to original frame_info.json")
    parser.add_argument("--embeddings", required=True, help="Path to original embeddings.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.2, help="Sampling ratio (default: 0.2)")
    
    args = parser.parse_args()
    create_subset_dataset(args.frame_info, args.embeddings, args.output_dir, args.ratio)

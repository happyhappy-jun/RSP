import json
from typing import Dict, List, Tuple
import copy
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_jsonl(filename: str) -> List[dict]:
    """Load data from JSONL file, skipping empty lines"""
    data = []
    with open(filename, 'r') as f:
        for line in tqdm(f, desc="Loading JSONL"):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    return data

def save_jsonl(data: List[dict], filename: str):
    """Save data as JSONL file, ensuring no empty lines"""
    with open(filename, 'w') as f:
        for item in tqdm(data, desc="Saving JSONL"):
            if item:  # Only write non-empty items
                try:
                    line = json.dumps(item).strip()
                    if line:  # Only write non-empty lines
                        f.write(line + '\n')
                except Exception as e:
                    logger.warning(f"Failed to write item: {e}")

def verify_pairs(data: List[dict]) -> Dict[str, List[int]]:
    """Verify pair indices for each video and return statistics"""
    video_pairs = defaultdict(list)
    for item in data:
        try:
            custom_id = item.get('custom_id', '')
            if '_pair_' in custom_id:
                video_id = custom_id.split('_pair_')[0]
                pair_idx = int(custom_id.split('_pair_')[1])
                video_pairs[video_id].append(pair_idx)
        except Exception as e:
            logger.warning(f"Error verifying pair: {e}")
    return video_pairs

def print_pair_statistics(video_pairs: Dict[str, List[int]], stage: str = ""):
    """Print statistics about pair indices"""
    logger.info(f"\n=== Pair Statistics {stage} ===")
    for video_id, pairs in list(video_pairs.items())[:5]:  # Show first 5 videos
        pairs.sort()
        logger.info(f"Video {video_id}: {pairs}")
    
    # Count frequency of each pair index
    pair_freq = defaultdict(int)
    for pairs in video_pairs.values():
        for idx in pairs:
            pair_freq[idx] += 1
    
    logger.info("\nPair index frequencies:")
    for idx in sorted(pair_freq.keys()):
        logger.info(f"Pair {idx}: {pair_freq[idx]} occurrences")

def print_random_captions(data: List[dict], samples_per_pair: int = 3):
    """Print random caption samples for each pair index"""
    # Group captions by pair index
    pair_captions = defaultdict(list)
    
    for item in data:
        try:
            custom_id = item.get('custom_id', '')
            if '_pair_' in custom_id:
                pair_idx = int(custom_id.split('_pair_')[1])
                caption = item.get('response', {}).get('body', {}).get('caption', '')
                if caption:
                    pair_captions[pair_idx].append({
                        'custom_id': custom_id,
                        'caption': caption
                    })
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
    
    # Print random samples for each pair
    logger.info("\n=== Random Caption Samples ===")
    for pair_idx in sorted(pair_captions.keys()):
        captions = pair_captions[pair_idx]
        samples = random.sample(captions, min(samples_per_pair, len(captions)))
        
        logger.info(f"\nPair Index {pair_idx} (Total captions: {len(captions)})")
        for i, sample in enumerate(samples, 1):
            logger.info(f"Sample {i}:")
            logger.info(f"ID: {sample['custom_id']}")
            logger.info(f"Caption: {sample['caption']}\n")

def process_captions(caption_data: List[dict]) -> List[dict]:
    """Process captions to fix pair indices based on creation time"""
    processed_data = []
    video_pairs = defaultdict(list)
    
    # Print initial statistics
    initial_pairs = verify_pairs(caption_data)
    print_pair_statistics(initial_pairs, "BEFORE PROCESSING")
    
    # Group pairs by video_id with creation time
    skipped = 0
    for item in tqdm(caption_data, desc="Grouping pairs"):
        try:
            custom_id = item.get('custom_id', '')
            if '_pair_' in custom_id:
                video_id = custom_id.split('_pair_')[0]
                pair_idx = int(custom_id.split('_pair_')[1])
                created_time = item['response']['body']['created']
                video_pairs[video_id].append((pair_idx, created_time, item))
        except Exception as e:
            logger.warning(f"Skipping item due to error: {e}")
            skipped += 1
    
    if skipped:
        logger.warning(f"Skipped {skipped} items due to errors")
    
    # Process each video
    for video_id, pairs in tqdm(video_pairs.items(), desc="Processing videos"):
        try:
            # Sort pairs by creation time
            pairs.sort(key=lambda x: x[1])
            
            # Process each pair
            for i, (pair_idx, created_time, item) in enumerate(pairs):
                new_item = copy.deepcopy(item)
                
                if i < 2:  # First two pairs (earlier creation time) stay as 0,1
                    new_item['custom_id'] = f"{video_id}_pair_{i}"
                else:  # Later pairs
                    if pair_idx <= 1:  # Later 0,1 become 2,3
                        new_idx = pair_idx + 2
                    else:  # Later 2,3,4,5 become 4,5,6,7
                        new_idx = pair_idx + 2
                    new_item['custom_id'] = f"{video_id}_pair_{new_idx}"
                
                processed_data.append(new_item)
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
    
    # Print final statistics
    final_pairs = verify_pairs(processed_data)
    print_pair_statistics(final_pairs, "AFTER PROCESSING")
    
    return processed_data

def main():
    parser = argparse.ArgumentParser(description='Fix caption pair indices based on creation time')
    parser.add_argument('input_file', type=str, help='Input JSONL file path')
    parser.add_argument('output_file', type=str, help='Output JSONL file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--samples', type=int, default=3, help='Number of random caption samples to print per pair')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.seed is not None:
        random.seed(args.seed)
    
    # Ensure input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Loading data from {args.input_file}")
        caption_data = load_jsonl(args.input_file)
        
        logger.info("Processing captions...")
        processed_data = process_captions(caption_data)
        
        logger.info("Printing random caption samples...")
        print_random_captions(processed_data, args.samples)
        
        logger.info(f"Saving processed data to {args.output_file}")
        save_jsonl(processed_data, args.output_file)
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
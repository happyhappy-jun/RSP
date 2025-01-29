import json
import glob
import os
import re
import argparse

from tqdm import tqdm


def extract_video_number(custom_id):
    """Extract the video number from custom_id string"""
    match = re.search(r'video_(\d+)', custom_id)
    return int(match.group(1)) if match else 0

def combine_caption_files(input_dir, output_file):
    # Get all json files matching the pattern
    json_files = glob.glob(os.path.join(input_dir, "caption_results_*.json"))
    
    if not json_files:
        print("No caption result files found!")
        return
    
    # Combined list to store all entries
    combined_data = []
    
    # Read and combine all JSON files
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    combined_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
                continue
    
    # Sort the combined data by video number
    combined_data.sort(key=lambda x: extract_video_number(x['custom_id']))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the combined and sorted data
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)
    
    print(f"Combined {len(json_files)} files into {output_file}")
    print(f"Total entries: {len(combined_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine and sort caption result JSON files')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing caption_results_*.json files')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the combined JSON file')
    
    args = parser.parse_args()
    combine_caption_files(args.input_dir, args.output_file)

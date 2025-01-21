import json
import glob
import os
from tqdm import tqdm
import re

# Input and output paths
input_dir = "/data/RSP/caption_batch_results"
output_file = "/data/RSP/captions_6_pair.jsonl"

def extract_video_pair_ids(custom_id):
    """Extract video ID and pair number from custom_id"""
    match = re.match(r'video_(\d+)_pair_(\d+)', custom_id)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float('inf'), float('inf'))

# Get all output files
output_files = glob.glob(os.path.join(input_dir, "*_output.jsonl"))
print(f"Found {len(output_files)} output files")

# Read and combine all data
all_data = []
print("\nReading files:")
for file_path in tqdm(output_files):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract custom_id directly from top level
                    custom_id = data.get('custom_id')
                    
                    if custom_id:
                        # Store sort key and original data
                        data['_sort_key'] = extract_video_pair_ids(custom_id)
                        all_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

print(f"\nTotal entries read: {len(all_data)}")

if len(all_data) == 0:
    print("No data was found. Exiting.")
    exit()

print("Sorting data...")
all_data.sort(key=lambda x: x['_sort_key'])

print("Writing sorted data...")
with open(output_file, 'w', encoding='utf-8') as f:
    for data in tqdm(all_data):
        # Remove the temporary sort key
        del data['_sort_key']
        f.write(json.dumps(data) + '\n')

print(f"\nDone! Combined and sorted data written to: {output_file}")

# Print statistics
video_ids = set()
pair_counts = {}

for data in all_data:
    custom_id = data.get('custom_id')
    if custom_id:
        match = re.match(r'video_(\d+)_pair_(\d+)', custom_id)
        if match:
            video_id = int(match.group(1))
            video_ids.add(video_id)
            pair_counts[video_id] = pair_counts.get(video_id, 0) + 1

print("\nStatistics:")
print(f"Total unique videos: {len(video_ids)}")
if video_ids:
    print(f"Video ID range: {min(video_ids)} to {max(video_ids)}")
    print("\nPair counts per video:")
    for video_id in sorted(pair_counts.keys())[:10]:
        print(f"Video {video_id}: {pair_counts[video_id]} pairs")
    if len(pair_counts) > 10:
        print("... (showing first 10 videos only)")

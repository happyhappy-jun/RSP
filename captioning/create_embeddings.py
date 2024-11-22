import json
import argparse
from pathlib import Path

MAX_REQUESTS_PER_FILE = 50000

def create_jsonl_for_embedding(json_path, output_file="embeddings.jsonl"):
    """
    Create JSONL files formatted for OpenAI's embedding API from frame analysis data.
    Files are automatically sharded if there are more than MAX_REQUESTS_PER_FILE entries.
    
    Args:
        json_path (str): Path to input JSON file containing frame analyses
        output_file (str): Base path for output JSONL file(s)
    """
    # Read the input JSON file
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    # Sort results by video_idx and pair_idx for consistency
    results = sorted(caption_data['results'], 
                    key=lambda x: (x['video_idx'], x['pair_idx']))
    
    # Calculate number of shards needed
    num_results = len(results)
    num_shards = max(1, (num_results + MAX_REQUESTS_PER_FILE - 1) // MAX_REQUESTS_PER_FILE)
    
    # Split the base filename
    output_path = Path(output_file)
    base_name = output_path.stem
    extension = output_path.suffix
    
    total_written = 0
    
    # Write the sharded JSONL files
    for shard in range(num_shards):
        start_idx = shard * MAX_REQUESTS_PER_FILE
        end_idx = min((shard + 1) * MAX_REQUESTS_PER_FILE, num_results)
        
        if num_shards > 1:
            shard_file = output_path.parent / f"{base_name}.{shard:03d}{extension}"
        else:
            shard_file = output_file
            
        with open(shard_file, 'w', encoding='utf-8') as f:
            for idx, result in enumerate(results[start_idx:end_idx], start=start_idx):
                request = {
                    "custom_id": f"request-{idx+1}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "input": result['analysis'].strip(),
                        "model": "text-embedding-3-small",
                        "encoding_format": "float",
                        "max_tokens": 1000
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
                total_written += 1
    
    if num_shards > 1:
        print(f"Created {num_shards} JSONL files with {total_written} total entries")
        print(f"Files: {base_name}.000{extension} to {base_name}.{num_shards-1:03d}{extension}")
    else:
        print(f"Created JSONL file with {total_written} entries at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to input JSON file containing frame analyses')
    parser.add_argument('--output_file', type=str, default="embeddings.jsonl",
                       help='Path to output JSONL file')
    
    args = parser.parse_args()
    create_jsonl_for_embedding(args.json_path, args.output_file)

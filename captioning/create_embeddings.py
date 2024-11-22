import json
import argparse
from pathlib import Path

def create_jsonl_for_embedding(json_path, output_file="embeddings.jsonl"):
    """
    Create a JSONL file formatted for OpenAI's embedding API from frame analysis data.
    Each line will be a JSON object with 'text' field containing the analysis.
    
    Args:
        json_path (str): Path to input JSON file containing frame analyses
        output_file (str): Path to output JSONL file
    """
    # Read the input JSON file
    with open(json_path, 'r') as f:
        caption_data = json.load(f)
    
    # Sort results by video_idx and pair_idx for consistency
    results = sorted(caption_data['results'], 
                    key=lambda x: (x['video_idx'], x['pair_idx']))
    
    # Write the JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = {"text": result['analysis'].strip()}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    print(f"Created JSONL file with {len(results)} entries at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to input JSON file containing frame analyses')
    parser.add_argument('--output_file', type=str, default="embeddings.jsonl",
                       help='Path to output JSONL file')
    
    args = parser.parse_args()
    create_jsonl_for_embedding(args.json_path, args.output_file)

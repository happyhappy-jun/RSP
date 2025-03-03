#!/usr/bin/env python
import os
import json
import numpy as np
import openai
import argparse
from typing import List

def npy_to_numpy_array(file_path: str):
    """Load numpy array from file, allowing for object arrays."""
    return np.load(file_path, allow_pickle=True)

def get_all_unique_captions(data_dir: str) -> List[str]:
    """Collect unique movement captions from all 'moves' npy files in the dataset directory."""
    unique_captions = set()
    for dir_name in os.listdir(data_dir):
        subdir = os.path.join(data_dir, dir_name)
        if os.path.isdir(subdir):
            for file_name in os.listdir(subdir):
                if file_name.startswith("moves"):
                    file_path = os.path.join(subdir, file_name)
                    captions = npy_to_numpy_array(file_path)
                    for caption in captions:
                        unique_captions.add(caption)
    return list(unique_captions)

def precompute_embeddings(data_dir: str, output_json: str, model: str, openai_api_key: str):
    """Precompute embeddings for unique captions using OpenAI embedding API."""
    openai.api_key = openai_api_key
    client = openai.OpenAI()
    unique_captions = get_all_unique_captions(data_dir)
    print(f"Found {len(unique_captions)} unique captions")
    embedding_map = {}
    for idx, caption in enumerate(unique_captions):
        try:
            response = client.embeddings.create(input=caption, model=model)
            embedding = response.data[0].embedding
            embedding_map[caption] = embedding
            print(f"Processed caption {idx + 1}/{len(unique_captions)}")
        except Exception as e:
            print(f"Error processing caption '{caption}': {e}")
    with open(output_json, "w") as f:
        json.dump(embedding_map, f, indent=2)
    print(f"Saved embeddings to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing trajectory subdirectories")
    parser.add_argument("--output_json", type=str, default="embedding_map.json", help="Output JSON file for embeddings")
    parser.add_argument("--model", type=str, default="text-embedding-3-large", help="OpenAI embedding model to use")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    args = parser.parse_args()

    precompute_embeddings(args.data_dir, args.output_json, args.model, args.openai_api_key)

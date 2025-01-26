import numpy as np
import os
import json
import ujson
from itertools import islice
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
import torch
from torchvision import transforms
from typing import Dict, List, Tuple
import h5py


def process_lines(lines, is_future=True):
    """Process a batch of lines from jsonl file."""
    results = {}
    for line in lines:
        try:
            record = ujson.loads(line)
            parts = record[-1]["custom_id"].split("_")
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            embedding = record[1]["data"][0]["embedding"]
            if is_future:
                results[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)
            else:
                results[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)[:512]
        except Exception as e:
            print(f"Error processing line: {e}")
    return results


def process_captions(lines):
    """Process caption data from jsonl file."""
    captions = {}
    for line in lines:
        try:
            data = ujson.loads(line)
            parts = data["custom_id"].split("_")
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            caption = data["response"]["body"]["choices"][0]["message"]["content"]
            captions[(video_idx, pair_idx)] = caption
        except Exception as e:
            print(f"Error processing caption: {e}")
    return captions


def create_memmap_dataset(
        frame_root: str,
        frame_info_path: str,
        embeddings_path: str,
        future_embeddings_path: str,
        future_captions_path: str,
        output_dir: str,
        frame_info_additional_path: str = None,
        embeddings_additional_path: str = None,
):
    """
    Preprocess the Kinetics dataset and save to memory-mapped files including captions
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load frame info and create video pairs structure
    print("Loading frame info...")
    results = defaultdict(list)

    def process_frame_info(frame_info_path, prefix="frames", pair_idx_offset=0):
        with open(frame_info_path, "r") as f:
            frame_info = json.load(f)
            for frame in tqdm(frame_info["videos"], desc=f"Processing {prefix} frame info"):
                video_idx = frame["video_idx"]
                frame_paths = frame["frame_paths"]
                pair_idx = frame["pair_idx"] + pair_idx_offset

                processed_paths = [
                    os.path.join(frame_root, f"{prefix}/{path}")
                    for path in frame_paths
                ]

                results[video_idx].append({
                    "video_idx": video_idx,
                    "pair_idx": pair_idx,
                    "frame_cur_path": processed_paths[0],
                    "frame_fut_path": processed_paths[1],
                })

    # Process main frame info
    process_frame_info(frame_info_path)

    # Process additional frame info if provided
    if frame_info_additional_path:
        process_frame_info(frame_info_additional_path,
                           prefix="frames_additional",
                           pair_idx_offset=2)

    # Step 2: Load embeddings and captions
    print("\nLoading embeddings and captions...")

    def load_embeddings(embeddings_path, is_future=False):
        chunk_size = 10000
        n_jobs = 30

        with open(embeddings_path, "r") as f:
            total_lines = sum(1 for _ in f)

        chunks = []
        with open(embeddings_path, "r") as f:
            while True:
                chunk = list(islice(f, chunk_size))
                if not chunk:
                    break
                chunks.append(chunk)

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_lines)(chunk, is_future)
            for chunk in chunks
        )

        combined_results = {}
        for r in results:
            combined_results.update(r)
        return combined_results

    def load_captions(captions_path):
        chunk_size = 10000
        n_jobs = 30

        chunks = []
        with open(captions_path, "r") as f:
            while True:
                chunk = list(islice(f, chunk_size))
                if not chunk:
                    break
                chunks.append(chunk)

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_captions)(chunk)
            for chunk in chunks
        )

        combined_results = {}
        for r in results:
            combined_results.update(r)
        return combined_results

    embeddings = load_embeddings(embeddings_path, False)
    future_embeddings = load_embeddings(future_embeddings_path, True)
    future_captions = load_captions(future_captions_path)

    if embeddings_additional_path:
        additional_embeddings = load_embeddings(embeddings_additional_path, False)
        embeddings.update(additional_embeddings)

    # Step 3: Create valid pairs structure
    print("\nCreating valid pairs structure...")
    video_pairs = defaultdict(list)
    for video_idx, frame_data in results.items():
        for pair in frame_data:
            key = (video_idx, pair["pair_idx"])
            if (key in embeddings and key in future_embeddings and key in future_captions):
                pair["embedding"] = embeddings[key]
                pair["future_embedding"] = future_embeddings[key]
                pair["future_caption"] = future_captions[key]
                video_pairs[video_idx].append(pair)

    valid_videos = [(video_idx, pairs) for video_idx, pairs in video_pairs.items()]

    # Step 4: Save metadata
    print("\nSaving metadata...")
    metadata = {
        'dataset_size': len(valid_videos),
        'video_indices': [v[0] for v in valid_videos],
        'pair_counts': [len(v[1]) for v in valid_videos]
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    # Step 5: Create HDF5 file for embeddings and captions
    print("\nCreating HDF5 file for embeddings and captions...")
    data_file = os.path.join(output_dir, 'dataset.h5')
    with h5py.File(data_file, 'w') as f:
        for video_idx, pairs in tqdm(valid_videos):
            grp = f.create_group(str(video_idx))

            # Create caption dataset with variable length string support
            caption_dt = h5py.special_dtype(vlen=str)
            captions_curr = grp.create_dataset('captions', (len(pairs),), dtype=caption_dt)
            captions_future = grp.create_dataset('future_captions', (len(pairs),), dtype=caption_dt)

            for i, pair in enumerate(pairs):
                # Store embeddings
                grp.create_dataset(f'embedding_{i}', data=pair['embedding'])
                grp.create_dataset(f'future_embedding_{i}', data=pair['future_embedding'])

                # Store captions
                captions_curr[i] = pair['caption']
                captions_future[i] = pair['future_caption']

    # Step 6: Save frame paths
    print("\nSaving frame paths...")
    frame_paths = {
        str(video_idx): {
            str(i): {
                'cur': pair['frame_cur_path'],
                'fut': pair['frame_fut_path']
            }
            for i, pair in enumerate(pairs)
        }
        for video_idx, pairs in valid_videos
    }
    with open(os.path.join(output_dir, 'frame_paths.json'), 'w') as f:
        json.dump(frame_paths, f)

    print(f"\nPreprocessing complete. Data saved to {output_dir}")
    return metadata


if __name__ == "__main__":
    # Example usage
    metadata = create_memmap_dataset(
        frame_root="/data/kinetics400caption",
        frame_info_path="/data/kinetics400caption/frame_info.json",
        embeddings_path="/data/kinetics400caption/embedding_large_512.jsonl",
        future_embeddings_path="/data/kinetics400caption/future_embedding_fixed.jsonl",
        future_captions_path="/data/kinetics400caption/future_caption_fixed.jsonl",
        frame_info_additional_path="/data/kinetics400caption/frame_info_additional.json",
        embeddings_additional_path="/data/kinetics400caption/embedding_6_pair_512.jsonl",
        output_dir="memmap_data_token"
    )
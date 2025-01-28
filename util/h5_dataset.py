import h5py
import os
import cv2
import json
import numpy as np
import tqdm
from joblib import Parallel, delayed
import ujson
from itertools import islice


def process_embeddings_chunk(chunk):
    """Process a batch of lines from jsonl file."""
    results = {}
    for line in chunk:
        try:
            record = ujson.loads(line)
            parts = record[-1]['custom_id'].split('_')
            video_idx = int(parts[1])
            pair_idx = int(parts[-1])
            embedding = record[1]['data'][0]['embedding']
            results[(video_idx, pair_idx)] = np.array(embedding, dtype=np.float32)[:512]
        except Exception as e:
            print(f"Error processing line: {e}")
    return results


def load_embeddings(embeddings_path, n_jobs=60, chunk_size=10000):
    """Load embeddings from jsonl file using parallel processing"""
    print(f"\nLoading embeddings from {embeddings_path}")

    embeddings = {}
    chunks = []

    # Read file in chunks
    with open(embeddings_path, 'r') as f:
        while True:
            chunk = list(islice(f, chunk_size))
            if not chunk:
                break
            chunks.append(chunk)

    # Process chunks in parallel
    print("\nProcessing chunks in parallel...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_embeddings_chunk)(chunk) for chunk in chunks
    )

    # Merge results
    print("\nMerging results...")
    for chunk_result in tqdm.tqdm(results):
        embeddings.update(chunk_result)

    return embeddings


def load_frame(frame_path):
    """Load and convert frame to RGB"""
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Failed to load frame from path: {frame_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def create_h5_dataset(
        output_path,
        frame_root,
        frame_info_path,
        embeddings_path,
        frame_info_additional_path=None,
        embeddings_additional_path=None,
        image_size=(224, 224)  # Default size for ViT
):
    """Create H5 dataset from frame info and embeddings"""

    # Load embeddings first
    print("Loading main embeddings...")
    embeddings = load_embeddings(embeddings_path)

    if embeddings_additional_path:
        print("Loading additional embeddings...")
        additional_embeddings = load_embeddings(embeddings_additional_path)
        embeddings.update(additional_embeddings)

    # Create H5 file
    with h5py.File(output_path, 'w') as f:
        # Create groups
        images_group = f.create_group('images')
        embeddings_group = f.create_group('embeddings')
        metadata_group = f.create_group('metadata')

        # Process frame info
        def process_frame_info(info_path, prefix="frames", pair_idx_offset=0):
            with open(info_path, 'r') as info_file:
                frame_info = json.load(info_file)
                videos = frame_info['videos']

                for video in tqdm.tqdm(videos, desc=f"Processing {prefix}"):
                    video_idx = video['video_idx']
                    frame_paths = video['frame_paths']
                    pair_idx = video['pair_idx'] + pair_idx_offset

                    key = (video_idx, pair_idx)
                    if key not in embeddings:
                        continue

                    # Create video group if it doesn't exist
                    video_group = images_group.require_group(str(video_idx))
                    pair_group = video_group.require_group(str(pair_idx))

                    # Load and save frames
                    try:
                        frame_cur = load_frame(os.path.join(frame_root, prefix, frame_paths[0]))
                        frame_fut = load_frame(os.path.join(frame_root, prefix, frame_paths[1]))

                        # Resize frames if needed
                        if image_size:
                            frame_cur = cv2.resize(frame_cur, image_size)
                            frame_fut = cv2.resize(frame_fut, image_size)

                        # Store frames
                        pair_group.create_dataset('frame_cur', data=frame_cur,
                                                  compression='gzip', compression_opts=9)
                        pair_group.create_dataset('frame_fut', data=frame_fut,
                                                  compression='gzip', compression_opts=9)

                        # Store embedding
                        embeddings_group.create_dataset(
                            f"{video_idx}_{pair_idx}",
                            data=embeddings[key],
                            compression='gzip',
                            compression_opts=9
                        )

                        # Store metadata
                        metadata = {
                            'frame_cur_path': os.path.join(prefix, frame_paths[0]),
                            'frame_fut_path': os.path.join(prefix, frame_paths[1])
                        }
                        metadata_group.create_dataset(
                            f"{video_idx}_{pair_idx}",
                            data=str(metadata),
                            dtype=h5py.special_dtype(vlen=str)
                        )

                    except Exception as e:
                        print(f"Error processing video {video_idx}, pair {pair_idx}: {e}")
                        continue

        # Process main frame info
        print("\nProcessing main frame info...")
        process_frame_info(frame_info_path)

        # Process additional frame info if provided
        if frame_info_additional_path:
            print("\nProcessing additional frame info...")
            process_frame_info(
                frame_info_additional_path,
                prefix="frames_additional",
                pair_idx_offset=2
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--frame_root', type=str, required=True)
    parser.add_argument('--frame_info_path', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--frame_info_additional_path', type=str)
    parser.add_argument('--embeddings_additional_path', type=str)
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224])

    args = parser.parse_args()

    create_h5_dataset(
        output_path=args.output_path,
        frame_root=args.frame_root,
        frame_info_path=args.frame_info_path,
        embeddings_path=args.embeddings_path,
        frame_info_additional_path=args.frame_info_additional_path,
        embeddings_additional_path=args.embeddings_additional_path,
        image_size=tuple(args.image_size)
    )
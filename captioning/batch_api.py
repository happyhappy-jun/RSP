import os
import json
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import time
import random
import argparse
import math
from concurrent.futures import ThreadPoolExecutor

def setup_batch_processing(project_root):
    """Setup directories for batch processing"""
    paths = {
        'batch_input': Path(project_root) / "artifacts" / "batch_input",
        'batch_output': Path(project_root) / "artifacts" / "batch_output",
        'frames': Path(project_root) / "artifacts" / "frames",
        'results': Path(project_root) / "artifacts" / "results",
        'shards': Path(project_root) / "artifacts" / "shards"
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    return paths

def create_batch_requests(frame_info):
    """Create all batch requests for 4 frames"""
    system_prompt = """You are a video scene analyzer. For the given sequence of 4 frames from a video, describe:
1. Main Action: Provide one clear sentence summarizing the overall activity or event
2. Temporal Changes: Describe how the scene evolves across the 4 frames
3. Movement Details:
   - Subject movements and position changes
   - Camera movements (if any)
   - Changes in background elements
Keep descriptions concise, specific, and focused on observable changes. Use precise spatial and temporal language."""

    all_requests = []
    frame_mapping = {}
    
    print("Creating batch requests...")
    for video_info in tqdm(frame_info['videos']):
        try:
            video_frames = video_info['frames']
            if len(video_frames) != 4:
                print(f"Skipping video {video_info['video_idx']} - expected 4 frames, got {len(video_frames)}")
                continue
                
            # Get paths for all 4 frames
            frame_paths = [os.path.join("/home/byungjun/RSP/artifacts/frames", frame['path']) 
                         for frame in video_frames]
            
            custom_id = f"video_{video_info['video_idx']}"
            
            # Store mapping
            frame_mapping[custom_id] = {
                'label': video_info['label'],
                'video_idx': video_info['video_idx'],
                'video_name': video_info['video_name'],
                'frame_paths': frame_paths,
                'frame_indices': [frame['video_frame_idx'] for frame in video_frames]
            }
            
            # Encode all 4 images
            contents = []
            for frame_path in frame_paths:
                with open(frame_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "low"
                    }
                })
            
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}]
                        },
                        {
                            "role": "user",
                            "content": contents
                        }
                    ],
                    "max_tokens": 8000
                }
            }
            
            all_requests.append(json.dumps(request))
            
        except Exception as e:
            print(f"Error creating request for video {video_info['video_idx']}: {str(e)}")
            continue
    
    return all_requests, frame_mapping

def create_shards(all_requests, num_shards, paths):
    """Divide requests into shards"""
    shard_size = math.ceil(len(all_requests) / num_shards)
    shards = []
    
    print(f"\nDividing {len(all_requests)} requests into {num_shards} shards...")
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, len(all_requests))
        
        shard_requests = all_requests[start_idx:end_idx]
        shard_path = paths['shards'] / f"shard_{i}.jsonl"
        
        with open(shard_path, "w") as f:
            f.write("\n".join(shard_requests))
        
        shards.append({
            'shard_index': i,
            'path': shard_path,
            'num_requests': len(shard_requests)
        })
    
    return shards

def upload_shard(client, shard):
    """Upload a single shard and start batch processing"""
    with open(shard['path'], "rb") as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"4-frame analysis shard {shard['shard_index']}"
        }
    )
    
    return {
        'shard_index': shard['shard_index'],
        'batch_id': batch.id,
        'status': batch.status,
        'num_requests': shard['num_requests']
    }


def monitor_batch_status(client, batch_info):
    """Monitor status of a single batch"""
    try:
        status = client.batches.retrieve(batch_info['batch_id'])
        return {
            'shard_index': batch_info['shard_index'],
            'batch_id': batch_info['batch_id'],
            'status': status.status,
            'completed': status.request_counts.completed,
            'failed': status.request_counts.failed,
            'total': status.request_counts.total
        }
    except Exception as e:
        print(f"Error checking status for shard {batch_info['shard_index']}: {str(e)}")
        return None

def process_results(client, batch_info, paths, frame_mapping):
    """Process results for a completed batch"""
    try:
        batch_data = client.batches.retrieve(batch_info['batch_id'])
        output_file = batch_data.output_file_id
        output_file = client.files.content(output_file)
        
        # Save raw output
        output_path = paths['batch_output'] / f"results_shard_{batch_info['shard_index']}.jsonl"
        with open(output_path, "w") as f:
            f.write(output_file.text)
        
        # Process results
        results = []
        for line in output_file.text.strip().split('\n'):
            result = json.loads(line)
            custom_id = result['custom_id']
            video_info = frame_mapping[custom_id]
            
            results.append({
                'custom_id': custom_id,
                'shard_index': batch_info['shard_index'],
                'label': video_info['label'],
                'video_idx': video_info['video_idx'],
                'video_name': video_info['video_name'],
                'frame_paths': video_info['frame_paths'],
                'frame_indices': video_info['frame_indices'],
                'analysis': result['response']['body']['choices'][0]['message']['content']
            })
        
        return results
    except Exception as e:
        print(f"Error processing results for shard {batch_info['shard_index']}: {str(e)}")
        return None

def main():
    load_dotenv()
    project_root = os.getenv('PROJECT_ROOT')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--check_interval', type=int, default=60)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()
    
    client = OpenAI(api_key="")
    paths = setup_batch_processing(project_root)
    
    if not args.resume:
        # Load frame info
        frame_info_path = "/home/byungjun/RSP/artifacts/frame_info.json"
        with open(frame_info_path) as f:
            frame_info = json.load(f)
        
        # Create batch requests
        all_requests, frame_mapping = create_batch_requests(frame_info)
        
        # Save frame mapping for resume
        mapping_path = paths['batch_input'] / "frame_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(frame_mapping, f, indent=2)
        
        # Create and upload shards
        shards = create_shards(all_requests, args.num_shards, paths)
        
        print("\nUploading shards...")
        # with ThreadPoolExecutor(max_workers=min(args.num_shards, 10)) as executor:
        #     batch_infos = list(filter(None, executor.map(
        #         lambda shard: upload_shard(client, shard),
        #         shards
        #     )))
        batch_infos = []
        for shard in shards:
            batch_infos.append(upload_shard(client, shard))
            
        
        # Save batch information
        batch_info_path = paths['batch_input'] / "batch_infos.json"
        with open(batch_info_path, "w") as f:
            json.dump(batch_infos, f, indent=2)
    
    else:
        print("Resuming from previous run...")
        batch_info_path = paths['batch_input'] / "batch_infos.json"
        mapping_path = paths['batch_input'] / "frame_mapping.json"
        
        if not batch_info_path.exists() or not mapping_path.exists():
            print("Error: Required files for resume not found!")
            return
        
        with open(batch_info_path) as f:
            batch_infos = json.load(f)
        with open(mapping_path) as f:
            frame_mapping = json.load(f)
    
    # Monitor progress
    print("\nMonitoring batch progress...")
    completed_batches = []
    active_batches = batch_infos.copy()
    
    while active_batches:
        statuses = []
        with ThreadPoolExecutor(max_workers=len(active_batches)) as executor:
            statuses = list(filter(None, executor.map(
                lambda info: monitor_batch_status(client, info),
                active_batches
            )))
        
        print("\nCurrent status:")
        for status in statuses:
            print(f"Shard {status['shard_index']}: {status['completed']}/{status['total']} "
                  f"completed, {status['failed']} failed - Status: {status['status']}")
        
        still_active = []
        for batch in active_batches:
            status = next((s for s in statuses if s['batch_id'] == batch['batch_id']), None)
            if status and status['status'] == 'completed':
                completed_batches.append(batch)
            elif status:
                still_active.append(batch)
        
        active_batches = still_active
        
        if active_batches:
            time.sleep(args.check_interval)
    
    # Process results
    print("\nProcessing results...")
    all_results = []
    
    with ThreadPoolExecutor(max_workers=len(completed_batches)) as executor:
        future_to_batch = {
            executor.submit(process_results, client, batch, paths, frame_mapping): batch
            for batch in completed_batches
        }
        
        for future in tqdm(future_to_batch):
            batch = future_to_batch[future]
            try:
                results = future.result()
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"Error processing batch {batch['shard_index']}: {str(e)}")
    
    # Save final results
    results_path = paths['results'] / "frame_analysis_results_complete.json"
    with open(results_path, "w") as f:
        json.dump({
            'metadata': {
                'total_videos': len(all_results),
                'total_shards': len(completed_batches),
                'batch_ids': [batch['batch_id'] for batch in completed_batches],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\nProcessing complete! Results saved to: {results_path}")
    print(f"Total processed videos: {len(all_results)}")

if __name__ == "__main__":
    main()
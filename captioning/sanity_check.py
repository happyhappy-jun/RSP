import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def create_test_request(frame_info):
    """Create a single test request with 4 frames from first video"""
    system_prompt = """You are a video scene analyzer. For the given sequence of 4 frames from a video, describe:
1. Main Action: Provide one clear sentence summarizing the overall activity or event
2. Temporal Changes: Describe how the scene evolves across the 4 frames
3. Movement Details:
   - Subject movements and position changes
   - Camera movements (if any)
   - Changes in background elements
Keep descriptions concise, specific, and focused on observable changes. Use precise spatial and temporal language."""

    # Get first video
    video_info = frame_info['videos'][0]
    video_frames = video_info['frames']
    
    # Get paths for all 4 frames
    frame_paths = [os.path.join("/home/byungjun/RSP/artifacts/frames", frame['path']) 
                  for frame in video_frames]
    
    custom_id = f"video_{video_info['video_idx']}"
    
    print("\nTest request details:")
    print(f"Video index: {video_info['video_idx']}")
    print(f"Label: {video_info['label']}")
    print("Frame paths:")
    for path in frame_paths:
        print(f"- {path}")
    
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
    
    return json.dumps(request)

def main():
    load_dotenv("/home/byungjun/RSP/.env")
    
    # Load frame info
    frame_info_path = "/home/byungjun/RSP/artifacts/frame_info.json"
    print(f"\nLoading frame info from: {frame_info_path}")
    with open(frame_info_path) as f:
        frame_info = json.load(f)
    
    print(f"\nFound {len(frame_info['videos'])} videos in frame info")
    
    # Create test request
    test_request = create_test_request(frame_info)
    
    # Save test request
    test_shard_path = "test_shard.jsonl"
    print(f"\nSaving test request to: {test_shard_path}")
    with open(test_shard_path, "w") as f:
        f.write(test_request)
    
    # Initialize OpenAI client
    client = OpenAI(api_key="")
    print("\nInitialized OpenAI client")
    
    # Upload file
    print("\nUploading test shard...")
    with open(test_shard_path, "rb") as f:
        batch_file = client.files.create(
            file=f,
            purpose="batch"
        )
    print(f"File uploaded with ID: {batch_file.id}")
    
    # Create batch
    print("\nCreating batch...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "test shard for sanity check"
        }
    )
    print(f"Batch created with ID: {batch.id}")
    
    # Get initial status
    print("\nRetrieving initial batch status...")
    status = client.batches.retrieve(batch.id)
    print(f"Status: {status.status}")
    print(f"Total requests: {status.request_counts.total}")
    print(f"Completed: {status.request_counts.completed}")
    print(f"Failed: {status.request_counts.failed}")
    print("\nFull status object:")
    print(json.dumps(status.model_dump(), indent=2))

if __name__ == "__main__":
    main()
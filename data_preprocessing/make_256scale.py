import os
import argparse
from joblib import Parallel, delayed
import subprocess
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def process_video(input_path, output_path):
    """Process a single video without audio"""
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',  '-hwaccel', 'cuda',
            '-i', input_path,
            '-vf', '"scale_cuda=256:256'
            '-c:a', 'copy', '-c:v', 'h264_nvenc',
            '-tune', 'hq', '-preset', 'p7',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, input_path
    except Exception as e:
        return False, f"{input_path}: {str(e)}"

def get_video_tasks(root, output_root, begin, end):
    """Collect all video processing tasks"""
    tasks = []
    
    # Get list of directories
    dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    dirs = dirs[begin:end]
    
    for dir_name in dirs:
        input_dir = os.path.join(root, dir_name)
        output_dir = os.path.join(output_root, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, filename)):
                continue
                
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Skip if output already exists
            if os.path.exists(output_path):
                continue
                
            tasks.append((input_path, output_path))
    
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=401)
    parser.add_argument('--datadir', type=str, default='/home/bjyoon/rsp-llm/data/sampled-256-resized-train/train2')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of tasks per batch')
    
    args = parser.parse_args()
    
    root = '/home/bjyoon/rsp-llm/data/sampled-train'
    
    # Create output directory
    os.makedirs(args.datadir, exist_ok=True)
    
    # Collect all tasks
    print("Collecting video processing tasks...")
    tasks = get_video_tasks(root, args.datadir, args.begin, args.end)
    print(f"Found {len(tasks)} videos to process")
    
    if not tasks:
        print("No videos to process. Exiting.")
        return
    
    # Process videos in parallel with progress bar
    print(f"Processing videos using {args.n_jobs} jobs...")
    results = Parallel(n_jobs=args.n_jobs, batch_size=args.batch_size)(
        delayed(process_video)(input_path, output_path)
        for input_path, output_path in tqdm(tasks, desc="Processing videos")
    )
    
    # Report results
    successes = sum(1 for success, _ in results if success)
    failures = [(msg, i) for i, (success, msg) in enumerate(results) if not success]
    
    print("\nProcessing completed!")
    print(f"Successfully processed: {successes}/{len(tasks)} videos")
    
    if failures:
        print("\nErrors occurred:")
        for msg, idx in failures:
            input_path, _ = tasks[idx]
            print(f"- {input_path}: {msg}")

if __name__ == '__main__':
    main()
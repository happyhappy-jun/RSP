import argparse
import glob
import os
import boto3
import questionary
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True,
                       help='Directory containing checkpoints (e.g., outputs/exp_name)')
    return parser.parse_args()

def upload_with_progress(s3_client, local_file, bucket_name, s3_key):
    """Upload file with progress bar"""
    file_size = os.path.getsize(local_file)
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(local_file))
    
    def callback(bytes_transferred):
        progress.update(bytes_transferred)
    
    try:
        s3_client.upload_file(
            local_file,
            bucket_name,
            s3_key,
            Callback=callback
        )
        progress.close()
        return True
    except Exception as e:
        progress.close()
        print(f"Error uploading {local_file}: {e}")
        return False

def upload_checkpoints_to_s3(target_dir):
    # Initialize S3 client
    s3_client = boto3.client('s3')
    bucket_name = 'junyoon-rsp'
    
    # Get experiment name from the path
    exp_name = os.path.basename(target_dir.rstrip('/'))
    
    # Create empty directory marker in S3
    try:
        s3_client.put_object(Bucket=bucket_name, Key=f'{exp_name}/')
        print(f"Created directory s3://{bucket_name}/{exp_name}/")
    except Exception as e:
        print(f"Error creating directory in S3: {e}")
        return
    
    # Check for checkpoint-199.pth first
    default_checkpoint = os.path.join(target_dir, 'checkpoint-199.pth')
    if os.path.exists(default_checkpoint):
        print("Found checkpoint-199.pth, uploading...")
        s3_key = f'{exp_name}/checkpoint-199.pth'
        upload_with_progress(s3_client, default_checkpoint, bucket_name, s3_key)
        return
    
    # If default checkpoint doesn't exist, find all checkpoint files
    checkpoint_pattern = os.path.join(target_dir, 'checkpoint-*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {target_dir}")
        return
    
    # Prepare choices for questionary with file sizes
    choices = []
    for checkpoint_file in checkpoint_files:
        size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)
        filename = os.path.basename(checkpoint_file)
        display = f"{filename} ({size_mb:.2f} MB)"
        choices.append({
            'name': display,
            'value': checkpoint_file
        })
    
    # Ask user which checkpoints to upload
    selected_checkpoints = questionary.checkbox(
        "Select checkpoints to upload:",
        choices=choices
    ).ask()
    
    if not selected_checkpoints:  # User cancelled
        return
    
    # Upload selected checkpoints
    for checkpoint_file in selected_checkpoints:
        filename = os.path.basename(checkpoint_file)
        s3_key = f'{exp_name}/{filename}'
        upload_with_progress(s3_client, checkpoint_file, bucket_name, s3_key)

def main():
    args = parse_args()
    upload_checkpoints_to_s3(args.target_dir)

if __name__ == '__main__':
    main()
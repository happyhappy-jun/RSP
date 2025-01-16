import argparse
import os
import boto3
import questionary
from botocore.exceptions import ClientError
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='Experiment name to download')
    parser.add_argument('--all', action='store_true', help='Download all checkpoints instead of just the latest')
    return parser.parse_args()

def list_s3_experiments(s3_client, bucket_name='junyoon-rsp'):
    """List all experiment directories in S3"""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        experiments = set()
        for obj in response.get('Contents', []):
            exp = obj['Key'].split('/')[0]
            if exp:
                experiments.add(exp)
        return sorted(list(experiments))
    except Exception as e:
        print(f"Error listing S3 contents: {e}")
        return []

def list_checkpoints(s3_client, bucket_name, exp_name):
    """List all checkpoint files for an experiment"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f'{exp_name}/'
        )
        checkpoints = []
        for obj in response.get('Contents', []):
            if 'checkpoint-' in obj['Key'] and obj['Key'].endswith('.pth'):
                checkpoints.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'display': f"{obj['Key']} ({obj['Size'] / 1024 / 1024:.2f} MB)"
                })
        return sorted(checkpoints, key=lambda x: int(x['key'].split('checkpoint-')[1].split('.')[0]))
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []

def download_with_progress(s3_client, bucket_name, s3_key, local_path, file_size):
    """Download file with progress bar"""
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(s3_key))
    
    def callback(bytes_transferred):
        progress.update(bytes_transferred)
    
    try:
        s3_client.download_file(
            bucket_name,
            s3_key,
            local_path,
            Callback=callback
        )
        progress.close()
        return True
    except Exception as e:
        progress.close()
        print(f"Error downloading {s3_key}: {e}")
        return False

def download_checkpoint(s3_client, bucket_name, checkpoint_info):
    """Download specific checkpoint"""
    s3_key = checkpoint_info['key']
    file_size = checkpoint_info['size']
    
    # Get experiment name and filename from s3_key
    exp_name = s3_key.split('/')[0]
    filename = os.path.basename(s3_key)
    local_dir = os.path.join('outputs', exp_name)
    local_path = os.path.join(local_dir, filename)
    
    # Check if local file already exists
    if os.path.exists(local_path):
        print(f"Checkpoint already exists locally: {local_path}")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading s3://junyoon-rsp/{s3_key} to {local_path}")
    success = download_with_progress(s3_client, bucket_name, s3_key, local_path, file_size)
    if success:
        print(f"Successfully downloaded {filename}")

def main():
    args = parse_args()
    s3_client = boto3.client('s3')
    bucket_name = 'junyoon-rsp'
    
    # If no experiment name provided, list available ones and ask user to select
    if not args.exp_name:
        experiments = list_s3_experiments(s3_client)
        if not experiments:
            print("No experiments found in S3")
            return
        
        # Allow multiple experiment selection
        selected_experiments = questionary.checkbox(
            "Select experiments to download:",
            choices=experiments
        ).ask()
        
        if not selected_experiments:  # User cancelled
            return
    else:
        selected_experiments = [args.exp_name]
    
    # Process each selected experiment
    for exp_name in selected_experiments:
        print(f"\nProcessing experiment: {exp_name}")
        checkpoints = list_checkpoints(s3_client, bucket_name, exp_name)
        
        if not checkpoints:
            print(f"No checkpoints found for experiment {exp_name}")
            continue
        
        if args.all:
            # Allow selecting multiple checkpoints
            selected_checkpoints = questionary.checkbox(
                f"Select checkpoints to download for {exp_name}:",
                choices=[ckpt['display'] for ckpt in checkpoints]
            ).ask()
            
            if selected_checkpoints:
                # Map selected displays back to checkpoint info
                selected_info = [
                    ckpt for ckpt in checkpoints
                    if ckpt['display'] in selected_checkpoints
                ]
                
                for checkpoint in selected_info:
                    download_checkpoint(s3_client, bucket_name, checkpoint)
        else:
            # Download only the latest checkpoint
            latest_checkpoint = checkpoints[-1]
            download_checkpoint(s3_client, bucket_name, latest_checkpoint)

if __name__ == '__main__':
    main()

import argparse
import os
import boto3
import questionary
from botocore.exceptions import ClientError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help='Experiment name to download')
    parser.add_argument('--all', action='store_true', help='Download all checkpoints instead of just 199')
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
                checkpoints.append(obj['Key'])
        return checkpoints
    except Exception as e:
        print(f"Error listing checkpoints: {e}")
        return []

def download_checkpoint(s3_client, bucket_name, s3_key):
    """Download specific checkpoint"""
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
    
    try:
        print(f"Downloading s3://junyoon-rsp/{s3_key} to {local_path}")
        s3_client.download_file(bucket_name, s3_key, local_path)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

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
            
        exp_name = questionary.select(
            "Select experiment to download:",
            choices=experiments
        ).ask()
        if not exp_name:  # User cancelled
            return
    else:
        exp_name = args.exp_name

    # Get list of available checkpoints
    checkpoints = list_checkpoints(s3_client, bucket_name, exp_name)
    if not checkpoints:
        print(f"No checkpoints found for experiment {exp_name}")
        return
    
    if args.all:
        # Download all checkpoints
        for checkpoint in checkpoints:
            download_checkpoint(s3_client, bucket_name, checkpoint)
    else:
        # Find and download only the latest checkpoint (assuming numeric naming)
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('checkpoint-')[1].split('.')[0]))
        download_checkpoint(s3_client, bucket_name, latest_checkpoint)

if __name__ == '__main__':
    main()

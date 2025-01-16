import argparse
import glob
import os
import boto3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True,
                       help='Directory containing checkpoints (e.g., outputs/exp_name)')
    return parser.parse_args()

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
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(target_dir, 'checkpoint-*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {target_dir}")
        return
    
    # Upload each checkpoint
    for checkpoint_file in checkpoint_files:
        # Construct S3 key: exp_name/filename
        filename = os.path.basename(checkpoint_file)
        s3_key = f'{exp_name}/{filename}'
        
        try:
            print(f"Uploading {checkpoint_file} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(checkpoint_file, bucket_name, s3_key)
            print(f"Successfully uploaded {filename}")
        except Exception as e:
            print(f"Error uploading {filename}: {e}")

def main():
    args = parse_args()
    upload_checkpoints_to_s3(args.target_dir)

if __name__ == '__main__':
    main()

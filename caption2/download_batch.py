import openai
from datetime import datetime
import os
from tqdm import tqdm

# Initialize the client using API key from environment variables
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Target batch ID and output directory
target_batch_id = "batch_678712a94738819082672369ddfe666c"
output_dir = "/data/RSP/caption_batch_results"
found_target = False

# Initialize variables
has_more = True
after = None
batch_size = 100
total_batches = 0
page_number = 0
batches_to_process = []  # List to store full batch objects

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

print(f"=== Phase 1: Scanning and Storing Batch Objects ===\n")

# Scan for batches with progress bar
pbar = tqdm(desc="Scanning batches", unit="page")

while has_more and not found_target:
    page_number += 1
    response = client.batches.list(limit=batch_size, after=after)
    
    for batch in response.data:
        total_batches += 1
        created_date = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d %H:%M:%S')
        
        # Store full batch object if we haven't found target yet
        if not found_target:
            batches_to_process.append(batch)
            pbar.set_postfix({"latest_batch": batch.id})
        
        if batch.id == target_batch_id:
            print(f"\nFound target batch: {batch.id}")
            found_target = True
            break

    if not found_target:
        has_more = response.has_more
        if has_more:
            after = response.last_id
            pbar.update(1)

pbar.close()

print(f"\n=== Phase 2: Downloading Batch Files ===")
print(f"Total batches to process: {len(batches_to_process)}\n")

def save_file_content(file_id, filepath):
    """Download and save file content"""
    # Skip if file already exists
    if os.path.exists(filepath):
        return "exists"
        
    try:
        response = client.files.content(file_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return "downloaded"
    except Exception as e:
        print(f"Error downloading file {file_id}: {str(e)}")
        return "error"

# Counters for summary
files_downloaded = 0
files_skipped = 0
files_error = 0

# Process batches in reverse order with progress bar
for idx, batch in enumerate(tqdm(reversed(batches_to_process), desc="Downloading files", total=len(batches_to_process))):
    created_date = datetime.fromtimestamp(batch.created_at).strftime('%Y-%m-%d')
    
    # Download output file
    if batch.output_file_id:
        output_path = os.path.join(output_dir, f"{idx}_{batch.id}_output.jsonl")
        result = save_file_content(batch.output_file_id, output_path)
        if result == "downloaded":
            files_downloaded += 1
            tqdm.write(f"Downloaded: {idx}_{batch.id}_output.jsonl")
        elif result == "exists":
            files_skipped += 1
            tqdm.write(f"Skipped existing: {idx}_{batch.id}_output.jsonl")
        else:
            files_error += 1
    
    # Download error file if it exists
    if batch.error_file_id:
        error_path = os.path.join(output_dir, f"{idx}_{batch.id}_error.jsonl")
        result = save_file_content(batch.error_file_id, error_path)
        if result == "downloaded":
            files_downloaded += 1
            tqdm.write(f"Downloaded: {idx}_{batch.id}_error.jsonl")
        elif result == "exists":
            files_skipped += 1
            tqdm.write(f"Skipped existing: {idx}_{batch.id}_error.jsonl")
        else:
            files_error += 1

print("\n=== Summary ===")
print(f"Total batches processed: {len(batches_to_process)}")
print(f"Files downloaded: {files_downloaded}")
print(f"Files skipped (already exist): {files_skipped}")
print(f"Files with errors: {files_error}")
print(f"Files saved to: {output_dir}")

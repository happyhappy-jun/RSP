import openai
from datetime import datetime
import os

# Initialize the client using API key from environment variables
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Get the timestamp for cutoff
cutoff = datetime(2025, 1, 17, 13, 0, 0).timestamp()

# Initialize counters and storage
total_files = 0
old_files = 0
file_ids_to_delete = []
has_more = True
after = None
limit = 100  # Number of files to fetch per request

print("=== Scanning Files ===\n")

# Paginate through all files
while has_more:
    # Get a page of files
    response = client.files.list(limit=limit, after=after, order='asc')
    
    # Process the current page
    for file in response.data:
        total_files += 1
        print(file)
        # Check if file matches both criteria: older than cutoff AND starts with "shard_"
        if file.created_at < cutoff and file.filename.startswith('shard_'):
            old_files += 1
            file_ids_to_delete.append({
                'id': file.id,
                'filename': file.filename,
                'created_at': datetime.fromtimestamp(file.created_at).strftime('%Y-%m-%d %H:%M:%S')
            })
            print(f"File {old_files}:")
            print(f"  ID: {file.id}")
            print(f"  Filename: {file.filename}")
            print(f"  Created: {datetime.fromtimestamp(file.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Purpose: {file.purpose}")
            print()

    # Check if there are more files to fetch
    if len(response.data) < limit:
        has_more = False
    else:
        # Set the after cursor to the last file's ID
        after = response.data[-1].id
        print(f"Fetched {total_files} files so far...")

print(f"=== Summary ===")
print(f"Total files scanned: {total_files}")
print(f"'shard_' files to be deleted: {old_files}")

if old_files > 0:
    confirm = input(f"\nWould you like to delete these {old_files} 'shard_' files? (Y/N): ").strip().upper()
    
    if confirm == 'Y':
        print("\n=== Deleting Files ===")
        deleted_count = 0
        error_count = 0
        
        for file_info in file_ids_to_delete:
            try:
                client.files.delete(file_info['id'])
                deleted_count += 1
                print(f"✓ ({deleted_count}/{old_files}) Deleted: {file_info['filename']} (Created: {file_info['created_at']})")
            except Exception as e:
                error_count += 1
                print(f"✗ Error deleting {file_info['filename']}: {str(e)}")
        
        print(f"\nDeletion process complete:")
        print(f"Successfully deleted: {deleted_count} 'shard_' files")
        if error_count > 0:
            print(f"Failed to delete: {error_count} files")
    else:
        print("\nOperation cancelled. No files were deleted.")
else:
    print(f"\nNo 'shard_' files found that were created before {datetime.fromtimestamp(cutoff).strftime('%Y-%m-%d %H:%M:%S')}")

# set up the openai environment
from openai import OpenAI
import os
import questionary
import json
import tempfile
from datetime import datetime
from collections import deque
from pathlib import Path
from tqdm import tqdm
#Setting the openAI key in the environement for creating the client

def batch2str(batch):
    return f'{batch.id} {batch.status} {datetime.fromtimestamp(batch.created_at).strftime("%Y-%m-%d %H:%M:%S")}'

def get_batch_page(after=None, limit=20):
    page = client.batches.list(limit=limit, after=after)
    return list(page.data)

def get_batch_id(choice):
    return choice.split()[0]

def get_all_batches_between(start_id, end_id):
    print(f"Getting batches between {start_id} and {end_id}")
    all_batches = []
    current_page = None
    start_index = None
    end_index = None
    
    # Collect all relevant batches
    while True:
        page_batches = get_batch_page(after=current_page)
        if not page_batches:
            break
            
        all_batches.extend(page_batches)
        
        # Check if we have both IDs
        if start_index is None:
            try:
                start_index = next(i for i, b in enumerate(all_batches) if b.id == start_id)
            except StopIteration:
                pass
                
        if end_index is None:
            try:
                end_index = next(i for i, b in enumerate(all_batches) if b.id == end_id)
            except StopIteration:
                pass
        
        # If we found both IDs, we can stop collecting
        if start_index is not None and end_index is not None:
            break
            
        current_page = page_batches[-1].id
    
    # If we didn't find both IDs
    if start_index is None or end_index is None:
        print("Warning: Could not find one or both batch IDs")
        return []
        
    # Return batches in correct order
    if start_index <= end_index:
        return all_batches[start_index:end_index + 1]
    else:
        return list(reversed(all_batches[end_index:start_index + 1]))

client = OpenAI()

# Select first batch
first_batch_choice = None
current_page = None
page_history = deque([None])  # Start with None as first page
print("Select the first batch:")
while True:
    batches = get_batch_page(after=current_page)
    batch_list = [batch2str(batch) for batch in batches]
    
    choices = []
    if len(page_history) > 1:  # If we have previous pages
        choices.append("Previous Page")
    if batch_list:  # If we have more pages ahead
        choices.append("Next Page")
    choices.extend(batch_list)
    
    choice = questionary.select(
        "Select starting batch (use Previous/Next Page to navigate):",
        choices=choices
    ).ask()
    
    if choice == "Next Page":
        if batch_list:
            current_page = batches[-1].id
            page_history.append(current_page)
    elif choice == "Previous Page":
        page_history.pop()  # Remove current page
        current_page = page_history[-1]  # Go back to previous page
    else:
        first_batch_choice = choice
        break

# Select last batch
last_batch_choice = None
current_page = None
page_history = deque([None])  # Start with None as first page
print("\nSelect the last batch:")
while True:
    batches = get_batch_page(after=current_page)
    batch_list = [batch2str(batch) for batch in batches]
    
    choices = []
    if len(page_history) > 1:  # If we have previous pages
        choices.append("Previous Page")
    if batch_list:  # If we have more pages ahead
        choices.append("Next Page")
    choices.extend(batch_list)
    
    choice = questionary.select(
        "Select ending batch (use Previous/Next Page to navigate):",
        choices=choices
    ).ask()
    
    if choice == "Next Page":
        if batch_list:
            current_page = batches[-1].id
            page_history.append(current_page)
    elif choice == "Previous Page":
        page_history.pop()  # Remove current page
        current_page = page_history[-1]  # Go back to previous page
    else:
        last_batch_choice = choice
        break

# Get all batches between first and last
start_id = get_batch_id(first_batch_choice)
end_id = get_batch_id(last_batch_choice)
selected_batches = get_all_batches_between(start_id, end_id)

def process_jsonl_file(file_path):
    """Process a JSONL file and return list of records"""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def combine_and_sort_outputs(batches):
    """Retrieve, combine and sort outputs from multiple batches"""
    all_records = []
    
    for batch in tqdm(batches, desc="Processing batches", unit="batch"):
        # Retrieve detailed batch info
        batch_detail = client.batches.retrieve(batch.id)
        
        # Create temp directory for downloading files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download output file
            output_file = Path(temp_dir) / "output.jsonl"
            
            # Get the output file URL from the batch
            if hasattr(batch_detail, 'output_file_id'):
                with open(output_file, 'wb') as f:
                    print(f"Downloading output file {batch_detail.output_file_id}")
                    response = client.files.content(batch_detail.output_file_id)
                    f.write(response.read())
                
                # Process the JSONL file
                records = process_jsonl_file(output_file)
                all_records.extend(records)
    
    # Sort all records by custom_id
    return sorted(all_records, key=lambda x: int(x.get('custom_id', '').split("-")[-1]))

# Print selected batches
print("\nSelected batches:")
for batch in selected_batches:
    print(batch2str(batch))

# Combine and sort all outputs
print("\nRetrieving and combining batch outputs...")
sorted_records = combine_and_sort_outputs(selected_batches)

# Save combined output
output_file = "combined_output.jsonl"
print(f"\nSaving combined and sorted output to {output_file}")
with open(output_file, 'w') as f:
    for record in sorted_records:
        json.dump(record, f)
        f.write('\n')

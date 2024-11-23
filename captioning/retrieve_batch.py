# set up the openai environment
from openai import OpenAI
import os
import questionary
from datetime import datetime
from collections import deque
#Setting the openAI key in the environement for creating the client

def batch2str(batch):
    return f'{batch.id} {batch.status} {datetime.fromtimestamp(batch.created_at).strftime("%Y-%m-%d %H:%M:%S")}'

def get_batch_page(after=None, limit=20):
    page = client.batches.list(limit=limit, after=after)
    return list(page.data)

def get_batch_id(choice):
    return choice.split()[0]

def get_all_batches_between(start_id, end_id):
    batches = []
    current_page = None
    found_start = False
    
    while True:
        page_batches = get_batch_page(after=current_page)
        if not page_batches:
            break
            
        for batch in page_batches:
            if batch.id == start_id:
                found_start = True
            
            if found_start:
                batches.append(batch)
                
            if batch.id == end_id:
                return batches
                
        current_page = page_batches[-1].id
    
    return batches

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

# Print selected batches
print("\nSelected batches:")
for batch in selected_batches:
    print(batch2str(batch))

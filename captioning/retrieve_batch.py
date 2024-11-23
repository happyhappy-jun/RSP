# set up the openai environment
from openai import OpenAI
import os
import questionary
from datetime import datetime
#Setting the openAI key in the environement for creating the client

def batch2str(batch):
    return f'{batch.id} {batch.status} {datetime.fromtimestamp(batch.created_at).strftime("%Y-%m-%d %H:%M:%S")}'

def get_batch_page(after=None, limit=20):
    return client.batches.list(limit=limit, after=after)

client = OpenAI()

current_page = None
while True:
    # Get the current page of batches
    batches = get_batch_page(after=current_page, limit=20)
    batch_list = [batch2str(batch) for batch in batches]
    
    # Store the ID of the last batch for pagination
    if batch_list:
        current_page = batches[-1].id
    
    choice = questionary.select(
        "Select a batch (or Next Page to see more):",
        choices=["Next Page"] + batch_list
    ).ask()
    
    if choice != "Next Page":
        break

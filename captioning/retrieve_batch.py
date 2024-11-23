# set up the openai environment
from openai import OpenAI
import os
import questionary
from datetime import datetime
#Setting the openAI key in the environement for creating the client

def batch2str(batch):
    return f'{batch.id} {batch.status} {datetime.fromtimestamp(batch.created_at).strftime("%Y-%m-%d %H:%M:%S")}'
    

client = OpenAI()

batches = client.batches.list()


questionary.select(
    "What do you want to do?",
    choices=["Next Page"] + [batch2str(batch) for batch in batches]
).ask()
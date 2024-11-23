# set up the openai environment
from openai import OpenAI
import os
#Setting the openAI key in the environement for creating the client
client = OpenAI()

# create the files for the batch job
batch_folder = '/home/junyoon/rsp-llm/artifacts'
batch_input_files = []
for file in os.listdir(batch_folder):
    if file.endswith(".jsonl"):
        batch_input_files.append(client.files.create(
            file=open(f'{batch_folder}/{file}', "rb"),
            purpose="batch"
        ))

# create the batch job
batch_file_ids= [batch_file.id for batch_file in batch_input_files] # we get the ids of the batch files
job_creations = []
for i,file_id in enumerate(batch_file_ids):
    job_creations.append(client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/embeddings",
    completion_window="24h", # currently only 24h is supported 
    metadata={
      "description": f"part_{i}_icd_embeddings"
    }
    ))

# We can see here the jobs created, they start with validation
for job in job_creations:
    print(job)

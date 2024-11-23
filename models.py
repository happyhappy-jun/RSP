from pydantic import BaseModel
import json
from typing import List, Optional
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage

class Response(BaseModel):
    """
    Response class is used to store the response of the API
    """
    status_code: int
    request_id: str
    body: CreateEmbeddingResponse

class BatchOutput(BaseModel):
    """
    BatchOutput class is used to store the output of the batch processing
    """
    id: str
    custom_id: str
    response: Response
    error: Optional[str] = None
    
if __name__ == "__main__":
    with open("/home/junyoon/rsp-llm/artifacts/combined_output.jsonl", "r") as f:
        lines = f.readlines()
    
    batch_outputs = []
    for line in lines[:10]:
        record = json.loads(line)
        print(record.keys())
        response = BatchOutput(**record).response.body.data[0].embedding
        print(response)

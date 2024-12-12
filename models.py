from pydantic import BaseModel, Field
import json
from typing import List, Optional, Dict

class EmbeddingData(BaseModel):
    """Data model for embedding output"""
    embedding: List[float]
    index: int = 0
    object: str = "embedding"

class Usage(BaseModel):
    """Usage statistics"""
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    """Response format matching embedding output"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str = "text-embedding-custom"
    usage: Usage

class Response(BaseModel):
    """Response class for batch processing"""
    status_code: int = 200
    request_id: str
    body: EmbeddingResponse

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

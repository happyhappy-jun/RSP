import os
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
import tiktoken
import time
from dataclasses import dataclass, field

@dataclass
class StatusTracker:
    """Tracks the status of async embedding creation"""
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int = 0
    total_tasks: int = 0
    
    def get_progress_str(self) -> str:
        """Get a formatted progress string"""
        return (f"Progress: {self.num_tasks_succeeded}/{self.total_tasks} "
                f"[Success: {self.num_tasks_succeeded}, "
                f"Failed: {self.num_tasks_failed}, "
                f"In Progress: {self.num_tasks_in_progress}]")

class EmbeddingCreator:
    """Creates embeddings using OpenAI's text-embedding-3-small model with async processing"""
    
    def __init__(self, api_key: str = None):
        """Initialize with optional API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "text-embedding-3-small"
        self.request_url = "https://api.openai.com/v1/embeddings"
        self.max_requests_per_minute = 10_000 * 0.75  # Using 75% of limit for safety
        self.max_tokens_per_minute = 10_000_000 * 0.75
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
        
    async def create_embedding(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        """Create embedding for a single text string"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        json_data = {"model": self.model, "input": text, "encoding_format": "float"}
        
        try:
            async with session.post(self.request_url, headers=headers, json=json_data) as response:
                response_data = await response.json()
                if "error" in response_data:
                    raise Exception(response_data["error"])
                return response_data["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"Error creating embedding: {str(e)}")
            return None
            
    async def process_caption_results(
        self,
        caption_results: List[Dict[str, Any]],
        output_dir: Path,
        max_attempts: int = 5
    ) -> List[Dict[str, Any]]:
        """Process caption results and create embeddings asynchronously"""
        
        status = StatusTracker(total_tasks=len(caption_results))
        embedding_results = []
        retry_queue = asyncio.Queue()
        
        # Initialize rate limiting
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()

        async with aiohttp.ClientSession() as session:
            # Create tasks for initial processing
            tasks = []
            for result in caption_results:
                try:
                    if 'response' in result:
                        # Handle combined_output.jsonl format
                        response_body = result['response']['body']
                        if isinstance(response_body, str):
                            response_body = json.loads(response_body)
                        caption = response_body['choices'][0]['message']['content']
                        custom_id = result['custom_id']
                    else:
                        # Handle direct API response format
                        caption = result['choices'][0]['message']['content']
                        custom_id = result.get('custom_id', str(result.get('created')))
                    
                    token_count = self.count_tokens(caption)
                    
                    # Check capacity
                    current_time = time.time()
                    time_elapsed = current_time - last_update_time
                    available_request_capacity = min(
                        available_request_capacity + self.max_requests_per_minute * time_elapsed / 60.0,
                        self.max_requests_per_minute
                    )
                    available_token_capacity = min(
                        available_token_capacity + self.max_tokens_per_minute * time_elapsed / 60.0,
                        self.max_tokens_per_minute
                    )
                    
                    if available_request_capacity >= 1 and available_token_capacity >= token_count:
                        available_request_capacity -= 1
                        available_token_capacity -= token_count
                        last_update_time = current_time
                        
                        task = asyncio.create_task(self.create_embedding(session, caption))
                        tasks.append((custom_id, caption, task))
                        status.num_tasks_started += 1
                        status.num_tasks_in_progress += 1
                    else:
                        await asyncio.sleep(0.001)  # Brief pause if at capacity
                        
                except Exception as e:
                    logging.error(f"Error queueing result {custom_id}: {str(e)}")
                    continue

            # Process results as they complete
            progress_bar = tqdm(tasks, desc="Creating embeddings")
            for custom_id, caption, task in progress_bar:
                progress_bar.set_postfix_str(status.get_progress_str())
                try:
                    embedding = await task
                    if embedding:
                        embedding_results.append({
                            'custom_id': custom_id,
                            'original_caption': caption,
                            'embedding': embedding
                        })
                        status.num_tasks_succeeded += 1
                    else:
                        status.num_tasks_failed += 1
                except Exception as e:
                    logging.error(f"Error processing result {custom_id}: {str(e)}")
                    status.num_tasks_failed += 1
                finally:
                    status.num_tasks_in_progress -= 1

        # Save results in JSONL format
        output_path = output_dir / "embedding_results.jsonl"
        with open(output_path, 'w') as f:
            for result in embedding_results:
                f.write(json.dumps(result) + '\n')
            
        logging.info(f"\nProcessing complete:")
        logging.info(f"Succeeded: {status.num_tasks_succeeded}")
        logging.info(f"Failed: {status.num_tasks_failed}")
        logging.info(f"Rate limits hit: {status.num_rate_limit_errors}")
        logging.info(f"Saved {len(embedding_results)} embeddings to {output_path}")
        
        return embedding_results

def create_embeddings(caption_results_path: str, output_dir: str) -> None:
    """Convenience function to create embeddings from caption results file"""
    
    # Load caption results
    caption_results = []
    with open(caption_results_path) as f:
        if caption_results_path.endswith('.jsonl'):
            for line in f:
                caption_results.append(json.loads(line))
        else:
            caption_results = json.load(f)
        
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings
    creator = EmbeddingCreator()
    creator.process_caption_results(caption_results, output_path)

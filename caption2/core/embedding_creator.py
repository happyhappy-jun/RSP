import os
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
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
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0
    total_tasks: int = 0
    
    def get_progress_str(self) -> str:
        """Get a formatted progress string"""
        return (f"Progress: {self.num_tasks_succeeded}/{self.total_tasks} "
                f"[Success: {self.num_tasks_succeeded}, "
                f"Failed: {self.num_tasks_failed}, "
                f"In Progress: {self.num_tasks_in_progress}, "
                f"Rate Limits: {self.num_rate_limit_errors}]")

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata"""
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results"""
        logging.debug(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url,
                headers=request_header,
                json=self.request_json
            ) as response:
                try:
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith('application/json'):
                        error_text = await response.text()
                        raise ValueError(f"Unexpected response type '{content_type}'. Status: {response.status}. Body: {error_text[:200]}")
                    
                    response = await response.json()
                except aiohttp.ContentTypeError as e:
                    raise ValueError(f"Failed to parse JSON response. Status: {response.status}. Error: {str(e)}")
                
            if "error" in response:
                logging.warning(f"Request {self.task_id} failed with error {response['error']}")
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "").lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors counted separately
        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.task_id} failed after all attempts. Saving errors: {self.result}")
                data = [self.request_json, [str(e) for e in self.result], self.metadata] if self.metadata else [self.request_json, [str(e) for e in self.result]]
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
                pbar.update(1)
        else:
            data = [self.request_json, response, self.metadata] if self.metadata else [self.request_json, response]
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            pbar.update(1)
            logging.debug(f"Request {self.task_id} completed successfully")

class EmbeddingCreator:
    """Creates embeddings using OpenAI's text-embedding-3-small model with async processing"""
    
    def __init__(self, max_requests_per_minute: int = 10000):
        """Initialize with rate limits"""
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = 10_000_000  # text-embedding-3-small limit
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    async def process_caption_results(
        self,
        caption_results: List[Dict[str, Any]],
        output_dir: Path,
        max_attempts: int = 5,
        seconds_to_sleep: float = 0.001
    ) -> None:
        """Process caption results and create embeddings asynchronously"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "embedding_results.jsonl"
        request_url = "https://api.openai.com/v1/embeddings"
        request_header = {"Authorization": f"Bearer {self.api_key}"}
        
        # Initialize trackers
        status = StatusTracker(total_tasks=len(caption_results))
        retry_queue = asyncio.Queue()
        task_id_counter = 0
        
        # Initialize progress bar
        pbar = tqdm(total=len(caption_results), desc="Creating embeddings")
        
        # Initialize rate limiting
        available_request_capacity = self.max_requests_per_minute
        available_token_capacity = self.max_tokens_per_minute
        last_update_time = time.time()
        seconds_to_pause_after_rate_limit = 15

        async with aiohttp.ClientSession() as session:
            next_request = None
            
            while True:
                # Get next request
                if next_request is None:
                    if not retry_queue.empty():
                        next_request = retry_queue.get_nowait()
                        logging.debug(f"Retrying request {next_request.task_id}")
                    elif task_id_counter < len(caption_results):
                        result = caption_results[task_id_counter]
                        if 'caption' not in result:
                            logging.error(f"Missing response field in result: {result}")
                            task_id_counter += 1
                            continue
                            
                        caption = result['caption']
                        custom_id = result['custom_id']
                        token_count = self.count_tokens(caption)
                        
                        request_json = {
                            "model": "text-embedding-3-small",
                            "input": caption,
                            "encoding_format": "float"
                        }
                        
                        next_request = APIRequest(
                            task_id=task_id_counter,
                            request_json=request_json,
                            token_consumption=token_count,
                            attempts_left=max_attempts,
                            metadata={"custom_id": custom_id, "original_caption": caption}
                        )
                        task_id_counter += 1
                        status.num_tasks_started += 1
                        status.num_tasks_in_progress += 1
                
                # Update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
                    self.max_requests_per_minute
                )
                available_token_capacity = min(
                    available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
                    self.max_tokens_per_minute
                )
                last_update_time = current_time

                # Process request if capacity available
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (available_request_capacity >= 1 and 
                        available_token_capacity >= next_request_tokens):
                        
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=retry_queue,
                                save_filepath=output_file,
                                status_tracker=status
                            )
                        )
                        next_request = None

                # Break if all tasks complete
                if status.num_tasks_in_progress == 0:
                    pbar.close()
                    break

                # Sleep briefly
                await asyncio.sleep(seconds_to_sleep)

                # Pause if rate limited recently
                seconds_since_rate_limit = time.time() - status.time_of_last_rate_limit_error
                if seconds_since_rate_limit < seconds_to_pause_after_rate_limit:
                    pause_time = seconds_to_pause_after_rate_limit - seconds_since_rate_limit
                    logging.warning(f"Rate limit hit - pausing for {pause_time:.1f}s")
                    await asyncio.sleep(pause_time)

        # Log final status
        logging.info(f"\nProcessing complete:")
        logging.info(f"Succeeded: {status.num_tasks_succeeded}")
        logging.info(f"Failed: {status.num_tasks_failed}")
        logging.info(f"Rate limits: {status.num_rate_limit_errors}")
        logging.info(f"Other API errors: {status.num_api_errors}")
        logging.info(f"Other errors: {status.num_other_errors}")
        logging.info(f"Results saved to: {output_file}")

def append_to_jsonl(data: Any, filename: str) -> None:
    """Append a json payload to a jsonl file"""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

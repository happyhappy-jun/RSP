import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from .metadata_store import MetadataStore

class BatchProcessor:
    """Wrapper for OpenAI's batch API"""
    
    def __init__(
        self,
        client: OpenAI,
        output_dir: str,
        max_retries: int = 3,
        check_interval: int = 60
    ):
        self.client = client
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.check_interval = check_interval
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_batch(
        self,
        requests: List[Dict[str, Any]],
        shard_idx: int,
        description: str = None
    ) -> str:
        """Create a batch job from requests"""
        # Save requests to JSONL
        shard_path = self.output_dir / f"shard_{shard_idx}.jsonl"
        with open(shard_path, "w") as f:
            for request in requests:
                json.dump(request, f)
                f.write("\n")
                
        # Upload file
        with open(shard_path, "rb") as f:
            batch_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
            
        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=requests[0]["url"],  # Use endpoint from first request
            completion_window="24h",
            metadata={
                "description": description or f"Batch shard {shard_idx}"
            }
        )
        
        return batch.id
        
    def monitor_batch(self, batch_id: str) -> Dict[str, Any]:
        """Monitor batch status until completion"""
        retries = 0
        while retries < self.max_retries:
            try:
                status = self.client.batches.retrieve(batch_id)
                if status.status == "completed":
                    return self._process_batch_output(status)
                elif status.status == "failed":
                    raise Exception(f"Batch {batch_id} failed")
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                retries += 1
                if retries == self.max_retries:
                    raise Exception(f"Max retries reached for batch {batch_id}: {str(e)}")
                time.sleep(self.check_interval)
                
        return None
        
    def _process_batch_output(self, batch: Any) -> Dict[str, Any]:
        """Process completed batch output"""
        output_file = self.client.files.content(batch.output_file_id)
        
        # Save raw output
        output_path = self.output_dir / f"output_{batch.id}.jsonl"
        with open(output_path, "wb") as f:
            f.write(output_file.read())
            
        # Process results
        results = []
        with open(output_path) as f:
            for line in f:
                results.append(json.loads(line))
                
        return {
            'batch_id': batch.id,
            'results': sorted(results, key=lambda x: x['custom_id'])
        }
        
    def _estimate_request_size(self, request: Dict[str, Any]) -> int:
        """Calculate exact size of request JSON in bytes"""
        # Convert request to JSON string and get its byte size
        request_json = json.dumps(request) + '\n'  # Add newline for JSONL format
        return len(request_json.encode('utf-8'))

    def process_requests(
        self,
        requests: List[Dict[str, Any]],
        max_batch_size: int = 100 * 1024 * 1024,  # 100MB in bytes
        num_workers: int = 4,
        description: str = None,
        sanity_check: bool = False
    ) -> List[Dict[str, Any]]:
        """Process large number of requests with concurrent batch processing"""
        # Initialize metadata store
        metadata_store = MetadataStore(self.output_dir)
        
        # Store metadata for each request
        print("\nProcessing metadata...")
        for request in tqdm(requests, desc="Storing metadata"):
            if 'metadata' in request['body']:
                metadata_store.add_metadata(request['custom_id'], request['body'].pop('metadata'))
        
        # For sanity check, use direct API call
        if sanity_check:
            print("\nRunning sanity check with first request only...")
            try:
                request = requests[0]
                response = self.client.chat.completions.create(**request['body'])
                result = {
                    'custom_id': request['custom_id'],
                    'response': {
                        'body': {
                            'choices': [{
                                'message': {
                                    'content': response.choices[0].message.content
                                }
                            }]
                        }
                    }
                }
                return metadata_store.merge_results([result])
            except Exception as e:
                print(f"Sanity check failed: {str(e)}")
                return []

        # Treat input requests as a single batch
        print("\nSubmitting batch...")
        with ThreadPoolExecutor(max_workers=num_workers) as submit_executor:
            # Submit single batch
            batch_futures = []
            future = submit_executor.submit(
                self.create_batch,
                requests,
                shard_idx=0,
                description=description or "Batch processing"
            )
            batch_futures.append(future)

            # Wait for all batch submissions to complete
            batch_ids = []
            for future in tqdm(batch_futures, desc="Waiting for batch submissions"):
                try:
                    batch_id = future.result()
                    batch_ids.append(batch_id)
                except Exception as e:
                    print(f"Error submitting batch: {str(e)}")

        # Monitor batches and return results directly
        results = []
        print("\nMonitoring batch progress...")
        with ThreadPoolExecutor(max_workers=num_workers) as monitor_executor:
            # Start monitoring all batches
            monitor_futures = [
                monitor_executor.submit(self.monitor_batch, batch_id)
                for batch_id in batch_ids
            ]
            
            # Process results as they complete
            for future in tqdm(monitor_futures, desc="Processing results"):
                try:
                    result = future.result()
                    if result and 'results' in result:
                        results.extend(result['results'])
                        # Clear memory
                        del result['results']
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
        
        # Sort results before returning
        results.sort(key=lambda x: x['custom_id'])
        return results

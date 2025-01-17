import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from .metadata_store import MetadataStore

__all__ = ['BatchProcessor']

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
        self.batch_ids = []  # Track submitted batch IDs
        self.status_file = self.output_dir / "batch_status.json"
        self._load_batch_status()

    def _load_batch_status(self):
        """Load or initialize batch status tracking"""
        if self.status_file.exists():
            with open(self.status_file) as f:
                self.batch_status = json.load(f)
        else:
            self.batch_status = {
                'submitted': [],    # Successfully submitted batch IDs
                'completed': [],    # Successfully completed batch IDs
                'failed': [],       # Failed batch IDs with errors
                'pending': []       # Batches waiting to be submitted
            }
            self._save_batch_status()

    def _save_batch_status(self):
        """Save current batch status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(self.batch_status, f, indent=2)
        
    def submit_batch(
        self,
        requests: List[Dict[str, Any]], 
        shard_idx: str,
        description: str = None,
        retry_failed: bool = True,
        original_shard_file: Optional[str] = None
    ) -> str:
        """Submit a batch job from requests"""
        # Save requests to JSONL using original shard filename if provided
        shard_path = self.output_dir / (original_shard_file or f"shard_{shard_idx}.jsonl")
        with open(shard_path, "w") as f:
            for request in requests:
                json.dump(request, f)
                f.write("\n")
                
        try:
            # Upload file
            with open(shard_path, "rb") as f:
                batch_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
                
            # Create batch
            # Determine endpoint and prepare request
            first_request = requests[0]
            if 'messages' in first_request['body']:
                endpoint = "/v1/chat/completions"
            else:
                endpoint = "/v1/embeddings"

            # Create batch with proper metadata
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint=endpoint,
                completion_window="24h",
                metadata={
                    "description": description or f"Batch shard {shard_idx}",
                    "shard_idx": str(shard_idx),
                    "total_requests": str(len(requests))
                }
            )
            
            batch_id = batch.id
            self.batch_ids.append(batch_id)  # Track the batch ID
            
            # Update status
            if batch_id not in self.batch_status['submitted']:
                self.batch_status['submitted'].append(batch_id)
                if batch_id in self.batch_status['failed']:
                    self.batch_status['failed'].remove(batch_id)
                self._save_batch_status()
                
            print(f"Successfully submitted batch {batch_id} for shard {shard_idx}")
            return batch_id
            
        except Exception as e:
            error_msg = f"Error submitting batch for shard {shard_idx}: {str(e)}"
            print(error_msg)
            
            # Track failed batch
            failed_info = {
                'shard_idx': shard_idx,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if shard_idx not in [f.get('shard_idx') for f in self.batch_status['failed']]:
                self.batch_status['failed'].append(failed_info)
                self._save_batch_status()
                
            raise Exception(error_msg)
        
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

    def submit_requests(
        self,
        requests: List[Dict[str, Any]],
        max_batch_size: int = 100 * 1024 * 1024,  # 100MB in bytes
        num_workers: int = 4,
        description: str = None,
        sanity_check: bool = False,
        shard_idx: Optional[int] = None,
        original_shard_file: Optional[str] = None
    ) -> List[str]:
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
                # Handle different request types
                if 'messages' in request['body']:
                    # Chat completion request
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
                else:
                    # Embedding request
                    response = self.client.embeddings.create(**request['body'])
                    result = {
                        'custom_id': request['custom_id'],
                        'response': {
                            'data': [{
                                'embedding': response.data[0].embedding
                            }]
                        }
                    }
                return metadata_store.merge_results([result])
            except Exception as e:
                print(f"Sanity check failed: {str(e)}")
                return []

        # Split into sub-batches of max 50,000 requests
        MAX_REQUESTS_PER_BATCH = 50000
        sub_batches = []
        for i in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
            sub_batches.append(requests[i:i + MAX_REQUESTS_PER_BATCH])
            
        print(f"\nSplitting into {len(sub_batches)} sub-batches of max {MAX_REQUESTS_PER_BATCH} requests...")
        
        with ThreadPoolExecutor(max_workers=num_workers) as submit_executor:
            # Submit all sub-batches
            batch_futures = []
            for sub_idx, sub_batch in enumerate(sub_batches):
                sub_shard_idx = f"{shard_idx}_{sub_idx}" if shard_idx is not None else str(sub_idx)
                future = submit_executor.submit(
                    self.submit_batch,
                    sub_batch,
                    shard_idx=sub_shard_idx,
                    description=f"{description or 'Batch processing'} (sub-batch {sub_idx + 1}/{len(sub_batches)})",
                    original_shard_file=original_shard_file
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

        return batch_ids

    def monitor_batches(
        self,
        batch_ids: Optional[List[str]] = None,
        num_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """Monitor and collect results from submitted batches"""
        if batch_ids is None:
            batch_ids = self.batch_ids

        if not batch_ids:
            raise ValueError("No batch IDs provided or stored for monitoring")

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

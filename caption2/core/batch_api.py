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
        """Estimate size of a request in bytes"""
        # Base size for request structure
        size = 1000  # Conservative base estimate
        
        # Add size of image data if present
        if 'body' in request and 'messages' in request['body']:
            for message in request['body']['messages']:
                if 'content' in message:
                    if isinstance(message['content'], list):
                        for content in message['content']:
                            if isinstance(content, dict) and 'image_url' in content:
                                # Extract base64 length and convert to bytes
                                img_url = content['image_url']['url']
                                if img_url.startswith('data:image/jpeg;base64,'):
                                    base64_str = img_url.split(',')[1]
                                    size += len(base64_str) * 3 // 4  # Convert base64 to bytes
                    
        return size

    def process_requests(
        self,
        requests: List[Dict[str, Any]],
        max_batch_size: int = 100 * 1024 * 1024,  # 512MB in bytes
        num_workers: int = 4,
        description: str = None,
        sanity_check: bool = False
    ) -> List[Dict[str, Any]]:
        """Process large number of requests with sharding based on size limit"""
        # Initialize metadata store
        metadata_store = MetadataStore(self.output_dir)
        
        # Store metadata for each request
        for request in requests:
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
                
        # Process batches normally
        shards = []
        current_shard = []
        current_size = 0
        
        print("\nCalculating request sizes...")
        for request in tqdm(requests):
            request_size = self._estimate_request_size(request)
            
            if current_size + request_size > max_batch_size and current_shard:
                shards.append(current_shard)
                current_shard = []
                current_size = 0
                
            current_shard.append(request)
            current_size += request_size
            
        if current_shard:
            shards.append(current_shard)

        # Create batches
        batch_ids = []
        print(f"\nCreating {len(shards)} batch shard{'s' if len(shards) > 1 else ''}...")
        for i, shard in enumerate(tqdm(shards)):
            batch_id = self.create_batch(
                shard,
                shard_idx=i,
                description=f"{description or 'Batch'} shard {i}"
            )
            batch_ids.append(batch_id)
            
        # Monitor batches
        all_results = []
        print("\nMonitoring batch progress...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.monitor_batch, batch_id) 
                      for batch_id in batch_ids]
            
            for future in tqdm(futures):
                try:
                    result = future.result()
                    if result:
                        all_results.extend(result['results'])
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    
        return sorted(all_results, key=lambda x: x['custom_id'])

"""
Batch retrieval and processing utility for OpenAI API responses.
Allows selecting and combining multiple batch outputs into a single JSONL file.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from pathlib import Path
import tempfile
import json
import logging

from openai import OpenAI
import questionary
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchInfo:
    """Container for batch information"""
    id: str
    status: str
    created_at: float
    completion_window: str
    metadata: Optional[Dict[str, Any]]
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    expired_at: Optional[float]

    @classmethod
    def from_api_batch(cls, batch) -> 'BatchInfo':
        return cls(
            id=batch.id,
            status=batch.status,
            created_at=batch.created_at,
            completion_window=batch.completion_window,
            metadata=batch.metadata,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            expired_at=batch.expired_at
        )

    def __str__(self) -> str:
        created = datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M:%S")
        return f'{self.id} {self.status} {created}'

class BatchManager:
    """Handles batch operations and processing"""
    
    def __init__(self, client: OpenAI):
        self.client = client

    def get_batch_page(self, after: Optional[str] = None, limit: int = 20) -> List[BatchInfo]:
        """Retrieve a page of batches from the API"""
        page = self.client.batches.list(limit=limit, after=after)
        return [BatchInfo.from_api_batch(batch) for batch in page.data]

    def get_batches_between(self, start_id: str, end_id: str) -> List[BatchInfo]:
        """Get all batches between two batch IDs inclusive"""
        logger.info(f"Getting batches between {start_id} and {end_id}")
        all_batches = []
        current_page = None
        start_index = end_index = None

        while True:
            page_batches = self.get_batch_page(after=current_page)
            if not page_batches:
                break
                
            all_batches.extend(page_batches)
            
            if start_index is None:
                try:
                    start_index = next(i for i, b in enumerate(all_batches) if b.id == start_id)
                except StopIteration:
                    pass
                    
            if end_index is None:
                try:
                    end_index = next(i for i, b in enumerate(all_batches) if b.id == end_id)
                except StopIteration:
                    pass
            
            if start_index is not None and end_index is not None:
                break
                
            current_page = page_batches[-1].id
        
        if start_index is None or end_index is None:
            logger.warning("Could not find one or both batch IDs")
            return []
            
        if start_index <= end_index:
            return all_batches[start_index:end_index + 1]
        else:
            return list(reversed(all_batches[end_index:start_index + 1]))

    def cancel_batch(self, batch_id: str) -> None:
        """Cancel a running batch"""
        try:
            self.client.batches.cancel(batch_id)
            logger.info(f"Successfully cancelled batch {batch_id}")
        except Exception as e:
            logger.error(f"Failed to cancel batch {batch_id}: {e}")

    def process_batch_outputs(self, batches: List[BatchInfo]) -> List[Dict[str, Any]]:
        """Process and combine outputs from multiple batches"""
        all_records = []
        errors = []
        
        for batch in tqdm(batches, desc="Processing batches", unit="batch"):
                
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process output file if it exists
                if batch.output_file_id:
                    output_file = Path(temp_dir) / "output.jsonl"
                    with open(output_file, 'wb') as f:
                        logger.info(f"Downloading output file {batch.output_file_id}")
                        response = self.client.files.content(batch.output_file_id)
                        f.write(response.read())
                    records = self._process_jsonl_file(output_file)
                    all_records.extend(records)
                
                # Process error file if it exists
                if batch.error_file_id:
                    error_file = Path(temp_dir) / "error.jsonl"
                    with open(error_file, 'wb') as f:
                        logger.info(f"Downloading error file {batch.error_file_id}")
                        response = self.client.files.content(batch.error_file_id)
                        f.write(response.read())
                    error_records = self._process_jsonl_file(error_file)
                    errors.extend(error_records)
                    
        if errors:
            logger.warning(f"Found {len(errors)} errors across all batches")
            error_file = Path("batch_errors.jsonl")
            with open(error_file, 'w') as f:
                for error in errors:
                    json.dump(error, f)
                    f.write('\n')
            logger.info(f"Wrote errors to {error_file}")
        
        def sort_key(x):
            custom_id = x.get('custom_id', '')
            try:
                # Extract video number and pair number
                parts = custom_id.split('_')
                if len(parts) >= 4:
                    video_num = int(parts[1])  # get number after 'video'
                    pair_num = int(parts[3])   # get number after 'pair'
                    return (video_num, pair_num)
                return (float('inf'), float('inf'))  # Put records without proper format at the end
            except (ValueError, IndexError):
                return (float('inf'), float('inf'))
                
        return sorted(all_records, key=sort_key)

    @staticmethod
    def _process_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
        """Process a JSONL file and return list of records"""
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                # Skip records that have batch_expired error
                if "error" in record and isinstance(record["error"], dict) and record["error"].get("code") == "batch_expired":
                    continue
                records.append(record)
        return records

class BatchSelector:
    """UI for selecting batches"""
    
    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager

    def select_batch(self, prompt: str) -> Optional[str]:
        """Interactive batch selection"""
        current_page = None
        page_history = deque([None])

        while True:
            batches = self.batch_manager.get_batch_page(after=current_page)
            batch_list = [str(batch) for batch in batches]
            
            choices = []
            if len(page_history) > 1:
                choices.append("Previous Page")
            if batch_list:
                choices.append("Next Page")
            choices.extend(batch_list)
            
            choice = questionary.select(
                prompt,
                choices=choices + ["Cancel Selection"]
            ).ask()
            
            if choice == "Cancel Selection":
                return None
            
            if choice == "Next Page":
                if batch_list:
                    current_page = batches[-1].id
                    page_history.append(current_page)
            elif choice == "Previous Page":
                page_history.pop()
                current_page = page_history[-1]
            else:
                return choice.split()[0]  # Return batch ID

def main():
    """Main execution function"""
    client = OpenAI()
    batch_manager = BatchManager(client)
    selector = BatchSelector(batch_manager)
    
    logger.info("Select the first batch:")
    start_id = selector.select_batch("Select starting batch (use Previous/Next Page to navigate):")
    if not start_id:
        logger.info("Selection cancelled")
        return
        
    logger.info("\nSelect the last batch:")
    end_id = selector.select_batch("Select ending batch (use Previous/Next Page to navigate):")
    if not end_id:
        logger.info("Selection cancelled")
        return
    
    selected_batches = batch_manager.get_batches_between(start_id, end_id)
    
    logger.info("\nSelected batches:")
    for batch in selected_batches:
        logger.info(str(batch))
    
    logger.info("\nRetrieving and combining batch outputs...")
    sorted_records = batch_manager.process_batch_outputs(selected_batches)
    
    output_file = Path("combined_output.jsonl")
    logger.info(f"\nSaving combined and sorted output to {output_file}")
    with open(output_file, 'w') as f:
        for record in sorted_records:
            json.dump(record, f)
            f.write('\n')
    
    logger.info("\nFirst 10 lines of combined output:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            logger.info(f"{i+1}: {line.strip()}")

if __name__ == "__main__":
    main()

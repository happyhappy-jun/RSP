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

    @classmethod
    def from_api_batch(cls, batch) -> 'BatchInfo':
        return cls(
            id=batch.id,
            status=batch.status,
            created_at=batch.created_at
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

    def process_batch_outputs(self, batches: List[BatchInfo]) -> List[Dict[str, Any]]:
        """Process and combine outputs from multiple batches"""
        all_records = []
        
        for batch in tqdm(batches, desc="Processing batches", unit="batch"):
            batch_detail = self.client.batches.retrieve(batch.id)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / "output.jsonl"
                
                if hasattr(batch_detail, 'output_file_id'):
                    with open(output_file, 'wb') as f:
                        logger.info(f"Downloading output file {batch_detail.output_file_id}")
                        response = self.client.files.content(batch_detail.output_file_id)
                        f.write(response.read())
                    
                    records = self._process_jsonl_file(output_file)
                    all_records.extend(records)
        
        return sorted(all_records, key=lambda x: int(x.get('custom_id', '').split("-")[-1]))

    @staticmethod
    def _process_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
        """Process a JSONL file and return list of records"""
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records

class BatchSelector:
    """UI for selecting batches"""
    
    def __init__(self, batch_manager: BatchManager):
        self.batch_manager = batch_manager

    def select_batch(self, prompt: str) -> str:
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
                choices=choices
            ).ask()
            
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
    
    logger.info("\nSelect the last batch:")
    end_id = selector.select_batch("Select ending batch (use Previous/Next Page to navigate):")
    
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

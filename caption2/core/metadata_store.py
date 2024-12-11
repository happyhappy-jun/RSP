import json
from pathlib import Path
from typing import Dict, Any, List

class MetadataStore:
    """Store and manage metadata separately from API requests"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "request_metadata.json"
        self.metadata = {}
        
    def add_metadata(self, custom_id: str, metadata: Dict[str, Any]):
        """Store metadata for a request"""
        self.metadata[custom_id] = metadata
        self._save_metadata()
        
    def get_metadata(self, custom_id: str) -> Dict[str, Any]:
        """Retrieve metadata for a request"""
        return self.metadata.get(custom_id, {})
        
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def load_metadata(self):
        """Load metadata from file"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
                
    def merge_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge API results with stored metadata"""
        merged = []
        for result in results:
            custom_id = result.get('custom_id')
            if custom_id:
                metadata = self.get_metadata(custom_id)
                merged.append({
                    **result,
                    'metadata': metadata
                })
            else:
                merged.append(result)
        return merged

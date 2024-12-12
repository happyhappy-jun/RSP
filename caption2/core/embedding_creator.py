import os
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI

class EmbeddingCreator:
    """Creates embeddings using OpenAI's text-embedding-3-small model"""
    
    def __init__(self, client: OpenAI = None):
        """Initialize with optional OpenAI client"""
        self.client = client or OpenAI()
        self.model = "text-embedding-3-small"
        
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text string"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            return None
            
    def process_caption_results(
        self,
        caption_results: List[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Process caption results and create embeddings"""
        
        embedding_results = []
        
        for result in tqdm(caption_results, desc="Creating embeddings"):
            try:
                # Extract caption text from result
                caption = result['response']['body']['choices'][0]['message']['content']
                custom_id = result['custom_id']
                
                # Create embedding
                embedding = self.create_embedding(caption)
                
                if embedding:
                    embedding_results.append({
                        'custom_id': custom_id,
                        'embedding': embedding,
                        'original_caption': caption
                    })
                    
            except Exception as e:
                print(f"Error processing result {result.get('custom_id', 'unknown')}: {str(e)}")
                continue
                
        # Save results
        output_path = output_dir / "embedding_results.json"
        with open(output_path, 'w') as f:
            json.dump(embedding_results, f, indent=2)
            
        print(f"\nSaved {len(embedding_results)} embeddings to {output_path}")
        return embedding_results

def create_embeddings(caption_results_path: str, output_dir: str) -> None:
    """Convenience function to create embeddings from caption results file"""
    
    # Load caption results
    with open(caption_results_path) as f:
        caption_results = json.load(f)
        
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create embeddings
    creator = EmbeddingCreator()
    creator.process_caption_results(caption_results, output_path)

from typing import List, Dict, Any
import base64
from pathlib import Path

class RequestBuilder:
    """Build API requests for different endpoints"""
    
    def __init__(self, model: str = "gpt-4-vision-preview"):
        self.model = model
        
    def build_caption_request(
        self, 
        frame_paths: List[str], 
        custom_id: str,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Build a request for frame captioning"""
        if system_prompt is None:
            system_prompt = """You are a video scene analyzer. For the given sequence of frames from a video, describe:
1. Main Action: Provide one clear sentence summarizing the overall activity or event
2. Temporal Changes: Describe how the scene evolves across the frames
3. Movement Details:
   - Subject movements and position changes
   - Camera movements (if any)
   - Changes in background elements
Keep descriptions concise, specific, and focused on observable changes. Use precise spatial and temporal language."""

        # Encode all images
        contents = []
        for frame_path in frame_paths:
            with open(frame_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "low"
                }
            })
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": contents
                    }
                ],
                "max_tokens": 8000
            }
        }
    
    def build_embedding_request(
        self,
        text: str,
        custom_id: str,
        model: str = "text-embedding-3-small"
    ) -> Dict[str, Any]:
        """Build a request for text embedding"""
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "input": text.strip(),
                "model": model,
                "encoding_format": "float"
            }
        }

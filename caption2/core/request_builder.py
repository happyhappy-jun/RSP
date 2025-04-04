from typing import List, Dict, Any
import base64
from pathlib import Path

class RequestBuilder:
    """Build API requests for different endpoints"""
    
    def __init__(self, model: str = "gpt-4o-mini", config=None):
        self.model = model
        self.config = config
        
    def build_caption_request(
        self, 
        frame_paths: List[str], 
        custom_id: str,
        metadata: Dict[str, Any] = None,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """Build a request for frame captioning"""
        if system_prompt is None and self.config:
            # prompt_type = self.config.prompt_config["caption"].get("default_prompt", "global")
            system_prompt = self.config.prompt_config["caption"]["prompts"].get("global")
        
        if system_prompt is None:
            system_prompt = self.config.prompt_config["caption"]["prompts"]["global"]

        # Encode all images
        contents = []
        for rel_path in frame_paths:
            full_path = Path(self.config.frame_output_dir) / rel_path
            with open(full_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}",
                    "detail": "low"
                }
            })
        
        request = {
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
                "max_tokens": 2000
            }
        }
        return request
    
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

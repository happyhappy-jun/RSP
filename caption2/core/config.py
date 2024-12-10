from typing import Dict, Any
import yaml
from pathlib import Path

DEFAULT_FRAME_CONFIG = {
    "uniform": {
        "num_frames": 4
    },
    "paired": {
        "min_gap": 4,
        "max_distance": 48,
        "num_pairs": 2
    }
}

DEFAULT_PROMPT_CONFIG = {
    "caption": {
        "system_prompt": """You are a video scene analyzer. For the given sequence of frames from a video, describe:
1. Main Action: Provide one clear sentence summarizing the overall activity or event
2. Temporal Changes: Describe how the scene evolves across the frames
3. Movement Details:
   - Subject movements and position changes
   - Camera movements (if any)
   - Changes in background elements
Keep descriptions concise, specific, and focused on observable changes. Use precise spatial and temporal language."""
    },
    "embedding": {
        "model": "text-embedding-3-small"
    }
}

class Config:
    """Configuration manager for frame sampling and prompts"""
    
    def __init__(
        self,
        config_path: str = None,
        frame_config: Dict[str, Any] = None,
        prompt_config: Dict[str, Any] = None
    ):
        self.frame_config = frame_config or DEFAULT_FRAME_CONFIG.copy()
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        if 'frame_config' in config:
            self.frame_config.update(config['frame_config'])
        if 'prompt_config' in config:
            self.prompt_config.update(config['prompt_config'])
            
    def save_config(self, config_path: str):
        """Save current configuration to YAML file"""
        config = {
            'frame_config': self.frame_config,
            'prompt_config': self.prompt_config
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
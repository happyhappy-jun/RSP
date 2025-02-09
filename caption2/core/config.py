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
        "num_pairs": 8
    }
}

DEFAULT_PROMPT_CONFIG = {
    "caption": {
        "prompts": {
            "default": """You are a movement analyzer specialized in comparing consecutive video frames. Analyze and describe changes between frames using the following guidelines:
FORMAT:
- Provide one clear sentence summarizing the overall dynamics and activity
- Describe qualitative and relative dynamics between frames
- Use precise directional terms (left, right, up, down, forward, backward)
- Focus on observable, concrete changes

ANALYZE THE FOLLOWING ELEMENTS:
- Main Subject/Object:
    - Position: Track center of mass movement
    - Rotation: Note any turns or spins
    - Orientation: Describe facing direction
    - State Changes: Document visible changes in:
        - Physical form or shape
        - Color or appearance
        - Expression or emotional state (if applicable)
- Background:
    - Note any changes in background elements
    - Identify moving vs static elements

Keep descriptions concise, objective, and focused on visible changes between frames.""",
            "future": """Caption this image""",
            "global": """You are a movement analyzer specialized in describing video content, context, and dynamics. Analyze and describe video using the following guidelines:
FORMAT:
- Aggregated one clear sentence summarizing the overall dynamics and activity in video
- Describe qualitative and relative dynamics in the video
- Use precise directional terms (left, right, up, down, forward, backward)
- Focus on observable, concrete changes
- Do not explain each frame independently
- Do not contain information that can be only derived from specific frame

ANALYZE THE FOLLOWING ELEMENTS:
- Main Subject/Object:
    - Position: Track center of mass movement
    - Rotation: Note any turns or spins
    - Orientation: Describe facing direction
    - State Changes: Document visible changes in:
        - Physical form or shape
        - Color or appearance
        - Expression or emotional state (if applicable)
- Background:
    - Note any changes in background elements
    - Identify moving vs static elements

Keep descriptions concise, objective, and focused on visible changes through out the video"""
        },
        "default_prompt": "default"
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
        prompt_config: Dict[str, Any] = None,
        frame_output_dir: str = None,
        data_root: str = None
    ):
        self.frame_config = frame_config or DEFAULT_FRAME_CONFIG.copy()
        self.prompt_config = prompt_config or DEFAULT_PROMPT_CONFIG.copy()
        self.frame_output_dir = frame_output_dir
        self.data_root = data_root
        
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

from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Any

class Config:
    """Wrapper class to provide object-like access to config while maintaining dict compatibility"""
    def __init__(self, cfg: DictConfig):
        self._cfg = OmegaConf.to_container(cfg, resolve=True)
        # Add distributed training parameters directly
        if 'distributed' in self._cfg:
            for k, v in self._cfg['distributed'].items():
                setattr(self, k, v)
        # Add all other parameters
        for k, v in self._cfg.items():
            if k != 'distributed':
                setattr(self, k, v)
    
    def __getattr__(self, name):
        if name not in self._cfg:
            raise AttributeError(f"'Config' object has no attribute '{name}'")
        return self._cfg[name]
    
    def __getitem__(self, key):
        return self._cfg[key]
    
    def get(self, key, default=None):
        return self._cfg.get(key, default)

from omegaconf import DictConfig, OmegaConf
from typing import Any, Optional

class Config:
    """Wrapper class to provide object-like access to config while maintaining dict compatibility"""
    def __init__(self, cfg: DictConfig):
        self._cfg = cfg  # Keep original DictConfig
        # Add distributed training parameters directly
        if 'distributed' in self._cfg:
            for k, v in self._cfg.distributed.items():
                setattr(self, k, v)
        # Add all other parameters
        for k, v in self._cfg.items():
            if k != 'distributed':
                setattr(self, k, v)
    
    def __getattr__(self, name: str) -> Any:
        """Attribute-style access with better error handling"""
        try:
            return OmegaConf.select(self._cfg, name)
        except Exception:
            raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        try:
            return self._cfg[key]
        except Exception:
            raise KeyError(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safe get with default value"""
        try:
            value = OmegaConf.select(self._cfg, key)
            return default if value is None else value
        except Exception:
            return default
    
    def to_dict(self) -> dict:
        """Convert config to plain Python dictionary"""
        return OmegaConf.to_container(self._cfg, resolve=True)

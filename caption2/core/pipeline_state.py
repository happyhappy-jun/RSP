from enum import Enum
from pathlib import Path
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Stages of the caption pipeline"""
    INIT = "init"
    FRAME_EXTRACTION = "frame_extraction" 
    REQUEST_CREATION = "request_creation"
    BATCH_PROCESSING = "batch_processing"
    EMBEDDING_CREATION = "embedding_creation"
    COMPLETE = "complete"
    FAILED = "failed"

class PipelineState:
    """Manages state and progress of the caption pipeline"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_file = output_dir / "pipeline_state.json"
        self.current_stage = PipelineStage.INIT
        self.stage_data: Dict[str, Any] = {}
        self._load_state()
        
    def _load_state(self) -> None:
        """Load state from file if it exists"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.current_stage = PipelineStage(data["current_stage"])
                self.stage_data = data.get("stage_data", {})
                logger.info(f"Loaded pipeline state: {self.current_stage.value}")
            except Exception as e:
                logger.error(f"Failed to load pipeline state: {e}")
                
    def _save_state(self) -> None:
        """Save current state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    "current_stage": self.current_stage.value,
                    "stage_data": self.stage_data
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")
                
    def update_stage(self, stage: PipelineStage, data: Optional[Dict[str, Any]] = None) -> None:
        """Update pipeline stage and optional stage data"""
        logger.info(f"Pipeline stage changing: {self.current_stage.value} -> {stage.value}")
        self.current_stage = stage
        if data:
            self.stage_data[stage.value] = data
        self._save_state()
        
    def get_stage_data(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get data for a specific pipeline stage"""
        return self.stage_data.get(stage.value, {})

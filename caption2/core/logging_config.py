import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> None:
    """Configure logging for the caption pipeline
    
    Args:
        log_file: Optional path to log file. If None, logs to stderr only
        level: Logging level to use
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Always log to stderr
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Optionally log to file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

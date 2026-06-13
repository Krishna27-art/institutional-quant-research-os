"""
Configuration module.
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class Config:
    """Default configuration."""
    log_level: str = "INFO"
    data_dir: str = "data"
    cache_dir: str = "cache"
    experiment_dir: Path = field(default_factory=lambda: Path("data/experiments"))
    
    def ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()

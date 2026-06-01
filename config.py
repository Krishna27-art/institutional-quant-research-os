"""Core configuration for the new market research system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENT_DIR = DATA_DIR / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports"
CATALOG_DIR = DATA_DIR / "catalog"


@dataclass(slots=True)
class SystemConfig:
    """Minimal system configuration for the first vertical slice."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    experiment_dir: Path = EXPERIMENT_DIR
    reports_dir: Path = REPORTS_DIR
    catalog_dir: Path = CATALOG_DIR
    initial_capital: float = 1_000_000.0
    max_daily_loss_pct: float = 0.02
    max_position_pct: float = 0.05
    gap_fade_min_gap_pct: float = 0.3
    gap_fade_max_gap_pct: float = 0.8

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.experiment_dir,
            self.reports_dir,
            self.catalog_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = SystemConfig()
DEFAULT_CONFIG.ensure_dirs()

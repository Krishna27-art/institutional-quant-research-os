"""Core configuration for the trading research OS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
RESEARCH_DATA_DIR = PROJECT_ROOT / "trading_research" / "data"

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
PRE_OPEN_START = "09:00"
AUCTION_END = "09:08"

GAP_THRESHOLD_PCT = 1.0
MIN_GAP_VOLUME = 1_000_000
LOOKBACK_DAYS = 252
MIN_TRADES_FOR_VALIDATION = 300
TARGET_TRADES_FOR_VALIDATION = 500
MAX_POSITION_SIZE_PCT = 0.01
MAX_SECTOR_EXPOSURE_PCT = 0.20
STOP_LOSS_PCT = 2.0
DATA_SOURCE = "nse"
USE_POINT_IN_TIME = True


@dataclass(slots=True)
class GapFadeV2Config:
    """Configuration for the simple gap fade strategy used in tests."""

    gap_down_threshold_pct: float = -1.0
    gap_up_threshold_pct: float = 1.0
    hold_bars: int = 1
    slippage_bps: float = 5.0
    cost_bps: float = 5.0


@dataclass(slots=True)
class WalkForwardConfig:
    """Walk-forward validation settings."""

    min_train_bars: int = 100
    test_bars: int = 20
    n_folds: int = 3


@dataclass(slots=True)
class Config:
    """Top-level system configuration."""

    strategy: GapFadeV2Config = field(default_factory=GapFadeV2Config)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    initial_capital: float = 500_000.0
    symbol: Optional[str] = None

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return DATA_DIR


for directory in (DATA_DIR, REFERENCE_DATA_DIR, RESEARCH_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

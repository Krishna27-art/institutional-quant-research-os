"""Simple gap fade strategy used for walk-forward validation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..config import GapFadeV2Config


@dataclass(slots=True)
class StrategyResult:
    returns: pd.Series
    signals: pd.Series
    positions: pd.Series


class GapFadeV2:
    """A compact, deterministic gap-fade strategy."""

    def __init__(self, config: Optional[GapFadeV2Config] = None):
        self.config = config or GapFadeV2Config()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        frame = data.copy()
        gap_pct = (frame["open"] - frame["close"].shift(1)) / frame["close"].shift(1) * 100.0
        signals = pd.Series(0, index=frame.index, dtype=int)
        signals[gap_pct <= self.config.gap_down_threshold_pct] = 1
        signals[gap_pct >= self.config.gap_up_threshold_pct] = -1
        return signals.fillna(0).astype(int)

    def backtest(self, data: pd.DataFrame) -> StrategyResult:
        frame = data.copy()
        signals = self.generate_signals(frame)
        returns = frame["close"].pct_change().fillna(0.0)
        positions = signals.shift(1).fillna(0).astype(float)
        strategy_returns = positions * returns
        return StrategyResult(strategy_returns, signals, positions)

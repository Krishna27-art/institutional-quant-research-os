"""Small legacy ORB backtester facade used by regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

import pandas as pd


@dataclass
class BacktestConfig:
    market_open: time = time(9, 15)
    orb_end: time = time(9, 20)
    market_close: time = time(15, 30)
    target_profit_pct: float = 0.015
    initial_capital: float = 10_000_000


class ORBBacktester:
    """Minimal path-exit simulator for the legacy ORB public API."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def _simulate_path_exit(
        self,
        path: pd.DataFrame,
        side: str,
        stop_loss: float,
        target: float,
    ) -> tuple[float, datetime, str]:
        if path.empty:
            raise ValueError("path cannot be empty")

        is_long = side.lower() == "long"
        for timestamp, bar in path.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            if is_long:
                if low <= stop_loss:
                    return float(stop_loss), timestamp, "stop_loss"
                if high >= target:
                    return float(target), timestamp, "target"
            else:
                if high >= stop_loss:
                    return float(stop_loss), timestamp, "stop_loss"
                if low <= target:
                    return float(target), timestamp, "target"

        last = path.iloc[-1]
        return float(last["close"]), path.index[-1], "end_of_day"


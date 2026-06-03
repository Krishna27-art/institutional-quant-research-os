"""Compatibility wrapper for the current ORB strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from .orb_zarattini import ORBBacktesterZarattini, ORBConfig


class ORBSignal(str, Enum):
    NO_SIGNAL = "NO_SIGNAL"
    LONG_BREAKOUT = "LONG_BREAKOUT"
    SHORT_BREAKOUT = "SHORT_BREAKOUT"


@dataclass(slots=True)
class ORBPosition:
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    rv: float | None = None
    atr: float | None = None


class ORBStrategy:
    """Thin strategy façade used by the orchestrator and vectorized backtester."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        orb_cfg = (config or {}).get("alpha", {}).get("orb", {})
        self.config = ORBConfig(
            orb_minutes=int(orb_cfg.get("orb_minutes", 5)),
            min_rv_threshold=float(orb_cfg.get("min_rv_threshold", 1.0)),
            high_rv_threshold=float(orb_cfg.get("high_rv_threshold", 3.0)),
        )
        self._backtester = ORBBacktesterZarattini(self.config)
        self._active_positions: dict[str, ORBPosition] = {}

    def scan_opening_range(self, intraday_data: dict[str, pd.DataFrame]) -> list[str]:
        candidates: list[str] = []
        for symbol, df in intraday_data.items():
            if len(df) < self.config.orb_minutes + 1:
                continue
            day_data = df.copy()
            if "volume" not in day_data.columns:
                continue
            orb_volume = float(day_data.iloc[: self.config.orb_minutes]["volume"].sum())
            avg_volume = float(day_data["volume"].rolling(20, min_periods=1).mean().iloc[-1])
            rv = self._backtester.calculate_relative_volume(orb_volume, avg_volume)
            if rv >= self.config.min_rv_threshold:
                candidates.append(symbol)
        return candidates

    def generate_signal(self, symbol: str, bar: pd.Series, regime_multiplier: float = 1.0):
        price = float(bar.get("close", bar.get("Close", bar.get("price", 0.0))))
        open_price = float(bar.get("open", bar.get("Open", price)))
        high = float(bar.get("high", bar.get("High", price)))
        low = float(bar.get("low", bar.get("Low", price)))
        if price > open_price and high > open_price * 1.001:
            position = ORBPosition(
                direction="long",
                entry_price=high,
                stop_loss=high * 0.995,
                target_price=high * 1.01,
            )
            self._active_positions[symbol] = position
            return ORBSignal.LONG_BREAKOUT, position
        if price < open_price and low < open_price * 0.999:
            position = ORBPosition(
                direction="short",
                entry_price=low,
                stop_loss=low * 1.005,
                target_price=low * 0.99,
            )
            self._active_positions[symbol] = position
            return ORBSignal.SHORT_BREAKOUT, position
        return ORBSignal.NO_SIGNAL, None

    def force_close_all(self, _: dict[str, Any]) -> None:
        self._active_positions.clear()


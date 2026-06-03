"""Compatibility wrapper for the current VWAP trend implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from .vwap_trend_zarattini import VWAPConfig


class VWAPSignal(str, Enum):
    NO_SIGNAL = "NO_SIGNAL"
    TREND_LONG = "TREND_LONG"
    TREND_SHORT = "TREND_SHORT"
    REVERSION_LONG = "REVERSION_LONG"
    REVERSION_SHORT = "REVERSION_SHORT"


@dataclass(slots=True)
class VWAPPosition:
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    vwap: float | None = None


class VWAPStrategy:
    """Thin façade used by the orchestrator and backtester."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        vwap_cfg = (config or {}).get("alpha", {}).get("vwap", {})
        self.config = VWAPConfig(
            vwap_threshold_sigma=float(vwap_cfg.get("vwap_threshold_sigma", 1.5)),
            min_holding_minutes=int(vwap_cfg.get("min_holding_minutes", 60)),
        )
        self._active_positions: dict[str, VWAPPosition] = {}

    def generate_signal(self, symbol: str, df: pd.DataFrame, vvg_label: str):
        if df.empty:
            return VWAPSignal.NO_SIGNAL, None

        bar = df.iloc[-1]
        price = float(bar.get("close", bar.get("Close", 0.0)))
        open_price = float(bar.get("open", bar.get("Open", price)))
        vwap = float((df["close"] * df["volume"]).cumsum().iloc[-1] / max(df["volume"].cumsum().iloc[-1], 1))

        if price > vwap and price > open_price:
            position = VWAPPosition(
                direction="long",
                entry_price=price,
                stop_loss=price * 0.995,
                target_price=price * 1.01,
                vwap=vwap,
            )
            self._active_positions[symbol] = position
            return VWAPSignal.TREND_LONG, position
        if price < vwap and price < open_price:
            position = VWAPPosition(
                direction="short",
                entry_price=price,
                stop_loss=price * 1.005,
                target_price=price * 0.99,
                vwap=vwap,
            )
            self._active_positions[symbol] = position
            return VWAPSignal.TREND_SHORT, position

        if vvg_label in {"choppy_volatile", "mean_reverting"}:
            if price < vwap:
                position = VWAPPosition("long", price, price * 0.995, price * 1.005, vwap)
                self._active_positions[symbol] = position
                return VWAPSignal.REVERSION_LONG, position
            position = VWAPPosition("short", price, price * 1.005, price * 0.995, vwap)
            self._active_positions[symbol] = position
            return VWAPSignal.REVERSION_SHORT, position

        return VWAPSignal.NO_SIGNAL, None

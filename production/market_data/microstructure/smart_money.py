"""Smart-money style structure signals without external dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class StructureSignal:
    signal_type: str
    direction: int
    strength: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyStructure:
    """Detect swing breaks, liquidity sweeps, and simple fair-value gaps."""

    def swing_points(self, frame: pd.DataFrame, window: int = 2) -> pd.DataFrame:
        df = frame.copy()
        df["swing_high"] = df["high"].eq(df["high"].rolling(window * 2 + 1, center=True).max())
        df["swing_low"] = df["low"].eq(df["low"].rolling(window * 2 + 1, center=True).min())
        return df

    def latest_break_of_structure(self, frame: pd.DataFrame, lookback: int = 20) -> StructureSignal | None:
        df = self.swing_points(frame).tail(lookback + 1)
        latest = df.iloc[-1]
        prior = df.iloc[:-1]
        swing_highs = prior.loc[prior["swing_high"], "high"]
        swing_lows = prior.loc[prior["swing_low"], "low"]
        if not swing_highs.empty and latest["close"] > swing_highs.max():
            return StructureSignal("break_of_structure", 1, 0.75, {"level": float(swing_highs.max())})
        if not swing_lows.empty and latest["close"] < swing_lows.min():
            return StructureSignal("break_of_structure", -1, 0.75, {"level": float(swing_lows.min())})
        return None

    def liquidity_sweep(self, frame: pd.DataFrame, lookback: int = 20) -> StructureSignal | None:
        df = frame.tail(lookback + 1)
        latest = df.iloc[-1]
        prior = df.iloc[:-1]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        if latest["high"] > prior_high and latest["close"] < prior_high:
            return StructureSignal("liquidity_sweep", -1, 0.8, {"swept_level": prior_high})
        if latest["low"] < prior_low and latest["close"] > prior_low:
            return StructureSignal("liquidity_sweep", 1, 0.8, {"swept_level": prior_low})
        return None

    def fair_value_gap(self, frame: pd.DataFrame) -> StructureSignal | None:
        if len(frame) < 3:
            return None
        a, _, c = frame.iloc[-3], frame.iloc[-2], frame.iloc[-1]
        if c["low"] > a["high"]:
            return StructureSignal("fair_value_gap", 1, 0.6, {"gap_low": float(a["high"]), "gap_high": float(c["low"])})
        if c["high"] < a["low"]:
            return StructureSignal("fair_value_gap", -1, 0.6, {"gap_low": float(c["high"]), "gap_high": float(a["low"])})
        return None


"""Intraday structure measurements."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class IntradayProfile:
    first_30m_range_fraction: float
    gap_fill_probability: float
    open_volume_share: float


class IntradayStructureAnalyzer:
    """Measure intraday behaviour from bar data."""

    def first_30m_range_fraction(self, intraday: pd.DataFrame) -> float:
        df = intraday.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        first_30 = df.groupby(df["timestamp"].dt.date).head(1)
        total_range = (df.groupby(df["timestamp"].dt.date)["high"].max() - df.groupby(df["timestamp"].dt.date)["low"].min()).replace(0, pd.NA)
        first_range = (first_30.groupby(first_30["timestamp"].dt.date)["high"].max() - first_30.groupby(first_30["timestamp"].dt.date)["low"].min())
        aligned = first_range.reindex(total_range.index)
        return float((aligned / total_range).dropna().mean())

    def gap_fill_probability(self, daily: pd.DataFrame, gap_threshold: float = 0.003) -> float:
        df = daily.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        prev_close = df["close"].shift()
        gap = (df["open"] - prev_close) / prev_close
        filled = ((df["low"] <= prev_close) & (gap.abs() >= gap_threshold)).astype(float)
        return float(filled.mean())

    def volume_share(self, intraday: pd.DataFrame, bucket_col: str = "bucket") -> pd.Series:
        if bucket_col not in intraday.columns:
            raise ValueError(f"Missing {bucket_col} column")
        grouped = intraday.groupby(bucket_col)["volume"].sum()
        return grouped / grouped.sum()

    def profile(self, daily: pd.DataFrame, intraday: pd.DataFrame) -> IntradayProfile:
        first_30m = self.first_30m_range_fraction(intraday)
        gap_fill = self.gap_fill_probability(daily)
        open_share = float(self.volume_share(intraday).iloc[0]) if not intraday.empty else 0.0
        return IntradayProfile(first_30m_range_fraction=first_30m, gap_fill_probability=gap_fill, open_volume_share=open_share)

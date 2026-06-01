"""Corporate action adjustment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .source import _to_naive_datetime


@dataclass(frozen=True, slots=True)
class CorporateAction:
    symbol: str
    date: pd.Timestamp
    action_type: str
    factor: float = 1.0
    cash_value: float = 0.0


class CorporateActionAdjuster:
    """Apply multiplicative price adjustments for splits and bonuses."""

    def adjust(self, frame: pd.DataFrame, actions: Iterable[CorporateAction]) -> pd.DataFrame:
        df = frame.copy()
        df["date"] = _to_naive_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for action in sorted(actions, key=lambda item: item.date):
            if action.factor <= 0:
                raise ValueError("corporate action factor must be positive")
            action_date = pd.Timestamp(action.date)
            if getattr(action_date, "tzinfo", None) is not None:
                action_date = action_date.tz_convert(None)
            mask = df["date"] < action_date
            if action.action_type.lower() in {"split", "bonus", "reverse_split"}:
                self._apply_factor(df, mask, action.factor)
        return df

    def _apply_factor(self, df: pd.DataFrame, mask: pd.Series, factor: float) -> None:
        price_cols = ["open", "high", "low", "close", "adjusted_close"]
        df.loc[mask, price_cols] = df.loc[mask, price_cols] / factor
        if "volume" in df.columns:
            df.loc[mask, "volume"] = df.loc[mask, "volume"] * factor

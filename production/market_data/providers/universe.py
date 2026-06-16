"""Point-in-time universe membership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .source import _to_naive_datetime


@dataclass(slots=True)
class UniverseMembership:
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp | None = None

    def active_on(self, date: str | pd.Timestamp) -> bool:
        ts = pd.Timestamp(date)
        if ts < self.start_date:
            return False
        if self.end_date is not None and ts > self.end_date:
            return False
        return True


class UniverseRegistry:
    """Load and query time-aware membership data."""

    def __init__(self, membership_file: str | Path | None = None) -> None:
        self.membership_file = Path(membership_file) if membership_file else None
        self._table = pd.DataFrame(columns=["symbol", "start_date", "end_date"])
        if self.membership_file and self.membership_file.exists():
            self.load(self.membership_file)

    def load(self, path: str | Path) -> None:
        df = pd.read_csv(path)
        required = {"symbol", "start_date"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Universe file missing columns: {sorted(missing)}")
        df["start_date"] = _to_naive_datetime(df["start_date"])
        if "end_date" in df.columns:
            df["end_date"] = _to_naive_datetime(df["end_date"])
        else:
            df["end_date"] = pd.NaT
        self._table = df[["symbol", "start_date", "end_date"]].copy()

    def symbols_on(self, date: str | pd.Timestamp) -> list[str]:
        ts = pd.Timestamp(date)
        active = self._table[
            (self._table["start_date"] <= ts)
            & ((self._table["end_date"].isna()) | (self._table["end_date"] >= ts))
        ]
        return sorted(active["symbol"].astype(str).unique().tolist())

    def is_active(self, symbol: str, date: str | pd.Timestamp) -> bool:
        return symbol.upper() in {item.upper() for item in self.symbols_on(date)}

    def to_frame(self) -> pd.DataFrame:
        return self._table.copy()

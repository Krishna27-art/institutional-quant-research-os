"""FII/DII participation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class ParticipationSnapshot:
    fii_net: float
    dii_net: float
    regime: str


class ParticipationAnalyzer:
    """Classify who is driving the market using flow series."""

    def classify(self, flows: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
        df = flows.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        df["fii_net_5d"] = df["fii_net"].rolling(rolling_window).sum()
        df["dii_net_5d"] = df["dii_net"].rolling(rolling_window).sum()
        ratio = df["fii_net_5d"].abs() / df["dii_net_5d"].abs().replace(0, pd.NA)
        df["regime"] = "balanced"
        df.loc[ratio > 1.5, "regime"] = "fii_dominant"
        df.loc[ratio < (1 / 1.5), "regime"] = "dii_dominant"
        return df

    def snapshot(self, fii_net: float, dii_net: float) -> ParticipationSnapshot:
        if abs(fii_net) > 1.5 * abs(dii_net):
            regime = "fii_dominant"
        elif abs(dii_net) > 1.5 * abs(fii_net):
            regime = "dii_dominant"
        else:
            regime = "balanced"
        return ParticipationSnapshot(fii_net=fii_net, dii_net=dii_net, regime=regime)

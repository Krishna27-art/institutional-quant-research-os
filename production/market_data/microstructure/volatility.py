"""Volatility estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class VolatilitySnapshot:
    realized_volatility: float
    ewma_volatility: float
    regime: str


class VolatilityForecaster:
    """Simple EWMA/GARCH-like volatility proxy for the first system slice."""

    def __init__(self, ewma_lambda: float = 0.94) -> None:
        self.ewma_lambda = ewma_lambda

    def realized_volatility(self, close: pd.Series, window: int = 20) -> pd.Series:
        returns = close.pct_change()
        return returns.rolling(window).std() * np.sqrt(252)

    def ewma_forecast(self, close: pd.Series) -> pd.Series:
        returns = close.pct_change().fillna(0.0)
        variance = []
        prev = returns.var()
        for r in returns:
            prev = self.ewma_lambda * prev + (1.0 - self.ewma_lambda) * (r * r)
            variance.append(prev)
        return pd.Series(np.sqrt(np.maximum(variance, 0.0)), index=close.index)

    def classify(self, daily_vol: float) -> str:
        if daily_vol < 0.008:
            return "low"
        if daily_vol < 0.015:
            return "medium"
        return "high"

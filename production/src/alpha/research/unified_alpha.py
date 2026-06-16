"""Unified trend/cycle alpha decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class TrendCycleDecomposition:
    """HP-filter trend/cycle decomposition."""

    trend: pd.Series
    cycle: pd.Series
    momentum_signal: pd.Series
    mean_reversion_signal: pd.Series


class UnifiedAlpha:
    """Decompose prices into momentum and mean-reversion components."""

    def __init__(self, lambda_hp: float = 1600.0, signal_window: int = 20) -> None:
        self.lambda_hp = lambda_hp
        self.signal_window = signal_window

    def hp_filter(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Solve (I + lambda D2' D2) trend = y with sparse linear algebra."""
        y = pd.Series(series, dtype=float).dropna()
        n = len(y)
        if n < 3:
            return y.copy(), pd.Series(0.0, index=y.index)
        identity = sparse.eye(n, format="csc")
        d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
        trend_values = sparse_linalg.spsolve(identity + self.lambda_hp * (d2.T @ d2), y.to_numpy(dtype=float))
        trend = pd.Series(trend_values, index=y.index)
        cycle = y - trend
        return trend, cycle

    def compute(self, close: pd.Series) -> TrendCycleDecomposition:
        """Return trend-following and mean-reversion signals."""
        trend, cycle = self.hp_filter(close)
        trend_return = trend.pct_change(self.signal_window).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cycle_scale = cycle.rolling(self.signal_window, min_periods=max(3, self.signal_window // 3)).std()
        cycle_z = (cycle / cycle_scale.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        momentum_signal = np.tanh(5.0 * trend_return)
        mean_reversion_signal = np.tanh(-cycle_z / 2.0)
        return TrendCycleDecomposition(
            trend=trend,
            cycle=cycle,
            momentum_signal=pd.Series(momentum_signal, index=trend.index).clip(-1.0, 1.0),
            mean_reversion_signal=pd.Series(mean_reversion_signal, index=trend.index).clip(-1.0, 1.0),
        )

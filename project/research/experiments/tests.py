"""Core statistical tests used by the research system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TestResults:
    t_stat: float
    p_value: float | None
    sharpe: float
    deflated_sharpe: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ADFResult:
    test_stat: float
    p_value: float | None
    lagged_coeff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def one_sample_t_test(values: Sequence[float], mu: float = 0.0) -> tuple[float, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0, None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std == 0:
        return 0.0, None
    t_stat = (mean - mu) / (std / sqrt(arr.size))
    return float(t_stat), None


def autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= lag:
        return 0.0
    x = arr[:-lag]
    y = arr[lag:]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def deflated_sharpe_ratio(values: Sequence[float], n_trials: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    sharpe = mean / std * np.sqrt(252)
    penalty = np.sqrt(2.0 * np.log(max(2, n_trials)) / arr.size)
    return float(sharpe - penalty)


def adf_like_test(values: Sequence[float]) -> ADFResult:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3:
        return ADFResult(0.0, None, 0.0)
    y = np.diff(arr)
    x = arr[:-1]
    x = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    lagged_coeff = float(beta[1])
    residuals = y - x @ beta
    denom = residuals.std(ddof=2)
    stat = lagged_coeff / denom if denom else 0.0
    return ADFResult(test_stat=float(stat), p_value=None, lagged_coeff=lagged_coeff)

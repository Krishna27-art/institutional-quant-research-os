"""Walk-forward splitting and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config


@dataclass(slots=True)
class WalkForwardResult:
    total_folds: int
    avg_sharpe: float
    fold_sharpes: List[float]


def walk_forward_split(data: pd.DataFrame, n_folds: int = 3, min_train_bars: int = 100):
    """Create expanding walk-forward splits."""
    n = len(data)
    if n <= min_train_bars:
        return []

    test_size = max((n - min_train_bars) // n_folds, 1)
    splits = []
    for fold in range(n_folds):
        train_end = min_train_bars + fold * test_size
        test_end = min(train_end + test_size, n)
        if train_end >= n or train_end >= test_end:
            break
        train = data.iloc[:train_end].copy()
        test = data.iloc[train_end:test_end].copy()
        splits.append((train, test))
    return splits


def contextual_walk_forward_split(data: pd.DataFrame, train_years: int = 2, test_years: int = 1):
    """Date-aware walk-forward split."""
    if data.empty:
        return []
    frame = data.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.sort_index()
    elif "Date" in frame.columns:
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.set_index("Date").sort_index()
    else:
        raise ValueError("Data must have a DatetimeIndex or a Date column")
    if frame.index.duplicated().any():
        frame = frame[~frame.index.duplicated(keep="first")]

    start = frame.index.min()
    end = frame.index.max()
    splits = []
    current_start = start
    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        train = frame.loc[current_start:train_end - pd.Timedelta(days=1)]
        test = frame.loc[train_end:test_end - pd.Timedelta(days=1)]
        if len(train) == 0 or len(test) == 0:
            break
        label = f"Train {train.index.min().date()} to {train.index.max().date()} | Test {test.index.min().date()} to {test.index.max().date()}"
        splits.append((train, test, label))
        current_start = test_end
        if current_start >= end:
            break
    return splits


def _sharpe(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))


def run_walk_forward(data: pd.DataFrame, strategy, cfg: Config, symbol: str, n_folds: int = 3, final_test_start: str | None = None):
    """Run a simple expanding walk-forward evaluation."""
    splits = walk_forward_split(data, n_folds=n_folds, min_train_bars=cfg.walk_forward.min_train_bars)
    fold_sharpes: List[float] = []
    for train, test in splits:
        eval_frame = pd.concat([train.tail(1), test], axis=0)
        if hasattr(strategy, "backtest"):
            result = strategy.backtest(eval_frame)
            fold_sharpes.append(_sharpe(result.returns))
        else:
            fold_sharpes.append(0.0)
    avg_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    return WalkForwardResult(total_folds=len(fold_sharpes), avg_sharpe=avg_sharpe, fold_sharpes=fold_sharpes)

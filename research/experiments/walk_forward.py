"""Purged walk-forward evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .tests import autocorrelation, deflated_sharpe_ratio


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_id: int
    start_index: int
    end_index: int
    mean_return: float
    std_return: float
    sharpe: float
    deflated_sharpe: float
    win_mechanism_mean: float
    loss_mechanism_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    windows: tuple[WalkForwardWindow, ...]
    summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "windows": [window.to_dict() for window in self.windows],
            "summary": dict(self.summary),
        }


class WalkForwardAnalyzer:
    """Evaluate a trade series in sequential purged windows."""

    def __init__(self, train_size: int = 100, test_size: int = 20, embargo: int = 0) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo

    def split_indices(self, n_rows: int) -> list[tuple[int, int, int, int]]:
        windows = []
        start = 0
        window_id = 0
        while start + self.train_size + self.test_size <= n_rows:
            train_start = start
            train_end = start + self.train_size
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size
            if test_end > n_rows:
                break
            windows.append((window_id, test_start, test_end, train_start, train_end))
            window_id += 1
            start = test_end
        return windows

    def analyze(
        self,
        frame: pd.DataFrame,
        return_col: str = "return",
        mechanism_col: str = "mechanism_score",
    ) -> WalkForwardResult:
        df = frame.copy().reset_index(drop=True)
        if return_col not in df.columns:
            raise ValueError(f"Missing required return column: {return_col}")
        if mechanism_col not in df.columns:
            df[mechanism_col] = 0.0

        windows: list[WalkForwardWindow] = []
        for window_id, test_start, test_end, *_ in self.split_indices(len(df)):
            test = df.iloc[test_start:test_end]
            returns = test[return_col].astype(float).to_numpy()
            mechanism = test[mechanism_col].astype(float).to_numpy()
            win_mask = returns > 0
            loss_mask = returns <= 0
            mean_return = float(np.mean(returns)) if returns.size else 0.0
            std_return = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
            sharpe = mean_return / std_return * np.sqrt(252) if std_return else 0.0
            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    start_index=test_start,
                    end_index=test_end,
                    mean_return=mean_return,
                    std_return=std_return,
                    sharpe=sharpe,
                    deflated_sharpe=deflated_sharpe_ratio(returns, n_trials=max(1, len(windows) + 1)),
                    win_mechanism_mean=float(mechanism[win_mask].mean()) if win_mask.any() else 0.0,
                    loss_mechanism_mean=float(mechanism[loss_mask].mean()) if loss_mask.any() else 0.0,
                )
            )

        all_returns = df[return_col].astype(float).to_numpy()
        summary = {
            "mean_return": float(np.mean(all_returns)) if all_returns.size else 0.0,
            "autocorrelation_1": autocorrelation(all_returns, lag=1),
            "deflated_sharpe": deflated_sharpe_ratio(all_returns, n_trials=max(1, len(windows))),
        }
        return WalkForwardResult(windows=tuple(windows), summary=summary)

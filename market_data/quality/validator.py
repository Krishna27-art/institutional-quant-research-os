"""Data validation and quality scoring."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .source import STANDARD_COLUMNS
from .source import _to_naive_datetime


@dataclass(slots=True)
class DataQualityReport:
    score: float
    issues: list[str] = field(default_factory=list)
    flagged_dates: list[pd.Timestamp] = field(default_factory=list)
    row_count: int = 0
    coverage_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "issues": list(self.issues),
            "flagged_dates": [str(d.date()) for d in self.flagged_dates],
            "row_count": self.row_count,
            "coverage_ratio": self.coverage_ratio,
        }


class DataValidator:
    """Validate OHLCV integrity before research use."""

    def __init__(self, max_gap_pct: float = 0.20, stale_run_limit: int = 3) -> None:
        self.max_gap_pct = max_gap_pct
        self.stale_run_limit = stale_run_limit

    def validate(self, frame: pd.DataFrame, expected_sessions: Iterable[str] | None = None) -> DataQualityReport:
        df = frame.copy()
        issues: list[str] = []
        flagged: list[pd.Timestamp] = []
        score = 100.0

        missing = [col for col in STANDARD_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Frame missing required columns: {missing}")

        df["date"] = _to_naive_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        duplicated = df["date"].duplicated()
        if duplicated.any():
            count = int(duplicated.sum())
            issues.append(f"duplicate_dates={count}")
            flagged.extend(df.loc[duplicated, "date"].tolist())
            score -= min(20.0, count * 2.0)

        if (df["close"] <= 0).any() or (df["open"] <= 0).any() or (df["high"] <= 0).any() or (df["low"] <= 0).any():
            issues.append("non_positive_prices")
            score -= 20.0

        if (df["volume"] <= 0).any():
            bad = int((df["volume"] <= 0).sum())
            issues.append(f"non_positive_volume={bad}")
            flagged.extend(df.loc[df["volume"] <= 0, "date"].tolist())
            score -= min(20.0, bad * 2.0)

        stale_mask = (
            df[["open", "high", "low", "close", "volume"]]
            .eq(df[["open", "high", "low", "close", "volume"]].shift())
            .all(axis=1)
        )
        stale_runs = self._run_lengths(stale_mask)
        if stale_runs:
            worst_run = max(stale_runs)
            if worst_run >= self.stale_run_limit:
                issues.append(f"stale_price_run={worst_run}")
                score -= min(15.0, worst_run * 3.0)

        pct_change = df["close"].pct_change().abs()
        extreme = pct_change > self.max_gap_pct
        if extreme.any():
            count = int(extreme.sum())
            issues.append(f"extreme_price_moves={count}")
            flagged.extend(df.loc[extreme, "date"].tolist())
            score -= min(20.0, count * 1.5)

        if expected_sessions is not None:
            expected = _to_naive_datetime(pd.Index(expected_sessions))
            observed = pd.Index(df["date"])
            missing_sessions = expected.difference(observed)
            if len(missing_sessions) > 0:
                issues.append(f"missing_sessions={len(missing_sessions)}")
                score -= min(20.0, len(missing_sessions) * 0.5)
                flagged.extend(list(missing_sessions[:10]))

        score = max(0.0, min(100.0, score))
        coverage_ratio = 1.0 if expected_sessions is None else 1.0 - (len(set(flagged)) / max(1, len(set(pd.to_datetime(expected_sessions)))))
        return DataQualityReport(
            score=score,
            issues=issues,
            flagged_dates=sorted(set(flagged)),
            row_count=len(df),
            coverage_ratio=max(0.0, min(1.0, coverage_ratio)),
        )

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        report = self.validate(frame)
        if report.score < 60:
            raise ValueError(f"Data quality too low: {report.score:.1f}")
        return frame.copy()

    @staticmethod
    def _run_lengths(mask: pd.Series) -> list[int]:
        runs: list[int] = []
        current = 0
        for value in mask.fillna(False):
            if value:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        if current:
            runs.append(current)
        return runs

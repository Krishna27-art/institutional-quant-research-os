"""Leakage and feature validation checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class LeakageReport:
    passed: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeakageGuard:
    """Detect common future-leak patterns in feature matrices."""

    def validate_feature_dates(self, frame: pd.DataFrame, feature_time_col: str = "feature_timestamp", decision_time_col: str = "decision_timestamp") -> LeakageReport:
        if feature_time_col not in frame.columns or decision_time_col not in frame.columns:
            return LeakageReport(True, metrics={"checked_rows": float(len(frame))})
        feature_time = pd.to_datetime(frame[feature_time_col], errors="coerce")
        decision_time = pd.to_datetime(frame[decision_time_col], errors="coerce")
        leaks = feature_time > decision_time
        issues = tuple(f"future_feature_row={idx}" for idx in frame.index[leaks].tolist()[:20])
        return LeakageReport(passed=not leaks.any(), issues=issues, metrics={"leak_rows": float(leaks.sum()), "checked_rows": float(len(frame))})

    def validate_target_shift(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str, max_abs_corr: float = 0.98) -> LeakageReport:
        if target_col not in frame.columns:
            raise ValueError(f"Missing target column: {target_col}")
        target = pd.to_numeric(frame[target_col], errors="coerce")
        issues: list[str] = []
        metrics: dict[str, float] = {}
        for col in feature_cols:
            if col not in frame.columns:
                continue
            feature = pd.to_numeric(frame[col], errors="coerce")
            valid = pd.concat([feature, target], axis=1).dropna()
            if len(valid) < 5 or valid.iloc[:, 0].std() == 0 or valid.iloc[:, 1].std() == 0:
                continue
            corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
            metrics[f"corr_{col}"] = corr
            if abs(corr) >= max_abs_corr:
                issues.append(f"{col}:suspicious_target_correlation={corr:.3f}")
        return LeakageReport(passed=not issues, issues=tuple(issues), metrics=metrics)


@dataclass(frozen=True, slots=True)
class FeatureValidationReport:
    passed: bool
    issues: tuple[str, ...]
    missing_rate: dict[str, float]
    constant_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureValidator:
    """Check feature matrix quality before statistical validation."""

    def validate(self, frame: pd.DataFrame, feature_cols: list[str], max_missing_rate: float = 0.05) -> FeatureValidationReport:
        issues: list[str] = []
        missing_rate: dict[str, float] = {}
        constant: list[str] = []
        for col in feature_cols:
            if col not in frame.columns:
                issues.append(f"{col}:missing_column")
                continue
            series = frame[col]
            rate = float(series.isna().mean())
            missing_rate[col] = rate
            if rate > max_missing_rate:
                issues.append(f"{col}:missing_rate={rate:.3f}")
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.nunique(dropna=True) <= 1:
                constant.append(col)
                issues.append(f"{col}:constant")
        return FeatureValidationReport(passed=not issues, issues=tuple(issues), missing_rate=missing_rate, constant_features=tuple(constant))


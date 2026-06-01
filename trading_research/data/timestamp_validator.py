"""Timestamp validation for anti-leakage checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import pandas as pd


@dataclass(slots=True)
class TimestampValidationResult:
    is_valid: bool
    future_timestamps: int
    issues: List[str] = field(default_factory=list)


class TimestampValidator:
    """Detects future timestamps in time series data."""

    def __init__(self, block_on_failure: bool = True, enforce_ist: bool = False):
        self.block_on_failure = block_on_failure
        self.enforce_ist = enforce_ist

    def validate(self, data: pd.DataFrame, symbol: str) -> TimestampValidationResult:
        if data.index.empty:
            return TimestampValidationResult(True, 0, [])

        idx = pd.to_datetime(data.index)
        now = pd.Timestamp.now(tz="Asia/Kolkata" if self.enforce_ist else None)
        if getattr(idx, "tz", None) is not None and now.tzinfo is None:
            now = now.tz_localize(idx.tz)
        if getattr(idx, "tz", None) is None and now.tzinfo is not None:
            now = now.tz_convert(None)
        future_mask = idx > now
        future_count = int(future_mask.sum())
        issues: List[str] = []
        if future_count:
            issues.append(f"{symbol}: found {future_count} future timestamps")
        return TimestampValidationResult(is_valid=future_count == 0, future_timestamps=future_count, issues=issues)

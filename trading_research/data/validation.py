"""High-level data validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .timestamp_validator import TimestampValidator


@dataclass(slots=True)
class DataValidationResult:
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    timestamp_result: object | None = None


class DataValidator:
    """Minimal validator used by the research tests."""

    def __init__(self, block_on_failure: bool = True, enforce_ist: bool = False):
        self.timestamp_validator = TimestampValidator(
            block_on_failure=block_on_failure,
            enforce_ist=enforce_ist,
        )

    def validate(self, data: pd.DataFrame, symbol: str) -> DataValidationResult:
        timestamp_result = self.timestamp_validator.validate(data, symbol)
        issues = list(timestamp_result.issues)
        return DataValidationResult(
            is_valid=timestamp_result.is_valid,
            issues=issues,
            timestamp_result=timestamp_result,
        )

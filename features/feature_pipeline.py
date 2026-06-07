"""Legacy-compatible point-in-time feature pipeline facade."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from market_data.feature_generation.feature_pipeline import (
    FeatureConfig as _FeatureConfig,
    FeaturePipeline as _FeaturePipeline,
)


@dataclass(slots=True)
class FeatureConfig(_FeatureConfig):
    """Feature config that accepts historical leakage flags.

    The canonical pipeline no longer needs these switches, but older callers
    still pass them. Keeping them here avoids silent API breakage.
    """

    enable_leakage_detection: bool = True
    enable_psi_detection: bool = True
    enable_future_info_check: bool = True


class FeaturePipeline(_FeaturePipeline):
    """Feature pipeline with support for the old `(symbol, data)` signature."""

    def compute_features(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        timestamp = kwargs.pop("timestamp", None)

        if len(args) >= 2 and isinstance(args[0], str):
            frame = self._point_in_time_frame(args[1], timestamp)
            return super().compute_features(frame, timestamp=timestamp, **kwargs)

        if args:
            return super().compute_features(args[0], timestamp=timestamp, **kwargs)

        return super().compute_features(timestamp=timestamp, **kwargs)

    def _point_in_time_frame(self, frame: pd.DataFrame, timestamp: datetime | None) -> pd.DataFrame:
        if frame.empty or timestamp is None:
            return frame

        normalized = self._normalize_frame(frame)
        if not isinstance(normalized.index, pd.DatetimeIndex):
            return normalized

        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return normalized.loc[normalized.index < ts]


"""Unified regime manager built on top of `hybrid_hmm_cpd`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from .hybrid_hmm_cpd import HybridHMMCPD


@dataclass(slots=True)
class RegimeSnapshot:
    """Standardized regime snapshot for downstream consumers."""

    regime: str
    probability: float
    is_change_point: bool
    change_point_confidence: float
    alpha_weights: dict[str, float]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegimeManager:
    """Wraps the hybrid HMM/CPD engine behind a clean interface."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = config or {}
        self.engine = HybridHMMCPD()
        self.current_snapshot: RegimeSnapshot | None = None
        self.is_fitted = False

    def fit(self, market_data: pd.DataFrame, window_days: int = 252) -> None:
        normalized = self._normalize(market_data)
        if normalized.empty:
            return
        try:
            self.engine.fit(normalized, window_days=window_days)
            self.is_fitted = True
        except Exception:
            # Synthetic or sparse data can produce non-PD covariances in HMM fitting.
            self.is_fitted = False
            self.current_snapshot = RegimeSnapshot(
                regime="sideways",
                probability=0.5,
                is_change_point=False,
                change_point_confidence=0.0,
                alpha_weights=dict(self.engine.regime_weights.get("sideways", {})),
                timestamp=self._latest_timestamp(normalized),
            )

    def detect(self, market_data: pd.DataFrame) -> RegimeSnapshot:
        normalized = self._normalize(market_data)
        if not self.is_fitted:
            snapshot = self.current_snapshot
            if snapshot is not None:
                return snapshot
            snapshot = RegimeSnapshot(
                regime="sideways",
                probability=0.5,
                is_change_point=False,
                change_point_confidence=0.0,
                alpha_weights=dict(self.engine.regime_weights.get("sideways", {})),
                timestamp=self._latest_timestamp(normalized),
            )
            self.current_snapshot = snapshot
            return snapshot
        result = self.engine.detect_regime(normalized)
        snapshot = RegimeSnapshot(
            regime=str(result.dominant_regime),
            probability=float(max(result.regime_probabilities.values()) if result.regime_probabilities else 0.0),
            is_change_point=bool(result.is_change_point),
            change_point_confidence=float(result.change_point_confidence),
            alpha_weights=dict(result.alpha_weights),
            timestamp=self._latest_timestamp(normalized),
        )
        self.current_snapshot = snapshot
        return snapshot

    def get_alpha_weights(self, regime: str | None = None) -> dict[str, float]:
        if regime:
            regime_key = str(regime).lower()
            if regime_key in self.engine.regime_weights:
                return dict(self.engine.regime_weights[regime_key])
        if self.current_snapshot is not None and self.current_snapshot.regime in self.engine.regime_weights:
            return dict(self.engine.regime_weights[self.current_snapshot.regime])
        return dict(self.engine.regime_weights.get("sideways", {}))

    def _normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        renamed = frame.copy()
        rename_map = {col: str(col).strip().lower() for col in renamed.columns}
        renamed = renamed.rename(columns=rename_map)
        if not isinstance(renamed.index, pd.DatetimeIndex) and "timestamp" in renamed.columns:
            renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="coerce")
            renamed = renamed.set_index("timestamp")
        return renamed.sort_index()

    def _latest_timestamp(self, frame: pd.DataFrame) -> datetime:
        if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
            return frame.index[-1].to_pydatetime()
        return datetime.utcnow()

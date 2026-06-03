"""Unified data manager for ingestion, normalization, and PIT universe handling.

This module keeps the current `core.data_layer.DataManager` available while
adding a higher-level orchestration surface for the rebuilt architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

import asyncio

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from core.data_layer import DataManager as LegacyDataManager
except Exception:  # pragma: no cover - keep import-safe in minimal envs
    LegacyDataManager = None  # type: ignore[assignment]

from .nse_adapter import NSELibAdapter


@dataclass(slots=True)
class DataManagerConfig:
    """Configuration for the unified data manager."""

    primary_feed: str = "yahoo"
    fallback_feed: str = "yahoo"
    backfill_days: int = 5
    expected_freq: str = "1min"
    universe: list[str] = field(default_factory=list)
    universe_membership: dict[str, dict[str, str]] = field(default_factory=dict)


class DataManager:
    """Consolidates feed access, gap handling, and point-in-time utilities."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        raw = config or {}
        data_cfg = raw.get("data", {}) if isinstance(raw, Mapping) else {}
        self.config = DataManagerConfig(
            primary_feed=str(data_cfg.get("primary_feed", "yahoo")),
            fallback_feed=str(data_cfg.get("fallback_feed", "yahoo")),
            backfill_days=int(data_cfg.get("backfill_days", 5)),
            expected_freq=str(data_cfg.get("expected_freq", "1min")),
            universe=list(data_cfg.get("universe", [])),
            universe_membership=dict(data_cfg.get("universe_membership", {})),
        )
        self.adapter = NSELibAdapter()
        self._legacy = None
        if LegacyDataManager is not None:
            try:
                self._legacy = LegacyDataManager(raw)
            except Exception:
                # Optional dependencies such as Arctic are not always installed in research-only envs.
                self._legacy = None

    async def initialize_feeds(self) -> None:
        """Initialize the underlying feeds if the legacy data manager is present."""
        if self._legacy is None:
            return
        try:
            await self._legacy.initialize_feeds()
        except Exception:
            # Live credentials are optional for research/backtest mode.
            return

    async def get_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical data and normalize the column layout."""
        if self._legacy is not None:
            try:
                frame = await self._legacy.get_data(symbol, start, end, interval=interval, use_cache=use_cache)
            except Exception:
                frame = pd.DataFrame()
        else:
            frame = pd.DataFrame()
        return self._normalize_ohlcv(frame)

    def detect_gaps(self, data: pd.DataFrame, expected_freq: str | None = None) -> pd.DatetimeIndex:
        """Return missing timestamps using a point-in-time view of the index."""
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return pd.DatetimeIndex([])

        freq = expected_freq or self.config.expected_freq
        expected = pd.date_range(data.index.min(), data.index.max(), freq=freq)
        return expected.difference(data.index)

    def backfill_missing_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch data and fill timestamp gaps with forward-filled prices."""
        frame = self._run_async(self.get_data(symbol, start, end, interval=interval, use_cache=use_cache))
        if frame.empty:
            return frame

        gaps = self.detect_gaps(frame)
        if gaps.empty:
            return frame

        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq=self.config.expected_freq)
        backfilled = frame.reindex(full_index)
        price_cols = [col for col in ["open", "high", "low", "close", "vwap"] if col in backfilled.columns]
        backfilled[price_cols] = backfilled[price_cols].ffill()
        if "volume" in backfilled.columns:
            backfilled["volume"] = backfilled["volume"].fillna(0)
        return backfilled

    def apply_corporate_actions(
        self,
        data: pd.DataFrame,
        corporate_actions: Iterable[Mapping[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Apply split/dividend adjustments using a forward-safe price adjustment."""
        if data.empty or not corporate_actions:
            return data.copy()

        adjusted = data.copy()
        for action in corporate_actions:
            action_type = str(action.get("event_type", action.get("type", ""))).lower()
            effective_date = pd.to_datetime(action.get("date") or action.get("effective_date"), errors="coerce")
            if pd.isna(effective_date):
                continue

            mask = adjusted.index < effective_date
            if action_type in {"split", "stock_split"}:
                ratio = float(action.get("ratio", action.get("split_ratio", 1.0)) or 1.0)
                if ratio > 0:
                    for col in ["open", "high", "low", "close", "vwap"]:
                        if col in adjusted.columns:
                            adjusted.loc[mask, col] = adjusted.loc[mask, col] / ratio
                    if "volume" in adjusted.columns:
                        adjusted.loc[mask, "volume"] = adjusted.loc[mask, "volume"] * ratio
            elif action_type in {"dividend", "cash_dividend"}:
                amount = float(action.get("amount", action.get("dividend", 0.0)) or 0.0)
                for col in ["open", "high", "low", "close", "vwap"]:
                    if col in adjusted.columns:
                        adjusted.loc[mask, col] = np.maximum(adjusted.loc[mask, col] - amount, 0.01)

        return adjusted

    def build_point_in_time_universe(
        self,
        as_of: datetime,
        base_universe: Iterable[str] | None = None,
    ) -> list[str]:
        """Build a point-in-time universe from the configured membership metadata."""
        candidates = list(base_universe or self.config.universe)
        if not candidates:
            return []

        result: list[str] = []
        for symbol in candidates:
            membership = self.config.universe_membership.get(symbol, {})
            start = pd.to_datetime(membership.get("start"), errors="coerce")
            end = pd.to_datetime(membership.get("end"), errors="coerce")
            if (pd.isna(start) or as_of >= start) and (pd.isna(end) or as_of <= end):
                result.append(symbol)
        return result

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize an arbitrary OHLCV frame to lower-case column names."""
        return self._normalize_ohlcv(frame)

    def _normalize_ohlcv(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()

        renamed = frame.copy()
        rename_map = {col: str(col).strip().lower() for col in renamed.columns}
        renamed = renamed.rename(columns=rename_map)
        if not isinstance(renamed.index, pd.DatetimeIndex) and "timestamp" in renamed.columns:
            renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="coerce")
            renamed = renamed.set_index("timestamp")

        renamed = renamed.sort_index()
        return renamed

    def _run_async(self, coro):
        """Run a coroutine in a local event loop when one is not already running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("backfill_missing_data cannot be called from an active event loop")

        return asyncio.run(coro)

"""
Data Quality Gate
=================
Validates incoming market data before it enters the signal pipeline.

Every batch of market data — live tick, 1-minute bar, or daily OHLCV —
must pass through this gate before signals are computed. If ANY check
returns REJECT, the trading system MUST NOT generate orders.

This prevents the most common source of live trading losses: trading
on stale, corrupted, or split-unadjusted data.

Checks (in order of severity):
  1. Timestamp ordering  — REJECT  if timestamps not monotonically increasing
  2. Gap detection       — REJECT  if any gap > max_gap_minutes
  3. OHLC sanity         — REJECT  if high < low or close outside [low, high]
  4. Price spike         — REJECT  if |pct_change| > spike_atr_multiple * ATR
  5. Zero volume         — WARN    if > max_zero_vol_bars consecutive zero-vol bars
  6. Staleness           — REJECT  if last bar timestamp > staleness_seconds behind wall clock

Usage
-----
    gate = DataQualityGate()
    result, reports = gate.check(df, is_live=True)
    if result == QualityResult.REJECT:
        logger.error("Data rejected: %s", [r.reason for r in reports if r.result == QualityResult.REJECT])
        return  # Do NOT generate signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityResult(Enum):
    ACCEPT = "accept"
    WARN   = "warn"
    REJECT = "reject"

    def __lt__(self, other: "QualityResult") -> bool:
        order = {QualityResult.ACCEPT: 0, QualityResult.WARN: 1, QualityResult.REJECT: 2}
        return order[self] < order[other]


@dataclass
class QualityReport:
    """Result of a single quality check."""
    check_name: str
    result: QualityResult
    reason: str
    details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.result.value.upper()}] {self.check_name}: {self.reason}"


class DataQualityGate:
    """
    Market data quality gate — run before every signal computation.

    Parameters
    ----------
    max_gap_minutes : int
        Maximum allowed gap between consecutive bars. Bars with gaps
        larger than this indicate a data feed outage or missing data.
    spike_atr_multiple : float
        Price changes beyond this multiple of the 20-bar ATR are
        flagged as likely unadjusted corporate actions (splits, bonuses).
    max_zero_vol_bars : int
        Number of consecutive zero-volume bars before issuing a WARN.
    staleness_seconds : int
        Maximum allowed age of the last bar vs wall clock (live mode only).
    atr_window : int
        Rolling window (bars) for ATR computation in spike detection.
    """

    def __init__(
        self,
        max_gap_minutes: int = 5,
        spike_atr_multiple: float = 20.0,
        max_zero_vol_bars: int = 5,
        staleness_seconds: int = 120,
        atr_window: int = 20,
    ):
        self.max_gap_minutes = max_gap_minutes
        self.spike_atr_multiple = spike_atr_multiple
        self.max_zero_vol_bars = max_zero_vol_bars
        self.staleness_seconds = staleness_seconds
        self.atr_window = atr_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        data: pd.DataFrame,
        is_live: bool = False,
    ) -> Tuple[QualityResult, List[QualityReport]]:
        """
        Run all quality checks on a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            The index OR a 'time' column must hold datetime values.
        is_live : bool
            If True, also runs the staleness check against wall clock.

        Returns
        -------
        (worst_result, all_reports)
            worst_result : QualityResult — the most severe result across all checks.
            all_reports  : List[QualityReport] — one entry per check.
        """
        if data is None or data.empty:
            report = QualityReport(
                check_name="non_empty",
                result=QualityResult.REJECT,
                reason="DataFrame is None or empty",
            )
            return QualityResult.REJECT, [report]

        df = self._normalise(data)
        reports: List[QualityReport] = []

        reports.append(self.check_timestamp_ordering(df))
        reports.append(self.check_gap(df))
        reports.append(self.check_ohlc_sanity(df))
        reports.append(self.check_price_spike(df))
        reports.append(self.check_zero_volume(df))

        if is_live:
            reports.append(self.check_staleness(df))

        worst = max(r.result for r in reports)

        # Log summary
        rejects = [r for r in reports if r.result == QualityResult.REJECT]
        warns   = [r for r in reports if r.result == QualityResult.WARN]
        if rejects:
            logger.error(
                "Data quality REJECT (%d checks failed): %s",
                len(rejects), "; ".join(r.reason for r in rejects),
            )
        elif warns:
            logger.warning(
                "Data quality WARN (%d checks): %s",
                len(warns), "; ".join(r.reason for r in warns),
            )
        else:
            logger.debug("Data quality: all checks PASSED (%d bars)", len(df))

        return worst, reports

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_timestamp_ordering(self, data: pd.DataFrame) -> QualityReport:
        """REJECT if timestamps are not strictly monotonically increasing."""
        times = data["time"]
        if not times.is_monotonic_increasing:
            n_violations = int((times.diff() < pd.Timedelta(0)).sum())
            return QualityReport(
                check_name="timestamp_ordering",
                result=QualityResult.REJECT,
                reason=f"Timestamps not monotonically increasing "
                       f"({n_violations} violations)",
                details={"n_violations": n_violations},
            )
        return QualityReport(
            check_name="timestamp_ordering",
            result=QualityResult.ACCEPT,
            reason="Timestamps are monotonically increasing",
        )

    def check_gap(self, data: pd.DataFrame) -> QualityReport:
        """REJECT if any inter-bar gap exceeds max_gap_minutes."""
        if len(data) < 2:
            return QualityReport(
                check_name="gap_detection",
                result=QualityResult.ACCEPT,
                reason="Only 1 bar — gap check skipped",
            )
        diffs = data["time"].diff().dropna()
        max_diff = diffs.max()
        max_minutes = max_diff.total_seconds() / 60.0

        if max_minutes > self.max_gap_minutes:
            gap_loc = diffs.idxmax()
            return QualityReport(
                check_name="gap_detection",
                result=QualityResult.REJECT,
                reason=(
                    f"Data gap of {max_minutes:.1f} min detected "
                    f"(limit={self.max_gap_minutes} min)"
                ),
                details={
                    "max_gap_minutes": max_minutes,
                    "gap_at_index": str(gap_loc),
                },
            )
        return QualityReport(
            check_name="gap_detection",
            result=QualityResult.ACCEPT,
            reason=f"Max gap {max_minutes:.1f} min within limit",
        )

    def check_ohlc_sanity(self, data: pd.DataFrame) -> QualityReport:
        """REJECT if high < low, or if close is outside [low, high]."""
        bad_hl = data["high"] < data["low"]
        bad_c_lo = data["close"] < data["low"]
        bad_c_hi = data["close"] > data["high"]
        total_bad = int((bad_hl | bad_c_lo | bad_c_hi).sum())

        if total_bad > 0:
            return QualityReport(
                check_name="ohlc_sanity",
                result=QualityResult.REJECT,
                reason=f"{total_bad} bars fail OHLC sanity (high<low or close outside range)",
                details={
                    "n_high_lt_low": int(bad_hl.sum()),
                    "n_close_below_low": int(bad_c_lo.sum()),
                    "n_close_above_high": int(bad_c_hi.sum()),
                },
            )
        return QualityReport(
            check_name="ohlc_sanity",
            result=QualityResult.ACCEPT,
            reason="All OHLC relationships valid",
        )

    def check_price_spike(self, data: pd.DataFrame) -> QualityReport:
        """
        REJECT if any close-to-close return exceeds spike_atr_multiple * ATR.

        Large sudden price changes usually indicate:
        - Unadjusted split or bonus issue
        - Corrupt data point from feed
        - Exchange data error

        A legitimate ±20x ATR move is astronomically unlikely intraday.
        """
        if len(data) < self.atr_window + 2:
            return QualityReport(
                check_name="price_spike",
                result=QualityResult.ACCEPT,
                reason=f"Insufficient data for ATR ({len(data)} bars < {self.atr_window + 2})",
            )

        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift()).abs(),
            (data["low"]  - data["close"].shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.atr_window).mean()
        abs_change = data["close"].diff().abs()
        spike_mask = abs_change > self.spike_atr_multiple * atr

        # Ignore first atr_window bars where ATR isn't yet established
        spike_mask.iloc[: self.atr_window] = False
        n_spikes = int(spike_mask.sum())

        if n_spikes > 0:
            spike_rows = data.loc[spike_mask, ["time", "close"]].head(3)
            return QualityReport(
                check_name="price_spike",
                result=QualityResult.REJECT,
                reason=(
                    f"{n_spikes} price spike(s) detected "
                    f"(>{self.spike_atr_multiple}x ATR) — likely unadjusted "
                    f"corporate action or corrupt data"
                ),
                details={
                    "n_spikes": n_spikes,
                    "first_spike_times": spike_rows["time"].astype(str).tolist(),
                },
            )
        return QualityReport(
            check_name="price_spike",
            result=QualityResult.ACCEPT,
            reason=f"No price spikes detected (threshold: {self.spike_atr_multiple}x ATR)",
        )

    def check_zero_volume(self, data: pd.DataFrame) -> QualityReport:
        """WARN if there are more than max_zero_vol_bars consecutive zero-volume bars."""
        if "volume" not in data.columns:
            return QualityReport(
                check_name="zero_volume",
                result=QualityResult.ACCEPT,
                reason="Volume column not present — check skipped",
            )

        zero_vol = data["volume"] == 0
        if not zero_vol.any():
            return QualityReport(
                check_name="zero_volume",
                result=QualityResult.ACCEPT,
                reason="No zero-volume bars",
            )

        # Find max run of consecutive zero-volume bars
        max_run = 0
        current_run = 0
        for v in zero_vol:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        total_zero = int(zero_vol.sum())
        pct_zero = total_zero / len(data)

        if max_run > self.max_zero_vol_bars:
            return QualityReport(
                check_name="zero_volume",
                result=QualityResult.WARN,
                reason=(
                    f"Max run of {max_run} consecutive zero-volume bars "
                    f"(limit={self.max_zero_vol_bars}); {pct_zero:.1%} of bars have zero volume"
                ),
                details={"max_consecutive_zero": max_run, "total_zero_bars": total_zero},
            )
        return QualityReport(
            check_name="zero_volume",
            result=QualityResult.ACCEPT,
            reason=f"{total_zero} zero-vol bars ({pct_zero:.1%}) — within tolerance",
        )

    def check_staleness(self, data: pd.DataFrame) -> QualityReport:
        """
        REJECT if the last bar's timestamp is more than staleness_seconds
        behind the current wall clock.

        Should only be called in live trading mode (is_live=True).
        """
        if data.empty:
            return QualityReport(
                check_name="staleness",
                result=QualityResult.REJECT,
                reason="Empty DataFrame — cannot check staleness",
            )

        last_ts = data["time"].iloc[-1]
        # Normalise to UTC-naive for comparison
        if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
            last_ts = last_ts.astimezone(timezone.utc).replace(tzinfo=None)
        if isinstance(last_ts, pd.Timestamp):
            last_ts = last_ts.to_pydatetime()

        now_utc = datetime.utcnow()
        age_seconds = (now_utc - last_ts).total_seconds()

        if age_seconds > self.staleness_seconds:
            return QualityReport(
                check_name="staleness",
                result=QualityResult.REJECT,
                reason=(
                    f"Last bar is {age_seconds:.0f}s old "
                    f"(limit={self.staleness_seconds}s) — data feed may be down"
                ),
                details={
                    "last_bar_time": str(last_ts),
                    "age_seconds": age_seconds,
                    "limit_seconds": self.staleness_seconds,
                },
            )
        return QualityReport(
            check_name="staleness",
            result=QualityResult.ACCEPT,
            reason=f"Data is fresh ({age_seconds:.0f}s old)",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise column names to lowercase and ensure a 'time' column
        from either a datetime index or an existing time/datetime column.
        """
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]

        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "time"}, inplace=True)
            elif "datetime" in df.columns:
                df.rename(columns={"datetime": "time"}, inplace=True)
            elif "timestamp" in df.columns:
                df.rename(columns={"timestamp": "time"}, inplace=True)

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        return df

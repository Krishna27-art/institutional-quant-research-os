"""
Data Quality Gate — strict boolean filter for market data.

This is the GATE, not the monitor. Data that fails any rule is REJECTED.
No alpha, no feature, no model sees rejected data.

5 Rules (from transformation audit):
1. No future data (point-in-time enforcement)
2. OHLC consistency (high >= open,close >= low)
3. No stale prices (>2 identical closes = stale feed)
4. Volume sanity (must be > 0 on trading days)
5. Price continuity (no >20% overnight gap without corporate action)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of running data through the quality gate."""
    passed: bool
    symbol: str
    rows_in: int
    rows_out: int
    violations: List[str] = field(default_factory=list)
    violation_counts: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.symbol}: {self.rows_out}/{self.rows_in} rows passed | "
            f"violations: {self.violation_counts}"
        )


class DataQualityGate:
    """
    Hard data quality gate. Data that fails is dropped, not flagged.

    Usage:
        gate = DataQualityGate()
        clean_df, result = gate.validate(symbol, raw_df)
        if not result.passed:
            logger.error(f"Data rejected: {result}")
            return  # do NOT run alpha on this data
    """

    def __init__(
        self,
        max_stale_closes: int = 2,
        max_overnight_gap_pct: float = 0.20,
        min_rows: int = 50,
        drop_bad_rows: bool = True,
    ):
        self.max_stale_closes = max_stale_closes
        self.max_overnight_gap_pct = max_overnight_gap_pct
        self.min_rows = min_rows
        self.drop_bad_rows = drop_bad_rows

    def validate(
        self,
        symbol: str,
        df: pd.DataFrame,
        as_of: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, GateResult]:
        """
        Run all 5 quality rules on the dataframe.

        Args:
            symbol: Ticker symbol
            df: DataFrame with OHLCV columns (open, high, low, close, volume).
                Index should be DatetimeIndex.
            as_of: Point-in-time cutoff. Rows after this are dropped.

        Returns:
            (clean_df, GateResult) — clean_df may have fewer rows than input.
        """
        violations: List[str] = []
        violation_counts: dict = {}
        rows_in = len(df)

        if df.empty:
            return df, GateResult(
                passed=False,
                symbol=symbol,
                rows_in=0,
                rows_out=0,
                violations=["empty_dataframe"],
                violation_counts={"empty": 1},
            )

        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        available = set(str(c) for c in df.columns)
        missing = required - available
        if missing:
            return df, GateResult(
                passed=False,
                symbol=symbol,
                rows_in=rows_in,
                rows_out=0,
                violations=[f"missing_columns: {missing}"],
                violation_counts={"missing_columns": len(missing)},
            )

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return df, GateResult(
                    passed=False,
                    symbol=symbol,
                    rows_in=rows_in,
                    rows_out=0,
                    violations=["non_datetime_index"],
                    violation_counts={"non_datetime_index": 1},
                )

        # Sort chronologically
        df = df.sort_index()

        # ── RULE 1: No future data ──────────────────────────────────
        if as_of is not None:
            future_mask = df.index > pd.Timestamp(as_of)
            n_future = future_mask.sum()
            if n_future > 0:
                violations.append(f"rule1_future_data: {n_future} rows after as_of")
                violation_counts["future_data"] = int(n_future)
                df = df[~future_mask]

        # ── RULE 2: OHLC consistency ────────────────────────────────
        ohlc_bad = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        n_ohlc = ohlc_bad.sum()
        if n_ohlc > 0:
            violations.append(f"rule2_ohlc_inconsistency: {n_ohlc} bars")
            violation_counts["ohlc_bad"] = int(n_ohlc)
            if self.drop_bad_rows:
                df = df[~ohlc_bad]

        # ── RULE 3: No stale prices ────────────────────────────────
        if len(df) > self.max_stale_closes:
            close_vals = df["close"].values
            stale_mask = pd.Series(False, index=df.index)
            for i in range(self.max_stale_closes, len(close_vals)):
                window = close_vals[i - self.max_stale_closes : i + 1]
                if len(set(window)) == 1:
                    stale_mask.iloc[i] = True

            n_stale = stale_mask.sum()
            if n_stale > 0:
                violations.append(f"rule3_stale_prices: {n_stale} bars with >{self.max_stale_closes} identical closes")
                violation_counts["stale_prices"] = int(n_stale)
                if self.drop_bad_rows:
                    df = df[~stale_mask]

        # ── RULE 4: Volume sanity ──────────────────────────────────
        zero_vol = df["volume"] <= 0
        n_zero = zero_vol.sum()
        if n_zero > 0:
            violations.append(f"rule4_zero_volume: {n_zero} bars")
            violation_counts["zero_volume"] = int(n_zero)
            if self.drop_bad_rows:
                df = df[~zero_vol]

        # ── RULE 5: Price continuity ───────────────────────────────
        if len(df) >= 2:
            prev_close = df["close"].shift(1)
            overnight_gap = ((df["open"] - prev_close) / prev_close).abs()
            # Skip first row (no previous close)
            gap_mask = overnight_gap > self.max_overnight_gap_pct
            gap_mask.iloc[0] = False  # first row has no prev

            n_gap = gap_mask.sum()
            if n_gap > 0:
                violations.append(
                    f"rule5_overnight_gap: {n_gap} bars with >{self.max_overnight_gap_pct:.0%} gap"
                )
                violation_counts["overnight_gap"] = int(n_gap)
                # Don't drop — just flag. Corporate actions cause legitimate gaps.
                # TODO: integrate corporate_actions table to whitelist known splits/dividends

        # ── Final assessment ────────────────────────────────────────
        rows_out = len(df)
        passed = rows_out >= self.min_rows and len(violations) == 0

        # Allow pass if violations were only soft (rule 5 gaps, small counts)
        if not passed and rows_out >= self.min_rows:
            hard_violations = {
                k for k in violation_counts
                if k not in ("overnight_gap",)  # overnight gap is soft
            }
            if not hard_violations:
                passed = True

        result = GateResult(
            passed=passed,
            symbol=symbol,
            rows_in=rows_in,
            rows_out=rows_out,
            violations=violations,
            violation_counts=violation_counts,
        )

        if not passed:
            logger.warning(f"Data quality gate REJECTED {symbol}: {result}")
        else:
            logger.debug(f"Data quality gate passed for {symbol}: {result}")

        return df, result

    def should_halt_signals(
        self,
        df: pd.DataFrame,
        last_update_time: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Check if we should halt signal generation entirely:
        1. Price moves > 25%
        2. Volume is 0 during market hours
        3. Data is > 15 mins stale
        """
        if df.empty:
            return True, "No data available"

        now = datetime.now()

        # 1. Stale data check
        if last_update_time is not None:
            if (now - last_update_time).total_seconds() > 15 * 60:
                return True, f"Data is stale by {(now - last_update_time).total_seconds() / 60:.1f} minutes"

        # 2. Extreme price movement (> 25%)
        if len(df) >= 2:
            prev_close = df["close"].iloc[-2]
            curr_close = df["close"].iloc[-1]
            move = abs(curr_close - prev_close) / prev_close
            if move > 0.25:
                return True, f"Extreme price movement detected: {move:.1%}"

        # 3. Zero volume during market hours (9:15 AM - 3:30 PM)
        if 9 <= now.hour < 16:
            if df["volume"].iloc[-1] <= 0:
                return True, "Zero volume detected during active market hours"

        return False, ""


# ── Module-level singleton ──────────────────────────────────────────
_gate: Optional[DataQualityGate] = None


def get_quality_gate() -> DataQualityGate:
    """Get the singleton quality gate instance."""
    global _gate
    if _gate is None:
        _gate = DataQualityGate()
    return _gate

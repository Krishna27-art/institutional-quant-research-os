"""
Universe Enforcer — Runtime Point-in-Time Guard
================================================
Ensures every signal, trade, and position reference only symbols that
were actually in the universe on that specific historical date.

This is the enforcement layer that makes UniverseMembershipTracker
actually useful in backtests. Without this guard, code can silently
trade symbols that weren't listed/liquid at the time, producing
survivorship-biased and look-ahead-contaminated results.

Usage
-----
    tracker = UniverseMembershipTracker()
    tracker.load_nifty_50_history('data/raw/nifty50_constituents_history.csv')

    enforcer = UniverseEnforcer(tracker, UniverseType.NIFTY_50, strict=False)

    # Filter a signal DataFrame before backtesting:
    clean_signals = enforcer.filter_signals(signals_df, date_col='timestamp')

    # Audit without filtering:
    report = enforcer.audit_signals(signals_df, 'timestamp', 'symbol')
    print(f"Violation rate: {report['violation_pct']:.1%}")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd

from data.universe_membership_tracker import UniverseMembershipTracker, UniverseType

logger = logging.getLogger(__name__)


class UniverseEnforcer:
    """
    Runtime guard that filters or audits signal DataFrames to ensure
    point-in-time universe compliance.

    Parameters
    ----------
    tracker : UniverseMembershipTracker
        Loaded tracker containing historical membership records.
    universe_type : UniverseType
        The universe to enforce (e.g. NIFTY_50, NIFTY_100).
    strict : bool
        If True, raises ValueError when an unknown symbol is found.
        If False (default for backtesting), logs a WARNING and drops it.
    cache_resolution : str
        Pandas offset string for caching universe snapshots.
        'D' = daily (accurate but slower), 'W' = weekly (faster).
    """

    def __init__(
        self,
        tracker: UniverseMembershipTracker,
        universe_type: UniverseType,
        strict: bool = False,
        cache_resolution: str = "D",
    ):
        self.tracker = tracker
        self.universe_type = universe_type
        self.strict = strict
        self.cache_resolution = cache_resolution
        # Cache universe snapshots to avoid re-querying for the same date
        self._cache: Dict[str, Set[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        signals: pd.DataFrame,
        date_col: str = "timestamp",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Filter a signal DataFrame to only include rows where the symbol
        was in the universe on the signal's date.

        Vectorized: groups by date so each date is queried once.

        Parameters
        ----------
        signals : pd.DataFrame
            Must contain `date_col` (datetime-like) and `symbol_col` (str).
        date_col : str
            Column name for the signal date/timestamp.
        symbol_col : str
            Column name for the symbol.

        Returns
        -------
        pd.DataFrame
            Filtered copy with invalid rows removed.
        """
        if signals.empty:
            return signals.copy()

        original_count = len(signals)
        valid_mask = self._build_validity_mask(signals, date_col, symbol_col)

        dropped = (~valid_mask).sum()
        if dropped > 0:
            msg = (
                f"UniverseEnforcer dropped {dropped}/{original_count} signals "
                f"({dropped/original_count:.1%}) not in {self.universe_type.value} "
                f"universe at their signal date."
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        return signals.loc[valid_mask].copy()

    def is_valid(self, symbol: str, date: datetime) -> bool:
        """
        Check if a single symbol was in the universe on a given date.

        Parameters
        ----------
        symbol : str
            Ticker symbol (case-insensitive match).
        date : datetime
            The point-in-time date to check.
        """
        universe = self._get_universe_on_date(date)
        return symbol.upper() in {s.upper() for s in universe}

    def audit_signals(
        self,
        signals: pd.DataFrame,
        date_col: str,
        symbol_col: str,
    ) -> Dict:
        """
        Audit a signal DataFrame for universe violations without modifying it.

        Returns a report dict suitable for logging or display.

        Returns
        -------
        dict with keys:
            total_signals : int
            violations : int
            violation_pct : float
            offending_symbols : List[str]  (unique symbols causing violations)
            offending_dates : List[str]    (unique dates with violations)
        """
        if signals.empty:
            return {
                "total_signals": 0,
                "violations": 0,
                "violation_pct": 0.0,
                "offending_symbols": [],
                "offending_dates": [],
            }

        valid_mask = self._build_validity_mask(signals, date_col, symbol_col)
        invalid = signals.loc[~valid_mask]

        return {
            "total_signals": len(signals),
            "violations": int((~valid_mask).sum()),
            "violation_pct": float((~valid_mask).sum() / len(signals)),
            "offending_symbols": sorted(invalid[symbol_col].unique().tolist()),
            "offending_dates": sorted(
                invalid[date_col].astype(str).unique().tolist()
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_validity_mask(
        self,
        signals: pd.DataFrame,
        date_col: str,
        symbol_col: str,
    ) -> pd.Series:
        """
        Build a boolean Series (same index as signals) marking valid rows.
        Groups by date bucket to minimise universe queries.
        """
        # Normalise date column to a consistent resolution for caching
        dates_series = pd.to_datetime(signals[date_col]).dt.normalize()
        symbols_series = signals[symbol_col].str.upper()

        valid = pd.Series(False, index=signals.index)

        for date_val, group_idx in dates_series.groupby(dates_series).groups.items():
            universe = self._get_universe_on_date(date_val.to_pydatetime())
            upper_universe = {s.upper() for s in universe}
            group_symbols = symbols_series.loc[group_idx]
            valid.loc[group_idx] = group_symbols.isin(upper_universe)

        return valid

    def _get_universe_on_date(self, date: datetime) -> Set[str]:
        """
        Fetch universe for a given date, using an in-memory cache.
        Cache key is the ISO date string (daily resolution).
        """
        key = date.strftime("%Y-%m-%d")
        if key not in self._cache:
            self._cache[key] = self.tracker.get_universe_at_date(
                self.universe_type, date
            )
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the in-memory date cache (call if tracker is updated)."""
        self._cache.clear()
        logger.debug("UniverseEnforcer cache cleared.")

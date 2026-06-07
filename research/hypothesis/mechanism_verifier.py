"""Mechanism verification system for gap fade strategy.

This module validates whether the claimed behavioral mechanism was actually
present during a trade, independent of whether the trade made money.

A trade that made money for the wrong reason is dangerous.
A trade that lost money but confirmed the thesis is valuable data.

This is NOT strategy validation. This is mechanism validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MechanismVerificationResult:
    """Result of mechanism verification for a single trade."""

    mechanism_claimed: str
    mechanism_confirmed: bool
    confidence: float

    # Timing metrics
    fade_time_minutes: Optional[float]
    max_fade_pct: float
    fade_velocity: float

    # Behavioral metrics
    volume_pattern_match: bool
    price_pattern_match: bool
    participant_regime_match: bool

    # Overall assessment
    thesis_correct: bool
    pnl_independent: bool  # Would thesis be correct regardless of exit?

    def to_dict(self) -> Dict:
        return {
            "mechanism_claimed": self.mechanism_claimed,
            "mechanism_confirmed": self.mechanism_confirmed,
            "confidence": self.confidence,
            "fade_time_minutes": self.fade_time_minutes,
            "max_fade_pct": self.max_fade_pct,
            "fade_velocity": self.fade_velocity,
            "volume_pattern_match": self.volume_pattern_match,
            "price_pattern_match": self.price_pattern_match,
            "participant_regime_match": self.participant_regime_match,
            "thesis_correct": self.thesis_correct,
            "pnl_independent": self.pnl_independent,
        }


class MechanismVerifier:
    """Verifies whether the claimed mechanism was actually present.

    For gap fade, this means:
    - Was retail panic actually present?
    - Did the gap actually fade (regardless of our exit)?
    - How fast did it fade?
    - What was the actual participant behavior?

    These two things are not the same:
    1. Did we make money?
    2. Was the mechanism active?
    """

    def __init__(
        self,
        fade_time_threshold_minutes: float = 45.0,
        min_fade_pct: float = 0.5,
        min_volume_shock: float = 1.5,
    ):
        self.fade_time_threshold = fade_time_threshold_minutes
        self.min_fade_pct = min_fade_pct
        self.min_volume_shock = min_volume_shock

    def verify_gap_fade(
        self,
        trade: Dict,
        intraday_data: pd.DataFrame,
        participant_regime: Optional[str] = None,
    ) -> MechanismVerificationResult:
        """Verify gap fade mechanism for a single trade.

        Args:
            trade: Trade dictionary with keys:
                - timestamp: Trade entry time
                - direction: 1 for long, -1 for short
                - entry_price: Entry price
                - gap_pct: Gap percentage
                - mechanism_claimed: Claimed mechanism (e.g., "retail_panic")
            intraday_data: Intraday OHLCV data for the trade day
            participant_regime: Participant regime from classifier (optional)

        Returns:
            MechanismVerificationResult with verification details
        """
        mechanism_claimed = trade.get("mechanism_claimed", "unknown")
        direction = trade.get("direction", 1)
        entry_price = trade.get("entry_price")
        gap_pct = trade.get("gap_pct", 0.0)

        # Extract intraday data for the trade day
        timestamp = trade.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        # Filter intraday data for the trade day
        trade_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
        day_data = intraday_data[
            intraday_data.index.date == trade_date
        ] if hasattr(intraday_data.index, 'date') else intraday_data

        if day_data.empty:
            return MechanismVerificationResult(
                mechanism_claimed=mechanism_claimed,
                mechanism_confirmed=False,
                confidence=0.0,
                fade_time_minutes=None,
                max_fade_pct=0.0,
                fade_velocity=0.0,
                volume_pattern_match=False,
                price_pattern_match=False,
                participant_regime_match=False,
                thesis_correct=False,
                pnl_independent=False,
            )

        # Measure fade time and magnitude
        fade_time, max_fade = self._measure_fade(
            day_data, entry_price, direction, gap_pct
        )

        # Calculate fade velocity (fade % per minute)
        if fade_time is not None and fade_time > 0:
            fade_velocity = max_fade / fade_time
        else:
            fade_velocity = 0.0

        # Verify volume pattern
        volume_match = self._verify_volume_pattern(day_data, trade)

        # Verify price pattern
        price_match = self._verify_price_pattern(day_data, trade)

        # Verify participant regime match
        regime_match = self._verify_regime_match(participant_regime, mechanism_claimed)

        # Determine if mechanism was confirmed
        mechanism_confirmed = self._determine_mechanism_confirmed(
            fade_time, max_fade, volume_match, price_match, regime_match
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            fade_time, max_fade, volume_match, price_match, regime_match
        )

        # Determine if thesis was correct (independent of P&L)
        thesis_correct = mechanism_confirmed and regime_match

        # Determine if result is P&L independent
        # (mechanism would be confirmed regardless of our exit timing)
        pnl_independent = (
            fade_time is not None 
            and fade_time <= self.fade_time_threshold
            and max_fade >= self.min_fade_pct
        )

        return MechanismVerificationResult(
            mechanism_claimed=mechanism_claimed,
            mechanism_confirmed=mechanism_confirmed,
            confidence=confidence,
            fade_time_minutes=fade_time,
            max_fade_pct=max_fade,
            fade_velocity=fade_velocity,
            volume_pattern_match=volume_match,
            price_pattern_match=price_match,
            participant_regime_match=regime_match,
            thesis_correct=thesis_correct,
            pnl_independent=pnl_independent,
        )

    def _measure_fade(
        self,
        day_data: pd.DataFrame,
        entry_price: float,
        direction: int,
        gap_pct: float,
    ) -> Tuple[Optional[float], float]:
        """Measure fade time and maximum fade magnitude.

        For long (gap down fade): time to reach previous close
        For short (gap up fade): time to reach previous close
        """
        if entry_price is None:
            return None, 0.0

        # Calculate target (previous close)
        if direction == 1:  # Long - gap down, target is previous close (higher)
            if gap_pct < 0:
                target_price = entry_price / (1 + gap_pct / 100.0)
            else:
                target_price = entry_price
        else:  # Short - gap up, target is previous close (lower)
            if gap_pct > 0:
                target_price = entry_price / (1 + gap_pct / 100.0)
            else:
                target_price = entry_price

        # Find when target was hit
        if direction == 1:
            # Long: looking for high >= target
            hit_mask = day_data["high"] >= target_price
        else:
            # Short: looking for low <= target
            hit_mask = day_data["low"] <= target_price

        if hit_mask.any():
            hit_idx = hit_mask.idxmax()
            # Convert to minutes from open (assuming minute data)
            if hasattr(hit_idx, 'time'):
                fade_time = hit_idx.hour * 60 + hit_idx.minute - 570  # 9:30 = 570 minutes
            else:
                # If index is integer, assume it's minute bars from open
                fade_time = float(hit_idx)
        else:
            fade_time = None

        # Calculate maximum fade achieved
        if direction == 1:
            # Long: max close - entry
            max_fade = ((day_data["close"].max() - entry_price) / entry_price) * 100.0
        else:
            # Short: entry - min close
            max_fade = ((entry_price - day_data["close"].min()) / entry_price) * 100.0

        return fade_time, max_fade

    def _verify_volume_pattern(
        self,
        day_data: pd.DataFrame,
        trade: Dict,
    ) -> bool:
        """Verify volume pattern matches claimed mechanism.

        Retail panic should show:
        - Elevated volume in first 15-30 minutes
        - Volume tapering off after initial surge
        """
        if "volume" not in day_data.columns:
            return True  # Can't verify, assume match

        # Check first 30 minutes volume
        first_30_min = day_data.head(30)
        if len(first_30_min) == 0:
            return True

        first_30_vol = first_30_min["volume"].sum()
        avg_vol = day_data["volume"].mean()

        if avg_vol == 0:
            return True

        volume_ratio = (first_30_vol / 30) / avg_vol

        # Retail panic should have elevated opening volume
        return volume_ratio >= self.min_volume_shock

    def _verify_price_pattern(
        self,
        day_data: pd.DataFrame,
        trade: Dict,
    ) -> bool:
        """Verify price pattern matches claimed mechanism.

        Retail panic fade should show:
        - Initial gap at open
        - Reversal within first 30-45 minutes
        - Not continuing in gap direction
        """
        direction = trade.get("direction", 1)
        gap_pct = trade.get("gap_pct", 0.0)

        if len(day_data) < 30:
            return True  # Can't verify with limited data

        # Check first 30 minutes trend
        first_30_close = day_data.iloc[29]["close"]
        first_30_open = day_data.iloc[0]["open"]

        if direction == 1:  # Long - should see upward movement
            first_30_return = (first_30_close - first_30_open) / first_30_open
            return first_30_return > 0
        else:  # Short - should see downward movement
            first_30_return = (first_30_close - first_30_open) / first_30_open
            return first_30_return < 0

    def _verify_regime_match(
        self,
        participant_regime: Optional[str],
        mechanism_claimed: str,
    ) -> bool:
        """Verify participant regime matches claimed mechanism."""
        if participant_regime is None:
            return True  # Can't verify, assume match

        # Map mechanisms to regimes
        mechanism_to_regime = {
            "retail_panic": "retail_panic",
            "institutional_absorption": "institutional_absorption",
            "expiry_mechanics": "expiry_mechanics",
        }

        expected_regime = mechanism_to_regime.get(mechanism_claimed)
        return participant_regime == expected_regime

    def _determine_mechanism_confirmed(
        self,
        fade_time: Optional[float],
        max_fade: float,
        volume_match: bool,
        price_match: bool,
        regime_match: bool,
    ) -> bool:
        """Determine if mechanism was confirmed based on all evidence."""
        # Must have price pattern match
        if not price_match:
            return False

        # Must have reasonable fade
        if max_fade < self.min_fade_pct:
            return False

        # If fade time is available, must be within threshold
        if fade_time is not None and fade_time > self.fade_time_threshold:
            return False

        # Volume and regime are supporting evidence
        # If both fail, mechanism is not confirmed
        if not volume_match and not regime_match:
            return False

        return True

    def _calculate_confidence(
        self,
        fade_time: Optional[float],
        max_fade: float,
        volume_match: bool,
        price_match: bool,
        regime_match: bool,
    ) -> float:
        """Calculate confidence score for mechanism confirmation."""
        confidence = 0.0

        # Price pattern is most important (40%)
        if price_match:
            confidence += 0.4

        # Fade magnitude (25%)
        fade_score = min(max_fade / 2.0, 1.0)
        confidence += 0.25 * fade_score

        # Fade time (15%)
        if fade_time is not None:
            time_score = max(1.0 - fade_time / 60.0, 0.0)
            confidence += 0.15 * time_score

        # Volume pattern (10%)
        if volume_match:
            confidence += 0.1

        # Regime match (10%)
        if regime_match:
            confidence += 0.1

        return min(confidence, 1.0)

    def verify_batch(
        self,
        trades: list,
        intraday_data_dict: Dict,
    ) -> list:
        """Verify mechanism for a batch of trades.

        Args:
            trades: List of trade dictionaries
            intraday_data_dict: Dict mapping timestamps to intraday data

        Returns:
            List of MechanismVerificationResult
        """
        results = []
        for trade in trades:
            timestamp = trade.get("timestamp")
            intraday_data = intraday_data_dict.get(timestamp)

            if intraday_data is None:
                continue

            result = self.verify_gap_fade(trade, intraday_data)
            results.append(result)

        return results

    def generate_mechanism_report(self, results: list) -> Dict:
        """Generate aggregate report on mechanism verification.

        Args:
            results: List of MechanismVerificationResult

        Returns:
            Dict with aggregate statistics
        """
        if not results:
            return {}

        total = len(results)
        confirmed = sum(1 for r in results if r.mechanism_confirmed)
        thesis_correct = sum(1 for r in results if r.thesis_correct)
        pnl_independent = sum(1 for r in results if r.pnl_independent)

        # Average fade time for confirmed trades
        confirmed_fade_times = [
            r.fade_time_minutes for r in results 
            if r.fade_time_minutes is not None
        ]
        avg_fade_time = np.mean(confirmed_fade_times) if confirmed_fade_times else None

        # Average max fade
        avg_max_fade = np.mean([r.max_fade_pct for r in results])

        # Breakdown by mechanism
        by_mechanism = {}
        for r in results:
            mech = r.mechanism_claimed
            if mech not in by_mechanism:
                by_mechanism[mech] = {"total": 0, "confirmed": 0, "thesis_correct": 0}
            by_mechanism[mech]["total"] += 1
            if r.mechanism_confirmed:
                by_mechanism[mech]["confirmed"] += 1
            if r.thesis_correct:
                by_mechanism[mech]["thesis_correct"] += 1

        return {
            "total_trades": total,
            "mechanism_confirmed_count": confirmed,
            "mechanism_confirmed_rate": confirmed / total if total > 0 else 0,
            "thesis_correct_count": thesis_correct,
            "thesis_correct_rate": thesis_correct / total if total > 0 else 0,
            "pnl_independent_count": pnl_independent,
            "pnl_independent_rate": pnl_independent / total if total > 0 else 0,
            "avg_fade_time_minutes": avg_fade_time,
            "avg_max_fade_pct": avg_max_fade,
            "by_mechanism": by_mechanism,
        }

"""Explicit participant models for opening gap behavior in Indian markets.

This module models WHO creates opening gap mispricing in Indian markets and WHY.
It is not a prediction model. It is a diagnostic that tells you which participant
regime you are in before you trade.

The focus is on the gap fade niche - understanding which participant behavior
creates fadeable gaps vs informational gaps that should not be faded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ParticipantRegime:
    """The dominant participant regime for a gap event."""
    regime: str
    confidence: float
    diagnostics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            **self.diagnostics
        }


class RetailPanicFlowDetector:
    """Detects when retail panic is the dominant force at open.

    Retail panic creates fading opportunities. Institutional accumulation
    at open does not.

    Indian market specifics:
    - Retail traders react emotionally to overnight news
    - Place market orders at open without considering price
    - Create artificial downward pressure that mean-reverts
    - High volume with weak conviction (low delivery percentage)
    """

    def __init__(
        self,
        min_gap_pct: float = -1.5,
        min_volume_shock: float = 1.5,
        max_delivery_pct: float = 35.0,
        min_breadth_panic: int = 5,
    ):
        self.min_gap_pct = min_gap_pct
        self.min_volume_shock = min_volume_shock
        self.max_delivery_pct = max_delivery_pct
        self.min_breadth_panic = min_breadth_panic

    def score(
        self,
        gap_pct: float,
        volume_ratio: float,
        delivery_pct: float,
        breadth_panic: int,
        vix_level: Optional[float] = None,
        prior_trend: Optional[float] = None,
    ) -> float:
        """Returns 0-1: probability that retail panic is causing the gap.

        Args:
            gap_pct: Gap percentage (negative for gap-down)
            volume_ratio: Current volume / 20-day average volume
            delivery_pct: Delivery percentage (lower = more intraday/retail)
            breadth_panic: Number of stocks gapping down simultaneously
            vix_level: India VIX level (optional)
            prior_trend: Prior 5-day trend (optional)

        Returns:
            Score 0-1 indicating retail panic probability
        """
        score = 0.0

        # Gap must be significant
        if gap_pct > self.min_gap_pct:
            return 0.0

        # Component 1: Gap severity (larger gaps = more panic)
        gap_severity = min(abs(gap_pct) / 5.0, 1.0)
        score += 0.25 * gap_severity

        # Component 2: Volume shock (elevated volume but not extreme)
        # Extreme volume (>10x) suggests institutional event, not retail panic
        if volume_ratio < 0.5:
            # Low volume = no panic
            volume_score = 0.0
        elif volume_ratio > 10.0:
            # Extreme volume = institutional event
            volume_score = 0.0
        else:
            # Sweet spot: 1.5-5x average volume
            volume_score = min((volume_ratio - 1.0) / 4.0, 1.0)
        score += 0.20 * volume_score

        # Component 3: Delivery percentage (low delivery = retail/intraday)
        # High delivery (>50%) = institutional buying
        if delivery_pct < self.max_delivery_pct:
            delivery_score = 1.0 - (delivery_pct / 50.0)
        else:
            delivery_score = 0.0
        score += 0.25 * delivery_score

        # Component 4: Breadth panic (multiple stocks gapping = market-wide panic)
        breadth_score = min(breadth_panic / 20.0, 1.0)
        score += 0.15 * breadth_score

        # Component 5: VIX level (elevated VIX = fear environment)
        if vix_level is not None:
            vix_score = min((vix_level - 15.0) / 20.0, 1.0)
            score += 0.10 * vix_score

        # Component 6: Prior trend (weak prior trend = panic, strong downtrend = continuation)
        if prior_trend is not None:
            # If prior trend was already bearish, gap is likely continuation
            if prior_trend < -5.0:
                prior_score = 0.0
            else:
                # Weak or neutral prior trend suggests panic
                prior_score = 1.0 - min(abs(prior_trend) / 10.0, 1.0)
            score += 0.05 * prior_score

        return min(score, 1.0)


class InstitutionalAbsorptionDetector:
    """Detects when large players are absorbing the gap move.

    When institutions absorb, fade succeeds faster.
    When they push further, fade fails.

    Indian market specifics:
    - FIIs/DII absorb during panic selling
    - High delivery percentage indicates institutional participation
    - Opening auction imbalance shows institutional intent
    - Price relative to VWAP shows absorption level
    """

    def __init__(
        self,
        min_delivery_pct: float = 50.0,
        min_auction_imbalance: float = 0.3,
    ):
        self.min_delivery_pct = min_delivery_pct
        self.min_auction_imbalance = min_auction_imbalance

    def score(
        self,
        delivery_pct: float,
        auction_imbalance: Optional[float] = None,
        open_vs_prev_vwap: Optional[float] = None,
        block_trades: Optional[int] = None,
    ) -> float:
        """Returns 0-1: probability that institutions are absorbing.

        Args:
            delivery_pct: Delivery percentage (high = institutional)
            auction_imbalance: Buy/sell imbalance in opening auction (optional)
            open_vs_prev_vwap: Open price relative to previous day VWAP (optional)
            block_trades: Number of block trades (optional)

        Returns:
            Score 0-1 indicating institutional absorption probability
        """
        score = 0.0

        # Component 1: Delivery percentage (primary signal)
        if delivery_pct >= self.min_delivery_pct:
            delivery_score = min((delivery_pct - 50.0) / 30.0, 1.0)
        else:
            delivery_score = 0.0
        score += 0.40 * delivery_score

        # Component 2: Auction imbalance (buy-side absorption)
        if auction_imbalance is not None:
            # Positive imbalance = more buy orders = absorption
            auction_score = max(auction_imbalance, 0.0)
            score += 0.30 * auction_score

        # Component 3: Open vs previous VWAP
        # If open is below VWAP but not extremely so, suggests absorption
        if open_vs_prev_vwap is not None:
            # Open below VWAP by 0-2% suggests absorption at discount
            if -2.0 <= open_vs_prev_vwap <= 0.0:
                vwap_score = 1.0 - abs(open_vs_prev_vwap) / 2.0
            else:
                vwap_score = 0.0
            score += 0.20 * vwap_score

        # Component 4: Block trades (indicates institutional activity)
        if block_trades is not None:
            block_score = min(block_trades / 10.0, 1.0)
            score += 0.10 * block_score

        return min(score, 1.0)


class ExpiryMechanicsDetector:
    """Detects expiry-driven gap pressure.

    Expiry week creates artificial gap pressure from F&O unwinding.
    These gaps fade more reliably within 15 minutes.

    Indian market specifics:
    - Monthly expiry on last Thursday
    - Weekly expiry on every Thursday
    - Gamma hedging by dealers creates artificial moves
    - Unwinding near expiry creates gap pressure
    """

    def __init__(
        self,
        expiry_window_days: int = 5,
        min_oi_change: float = 10.0,
    ):
        self.expiry_window_days = expiry_window_days
        self.min_oi_change = min_oi_change

    def score(
        self,
        days_to_expiry: int,
        oi_change_pct: float,
        basis_spread: Optional[float] = None,
        option_pcr: Optional[float] = None,
    ) -> float:
        """Returns 0-1: probability that expiry mechanics are causing the gap.

        Args:
            days_to_expiry: Days to next F&O expiry
            oi_change_pct: Percentage change in open interest from previous day
            basis_spread: Futures-spot basis spread (optional)
            option_pcr: Put-call ratio (optional)

        Returns:
            Score 0-1 indicating expiry mechanics probability
        """
        score = 0.0

        # Component 1: Days to expiry (strongest signal)
        if days_to_expiry <= 0:
            # Expiry day
            expiry_score = 1.0
        elif days_to_expiry <= 1:
            # Day before expiry
            expiry_score = 0.9
        elif days_to_expiry <= self.expiry_window_days:
            # Expiry week
            expiry_score = 0.7
        else:
            expiry_score = 0.0
        score += 0.40 * expiry_score

        # Component 2: OI change (large changes indicate positioning/unwinding)
        if abs(oi_change_pct) >= self.min_oi_change:
            oi_score = min(abs(oi_change_pct) / 30.0, 1.0)
        else:
            oi_score = 0.0
        score += 0.30 * oi_score

        # Component 3: Basis spread (widening basis = expiry pressure)
        if basis_spread is not None:
            # Basis spread > 0.5% suggests expiry pressure
            basis_score = min(abs(basis_spread) / 1.0, 1.0)
            score += 0.20 * basis_score

        # Component 4: PCR (extreme PCR indicates positioning pressure)
        if option_pcr is not None:
            # PCR > 1.5 or < 0.5 indicates extreme positioning
            if option_pcr > 1.5 or option_pcr < 0.5:
                pcr_score = 1.0
            else:
                pcr_score = 0.0
            score += 0.10 * pcr_score

        return min(score, 1.0)


class ParticipantRegimeClassifier:
    """Classifies the participant regime for a gap event.

    This is the main entry point for participant modeling.
    It combines the three detectors to determine which regime is dominant.
    """

    def __init__(self):
        self.retail_panic = RetailPanicFlowDetector()
        self.institutional_absorption = InstitutionalAbsorptionDetector()
        self.expiry_mechanics = ExpiryMechanicsDetector()

    def classify(
        self,
        gap_pct: float,
        volume_ratio: float,
        delivery_pct: float,
        breadth_panic: int,
        days_to_expiry: int,
        oi_change_pct: float,
        vix_level: Optional[float] = None,
        prior_trend: Optional[float] = None,
        auction_imbalance: Optional[float] = None,
        open_vs_prev_vwap: Optional[float] = None,
        block_trades: Optional[int] = None,
        basis_spread: Optional[float] = None,
        option_pcr: Optional[float] = None,
    ) -> ParticipantRegime:
        """Classify the participant regime for a gap event.

        Args:
            gap_pct: Gap percentage
            volume_ratio: Volume ratio vs average
            delivery_pct: Delivery percentage
            breadth_panic: Number of stocks gapping down
            days_to_expiry: Days to F&O expiry
            oi_change_pct: Open interest change percentage
            vix_level: India VIX level (optional)
            prior_trend: Prior 5-day trend (optional)
            auction_imbalance: Opening auction imbalance (optional)
            open_vs_prev_vwap: Open vs previous VWAP (optional)
            block_trades: Number of block trades (optional)
            basis_spread: Futures-spot basis (optional)
            option_pcr: Put-call ratio (optional)

        Returns:
            ParticipantRegime with dominant regime and confidence
        """
        # Score each detector
        retail_score = self.retail_panic.score(
            gap_pct, volume_ratio, delivery_pct, breadth_panic, vix_level, prior_trend
        )
        absorption_score = self.institutional_absorption.score(
            delivery_pct, auction_imbalance, open_vs_prev_vwap, block_trades
        )
        expiry_score = self.expiry_mechanics.score(
            days_to_expiry, oi_change_pct, basis_spread, option_pcr
        )

        # Determine dominant regime
        scores = {
            "retail_panic": retail_score,
            "institutional_absorption": absorption_score,
            "expiry_mechanics": expiry_score,
        }

        dominant = max(scores.items(), key=lambda x: x[1])
        regime = dominant[0]
        confidence = dominant[1]

        # Build diagnostics
        diagnostics = {
            "retail_panic_score": retail_score,
            "institutional_absorption_score": absorption_score,
            "expiry_mechanics_score": expiry_score,
            "gap_pct": gap_pct,
            "volume_ratio": volume_ratio,
            "delivery_pct": delivery_pct,
            "breadth_panic": breadth_panic,
            "days_to_expiry": days_to_expiry,
            "oi_change_pct": oi_change_pct,
        }

        return ParticipantRegime(
            regime=regime,
            confidence=confidence,
            diagnostics=diagnostics,
        )

    def classify_from_dataframe(
        self,
        df: pd.DataFrame,
        timestamp,
    ) -> ParticipantRegime:
        """Classify participant regime from a DataFrame row.

        This is a convenience method for backtesting.
        It extracts required fields from the DataFrame.
        """
        row = df.loc[timestamp]

        return self.classify(
            gap_pct=row.get("gap_pct", 0.0),
            volume_ratio=row.get("volume_ratio", 1.0),
            delivery_pct=row.get("delivery_pct", 40.0),
            breadth_panic=int(row.get("breadth_panic", 0)),
            days_to_expiry=int(row.get("days_to_expiry", 30)),
            oi_change_pct=row.get("oi_change_pct", 0.0),
            vix_level=row.get("vix_level"),
            prior_trend=row.get("prior_trend"),
            auction_imbalance=row.get("auction_imbalance"),
            open_vs_prev_vwap=row.get("open_vs_prev_vwap"),
            block_trades=row.get("block_trades"),
            basis_spread=row.get("basis_spread"),
            option_pcr=row.get("option_pcr"),
        )

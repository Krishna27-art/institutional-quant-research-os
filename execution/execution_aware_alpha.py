"""Execution-Aware Alpha Systems.

Traditional alpha systems generate signals assuming perfect execution at
the signal price. Execution-aware alpha systems incorporate:

1. Expected slippage based on order size and liquidity
2. Market impact estimation for position sizing
3. Execution timing optimization (best time to enter)
4. Order type selection (market vs limit vs iceberg)
5. Partial fill modeling and handling
6. Execution quality feedback to signal generation

This transforms signals from "theoretical edge" to "realizable edge" by
incorporating execution costs and constraints into the signal itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    ICEBERG = "iceberg"
    VWAP = "vwap"
    TWAP = "twap"


class ExecutionVenue(Enum):
    """Execution venues for Indian markets."""
    NSE = "nse"
    BSE = "bse"
    BOTH = "both"


@dataclass
class ExecutionConstraints:
    """Constraints for executing a signal."""
    max_position_size: float  # Maximum shares/quantity
    max_order_value: float  # Maximum INR value per order
    max_slippage_pct: float  # Maximum acceptable slippage
    min_fill_pct: float  # Minimum fill percentage to proceed
    execution_window_minutes: int  # Time window to complete execution

    # Venue preferences
    preferred_venue: ExecutionVenue = ExecutionVenue.BOTH

    # Order type preferences
    preferred_order_type: OrderType = OrderType.MARKET

    # Time-of-day preferences
    avoid_first_minutes: int = 5  # Avoid first X minutes
    avoid_last_minutes: int = 5  # Avoid last X minutes


@dataclass
class ExecutionEstimate:
    """Estimate of execution quality for a signal."""
    expected_slippage_pct: float
    expected_fill_pct: float
    expected_market_impact_pct: float
    execution_time_minutes: float

    # Cost estimates
    total_cost_pct: float  # Slippage + impact + fees
    net_edge_after_execution: float  # Original edge minus execution costs

    # Confidence in estimate
    estimate_confidence: float  # 0-1

    # Recommendation
    recommended_order_type: OrderType
    recommended_venue: ExecutionVenue
    recommended_position_size: float

    def is_executable(self, min_edge_threshold: float = 0.005) -> bool:
        """Check if signal is executable after execution costs."""
        return self.net_edge_after_execution >= min_edge_threshold


class ExecutionAwareSignal:
    """An alpha signal with execution awareness."""
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        confidence: float,
        original_edge: float,  # Theoretical edge before execution costs
    ):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.confidence = confidence
        self.original_edge = original_edge

        # Execution estimates
        self.execution_estimate: Optional[ExecutionEstimate] = None
        self.execution_constraints: Optional[ExecutionConstraints] = None

        # Adjusted signal after execution awareness
        self.adjusted_entry_price: float = entry_price
        self.adjusted_confidence: float = confidence
        self.is_executable: bool = True
        self.execution_reason: str = ""

    def apply_execution_awareness(
        self,
        estimate: ExecutionEstimate,
        constraints: ExecutionConstraints
    ) -> None:
        """Apply execution awareness to the signal."""
        self.execution_estimate = estimate
        self.execution_constraints = constraints

        # Adjust entry price for expected slippage
        if self.direction == "long":
            self.adjusted_entry_price = self.entry_price * (1 + estimate.expected_slippage_pct)
        else:
            self.adjusted_entry_price = self.entry_price * (1 - estimate.expected_slippage_pct)

        # Adjust confidence based on execution quality
        self.adjusted_confidence = self.confidence * estimate.estimate_confidence

        # Check if executable
        self.is_executable = estimate.is_executable()

        if not self.is_executable:
            self.execution_reason = (
                f"Execution costs ({estimate.total_cost_pct:.2%}) exceed edge "
                f"({self.original_edge:.2%})"
            )
        else:
            self.execution_reason = "Signal executable after execution costs"


class ExecutionModel:
    """Model for estimating execution quality."""

    def __init__(self):
        # Historical execution data for calibration
        self.execution_history: List[Dict[str, Any]] = []

        # Default parameters (can be calibrated from data)
        self.base_slippage_bps = 5  # Base slippage in basis points
        self.impact_factor = 0.01  # Market impact per % of ADV
        self.fill_rate_base = 0.95  # Base fill rate

        logger.info("ExecutionModel initialized")

    def estimate_execution(
        self,
        signal: ExecutionAwareSignal,
        position_size: float,
        avg_daily_volume: float,
        avg_daily_value: float,
        current_time: datetime,
        constraints: Optional[ExecutionConstraints] = None
    ) -> ExecutionEstimate:
        """Estimate execution quality for a signal.

        Args:
            signal: The signal to execute
            position_size: Position size in shares
            avg_daily_volume: Average daily volume for the symbol
            avg_daily_value: Average daily value traded
            current_time: Current time
            constraints: Execution constraints

        Returns:
            ExecutionEstimate with quality metrics
        """
        constraints = constraints or ExecutionConstraints(
            max_position_size=position_size,
            max_order_value=avg_daily_value * 0.1,  # 10% of ADV
            max_slippage_pct=0.005,  # 0.5%
            min_fill_pct=0.8,
        )

        # Calculate position size as % of ADV
        position_pct_adv = position_size / (avg_daily_volume + 1e-6)
        position_value = position_size * signal.entry_price
        position_pct_value = position_value / (avg_daily_value + 1e-6)

        # Estimate slippage
        # Base slippage + impact from position size
        slippage_bps = self.base_slippage_bps + (position_pct_adv * 100 * self.impact_factor * 100)
        expected_slippage_pct = slippage_bps / 10000

        # Estimate market impact
        expected_impact_pct = position_pct_value * 0.5  # Simplified model

        # Estimate fill rate
        # Larger positions have lower fill rates
        fill_rate = self.fill_rate_base * (1 - position_pct_adv * 0.5)
        fill_rate = max(constraints.min_fill_pct, fill_rate)

        # Estimate execution time
        # Larger positions take longer
        execution_time_minutes = 5 + (position_pct_adv * 60)

        # Total cost
        total_cost_pct = expected_slippage_pct + expected_impact_pct + 0.001  # + 0.1% fees

        # Net edge after execution
        net_edge = signal.original_edge - total_cost_pct

        # Estimate confidence (higher for smaller positions in liquid stocks)
        liquidity_score = min(avg_daily_value / 10_000_000, 1.0)  # Normalize
        size_penalty = min(position_pct_adv / 0.1, 1.0)
        estimate_confidence = liquidity_score * (1 - size_penalty * 0.5)

        # Recommend order type
        if position_pct_adv < 0.01:
            recommended_order_type = OrderType.MARKET
        elif position_pct_adv < 0.05:
            recommended_order_type = OrderType.LIMIT
        else:
            recommended_order_type = OrderType.ICEBERG

        return ExecutionEstimate(
            expected_slippage_pct=expected_slippage_pct,
            expected_fill_pct=fill_rate,
            expected_market_impact_pct=expected_impact_pct,
            execution_time_minutes=execution_time_minutes,
            total_cost_pct=total_cost_pct,
            net_edge_after_execution=net_edge,
            estimate_confidence=estimate_confidence,
            recommended_order_type=recommended_order_type,
            recommended_venue=ExecutionVenue.BOTH,
            recommended_position_size=min(position_size, constraints.max_position_size),
        )

    def record_execution(
        self,
        signal: ExecutionAwareSignal,
        actual_slippage_pct: float,
        actual_fill_pct: float,
        execution_time_minutes: float
    ) -> None:
        """Record actual execution for model calibration."""
        self.execution_history.append({
            "symbol": signal.symbol,
            "strategy": signal.strategy_name,
            "position_size": signal.execution_estimate.recommended_position_size if signal.execution_estimate else 0,
            "expected_slippage": signal.execution_estimate.expected_slippage_pct if signal.execution_estimate else 0,
            "actual_slippage": actual_slippage_pct,
            "expected_fill": signal.execution_estimate.expected_fill_pct if signal.execution_estimate else 0,
            "actual_fill": actual_fill_pct,
            "expected_time": signal.execution_estimate.execution_time_minutes if signal.execution_estimate else 0,
            "actual_time": execution_time_minutes,
            "timestamp": datetime.now(),
        })

        # Trim history
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-10000:]

        logger.debug(f"Recorded execution for {signal.symbol}")

    def calibrate_model(self) -> Dict[str, float]:
        """Calibrate model parameters from execution history."""
        if len(self.execution_history) < 100:
            return {"status": "insufficient_data"}

        # Calculate slippage error
        slippage_errors = [
            h["actual_slippage"] - h["expected_slippage"]
            for h in self.execution_history
        ]
        avg_slippage_error = np.mean(slippage_errors)

        # Adjust base slippage
        self.base_slippage_bps = max(1, self.base_slippage_bps + avg_slippage_error * 10000)

        # Calculate fill rate error
        fill_errors = [
            h["actual_fill"] - h["expected_fill"]
            for h in self.execution_history
        ]
        avg_fill_error = np.mean(fill_errors)

        # Adjust base fill rate
        self.fill_rate_base = max(0.8, min(0.99, self.fill_rate_base + avg_fill_error))

        logger.info(
            f"Calibrated execution model: slippage_bps={self.base_slippage_bps:.1f}, "
            f"fill_rate={self.fill_rate_base:.3f}"
        )

        return {
            "status": "calibrated",
            "base_slippage_bps": self.base_slippage_bps,
            "fill_rate": self.fill_rate_base,
            "samples": len(self.execution_history),
        }


class ExecutionAwareAlphaSystem:
    """System for making alpha signals execution-aware.

    This system:
    1. Receives raw alpha signals
    2. Estimates execution quality
    3. Adjusts signals based on execution constraints
    4. Filters out signals that are not executable
    5. Records actual executions for model calibration
    """

    def __init__(self):
        self.execution_model = ExecutionModel()

        # Signal history
        self.signal_history: List[ExecutionAwareSignal] = []

        # Default constraints
        self.default_constraints = ExecutionConstraints()

        logger.info("ExecutionAwareAlphaSystem initialized")

    def process_signal(
        self,
        raw_signal: Dict[str, Any],
        market_data: Dict[str, float],
        current_time: datetime,
        constraints: Optional[ExecutionConstraints] = None
    ) -> Optional[ExecutionAwareSignal]:
        """Process a raw signal through execution awareness.

        Args:
            raw_signal: Raw signal data (symbol, direction, entry, etc.)
            market_data: Market data (ADV, etc.)
            current_time: Current time
            constraints: Execution constraints

        Returns:
            ExecutionAwareSignal if executable, None otherwise
        """
        constraints = constraints or self.default_constraints

        # Create execution-aware signal
        signal = ExecutionAwareSignal(
            strategy_name=raw_signal.get("strategy", "unknown"),
            symbol=raw_signal.get("symbol", "UNKNOWN"),
            direction=raw_signal.get("direction", "long"),
            entry_price=raw_signal.get("entry_price", 0.0),
            stop_price=raw_signal.get("stop_price", 0.0),
            target_price=raw_signal.get("target_price", 0.0),
            confidence=raw_signal.get("confidence", 0.5),
            original_edge=raw_signal.get("edge", 0.01),
        )

        # Estimate execution
        position_size = raw_signal.get("position_size", 1000)
        avg_daily_volume = market_data.get("adv", 1000000)
        avg_daily_value = market_data.get("adv_value", avg_daily_volume * signal.entry_price)

        estimate = self.execution_model.estimate_execution(
            signal=signal,
            position_size=position_size,
            avg_daily_volume=avg_daily_volume,
            avg_daily_value=avg_daily_value,
            current_time=current_time,
            constraints=constraints,
        )

        # Apply execution awareness
        signal.apply_execution_awareness(estimate, constraints)

        # Store signal if executable
        if signal.is_executable:
            self.signal_history.append(signal)

        return signal if signal.is_executable else None

    def record_execution_result(
        self,
        signal: ExecutionAwareSignal,
        actual_slippage_pct: float,
        actual_fill_pct: float,
        execution_time_minutes: float
    ) -> None:
        """Record actual execution result."""
        self.execution_model.record_execution(
            signal, actual_slippage_pct, actual_fill_pct, execution_time_minutes
        )

    def calibrate(self) -> Dict[str, float]:
        """Calibrate the execution model."""
        return self.execution_model.calibrate_model()

    def get_execution_report(self) -> str:
        """Generate an execution report."""
        lines = []
        lines.append("=" * 70)
        lines.append("EXECUTION-AWARE ALPHA REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Total Signals Processed: {len(self.signal_history)}")
        lines.append(f"Execution History: {len(self.execution_model.execution_history)}")
        lines.append("")

        lines.append("MODEL PARAMETERS")
        lines.append("-" * 70)
        lines.append(f"Base Slippage: {self.execution_model.base_slippage_bps:.1f} bps")
        lines.append(f"Impact Factor: {self.execution_model.impact_factor}")
        lines.append(f"Base Fill Rate: {self.execution_model.fill_rate_base:.3f}")
        lines.append("")

        # Recent signals
        if self.signal_history:
            lines.append("RECENT EXECUTION-AWARE SIGNALS")
            lines.append("-" * 70)
            for signal in self.signal_history[-10:]:
                est = signal.execution_estimate
                if est:
                    lines.append(
                        f"{signal.symbol} {signal.direction}: "
                        f"edge={signal.original_edge:.2%} -> "
                        f"net={est.net_edge_after_execution:.2%}, "
                        f"slippage={est.expected_slippage_pct:.2%}, "
                        f"executable={signal.is_executable}"
                    )
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

"""Capacity Intelligence.

Capacity intelligence estimates and monitors the trading capacity of strategies,
ensuring that capital allocation does not exceed what the strategy can handle
without significant performance degradation.

This module provides:
1. Capacity estimation based on liquidity and market impact
2. Capacity monitoring and alerts
3. Capacity utilization tracking
4. Diminishing returns modeling (performance vs size)
5. Capacity expansion strategies
6. Multi-strategy capacity optimization

This prevents the "too much capital" problem where strategies that work
well with small amounts of capital degrade significantly when scaled up.
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


class CapacityConstraint(Enum):
    """Types of capacity constraints."""
    LIQUIDITY = "liquidity"  # Limited by market liquidity
    MARKET_IMPACT = "market_impact"  # Limited by market impact
    EXECUTION = "execution"  # Limited by execution infrastructure
    RISK = "risk"  # Limited by risk management
    REGULATORY = "regulatory"  # Limited by regulatory constraints


@dataclass
class CapacityEstimate:
    """Capacity estimate for a strategy."""
    strategy_name: str

    # Capacity metrics
    estimated_capacity: float  # INR value
    capacity_per_trade: float  # Max per trade
    daily_capacity: float  # Max per day
    monthly_capacity: float  # Max per month

    # Constraints
    primary_constraint: CapacityConstraint
    constraint_details: Dict[str, float] = field(default_factory=dict)

    # Utilization
    current_utilization: float = 0.0  # Current capital / capacity
    utilization_history: List[float] = field(default_factory=list)

    # Performance degradation
    degradation_curve: List[Tuple[float, float]] = field(default_factory=list)  # (size, performance)

    # Confidence
    estimate_confidence: float = 0.0  # 0-1

    timestamp: datetime = field(default_factory=datetime.now)

    def get_headroom(self) -> float:
        """Get remaining capacity headroom."""
        return max(0, self.estimated_capacity - (self.estimated_capacity * self.current_utilization))

    def get_headroom_pct(self) -> float:
        """Get remaining capacity as percentage."""
        return max(0, 1.0 - self.current_utilization)

    def is_near_capacity(self, threshold: float = 0.9) -> bool:
        """Check if near capacity limit."""
        return self.current_utilization >= threshold


class CapacityModel:
    """Model for estimating strategy capacity."""

    def __init__(self):
        # Historical capacity data
        self.capacity_history: List[CapacityEstimate] = []

        # Default parameters
        self.liquidity_capacity_factor = 0.005  # 0.5% of ADV
        self.market_impact_factor = 0.02  # 2% impact threshold
        self.risk_capacity_factor = 0.1  # 10% of portfolio risk

        logger.info("CapacityModel initialized")

    def estimate_capacity(
        self,
        strategy_name: str,
        avg_daily_volume: float,
        avg_daily_value: float,
        avg_trade_size: float,
        volatility: float,
        holding_period_days: int = 5,
        current_capital: float = 0.0
    ) -> CapacityEstimate:
        """Estimate capacity for a strategy.

        Args:
            strategy_name: Name of the strategy
            avg_daily_volume: Average daily volume in shares
            avg_daily_value: Average daily value in INR
            avg_trade_size: Average trade size in shares
            volatility: Annualized volatility
            holding_period_days: Average holding period
            current_capital: Current capital allocated

        Returns:
            CapacityEstimate with capacity metrics
        """
        # Liquidity-based capacity
        liquidity_capacity = avg_daily_value * self.liquidity_capacity_factor * 20  # 20 days

        # Market impact-based capacity
        # Assume 2% market impact is acceptable
        impact_capacity = avg_daily_value * 0.5  # Can trade up to 50% of ADV with acceptable impact

        # Risk-based capacity
        # Assume strategy can handle 10% of portfolio risk
        risk_capacity = current_capital * 5 if current_capital > 0 else 10_000_000  # 5x current or 10M default

        # Take minimum of constraints
        capacity = min(liquidity_capacity, impact_capacity, risk_capacity)

        # Determine primary constraint
        if capacity == liquidity_capacity:
            primary_constraint = CapacityConstraint.LIQUIDITY
        elif capacity == impact_capacity:
            primary_constraint = CapacityConstraint.MARKET_IMPACT
        else:
            primary_constraint = CapacityConstraint.RISK

        # Calculate per-trade capacity
        capacity_per_trade = min(avg_trade_size * 10, capacity * 0.1)  # Max 10x avg trade or 10% of capacity

        # Calculate daily capacity (assume 5 trades per day)
        daily_capacity = capacity_per_trade * 5

        # Calculate monthly capacity (20 trading days)
        monthly_capacity = daily_capacity * 20

        # Current utilization
        utilization = current_capital / capacity if capacity > 0 else 0.0

        # Estimate confidence based on data quality
        confidence = min(1.0, (avg_daily_value / 1_000_000) / 10)  # Higher for larger stocks

        estimate = CapacityEstimate(
            strategy_name=strategy_name,
            estimated_capacity=capacity,
            capacity_per_trade=capacity_per_trade,
            daily_capacity=daily_capacity,
            monthly_capacity=monthly_capacity,
            primary_constraint=primary_constraint,
            constraint_details={
                "liquidity_capacity": liquidity_capacity,
                "impact_capacity": impact_capacity,
                "risk_capacity": risk_capacity,
            },
            current_utilization=utilization,
            estimate_confidence=confidence,
        )

        # Generate degradation curve (simplified)
        # Performance degrades as utilization increases
        for util in np.linspace(0, 1.5, 10):
            if util <= 0.5:
                performance = 1.0  # No degradation
            elif util <= 0.8:
                performance = 1.0 - (util - 0.5) * 0.5  # Linear degradation
            elif util <= 1.0:
                performance = 0.85 - (util - 0.8) * 1.5  # Accelerated degradation
            else:
                performance = 0.55 - (util - 1.0) * 1.1  # Severe degradation

            performance = max(0, performance)
            estimate.degradation_curve.append((util, performance))

        return estimate

    def update_utilization(
        self,
        strategy_name: str,
        current_capital: float
    ) -> Optional[CapacityEstimate]:
        """Update utilization for a strategy.

        Args:
            strategy_name: Strategy name
            current_capital: Current capital allocated

        Returns:
            Updated capacity estimate if found
        """
        for estimate in self.capacity_history:
            if estimate.strategy_name == strategy_name:
                estimate.current_utilization = current_capital / estimate.estimated_capacity
                estimate.utilization_history.append(estimate.current_utilization)

                # Trim history
                if len(estimate.utilization_history) > 100:
                    estimate.utilization_history = estimate.utilization_history[-100:]

                return estimate

        return None

    def check_capacity_alerts(self) -> List[str]:
        """Check for capacity alerts.

        Returns:
            List of alert messages
        """
        alerts = []

        for estimate in self.capacity_history:
            # Near capacity alert
            if estimate.is_near_capacity(threshold=0.9):
                alerts.append(
                    f"CAPACITY ALERT: {estimate.strategy_name} is at "
                    f"{estimate.current_utilization:.1%} capacity "
                    f"(constraint: {estimate.primary_constraint.value})"
                )

            # Over capacity alert
            if estimate.current_utilization > 1.0:
                alerts.append(
                    f"CAPACITY BREACH: {estimate.strategy_name} exceeds capacity "
                    f"({estimate.current_utilization:.1%} of {estimate.estimated_capacity:,.0f})"
                )

        return alerts

    def get_capacity_report(self) -> str:
        """Generate a capacity report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CAPACITY INTELLIGENCE REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Strategies Monitored: {len(self.capacity_history)}")
        lines.append("")

        # Capacity summary
        lines.append("CAPACITY SUMMARY")
        lines.append("-" * 70)
        for estimate in sorted(self.capacity_history, key=lambda x: x.estimated_capacity, reverse=True):
            lines.append(
                f"{estimate.strategy_name}: "
                f"Capacity: ₹{estimate.estimated_capacity:,.0f}, "
                f"Utilization: {estimate.current_utilization:.1%}, "
                f"Constraint: {estimate.primary_constraint.value}"
            )
        lines.append("")

        # Alerts
        alerts = self.check_capacity_alerts()
        if alerts:
            lines.append("CAPACITY ALERTS")
            lines.append("-" * 70)
            for alert in alerts:
                lines.append(f"  ⚠ {alert}")
            lines.append("")

        # Degradation curves
        lines.append("PERFORMANCE DEGRADATION CURVES")
        lines.append("-" * 70)
        for estimate in self.capacity_history[:3]:  # Show first 3
            lines.append(f"\n{estimate.strategy_name}:")
            for util, perf in estimate.degradation_curve:
                if util % 0.3 < 0.1:  # Show every 30%
                    lines.append(f"  {util:.0%} utilization -> {perf:.0%} performance")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


class CapacityIntelligenceSystem:
    """System for managing capacity intelligence across strategies.

    This system:
    1. Estimates capacity for each strategy
    2. Monitors capacity utilization
    3. Alerts on capacity breaches
    4. Optimizes multi-strategy capacity allocation
    5. Recommends capacity expansion strategies
    """

    def __init__(self):
        self.capacity_model = CapacityModel()

        # Strategy capacity estimates
        self.capacity_estimates: Dict[str, CapacityEstimate] = {}

        # Total portfolio capacity
        self.total_capacity: float = 0.0
        self.total_utilization: float = 0.0

        logger.info("CapacityIntelligenceSystem initialized")

    def register_strategy(
        self,
        strategy_name: str,
        avg_daily_volume: float,
        avg_daily_value: float,
        avg_trade_size: float,
        volatility: float,
        holding_period_days: int = 5
    ) -> CapacityEstimate:
        """Register a strategy for capacity monitoring.

        Args:
            strategy_name: Strategy name
            avg_daily_volume: Average daily volume
            avg_daily_value: Average daily value
            avg_trade_size: Average trade size
            volatility: Volatility
            holding_period_days: Holding period

        Returns:
            CapacityEstimate for the strategy
        """
        estimate = self.capacity_model.estimate_capacity(
            strategy_name=strategy_name,
            avg_daily_volume=avg_daily_volume,
            avg_daily_value=avg_daily_value,
            avg_trade_size=avg_trade_size,
            volatility=volatility,
            holding_period_days=holding_period_days,
        )

        self.capacity_estimates[strategy_name] = estimate
        self.capacity_model.capacity_history.append(estimate)

        # Update total capacity
        self.total_capacity += estimate.estimated_capacity

        logger.info(
            f"Registered {strategy_name} with capacity ₹{estimate.estimated_capacity:,.0f}"
        )

        return estimate

    def update_strategy_capital(
        self,
        strategy_name: str,
        capital: float
    ) -> None:
        """Update capital allocation for a strategy.

        Args:
            strategy_name: Strategy name
            capital: New capital allocation
        """
        if strategy_name in self.capacity_estimates:
            estimate = self.capacity_estimates[strategy_name]
            old_utilization = estimate.current_utilization

            self.capacity_model.update_utilization(strategy_name, capital)

            # Update total utilization
            self.total_utilization = self.total_utilization - old_utilization + estimate.current_utilization

            logger.debug(
                f"Updated {strategy_name} capital: ₹{capital:,.0f} "
                f"(utilization: {old_utilization:.1%} -> {estimate.current_utilization:.1%})"
            )

    def optimize_capacity_allocation(
        self,
        total_capital: float,
        min_utilization: float = 0.5,
        max_utilization: float = 0.8
    ) -> Dict[str, float]:
        """Optimize capital allocation across strategies based on capacity.

        Args:
            total_capital: Total capital to allocate
            min_utilization: Minimum target utilization per strategy
            max_utilization: Maximum target utilization per strategy

        Returns:
            Dictionary of optimal allocations by strategy
        """
        if not self.capacity_estimates:
            return {}

        # Calculate target total capacity utilization
        target_utilization = (min_utilization + max_utilization) / 2
        target_total_capacity = total_capital / target_utilization

        # If total capacity is insufficient, alert
        if self.total_capacity < target_total_capacity:
            logger.warning(
                f"Total capacity (₹{self.total_capacity:,.0f}) insufficient "
                f"for target utilization of {target_utilization:.1%}"
            )

        # Allocate proportionally to capacity
        allocations = {}
        for strategy_name, estimate in self.capacity_estimates.items():
            allocation = estimate.estimated_capacity * target_utilization
            allocations[strategy_name] = allocation

        return allocations

    def get_system_report(self) -> str:
        """Generate a comprehensive capacity intelligence report."""
        lines = []
        lines.append("=" * 70)
        lines.append("CAPACITY INTELLIGENCE SYSTEM REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Total Portfolio Capacity: ₹{self.total_capacity:,.0f}")
        lines.append(f"Total Utilization: {self.total_utilization:.1%}")
        lines.append("")

        # Individual strategy reports
        for strategy_name, estimate in self.capacity_estimates.items():
            lines.append(f"\n{strategy_name}")
            lines.append("-" * 70)
            lines.append(f"Estimated Capacity: ₹{estimate.estimated_capacity:,.0f}")
            lines.append(f"Per-Trade Capacity: ₹{estimate.capacity_per_trade:,.0f}")
            lines.append(f"Daily Capacity: ₹{estimate.daily_capacity:,.0f}")
            lines.append(f"Monthly Capacity: ₹{estimate.monthly_capacity:,.0f}")
            lines.append(f"Current Utilization: {estimate.current_utilization:.1%}")
            lines.append(f"Headroom: ₹{estimate.get_headroom():,.0f} ({estimate.get_headroom_pct():.1%})")
            lines.append(f"Primary Constraint: {estimate.primary_constraint.value}")
            lines.append(f"Estimate Confidence: {estimate.estimate_confidence:.2f}")

        lines.append("")
        lines.append(self.capacity_model.get_capacity_report())

        return "\n".join(lines)

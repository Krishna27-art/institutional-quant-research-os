"""
Capacity Analysis
Systematic analysis of strategy scalability with capacity curves and capacity limits.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd


@dataclass
class ImpactModel:
    """Market impact model for capacity analysis"""
    model_type: str = "square_root"  # "square_root", "linear", "power"
    
    # Square root model parameters
    k: float = 0.001  # Impact coefficient
    
    # Indian market calibration
    base_slippage_bps: float = 2.0  # Base slippage for large caps
    mid_cap_multiplier: float = 2.5  # Multiplier for mid caps
    
    def calculate_impact(
        self,
        order_notional: float,
        avg_daily_volume_notional: float,
        daily_volatility: float = 0.02
    ) -> float:
        """
        Calculate market impact in basis points.
        
        Args:
            order_notional: Order size in currency
            avg_daily_volume_notional: Average daily volume in currency
            daily_volatility: Daily volatility
        
        Returns:
            Impact in basis points
        """
        if self.model_type == "square_root":
            # Square root law: impact = k * sqrt(order / ADV)
            participation_rate = order_notional / avg_daily_volume_notional
            impact_bps = self.k * np.sqrt(participation_rate) * 10000
        elif self.model_type == "linear":
            # Linear model: impact = k * (order / ADV)
            participation_rate = order_notional / avg_daily_volume_notional
            impact_bps = self.k * participation_rate * 10000
        else:
            # Power law: impact = k * (order / ADV)^alpha
            participation_rate = order_notional / avg_daily_volume_notional
            impact_bps = self.k * (participation_rate ** 0.5) * 10000
        
        return impact_bps
    
    def calculate_total_cost(
        self,
        order_notional: float,
        avg_daily_volume_notional: float,
        is_mid_cap: bool = False
    ) -> float:
        """
        Calculate total transaction cost including impact.
        
        Args:
            order_notional: Order size in currency
            avg_daily_volume_notional: Average daily volume in currency
            is_mid_cap: Whether the instrument is mid-cap
        
        Returns:
            Total cost in basis points
        """
        impact = self.calculate_impact(order_notional, avg_daily_volume_notional)
        
        base_slippage = self.base_slippage_bps
        if is_mid_cap:
            base_slippage *= self.mid_cap_multiplier
        
        total_cost = base_slippage + impact
        
        return total_cost


@dataclass
class CapacityCurve:
    """Capacity curve for a strategy"""
    strategy_id: str
    aum_levels: List[float] = field(default_factory=list)
    sharpe_ratios: List[float] = field(default_factory=list)
    costs_bps: List[float] = field(default_factory=list)
    
    # Fitted parameters
    peak_sharpe: float = 0.0
    peak_aum: float = 0.0
    capacity_limit: float = 0.0  # AUM where Sharpe drops 20% from peak
    decay_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "aum_levels": self.aum_levels,
            "sharpe_ratios": self.sharpe_ratios,
            "costs_bps": self.costs_bps,
            "peak_sharpe": self.peak_sharpe,
            "peak_aum": self.peak_aum,
            "capacity_limit": self.capacity_limit,
            "decay_rate": self.decay_rate,
        }
    
    def predict_sharpe(self, aum: float) -> float:
        """
        Predict Sharpe at a given AUM using fitted curve.
        
        Args:
            aum: Assets under management in currency
        
        Returns:
            Predicted Sharpe ratio
        """
        if self.decay_rate == 0:
            return self.peak_sharpe
        
        # Exponential decay model: Sharpe(AUM) = Sharpe0 * exp(-AUM / capacity_limit)
        predicted = self.peak_sharpe * np.exp(-aum / self.capacity_limit)
        return max(0.0, predicted)


@dataclass
class CapacityResult:
    """Results of capacity analysis for a strategy"""
    strategy_id: str
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    # Capacity curve
    capacity_curve: Optional[CapacityCurve] = None
    
    # Current status
    current_aum: float = 0.0
    current_sharpe: float = 0.0
    utilization_pct: float = 0.0  # Current AUM as % of capacity limit
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            "capacity_curve": self.capacity_curve.to_dict() if self.capacity_curve else None,
            "current_aum": self.current_aum,
            "current_sharpe": self.current_sharpe,
            "utilization_pct": self.utilization_pct,
            "alerts": self.alerts,
            "recommendations": self.recommendations,
        }


class CapacityAnalysis:
    """
    Systematic analysis of strategy scalability.
    Runs backtests at increasing AUM levels to determine capacity limits.
    """
    
    # Standard AUM levels for testing (in crores)
    AUM_LEVELS_CRORES = [1, 5, 10, 25, 50, 100, 200, 500]
    
    def __init__(
        self,
        impact_model: Optional[ImpactModel] = None,
        utilization_threshold: float = 0.7  # Alert at 70% of capacity
    ):
        self.impact_model = impact_model or ImpactModel()
        self.utilization_threshold = utilization_threshold
        
        self.capacity_results: Dict[str, CapacityResult] = {}
        self.backtest_function: Optional[Callable] = None
    
    def set_backtest_function(self, backtest_function: Callable) -> None:
        """
        Set backtest function for strategy evaluation.
        
        Args:
            backtest_function: Function that takes (aum, impact_model) and returns Sharpe
        """
        self.backtest_function = backtest_function
    
    def run_capacity_analysis(
        self,
        strategy_id: str,
        base_sharpe: float,
        avg_daily_volume_notional: float,
        is_mid_cap: bool = False,
        current_aum: float = 0.0,
        aum_levels: Optional[List[float]] = None
    ) -> CapacityResult:
        """
        Run capacity analysis for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            base_sharpe: Sharpe at minimal AUM (no impact)
            avg_daily_volume_notional: Average daily volume in currency
            is_mid_cap: Whether strategy trades mid-caps
            current_aum: Current AUM
            aum_levels: Custom AUM levels to test (default: standard levels)
        
        Returns:
            CapacityResult with analysis results
        """
        if aum_levels is None:
            aum_levels = self.AUM_LEVELS_CRORES
        
        result = CapacityResult(
            strategy_id=strategy_id,
            current_aum=current_aum
        )
        
        # Run backtests at each AUM level
        aum_levels_tested = []
        sharpes = []
        costs = []
        
        for aum in aum_levels:
            # Calculate cost at this AUM
            # Assume position size scales with AUM
            order_notional = aum * 1e7  # Convert crores to currency
            cost_bps = self.impact_model.calculate_total_cost(
                order_notional,
                avg_daily_volume_notional,
                is_mid_cap
            )
            costs.append(cost_bps)
            
            # Calculate Sharpe with impact
            # Simplified: Sharpe degrades linearly with cost
            # Sharpe = base_sharpe - (cost_bps / 100) * base_sharpe
            sharpe_with_impact = base_sharpe * (1 - cost_bps / 10000)
            sharpes.append(max(0.0, sharpe_with_impact))
            
            aum_levels_tested.append(aum)
        
        # Create capacity curve
        curve = CapacityCurve(
            strategy_id=strategy_id,
            aum_levels=aum_levels_tested,
            sharpe_ratios=sharpes,
            costs_bps=costs
        )
        
        # Fit curve parameters
        curve.peak_sharpe = max(sharpes)
        peak_idx = sharpes.index(curve.peak_sharpe)
        curve.peak_aum = aum_levels_tested[peak_idx]
        
        # Find capacity limit (20% drop from peak)
        target_sharpe = curve.peak_sharpe * 0.8
        for i, sharpe in enumerate(sharpes):
            if sharpe < target_sharpe and i > 0:
                # Interpolate to find exact capacity limit
                x1, x2 = aum_levels_tested[i-1], aum_levels_tested[i]
                y1, y2 = sharpes[i-1], sharpes[i]
                curve.capacity_limit = x1 + (x2 - x1) * (target_sharpe - y1) / (y2 - y1)
                break
        else:
            # If no 20% drop found, use max AUM
            curve.capacity_limit = max(aum_levels_tested)
        
        # Fit decay rate
        if curve.capacity_limit > 0:
            curve.decay_rate = 1.0 / curve.capacity_limit
        
        result.capacity_curve = curve
        
        # Calculate current utilization
        if current_aum > 0:
            result.utilization_pct = (current_aum / curve.capacity_limit) * 100
            result.current_sharpe = curve.predict_sharpe(current_aum)
        else:
            result.utilization_pct = 0.0
            result.current_sharpe = base_sharpe
        
        # Generate alerts
        if result.utilization_pct > self.utilization_threshold * 100:
            result.alerts.append(
                f"Current AUM {current_aum}Cr is {result.utilization_pct:.1f}% of capacity limit "
                f"{curve.capacity_limit:.0f}Cr. Consider reducing allocation or capping inflows."
            )
        
        if result.utilization_pct > 90:
            result.alerts.append(
                f"CRITICAL: Utilization at {result.utilization_pct:.1f}%. "
                f"Sharpe degradation expected."
            )
        
        # Generate recommendations
        if result.utilization_pct > 50:
            result.recommendations.append(
                f"Monitor capacity utilization closely. "
                f"Consider scaling strategy or reducing position sizes."
            )
        
        if curve.capacity_limit < 50:
            result.recommendations.append(
                f"Strategy has low capacity ({curve.capacity_limit:.0f}Cr). "
                f"Suitable for smaller AUM only."
            )
        
        if curve.capacity_limit >= 200:
            result.recommendations.append(
                f"Strategy has high capacity ({curve.capacity_limit:.0f}Cr). "
                f"Suitable for institutional scale."
            )
        
        # Store result
        self.capacity_results[strategy_id] = result
        
        return result
    
    def get_capacity_result(self, strategy_id: str) -> Optional[CapacityResult]:
        """Get capacity analysis result for a strategy"""
        return self.capacity_results.get(strategy_id)
    
    def get_all_strategies(self) -> List[str]:
        """Get list of all analyzed strategies"""
        return list(self.capacity_results.keys())
    
    def generate_summary_report(self) -> Dict:
        """
        Generate summary report for all strategies.
        
        Returns:
            Summary with aggregate statistics
        """
        if not self.capacity_results:
            return {
                "status": "No capacity analyses run",
                "strategies_analyzed": 0
            }
        
        strategies = list(self.capacity_results.values())
        
        total_capacity = sum(
            r.capacity_curve.capacity_limit for r in strategies
            if r.capacity_curve
        )
        avg_capacity = total_capacity / len(strategies) if strategies else 0
        
        high_capacity_strategies = [
            r.strategy_id for r in strategies
            if r.capacity_curve and r.capacity_curve.capacity_limit >= 200
        ]
        
        low_capacity_strategies = [
            r.strategy_id for r in strategies
            if r.capacity_curve and r.capacity_curve.capacity_limit < 50
        ]
        
        at_capacity_strategies = [
            r.strategy_id for r in strategies
            if r.utilization_pct > self.utilization_threshold * 100
        ]
        
        return {
            "strategies_analyzed": len(strategies),
            "total_capacity_cr": total_capacity,
            "avg_capacity_cr": avg_capacity,
            "high_capacity_strategies": high_capacity_strategies,
            "low_capacity_strategies": low_capacity_strategies,
            "at_capacity_strategies": at_capacity_strategies,
            "utilization_threshold_pct": self.utilization_threshold * 100,
            "generated_at": datetime.now().isoformat(),
        }
    
    def clear_results(self, strategy_id: Optional[str] = None) -> None:
        """Clear capacity results, optionally filtered by strategy"""
        if strategy_id is None:
            self.capacity_results.clear()
        else:
            self.capacity_results.pop(strategy_id, None)


def calculate_capacity_limit_from_curve(
    aum_levels: List[float],
    sharpe_ratios: List[float],
    drop_threshold: float = 0.2
) -> float:
    """
    Calculate capacity limit from a capacity curve.
    
    Args:
        aum_levels: AUM levels tested
        sharpe_ratios: Sharpe ratios at each AUM level
        drop_threshold: Percentage drop from peak to define capacity limit
    
    Returns:
        Capacity limit in same units as aum_levels
    """
    peak_sharpe = max(sharpe_ratios)
    target_sharpe = peak_sharpe * (1 - drop_threshold)
    
    for i, sharpe in enumerate(sharpe_ratios):
        if sharpe < target_sharpe and i > 0:
            # Interpolate
            x1, x2 = aum_levels[i-1], aum_levels[i]
            y1, y2 = sharpe_ratios[i-1], sharpe_ratios[i]
            capacity_limit = x1 + (x2 - x1) * (target_sharpe - y1) / (y2 - y1)
            return capacity_limit
    
    # If no drop found, return max AUM
    return max(aum_levels)

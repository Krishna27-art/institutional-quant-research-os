"""
Performance Attribution
Decompose daily PnL into alpha, regime, execution, and luck.

Critical for institutional performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class PnLComponent(Enum):
    """PnL decomposition components"""
    ALPHA = "alpha"
    REGIME = "regime"
    EXECUTION = "execution"
    LUCK = "luck"
    COSTS = "costs"


@dataclass
class DailyAttribution:
    """Daily PnL attribution"""
    date: datetime
    total_pnl: float
    alpha_pnl: float
    regime_pnl: float
    execution_pnl: float
    luck_pnl: float
    costs_pnl: float
    sharpe_contribution: float


class PerformanceAttributor:
    """
    Performance Attributor
    
    Decomposes daily PnL into components:
    - Alpha: Strategy edge
    - Regime: Market regime impact
    - Execution: Slippage and timing
    - Luck: Random variance
    - Costs: Transaction costs
    
    Methods:
    - Regression-based attribution
    - Factor-based attribution
    - Transaction cost attribution
    """
    
    def __init__(self):
        self.attribution_history: List[DailyAttribution] = []
        self.regime_history: List[str] = []
        self.execution_cost_history: List[float] = []
    
    def attribute_daily_pnl(self, date: datetime, total_pnl: float,
                          gross_pnl: float, transaction_costs: float,
                          regime: str, market_return: float,
                          strategy_beta: float = 1.0) -> DailyAttribution:
        """
        Attribute daily PnL to components.
        
        Args:
            date: Date of attribution
            total_pnl: Total PnL after costs
            gross_pnl: PnL before costs
            transaction_costs: Transaction costs
            regime: Market regime
            market_return: Market return
            strategy_beta: Strategy beta to market
        
        Returns:
            DailyAttribution
        """
        # Costs component (direct)
        costs_pnl = -transaction_costs
        
        # Execution component (slippage)
        # Simplified: execution = gross_pnl - (alpha + regime)
        # We'll estimate alpha as gross_pnl - market_return * beta
        market_contribution = market_return * strategy_beta
        alpha_pnl = gross_pnl - market_contribution
        
        # Regime component (market contribution adjusted for regime)
        regime_adjustment = self._get_regime_adjustment(regime)
        regime_pnl = market_contribution * regime_adjustment
        
        # Execution component (residual after alpha, regime, costs)
        execution_pnl = total_pnl - alpha_pnl - regime_pnl - costs_pnl
        
        # Luck component (unexplained variance)
        # Simplified: assume 20% of residual is luck
        luck_pnl = execution_pnl * 0.2
        execution_pnl -= luck_pnl
        
        # Sharpe contribution (simplified)
        sharpe_contribution = alpha_pnl / (abs(alpha_pnl) + 1e-6)
        
        attribution = DailyAttribution(
            date=date,
            total_pnl=total_pnl,
            alpha_pnl=alpha_pnl,
            regime_pnl=regime_pnl,
            execution_pnl=execution_pnl,
            luck_pnl=luck_pnl,
            costs_pnl=costs_pnl,
            sharpe_contribution=sharpe_contribution
        )
        
        self.attribution_history.append(attribution)
        self.regime_history.append(regime)
        self.execution_cost_history.append(transaction_costs)
        
        return attribution
    
    def _get_regime_adjustment(self, regime: str) -> float:
        """Get regime adjustment factor"""
        regime_adjustments = {
            "bull": 1.2,
            "bear": 0.8,
            "normal": 1.0,
            "high_vol": 0.7,
            "low_vol": 1.3
        }
        return regime_adjustments.get(regime, 1.0)
    
    def get_attribution_summary(self, n_days: int = 30) -> Dict:
        """Get attribution summary over recent days"""
        if not self.attribution_history:
            return {}
        
        recent = self.attribution_history[-n_days:]
        
        summary = {
            "total_pnl": sum(d.total_pnl for d in recent),
            "alpha_pnl": sum(d.alpha_pnl for d in recent),
            "regime_pnl": sum(d.regime_pnl for d in recent),
            "execution_pnl": sum(d.execution_pnl for d in recent),
            "luck_pnl": sum(d.luck_pnl for d in recent),
            "costs_pnl": sum(d.costs_pnl for d in recent),
            "alpha_ratio": sum(d.alpha_pnl for d in recent) / sum(d.total_pnl for d in recent) if sum(d.total_pnl for d in recent) != 0 else 0,
            "execution_ratio": sum(d.execution_pnl for d in recent) / sum(d.total_pnl for d in recent) if sum(d.total_pnl for d in recent) != 0 else 0,
            "costs_ratio": sum(d.costs_pnl for d in recent) / sum(d.total_pnl for d in recent) if sum(d.total_pnl for d in recent) != 0 else 0
        }
        
        return summary
    
    def get_regime_attribution(self) -> Dict[str, Dict]:
        """Get attribution by regime"""
        regime_attribution = {}
        
        for attribution, regime in zip(self.attribution_history, self.regime_history):
            if regime not in regime_attribution:
                regime_attribution[regime] = {
                    "total_pnl": 0.0,
                    "alpha_pnl": 0.0,
                    "regime_pnl": 0.0,
                    "execution_pnl": 0.0,
                    "count": 0
                }
            
            regime_attribution[regime]["total_pnl"] += attribution.total_pnl
            regime_attribution[regime]["alpha_pnl"] += attribution.alpha_pnl
            regime_attribution[regime]["regime_pnl"] += attribution.regime_pnl
            regime_attribution[regime]["execution_pnl"] += attribution.execution_pnl
            regime_attribution[regime]["count"] += 1
        
        return regime_attribution
    
    def identify_performance_issues(self) -> List[str]:
        """Identify performance issues from attribution"""
        issues = []
        
        summary = self.get_attribution_summary()
        
        # Check if execution costs are too high
        if summary.get("execution_ratio", 0) < -0.3:  # Execution > 30% of losses
            issues.append("High execution costs detected")
        
        # Check if alpha is negative
        if summary.get("alpha_pnl", 0) < 0:
            issues.append("Negative alpha - strategy edge degraded")
        
        # Check if regime impact is large
        if abs(summary.get("regime_pnl", 0)) > abs(summary.get("alpha_pnl", 0)):
            issues.append("Regime impact exceeds alpha - consider regime hedging")
        
        # Check if costs are too high
        if summary.get("costs_ratio", 0) < -0.2:  # Costs > 20% of losses
            issues.append("High transaction costs - consider reducing turnover")
        
        return issues
    
    def generate_report(self) -> str:
        """Generate attribution report"""
        summary = self.get_attribution_summary()
        regime_attribution = self.get_regime_attribution()
        issues = self.identify_performance_issues()
        
        report = f"""
Performance Attribution Report
{'=' * 50}
Total PnL: {summary.get('total_pnl', 0):.2f}
Alpha PnL: {summary.get('alpha_pnl', 0):.2f} ({summary.get('alpha_ratio', 0):.1%})
Regime PnL: {summary.get('regime_pnl', 0):.2f}
Execution PnL: {summary.get('execution_pnl', 0):.2f} ({summary.get('execution_ratio', 0):.1%})
Luck PnL: {summary.get('luck_pnl', 0):.2f}
Costs PnL: {summary.get('costs_pnl', 0):.2f} ({summary.get('costs_ratio', 0):.1%})

Regime Attribution:
{'-' * 50}
"""
        
        for regime, attrib in regime_attribution.items():
            report += f"{regime}: {attrib['total_pnl']:.2f} "
            report += f"(Alpha: {attrib['alpha_pnl']:.2f}, "
            report += f"Regime: {attrib['regime_pnl']:.2f}, "
            report += f"Execution: {attrib['execution_pnl']:.2f})\n"
        
        if issues:
            report += f"\nPerformance Issues:\n{'-' * 50}\n"
            for issue in issues:
                report += f"- {issue}\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    attributor = PerformanceAttributor()
    
    # Simulate daily attribution
    print("Simulating daily performance attribution...")
    for day in range(30):
        date = datetime.now() - pd.Timedelta(days=30-day)
        
        # Simulate PnL components
        total_pnl = np.random.randn() * 10000
        gross_pnl = total_pnl + np.random.randn() * 2000
        transaction_costs = abs(np.random.randn()) * 1000
        
        regime = np.random.choice(["bull", "bear", "normal", "high_vol", "low_vol"])
        market_return = np.random.randn() * 0.02
        
        attribution = attributor.attribute_daily_pnl(
            date=date,
            total_pnl=total_pnl,
            gross_pnl=gross_pnl,
            transaction_costs=transaction_costs,
            regime=regime,
            market_return=market_return
        )
    
    print(attributor.generate_report())

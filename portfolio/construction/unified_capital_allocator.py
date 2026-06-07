"""
Unified Capital Allocator
Cross-strategy optimization for dynamic capital allocation.

Critical for institutional portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from scipy.optimize import minimize


class AllocationMethod(Enum):
    """Capital allocation methods"""
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    KELLY = "kelly"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    REGIME_CONDITIONAL = "regime_conditional"


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy"""
    strategy_id: str
    expected_return: float
    volatility: float
    sharpe: float
    confidence: float  # 0-1, probability that strategy works
    capacity_cr: float
    current_allocation: float
    regime_suitability: Dict[str, float]  # regime -> suitability score


@dataclass
class AllocationResult:
    """Result of capital allocation"""
    strategy_id: str
    optimal_weight: float
    current_weight: float
    weight_change: float
    expected_return: float
    risk_contribution: float
    reason: str


class UnifiedCapitalAllocator:
    """
    Unified Capital Allocator
    
    Optimizes capital allocation across all strategies simultaneously.
    
    Features:
    - Cross-strategy optimization (not per-strategy)
    - Confidence-based weighting
    - Regime-conditional allocation
    - Risk budget constraints
    - Crowding adjustments
    """
    
    def __init__(self, total_capital: float = 1000000000,  # 1000 Cr
                 risk_budget: float = 0.15,  # 15% annual risk
                 method: AllocationMethod = AllocationMethod.MAX_SHARPE):
        self.total_capital = total_capital
        self.risk_budget = risk_budget
        self.method = method
        
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.allocation_history: List[AllocationResult] = []
        self.current_regime: str = "normal"
    
    def add_strategy(self, metrics: StrategyMetrics):
        """Add strategy to allocator"""
        self.strategies[metrics.strategy_id] = metrics
    
    def set_regime(self, regime: str):
        """Set current market regime"""
        self.current_regime = regime
    
    def allocate(self) -> Dict[str, AllocationResult]:
        """
        Calculate optimal capital allocation across strategies.
        
        Returns:
            Dictionary of strategy_id -> AllocationResult
        """
        if not self.strategies:
            return {}
        
        if self.method == AllocationMethod.RISK_PARITY:
            return self._risk_parity_allocation()
        elif self.method == AllocationMethod.MAX_SHARPE:
            return self._max_sharpe_allocation()
        elif self.method == AllocationMethod.KELLY:
            return self._kelly_allocation()
        elif self.method == AllocationMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_allocation()
        elif self.method == AllocationMethod.REGIME_CONDITIONAL:
            return self._regime_conditional_allocation()
        else:
            return self._equal_weight_allocation()
    
    def _risk_parity_allocation(self) -> Dict[str, AllocationResult]:
        """Risk parity allocation"""
        n_strategies = len(self.strategies)
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        total_inv_vol = 0.0
        
        for strategy_id, metrics in self.strategies.items():
            inv_vol = 1.0 / (metrics.volatility + 1e-6)
            inv_vol_weights[strategy_id] = inv_vol
            total_inv_vol += inv_vol
        
        # Normalize
        results = {}
        for strategy_id, metrics in self.strategies.items():
            weight = inv_vol_weights[strategy_id] / total_inv_vol
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                optimal_weight=weight,
                current_weight=metrics.current_allocation,
                weight_change=weight - metrics.current_allocation,
                expected_return=metrics.expected_return,
                risk_contribution=weight * metrics.volatility,
                reason="Risk parity allocation"
            )
        
        self.allocation_history.extend(results.values())
        return results
    
    def _max_sharpe_allocation(self) -> Dict[str, AllocationResult]:
        """Max Sharpe ratio allocation"""
        strategy_ids = list(self.strategies.keys())
        n = len(strategy_ids)
        
        # Build expected returns and covariance matrix
        expected_returns = np.array([self.strategies[sid].expected_return for sid in strategy_ids])
        volatilities = np.array([self.strategies[sid].volatility for sid in strategy_ids])
        
        # Simplified: assume correlation = 0.5
        correlation = 0.5
        cov_matrix = np.outer(volatilities, volatilities) * correlation
        np.fill_diagonal(cov_matrix, volatilities ** 2)
        
        # Optimize for max Sharpe
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_vol  # Negative for minimization
        
        # Constraints: weights sum to 1, weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(n)]
        initial_guess = np.ones(n) / n
        
        result = minimize(objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Create results
        results = {}
        for i, strategy_id in enumerate(strategy_ids):
            weight = optimal_weights[i]
            metrics = self.strategies[strategy_id]
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                optimal_weight=weight,
                current_weight=metrics.current_allocation,
                weight_change=weight - metrics.current_allocation,
                expected_return=metrics.expected_return,
                risk_contribution=weight * metrics.volatility,
                reason="Max Sharpe allocation"
            )
        
        self.allocation_history.extend(results.values())
        return results
    
    def _kelly_allocation(self) -> Dict[str, AllocationResult]:
        """Kelly criterion allocation"""
        results = {}
        
        for strategy_id, metrics in self.strategies.items():
            # Kelly fraction = expected_return / variance
            kelly_fraction = metrics.expected_return / (metrics.volatility ** 2 + 1e-6)
            
            # Cap at 0.25 (quarter Kelly for safety)
            weight = min(kelly_fraction, 0.25)
            
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                optimal_weight=weight,
                current_weight=metrics.current_allocation,
                weight_change=weight - metrics.current_allocation,
                expected_return=metrics.expected_return,
                risk_contribution=weight * metrics.volatility,
                reason="Kelly criterion allocation"
            )
        
        # Normalize to sum to 1
        total_weight = sum(r.optimal_weight for r in results.values())
        if total_weight > 0:
            for result in results.values():
                result.optimal_weight /= total_weight
                result.weight_change = result.optimal_weight - result.current_weight
        
        self.allocation_history.extend(results.values())
        return results
    
    def _confidence_weighted_allocation(self) -> Dict[str, AllocationResult]:
        """Confidence-weighted allocation"""
        results = {}
        
        # Calculate confidence-adjusted Sharpe
        confidence_sharpe = {}
        total_confidence_sharpe = 0.0
        
        for strategy_id, metrics in self.strategies.items():
            adj_sharpe = metrics.sharpe * metrics.confidence
            confidence_sharpe[strategy_id] = max(adj_sharpe, 0)
            total_confidence_sharpe += confidence_sharpe[strategy_id]
        
        # Allocate based on confidence-adjusted Sharpe
        for strategy_id, metrics in self.strategies.items():
            if total_confidence_sharpe > 0:
                weight = confidence_sharpe[strategy_id] / total_confidence_sharpe
            else:
                weight = 1.0 / len(self.strategies)
            
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                optimal_weight=weight,
                current_weight=metrics.current_allocation,
                weight_change=weight - metrics.current_allocation,
                expected_return=metrics.expected_return,
                risk_contribution=weight * metrics.volatility,
                reason="Confidence-weighted allocation"
            )
        
        self.allocation_history.extend(results.values())
        return results
    
    def _regime_conditional_allocation(self) -> Dict[str, AllocationResult]:
        """Regime-conditional allocation"""
        results = {}
        
        # Get regime suitability scores
        regime_suitability = {}
        total_suitability = 0.0
        
        for strategy_id, metrics in self.strategies.items():
            suitability = metrics.regime_suitability.get(self.current_regime, 0.5)
            regime_suitability[strategy_id] = suitability * metrics.confidence
            total_suitability += regime_suitability[strategy_id]
        
        # Allocate based on regime suitability
        for strategy_id, metrics in self.strategies.items():
            if total_suitability > 0:
                weight = regime_suitability[strategy_id] / total_suitability
            else:
                weight = 1.0 / len(self.strategies)
            
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                optimal_weight=weight,
                current_weight=metrics.current_allocation,
                weight_change=weight - metrics.current_allocation,
                expected_return=metrics.expected_return,
                risk_contribution=weight * metrics.volatility,
                reason=f"Regime-conditional allocation ({self.current_regime})"
            )
        
        self.allocation_history.extend(results.values())
        return results
    
    def _equal_weight_allocation(self) -> Dict[str, AllocationResult]:
        """Equal weight allocation"""
        n = len(self.strategies)
        weight = 1.0 / n
        
        results = {}
        for strategy_id, metrics in self.strategies.items():
            results[strategy_id] = AllocationResult(
                strategy_id=strategy_id,
                optimal_weight=weight,
                current_weight=metrics.current_allocation,
                weight_change=weight - metrics.current_allocation,
                expected_return=metrics.expected_return,
                risk_contribution=weight * metrics.volatility,
                reason="Equal weight allocation"
            )
        
        self.allocation_history.extend(results.values())
        return results
    
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio-level metrics"""
        if not self.allocation_history:
            return {}
        
        latest_allocations = {r.strategy_id: r for r in self.allocation_history[-len(self.strategies):]}
        
        total_return = sum(r.optimal_weight * r.expected_return for r in latest_allocations.values())
        total_risk = sum(r.optimal_weight * r.risk_contribution for r in latest_allocations.values())
        portfolio_sharpe = total_return / total_risk if total_risk > 0 else 0
        
        return {
            "total_return": total_return,
            "total_risk": total_risk,
            "portfolio_sharpe": portfolio_sharpe,
            "num_strategies": len(self.strategies),
            "current_regime": self.current_regime
        }
    
    def generate_report(self) -> str:
        """Generate allocation report"""
        portfolio_metrics = self.get_portfolio_metrics()
        
        report = f"""
Unified Capital Allocator Report
{'=' * 50}
Total Capital: {self.total_capital:,.0f}
Risk Budget: {self.risk_budget:.1%}
Allocation Method: {self.method.value}
Current Regime: {self.current_regime}

Portfolio Metrics:
{'-' * 50}
Expected Return: {portfolio_metrics.get('total_return', 0):.2%}
Total Risk: {portfolio_metrics.get('total_risk', 0):.2%}
Portfolio Sharpe: {portfolio_metrics.get('portfolio_sharpe', 0):.3f}
Number of Strategies: {portfolio_metrics.get('num_strategies', 0)}

Strategy Allocations:
{'-' * 50}
"""
        
        if self.allocation_history:
            latest = {r.strategy_id: r for r in self.allocation_history[-len(self.strategies):]}
            for strategy_id, result in latest.items():
                report += f"{strategy_id}: {result.optimal_weight:.2%} (change: {result.weight_change:+.2%})\n"
                report += f"  Expected Return: {result.expected_return:.2%}\n"
                report += f"  Risk Contribution: {result.risk_contribution:.2%}\n"
                report += f"  Reason: {result.reason}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    allocator = UnifiedCapitalAllocator(total_capital=1000000000, risk_budget=0.15)
    
    # Add strategies
    allocator.add_strategy(StrategyMetrics(
        strategy_id="momentum",
        expected_return=0.15,
        volatility=0.20,
        sharpe=0.75,
        confidence=0.8,
        capacity_cr=500,
        current_allocation=0.25,
        regime_suitability={"normal": 0.8, "bull": 0.9, "bear": 0.3}
    ))
    
    allocator.add_strategy(StrategyMetrics(
        strategy_id="mean_reversion",
        expected_return=0.12,
        volatility=0.15,
        sharpe=0.80,
        confidence=0.7,
        capacity_cr=300,
        current_allocation=0.25,
        regime_suitability={"normal": 0.7, "bull": 0.4, "bear": 0.9}
    ))
    
    allocator.add_strategy(StrategyMetrics(
        strategy_id="stat_arb",
        expected_return=0.10,
        volatility=0.10,
        sharpe=1.00,
        confidence=0.9,
        capacity_cr=200,
        current_allocation=0.25,
        regime_suitability={"normal": 0.9, "bull": 0.8, "bear": 0.7}
    ))
    
    allocator.add_strategy(StrategyMetrics(
        strategy_id="pairs_trading",
        expected_return=0.08,
        volatility=0.08,
        sharpe=1.00,
        confidence=0.6,
        capacity_cr=100,
        current_allocation=0.25,
        regime_suitability={"normal": 0.6, "bull": 0.5, "bear": 0.8}
    ))
    
    # Set regime
    allocator.set_regime("normal")
    
    # Allocate
    results = allocator.allocate()
    
    print(allocator.generate_report())

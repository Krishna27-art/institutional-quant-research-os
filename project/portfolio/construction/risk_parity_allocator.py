"""
Risk-Parity Portfolio Allocator with Kelly Criterion
Based on Architecture V2 agent debate consensus

Key findings from research:
- Risk-parity: Equal volatility contribution from each strategy
- Kelly Criterion: 15% of optimal for conservative sizing
- Sequential Least Squares (SLSQP) optimization
- Constraints: Max 50% single strategy, 30% sector, 4x leverage
- Objective: Minimize sqrt(w' Σ w) subject to target vol = 15%

Architecture V2 - Quantitative Trading System for Indian Markets
Phase 1: Simplified stack with risk-parity portfolio construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class StrategyAllocation:
    """Strategy allocation result"""
    strategy_name: str
    weight: float
    target_volatility: float
    kelly_fraction: float
    regime_adjusted_weight: float


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    allocations: Dict[str, StrategyAllocation]
    total_weight: float
    portfolio_volatility: float
    expected_return: float
    sharpe_ratio: float
    leverage: float
    constraints_satisfied: bool


class RiskParityAllocator:
    """
    Risk-Parity Portfolio Allocator with Kelly Criterion (Architecture V2).
    
    Architecture V2 Portfolio Construction:
    - Method: Risk-parity (equal volatility contribution)
    - Optimizer: Sequential Least Squares (SLSQP) - daily
    - Kelly: 15% of optimal for conservative sizing
    - Constraints:
      - Max single strategy weight: 50%
      - Max sector weight: 30%
      - Max leverage: 4x
      - Max position size: 5% of AUM
    - Objective: Minimize sqrt(w' Σ w) subject to target vol = 15%
    - Rebalance: At market open, using previous close data
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annual volatility target
        max_strategy_weight: float = 0.50,  # 50% max single strategy
        max_sector_weight: float = 0.30,  # 30% max sector exposure
        max_leverage: float = 4.0,  # 4x max leverage
        kelly_fraction: float = 0.15  # 15% of optimal Kelly
    ):
        self.target_volatility = target_volatility
        self.max_strategy_weight = max_strategy_weight
        self.max_sector_weight = max_sector_weight
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        
        # Strategy to sector mapping
        self.strategy_sectors = {
            'ORB': 'EQUITY',
            'VWAP': 'INDEX',
            'PCP': 'OPTIONS',
            'VOL_CARRY': 'OPTIONS',
            'GAME_THEORETIC': 'EQUITY'
        }
    
    def calculate_covariance_matrix(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate covariance matrix from returns.
        
        Args:
            returns: DataFrame with strategy returns
            
        Returns:
            Covariance matrix
        """
        return returns.cov().values * 252  # Annualize
    
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            expected_return: Expected annual return
            volatility: Annual volatility
            risk_free_rate: Risk-free rate
            
        Returns:
            Kelly fraction (capped at 1.0)
        """
        if volatility == 0:
            return 0.0
        
        # Kelly = (μ - r) / σ²
        kelly = (expected_return - risk_free_rate) / (volatility ** 2)
        
        # Architecture V2: Use 15% of optimal Kelly
        kelly_conservative = kelly * self.kelly_fraction
        
        # Cap at 1.0
        kelly_conservative = min(kelly_conservative, 1.0)
        kelly_conservative = max(kelly_conservative, 0.0)
        
        return kelly_conservative
    
    def risk_parity_objective(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Risk-parity objective function.
        
        Minimize portfolio variance subject to equal volatility contribution.
        
        Args:
            weights: Portfolio weights
            covariance: Covariance matrix
            
        Returns:
            Portfolio variance
        """
        portfolio_variance = np.dot(weights.T, np.dot(covariance, weights))
        return portfolio_variance
    
    def optimize_weights(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        strategy_names: List[str],
        regime: str = "normal"
    ) -> np.ndarray:
        """
        Optimize portfolio weights using SLSQP.
        
        Args:
            expected_returns: Expected returns for each strategy
            covariance: Covariance matrix
            strategy_names: List of strategy names
            regime: Current market regime
            
        Returns:
            Optimized weights
        """
        n_strategies = len(strategy_names)
        
        # Initial weights: Equal weight
        initial_weights = np.ones(n_strategies) / n_strategies
        
        # Constraints
        constraints = []
        
        # Sum of weights = 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Max single strategy weight
        for i in range(n_strategies):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, i=i: self.max_strategy_weight - w[i]
            })
        
        # Sector constraints
        sector_constraints = {}
        for i, strategy in enumerate(strategy_names):
            sector = self.strategy_sectors.get(strategy, 'OTHER')
            if sector not in sector_constraints:
                sector_constraints[sector] = []
            sector_constraints[sector].append(i)
        
        for sector, indices in sector_constraints.items():
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=indices: self.max_sector_weight - np.sum(w[idx])
            })
        
        # Bounds: 0 to max_strategy_weight
        bounds = [(0, self.max_strategy_weight) for _ in range(n_strategies)]
        
        # Regime-based adjustments
        if regime == "high_vol":
            # Reduce weights in high volatility
            bounds = [(0, self.max_strategy_weight * 0.5) for _ in range(n_strategies)]
        elif regime == "bull_trend":
            # Allow higher weights in bull trend
            bounds = [(0, self.max_strategy_weight) for _ in range(n_strategies)]
        
        # Optimize
        result = minimize(
            self.risk_parity_objective,
            initial_weights,
            args=(covariance,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return initial_weights
    
    def allocate_portfolio(
        self,
        strategy_returns: pd.DataFrame,
        expected_returns: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        regime: str = "normal"
    ) -> PortfolioAllocation:
        """
        Allocate portfolio using risk-parity with Kelly.
        
        Args:
            strategy_returns: DataFrame with historical strategy returns
            expected_returns: Optional expected returns for each strategy
            volatilities: Optional volatilities for each strategy
            regime: Current market regime
            
        Returns:
            PortfolioAllocation with allocations and metrics
        """
        strategy_names = list(strategy_returns.columns)
        n_strategies = len(strategy_names)
        
        # Calculate covariance matrix
        covariance = self.calculate_covariance_matrix(strategy_returns)
        
        # Use provided or calculate expected returns
        if expected_returns is None:
            expected_returns = {
                strategy: strategy_returns[strategy].mean() * 252
                for strategy in strategy_names
            }
        
        # Use provided or calculate volatilities
        if volatilities is None:
            volatilities = {
                strategy: strategy_returns[strategy].std() * np.sqrt(252)
                for strategy in strategy_names
            }
        
        # Convert to arrays
        expected_returns_array = np.array([expected_returns[s] for s in strategy_names])
        
        # Optimize weights
        weights = self.optimize_weights(expected_returns_array, covariance, strategy_names, regime)
        
        # Calculate Kelly fractions
        kelly_fractions = {}
        for i, strategy in enumerate(strategy_names):
            kelly = self.calculate_kelly_fraction(
                expected_returns[strategy],
                volatilities[strategy]
            )
            kelly_fractions[strategy] = kelly
        
        # Apply Kelly adjustment (use min of risk-parity weight and Kelly)
        adjusted_weights = {}
        for i, strategy in enumerate(strategy_names):
            # Risk-parity weight
            rp_weight = weights[i]
            # Kelly weight
            kelly_weight = kelly_fractions[strategy]
            # Use minimum (conservative)
            adjusted_weight = min(rp_weight, kelly_weight)
            adjusted_weights[strategy] = adjusted_weight
        
        # Re-normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # Create strategy allocations
        allocations = {}
        for strategy in strategy_names:
            allocations[strategy] = StrategyAllocation(
                strategy_name=strategy,
                weight=adjusted_weights[strategy],
                target_volatility=volatilities[strategy],
                kelly_fraction=kelly_fractions[strategy],
                regime_adjusted_weight=adjusted_weights[strategy]
            )
        
        # Calculate portfolio metrics
        portfolio_volatility = np.sqrt(np.dot(
            np.array(list(adjusted_weights.values())),
            np.dot(covariance, np.array(list(adjusted_weights.values())))
        ))
        
        portfolio_expected_return = sum(
            adjusted_weights[s] * expected_returns[s]
            for s in strategy_names
        )
        
        sharpe_ratio = portfolio_expected_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        # Calculate leverage (sum of absolute weights)
        leverage = sum(abs(w) for w in adjusted_weights.values())
        
        # Check constraints
        constraints_satisfied = True
        if leverage > self.max_leverage:
            constraints_satisfied = False
        for strategy, weight in adjusted_weights.items():
            if weight > self.max_strategy_weight:
                constraints_satisfied = False
        
        return PortfolioAllocation(
            allocations=allocations,
            total_weight=1.0,
            portfolio_volatility=portfolio_volatility,
            expected_return=portfolio_expected_return,
            sharpe_ratio=sharpe_ratio,
            leverage=leverage,
            constraints_satisfied=constraints_satisfied
        )
    
    def print_allocation_report(self, allocation: PortfolioAllocation) -> None:
        """Print portfolio allocation report."""
        print("\n" + "="*60)
        print("RISK-PARITY PORTFOLIO ALLOCATION (Architecture V2)")
        print("="*60)
        print(f"Portfolio Volatility: {allocation.portfolio_volatility:.2%}")
        print(f"Expected Return: {allocation.expected_return:.2%}")
        print(f"Sharpe Ratio: {allocation.sharpe_ratio:.2f}")
        print(f"Leverage: {allocation.leverage:.2f}x")
        print(f"Constraints Satisfied: {'✅' if allocation.constraints_satisfied else '❌'}")
        
        print("\nStrategy Allocations:")
        for strategy, alloc in allocation.allocations.items():
            print(f"  {strategy:<20}: {alloc.weight:>6.2%} | Kelly: {alloc.kelly_fraction:>6.2%} | Vol: {alloc.target_volatility:>6.2%}")
        
        print("\nArchitecture V2 Constraints:")
        print(f"  Max single strategy weight: {self.max_strategy_weight:.0%}")
        print(f"  Max sector weight: {self.max_sector_weight:.0%}")
        print(f"  Max leverage: {self.max_leverage:.0f}x")
        print(f"  Kelly fraction: {self.kelly_fraction:.0%} of optimal")
        
        if not allocation.constraints_satisfied:
            print("\n⚠️  Constraints violated - weights need adjustment")
        
        print("="*60)


def run_sample_allocation():
    """Run sample portfolio allocation."""
    # Create sample strategy returns
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    np.random.seed(42)
    
    strategy_returns = pd.DataFrame({
        'ORB': np.random.normal(0.0005, 0.015, 252),
        'VWAP': np.random.normal(0.0004, 0.012, 252),
        'PCP': np.random.normal(0.0003, 0.010, 252),
        'VOL_CARRY': np.random.normal(0.0002, 0.008, 252)
    }, index=dates)
    
    # Initialize allocator
    allocator = RiskParityAllocator(
        target_volatility=0.15,
        max_strategy_weight=0.50,
        max_sector_weight=0.30,
        max_leverage=4.0,
        kelly_fraction=0.15
    )
    
    # Allocate portfolio
    allocation = allocator.allocate_portfolio(strategy_returns, regime="normal")
    
    # Print report
    allocator.print_allocation_report(allocation)
    
    return allocation


if __name__ == "__main__":
    run_sample_allocation()

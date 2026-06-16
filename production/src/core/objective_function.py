"""
Single Objective Function for the Quant System
Based on the critique: Maximize risk-adjusted after-cost profit

Objective:
Maximize risk-adjusted after-cost profit
Subject to:
    - Drawdown < X
    - Capacity > Y
    - Turnover < Z

This is the single objective that drives all decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ObjectiveConstraints:
    """Constraints for the objective function."""
    max_drawdown: float = 0.15  # 15% max drawdown
    min_capacity: float = 1000000.0  # $1M minimum capacity
    max_turnover: float = 3.0  # 3x annual turnover
    min_sharpe: float = 1.0  # Minimum Sharpe ratio
    max_leverage: float = 2.0  # Maximum leverage
    min_liquidity: float = 100000.0  # Minimum daily liquidity per trade


@dataclass
class ObjectiveScore:
    """Score for a strategy or decision."""
    objective_value: float  # Risk-adjusted after-cost profit
    sharpe_ratio: float
    max_drawdown: float
    capacity: float
    turnover: float
    leverage: float
    liquidity: float
    is_feasible: bool
    constraint_violations: List[str]


class ObjectiveFunction:
    """
    Single Objective Function for the quant system.
    
    Objective: Maximize risk-adjusted after-cost profit
    
    Score = (After-cost Return / Drawdown) * sqrt(252) / Std(Returns)
    
    Subject to constraints on drawdown, capacity, turnover, leverage, liquidity.
    """
    
    def __init__(self, constraints: ObjectiveConstraints = None):
        self.constraints = constraints or ObjectiveConstraints()
    
    def calculate_objective(
        self,
        returns: pd.Series,
        costs: pd.Series,
        positions: pd.Series,
        volumes: pd.Series = None
    ) -> ObjectiveScore:
        """
        Calculate the objective score for a strategy.
        
        Args:
            returns: Strategy returns
            costs: Transaction costs
            positions: Position sizes
            volumes: Trading volumes (for capacity check)
            
        Returns:
            ObjectiveScore with all metrics
        """
        # Calculate after-cost returns
        after_cost_returns = returns - costs
        
        # Calculate metrics
        total_return = after_cost_returns.sum()
        mean_return = after_cost_returns.mean()
        std_return = after_cost_returns.std()
        
        # Sharpe ratio (risk-adjusted return)
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + after_cost_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Capacity (average position size)
        capacity = positions.abs().mean()
        
        # Turnover (sum of absolute position changes)
        turnover = positions.diff().abs().sum()
        
        # Leverage (max position size relative to capital)
        leverage = positions.abs().max()
        
        # Liquidity (minimum volume)
        liquidity = volumes.min() if volumes is not None else float('inf')
        
        # Check constraints
        violations = []
        is_feasible = True
        
        if max_drawdown > self.constraints.max_drawdown:
            violations.append(f"Drawdown {max_drawdown:.2%} exceeds limit {self.constraints.max_drawdown:.2%}")
            is_feasible = False
        
        if capacity < self.constraints.min_capacity:
            violations.append(f"Capacity {capacity:.0f} below minimum {self.constraints.min_capacity:.0f}")
            is_feasible = False
        
        if turnover > self.constraints.max_turnover:
            violations.append(f"Turnover {turnover:.2f} exceeds limit {self.constraints.max_turnover:.2f}")
            is_feasible = False
        
        if leverage > self.constraints.max_leverage:
            violations.append(f"Leverage {leverage:.2f} exceeds limit {self.constraints.max_leverage:.2f}")
            is_feasible = False
        
        if liquidity < self.constraints.min_liquidity:
            violations.append(f"Liquidity {liquidity:.0f} below minimum {self.constraints.min_liquidity:.0f}")
            is_feasible = False
        
        if sharpe < self.constraints.min_sharpe:
            violations.append(f"Sharpe {sharpe:.2f} below minimum {self.constraints.min_sharpe:.2f}")
            is_feasible = False
        
        # Objective value: risk-adjusted after-cost profit
        # Higher is better
        objective_value = sharpe if is_feasible else -float('inf')
        
        return ObjectiveScore(
            objective_value=objective_value,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            capacity=capacity,
            turnover=turnover,
            leverage=leverage,
            liquidity=liquidity,
            is_feasible=is_feasible,
            constraint_violations=violations
        )
    
    def rank_strategies(
        self,
        strategy_results: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]]
    ) -> List[Tuple[str, ObjectiveScore]]:
        """
        Rank strategies by objective score.
        
        Args:
            strategy_results: Dictionary of strategy_name -> (returns, costs, positions)
            
        Returns:
            List of (strategy_name, score) sorted by objective value
        """
        scores = []
        
        for strategy_name, (returns, costs, positions) in strategy_results.items():
            score = self.calculate_objective(returns, costs, positions)
            scores.append((strategy_name, score))
        
        # Sort by objective value descending
        scores.sort(key=lambda x: x[1].objective_value, reverse=True)
        
        return scores
    
    def get_optimal_portfolio(
        self,
        strategy_scores: List[Tuple[str, ObjectiveScore]]
    ) -> List[str]:
        """
        Get optimal portfolio of strategies.
        
        Returns only feasible strategies, sorted by objective value.
        """
        feasible = [(name, score) for name, score in strategy_scores if score.is_feasible]
        return [name for name, score in feasible]


if __name__ == "__main__":
    # Test the Objective Function
    print("Testing Single Objective Function...")
    
    objective = ObjectiveFunction()
    
    # Generate sample data
    np.random.seed(42)
    n = 252  # 1 year of daily data
    
    returns = pd.Series(np.random.normal(0.0005, 0.01, n))
    costs = pd.Series(np.random.uniform(0.0001, 0.0003, n))
    positions = pd.Series(np.random.uniform(-0.5, 0.5, n))
    volumes = pd.Series(np.random.uniform(500000, 2000000, n))
    
    # Calculate objective
    score = objective.calculate_objective(returns, costs, positions, volumes)
    
    print(f"\nObjective Score: {score.objective_value:.2f}")
    print(f"Sharpe Ratio: {score.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {score.max_drawdown:.2%}")
    print(f"Capacity: {score.capacity:.0f}")
    print(f"Turnover: {score.turnover:.2f}")
    print(f"Leverage: {score.leverage:.2f}")
    print(f"Liquidity: {score.liquidity:.0f}")
    print(f"Feasible: {score.is_feasible}")
    
    if score.constraint_violations:
        print(f"\nConstraint Violations:")
        for violation in score.constraint_violations:
            print(f"  - {violation}")
    
    # Test ranking multiple strategies
    print("\nRanking multiple strategies...")
    strategy_results = {
        "ORB": (pd.Series(np.random.normal(0.0008, 0.015, n)),
                pd.Series(np.random.uniform(0.0001, 0.0003, n)),
                pd.Series(np.random.uniform(-0.3, 0.3, n))),
        "VWAP": (pd.Series(np.random.normal(0.0004, 0.008, n)),
                 pd.Series(np.random.uniform(0.0001, 0.0003, n)),
                 pd.Series(np.random.uniform(-0.4, 0.4, n))),
        "Momentum": (pd.Series(np.random.normal(0.0002, 0.012, n)),
                    pd.Series(np.random.uniform(0.0001, 0.0003, n)),
                    pd.Series(np.random.uniform(-0.5, 0.5, n)))
    }
    
    ranked = objective.rank_strategies(strategy_results)
    
    print("\nStrategy Rankings:")
    for name, score in ranked:
        print(f"  {name}: Objective={score.objective_value:.2f}, Sharpe={score.sharpe_ratio:.2f}, Feasible={score.is_feasible}")
    
    # Get optimal portfolio
    optimal = objective.get_optimal_portfolio(ranked)
    print(f"\nOptimal Portfolio: {optimal}")

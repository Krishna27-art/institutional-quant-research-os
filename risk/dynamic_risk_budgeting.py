"""
Dynamic Risk Budgeting

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#42)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Dynamic risk allocation across strategies
- Risk parity with time-varying volatilities
- Conditional Value-at-Risk (CVaR) budgeting
- Adaptive risk limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Install with: pip install scipy")


@dataclass
class RiskBudgetConfig:
    """Configuration for Dynamic Risk Budgeting"""
    # Risk budget parameters
    target_portfolio_volatility: float = 0.15  # 15% annual volatility
    risk_free_rate: float = 0.05
    
    # CVaR parameters
    cvar_confidence: float = 0.95  # 95% confidence
    cvar_lookback: int = 252  # 1 year lookback
    
    # Risk limits
    max_strategy_risk: float = 0.05  # 5% max risk per strategy
    min_strategy_risk: float = 0.01  # 1% min risk per strategy
    
    # Rebalancing
    rebalance_frequency: str = "weekly"
    volatility_window: int = 20  # Volatility estimation window
    
    # Risk budgeting method
    method: str = "risk_parity"  # "risk_parity", "cvar", "equal_risk"


class DynamicRiskBudgeting:
    """
    Dynamic Risk Budgeting Engine
    
    Dynamically allocates risk budget across strategies
    based on their risk characteristics and correlations.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: RiskBudgetConfig):
        self.config = config
        
        # Current risk budgets
        self.current_risk_budgets: Optional[pd.Series] = None
        
        # Strategy volatilities
        self.strategy_volatilities: Optional[pd.Series] = None
    
    def estimate_volatilities(self, returns: pd.DataFrame) -> pd.Series:
        """
        Estimate strategy volatilities
        
        Args:
            returns: Strategy returns
            
        Returns:
            Volatility estimates
        """
        vol = returns.tail(self.config.volatility_window).std() * np.sqrt(252)
        self.strategy_volatilities = vol
        return vol
    
    def calculate_risk_parity_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate risk parity weights
        
        Args:
            returns: Strategy returns
            
        Returns:
            Risk parity weights
        """
        if not SCIPY_AVAILABLE:
            return self._equal_risk_allocation(returns)
        
        cov = returns.tail(self.config.cvar_lookback).cov()
        n_strategies = cov.shape[0]
        
        # Objective function (minimize risk contribution variance)
        def objective(weights):
            portfolio_std = np.sqrt(weights @ cov @ weights)
            marginal_risk = cov @ weights / portfolio_std
            risk_contributions = weights * marginal_risk
            target_risk = portfolio_std / n_strategies
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n_strategies)]
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n_strategies) / n_strategies
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=cov.index)
            return weights
        else:
            return self._equal_risk_allocation(returns)
    
    def calculate_cvar_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate CVaR-based weights
        
        Args:
            returns: Strategy returns
            
        Returns:
            CVaR-based weights
        """
        if not SCIPY_AVAILABLE:
            return self._equal_risk_allocation(returns)
        
        # Calculate CVaR for each strategy
        cvar_values = {}
        for strategy in returns.columns:
            strategy_returns = returns[strategy].tail(self.config.cvar_lookback)
            
            # Calculate VaR
            var = np.percentile(strategy_returns, (1 - self.config.cvar_confidence) * 100)
            
            # Calculate CVaR (average of losses beyond VaR)
            losses = strategy_returns[strategy_returns < var]
            if len(losses) > 0:
                cvar = losses.mean()
            else:
                cvar = var
            
            cvar_values[strategy] = abs(cvar)
        
        # Inverse CVaR weighting (lower CVaR = higher weight)
        inv_cvar = 1 / pd.Series(cvar_values)
        weights = inv_cvar / inv_cvar.sum()
        
        return weights
    
    def _equal_risk_allocation(self, returns: pd.DataFrame) -> pd.Series:
        """Fallback: equal risk allocation"""
        n_strategies = len(returns.columns)
        weights = pd.Series(np.ones(n_strategies) / n_strategies, index=returns.columns)
        return weights
    
    def allocate_risk_budget(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Allocate risk budget across strategies
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dictionary of strategy -> risk budget
        """
        # Estimate volatilities
        volatilities = self.estimate_volatilities(returns)
        
        # Calculate weights based on method
        if self.config.method == "risk_parity":
            weights = self.calculate_risk_parity_weights(returns)
        elif self.config.method == "cvar":
            weights = self.calculate_cvar_weights(returns)
        else:
            weights = self._equal_risk_allocation(returns)
        
        # Calculate risk budgets
        risk_budgets = weights * self.config.target_portfolio_volatility
        
        # Apply risk limits
        risk_budgets = risk_budgets.clip(
            lower=self.config.min_strategy_risk,
            upper=self.config.max_strategy_risk
        )
        
        # Normalize
        risk_budgets = risk_budgets / risk_budgets.sum() * self.config.target_portfolio_volatility
        
        self.current_risk_budgets = risk_budgets
        return risk_budgets.to_dict()
    
    def calculate_position_sizes(self, risk_budgets: Dict[str, float], 
                                returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate position sizes from risk budgets
        
        Args:
            risk_budgets: Risk budget per strategy
            returns: Strategy returns
            
        Returns:
            Dictionary of strategy -> position size
        """
        position_sizes = {}
        
        for strategy, risk_budget in risk_budgets.items():
            if strategy in self.strategy_volatilities.index:
                vol = self.strategy_volatilities[strategy]
                
                # Position size = risk_budget / volatility
                position_size = risk_budget / vol
                position_sizes[strategy] = position_size
        
        return position_sizes
    
    def backtest(self, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest dynamic risk budgeting
        
        Args:
            returns: Strategy returns
            
        Returns:
            Strategy returns
        """
        strategy_returns = []
        
        for i in range(self.config.cvar_lookback, len(returns)):
            window_returns = returns.iloc[i-self.config.cvar_lookback:i]
            
            # Allocate risk budget
            risk_budgets = self.allocate_risk_budget(window_returns)
            
            # Calculate position sizes
            position_sizes = self.calculate_position_sizes(risk_budgets, window_returns)
            
            # Calculate return for next period
            next_return = returns.iloc[i]
            strategy_return = sum(position_sizes.get(strategy, 0) * next_return[strategy] 
                                for strategy in position_sizes.keys())
            
            strategy_returns.append(strategy_return)
        
        return pd.Series(strategy_returns, index=returns.index[self.config.cvar_lookback:])
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary"""
        if self.current_risk_budgets is None:
            return {}
        
        return {
            "total_risk_budget": self.current_risk_budgets.sum(),
            "strategy_risk_budgets": self.current_risk_budgets.to_dict(),
            "num_strategies": len(self.current_risk_budgets)
        }


def simulate_strategy_returns(n_strategies: int = 5, n_days: int = 500) -> pd.DataFrame:
    """Simulate strategy returns for testing"""
    np.random.seed(42)
    
    strategy_names = [f"STRATEGY_{i}" for i in range(n_strategies)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.3, 0.2, 0.4, 0.1],
        [0.3, 1.0, 0.3, 0.2, 0.2],
        [0.2, 0.3, 1.0, 0.3, 0.3],
        [0.4, 0.2, 0.3, 1.0, 0.2],
        [0.1, 0.2, 0.3, 0.2, 1.0]
    ])
    
    L = np.linalg.cholesky(correlation_matrix)
    
    independent_returns = np.random.randn(n_days, n_strategies)
    correlated_returns = independent_returns @ L.T
    
    # Add drift and scale
    drifts = np.array([0.0003, 0.0002, 0.0004, 0.0001, 0.0002])
    scales = np.array([0.015, 0.01, 0.02, 0.012, 0.018])
    
    returns = pd.DataFrame(
        correlated_returns * scales + drifts,
        index=dates,
        columns=strategy_names
    )
    
    return returns


if __name__ == "__main__":
    # Example usage
    config = RiskBudgetConfig(
        target_portfolio_volatility=0.15,
        method="risk_parity",
        cvar_confidence=0.95
    )
    
    risk_budgeting = DynamicRiskBudgeting(config)
    
    # Simulate data
    print("Simulating strategy returns...")
    returns = simulate_strategy_returns(5, 500)
    
    # Allocate risk budget
    print("\nAllocating risk budget...")
    risk_budgets = risk_budgeting.allocate_risk_budget(returns)
    
    print(f"\nRisk Budgets:")
    for strategy, budget in risk_budgets.items():
        print(f"  {strategy}: {budget:.4f}")
    
    # Risk summary
    print("\nRisk Summary:")
    summary = risk_budgeting.get_risk_summary()
    for key, value in summary.items():
        if key != "strategy_risk_budgets":
            print(f"  {key}: {value}")
    
    # Backtest
    print("\nBacktesting dynamic risk budgeting...")
    strategy_returns = risk_budgeting.backtest(returns)
    
    # Performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {total_return:.4f}")
    print(f"  Sharpe Ratio: {sharpe:.4f}")

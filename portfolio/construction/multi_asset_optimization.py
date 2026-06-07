"""
Multi-Asset Portfolio Optimization

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#41)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Multi-asset class optimization
- Cross-asset correlation modeling
- Dynamic asset allocation
- Risk parity across asset classes
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
class MultiAssetConfig:
    """Configuration for Multi-Asset Optimization"""
    # Asset classes
    asset_classes: List[str] = None
    
    # Optimization parameters
    lookback_window: int = 252  # 1 year lookback
    rebalance_frequency: str = "monthly"
    
    # Risk parameters
    risk_free_rate: float = 0.05
    max_weight_per_asset: float = 0.4  # 40% max weight per asset
    min_weight_per_asset: float = 0.0  # 0% min weight
    
    # Risk parity parameters
    use_risk_parity: bool = True
    risk_parity_tolerance: float = 0.05
    
    # Constraints
    max_leverage: float = 1.5  # Maximum leverage


class MultiAssetOptimizer:
    """
    Multi-Asset Portfolio Optimizer
    
    Optimizes portfolio weights across multiple asset classes
    considering correlations and risk contributions.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: MultiAssetConfig):
        self.config = config
        
        if config.asset_classes is None:
            self.config.asset_classes = ["Equity", "Bonds", "Gold", "REITs", "Commodities"]
        
        # Current weights
        self.current_weights: Optional[pd.Series] = None
        
        # Covariance matrix
        self.covariance_matrix: Optional[pd.DataFrame] = None
    
    def calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate covariance matrix
        
        Args:
            returns: Asset returns
            
        Returns:
            Covariance matrix
        """
        cov = returns.tail(self.config.lookback_window).cov()
        self.covariance_matrix = cov
        return cov
    
    def optimize_max_sharpe(self, returns: pd.DataFrame) -> pd.Series:
        """
        Optimize for maximum Sharpe ratio
        
        Args:
            returns: Asset returns
            
        Returns:
            Optimal weights
        """
        if not SCIPY_AVAILABLE:
            return self._equal_weight_allocation(returns)
        
        cov = self.calculate_covariance(returns)
        mean_returns = returns.tail(self.config.lookback_window).mean()
        
        n_assets = len(mean_returns)
        
        # Objective function (negative Sharpe)
        def objective(weights):
            portfolio_return = mean_returns @ weights
            portfolio_std = np.sqrt(weights @ cov @ weights)
            sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_std
            return -sharpe
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                 for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=mean_returns.index)
            self.current_weights = weights
            return weights
        else:
            return self._equal_weight_allocation(returns)
    
    def optimize_risk_parity(self, returns: pd.DataFrame) -> pd.Series:
        """
        Optimize for risk parity (equal risk contribution)
        
        Args:
            returns: Asset returns
            
        Returns:
            Optimal weights
        """
        if not SCIPY_AVAILABLE:
            return self._equal_weight_allocation(returns)
        
        cov = self.calculate_covariance(returns)
        n_assets = cov.shape[0]
        
        # Objective function (minimize risk contribution variance)
        def objective(weights):
            portfolio_std = np.sqrt(weights @ cov @ weights)
            marginal_risk = cov @ weights / portfolio_std
            risk_contributions = weights * marginal_risk
            target_risk = portfolio_std / n_assets
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                 for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n_assets) / n_assets
        
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
            self.current_weights = weights
            return weights
        else:
            return self._equal_weight_allocation(returns)
    
    def optimize_minimum_variance(self, returns: pd.DataFrame) -> pd.Series:
        """
        Optimize for minimum variance
        
        Args:
            returns: Asset returns
            
        Returns:
            Optimal weights
        """
        if not SCIPY_AVAILABLE:
            return self._equal_weight_allocation(returns)
        
        cov = self.calculate_covariance(returns)
        n_assets = cov.shape[0]
        
        # Objective function (portfolio variance)
        def objective(weights):
            return weights @ cov @ weights
        
        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight_per_asset, self.config.max_weight_per_asset) 
                 for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n_assets) / n_assets
        
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
            self.current_weights = weights
            return weights
        else:
            return self._equal_weight_allocation(returns)
    
    def _equal_weight_allocation(self, returns: pd.DataFrame) -> pd.Series:
        """Fallback: equal weight allocation"""
        n_assets = len(returns.columns)
        weights = pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)
        self.current_weights = weights
        return weights
    
    def optimize_combined(self, returns: pd.DataFrame) -> pd.Series:
        """
        Combined optimization (mix of strategies)
        
        Args:
            returns: Asset returns
            
        Returns:
            Optimal weights
        """
        if self.config.use_risk_parity:
            return self.optimize_risk_parity(returns)
        else:
            return self.optimize_max_sharpe(returns)
    
    def backtest(self, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest multi-asset strategy
        
        Args:
            returns: Asset returns
            
        Returns:
            Strategy returns
        """
        strategy_returns = []
        
        for i in range(self.config.lookback_window, len(returns)):
            window_returns = returns.iloc[i-self.config.lookback_window:i]
            
            # Optimize weights
            weights = self.optimize_combined(window_returns)
            
            # Calculate return for next period
            next_return = returns.iloc[i]
            strategy_return = (weights * next_return).sum()
            
            strategy_returns.append(strategy_return)
        
        return pd.Series(strategy_returns, index=returns.index[self.config.lookback_window:])
    
    def get_performance_metrics(self, strategy_returns: pd.Series) -> Dict:
        """Get performance metrics"""
        if len(strategy_returns) == 0:
            return {}
        
        total_return = (1 + strategy_returns).prod() - 1
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
        
        # Drawdown
        cum_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }


def simulate_multi_asset_data(n_assets: int = 5, n_days: int = 500) -> pd.DataFrame:
    """Simulate multi-asset data for testing"""
    np.random.seed(42)
    
    asset_names = ["Equity", "Bonds", "Gold", "REITs", "Commodities"]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.3, 0.1, 0.4, 0.2],
        [0.3, 1.0, 0.2, 0.3, 0.1],
        [0.1, 0.2, 1.0, 0.2, 0.5],
        [0.4, 0.3, 0.2, 1.0, 0.3],
        [0.2, 0.1, 0.5, 0.3, 1.0]
    ])
    
    # Cholesky decomposition
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate independent returns
    independent_returns = np.random.randn(n_days, n_assets)
    
    # Apply correlation
    correlated_returns = independent_returns @ L.T
    
    # Add drift and scale
    drifts = np.array([0.0005, 0.0002, 0.0001, 0.0003, 0.0001])
    scales = np.array([0.02, 0.01, 0.015, 0.025, 0.02])
    
    returns = pd.DataFrame(
        correlated_returns * scales + drifts,
        index=dates,
        columns=asset_names
    )
    
    return returns


if __name__ == "__main__":
    # Example usage
    config = MultiAssetConfig(
        asset_classes=["Equity", "Bonds", "Gold", "REITs", "Commodities"],
        lookback_window=252,
        use_risk_parity=True,
        max_weight_per_asset=0.4
    )
    
    optimizer = MultiAssetOptimizer(config)
    
    # Simulate data
    print("Simulating multi-asset data...")
    returns = simulate_multi_asset_data(5, 500)
    
    # Optimize weights
    print("\nOptimizing portfolio weights...")
    weights = optimizer.optimize_combined(returns)
    
    print(f"\nOptimal Weights:")
    for asset, weight in weights.items():
        print(f"  {asset}: {weight:.2%}")
    
    # Backtest
    print("\nBacktesting multi-asset strategy...")
    strategy_returns = optimizer.backtest(returns)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = optimizer.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

"""
Volatility Targeting Module

Based on Comprehensive Upgrade Analysis - Tier 1 Upgrade (#3)
Expected Sharpe improvement: +0.3–0.5
Smooths returns, reduces max drawdown

Methodology:
- Scale positions inversely with realized volatility
- Target constant volatility (e.g., 15% annual)
- Use exponential weighting for volatility estimation
- Apply leverage caps and position limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque


@dataclass
class VolTargetConfig:
    """Configuration for Volatility Targeting"""
    target_volatility: float = 0.15  # 15% annual target volatility
    vol_estimation_window: int = 20  # 20-day window for volatility estimation
    vol_update_frequency: str = "daily"  # "daily" or "intraday"
    max_leverage: float = 2.0  # Maximum leverage cap
    min_leverage: float = 0.5  # Minimum leverage (don't go below 50%)
    leverage_smoothing: float = 0.5  # EWMA smoothing factor for leverage
    use_ewma: bool = True  # Use EWMA for volatility estimation
    ewma_lambda: float = 0.94  # EWMA decay factor (RiskMetrics standard)
    min_vol_floor: float = 0.05  # 5% minimum volatility floor
    max_vol_cap: float = 0.50  # 50% maximum volatility cap


class VolatilityTargeting:
    """
    Volatility Targeting Engine
    
    Scales portfolio positions to achieve target volatility.
    
    Formula:
    leverage = target_vol / realized_vol
    
    With constraints:
    - min_leverage <= leverage <= max_leverage
    - min_vol_floor <= realized_vol <= max_vol_cap
    - Leverage smoothing to avoid abrupt changes
    
    Expected Sharpe improvement: +0.3–0.5
    """
    
    def __init__(self, config: VolTargetConfig):
        self.config = config
        
        # Volatility history
        self.vol_history: deque = deque(maxlen=config.vol_estimation_window)
        self.ewma_vol: Optional[float] = None
        
        # Leverage history
        self.leverage_history: deque = deque(maxlen=100)
        self.current_leverage: float = 1.0
        
        # Return history for volatility estimation
        self.return_history: deque = deque(maxlen=config.vol_estimation_window)
    
    def update_returns(self, returns: pd.Series) -> None:
        """
        Update return history for volatility estimation
        
        Args:
            returns: Series of returns
        """
        for ret in returns:
            self.return_history.append(ret)
    
    def estimate_volatility(self) -> float:
        """
        Estimate realized volatility
        
        Returns:
            Annualized volatility estimate
        """
        if len(self.return_history) < 2:
            return self.config.target_volatility
        
        returns = np.array(self.return_history)
        
        if self.config.use_ewma:
            # EWMA volatility estimation (RiskMetrics)
            if self.ewma_vol is None:
                # Initialize with simple std
                self.ewma_vol = np.std(returns) * np.sqrt(252)
            else:
                # Update EWMA
                lambda_factor = self.config.ewma_lambda
                latest_return = returns[-1]
                self.ewma_vol = np.sqrt(
                    lambda_factor * self.ewma_vol**2 + 
                    (1 - lambda_factor) * latest_return**2
                ) * np.sqrt(252)
            
            vol = self.ewma_vol
        else:
            # Simple rolling standard deviation
            vol = np.std(returns) * np.sqrt(252)
        
        # Apply volatility floor and cap
        vol = np.clip(vol, self.config.min_vol_floor, self.config.max_vol_cap)
        
        self.vol_history.append(vol)
        
        return vol
    
    def compute_leverage(self, realized_vol: float) -> float:
        """
        Compute leverage based on volatility targeting
        
        Args:
            realized_vol: Current realized volatility (annualized)
            
        Returns:
            Target leverage
        """
        # Basic volatility targeting formula
        target_leverage = self.config.target_volatility / realized_vol
        
        # Apply leverage constraints
        target_leverage = np.clip(target_leverage, 
                                  self.config.min_leverage, 
                                  self.config.max_leverage)
        
        # Apply leverage smoothing
        if self.config.leverage_smoothing > 0:
            smoothed_leverage = (
                self.config.leverage_smoothing * target_leverage +
                (1 - self.config.leverage_smoothing) * self.current_leverage
            )
            target_leverage = smoothed_leverage
        
        self.current_leverage = target_leverage
        self.leverage_history.append(target_leverage)
        
        return target_leverage
    
    def get_position_scale(self) -> float:
        """
        Get current position scaling factor
        
        Returns:
            Scaling factor for positions
        """
        realized_vol = self.estimate_volatility()
        leverage = self.compute_leverage(realized_vol)
        return leverage
    
    def scale_positions(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        Scale positions by volatility targeting factor
        
        Args:
            positions: Dictionary of symbol -> position size
            
        Returns:
            Scaled positions
        """
        scale_factor = self.get_position_scale()
        
        scaled_positions = {
            symbol: size * scale_factor
            for symbol, size in positions.items()
        }
        
        return scaled_positions
    
    def get_metrics(self) -> Dict:
        """Get volatility targeting metrics"""
        realized_vol = self.estimate_volatility()
        
        return {
            "target_volatility": self.config.target_volatility,
            "realized_volatility": realized_vol,
            "current_leverage": self.current_leverage,
            "leverage_history": list(self.leverage_history),
            "vol_history": list(self.vol_history),
            "vol_gap": realized_vol - self.config.target_volatility
        }


class VolatilityManagedPortfolio:
    """
    Volatility-Managed Portfolio
    
    Combines volatility targeting with position sizing and risk limits.
    """
    
    def __init__(self, config: VolTargetConfig):
        self.config = config
        self.vol_targeting = VolatilityTargeting(config)
        
        # Portfolio state
        self.current_positions: Dict[str, float] = {}
        self.portfolio_value: float = 1_000_000  # Starting with $1M
        
        # Performance tracking
        self.returns_history: List[float] = []
        self.leverage_history: List[float] = []
    
    def update(self, returns: Dict[str, float]) -> float:
        """
        Update portfolio with new returns
        
        Args:
            returns: Dictionary of symbol -> return
            
        Returns:
            Portfolio return
        """
        # Update return history for volatility estimation
        portfolio_return = sum(
            self.current_positions.get(symbol, 0) * ret
            for symbol, ret in returns.items()
        )
        
        self.returns_history.append(portfolio_return)
        self.vol_targeting.update_returns(pd.Series([portfolio_return]))
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Record leverage
        self.leverage_history.append(self.vol_targeting.current_leverage)
        
        return portfolio_return
    
    def rebalance(self, target_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Rebalance portfolio with volatility targeting
        
        Args:
            target_positions: Target positions (before volatility scaling)
            
        Returns:
            Scaled target positions
        """
        # Scale positions by volatility targeting
        scaled_positions = self.vol_targeting.scale_positions(target_positions)
        
        # Update current positions
        self.current_positions = scaled_positions
        
        return scaled_positions
    
    def get_portfolio_metrics(self) -> Dict:
        """Get portfolio performance metrics"""
        if len(self.returns_history) < 2:
            return {
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "current_leverage": self.vol_targeting.current_leverage
            }
        
        returns = np.array(self.returns_history)
        
        # Total return
        total_return = np.sum(returns)
        
        # Sharpe ratio
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "current_leverage": self.vol_targeting.current_leverage,
            "portfolio_value": self.portfolio_value
        }


def backtest_volatility_targeting(
    returns_data: pd.DataFrame,
    config: VolTargetConfig
) -> Dict:
    """
    Backtest volatility targeting on historical returns
    
    Args:
        returns_data: DataFrame with returns (index can be datetime)
        config: Volatility targeting configuration
        
    Returns:
        Dictionary with backtest results
    """
    portfolio = VolatilityManagedPortfolio(config)
    
    # Simulate trading
    for date, row in returns_data.iterrows():
        # Assume equal-weighted target positions
        n_assets = len(row)
        target_positions = {col: 1.0/n_assets for col in row.index}
        
        # Rebalance with volatility targeting
        scaled_positions = portfolio.rebalance(target_positions)
        
        # Update with returns
        portfolio.update(row.to_dict())
    
    # Get metrics
    metrics = portfolio.get_portfolio_metrics()
    vol_metrics = portfolio.vol_targeting.get_metrics()
    
    return {
        "portfolio_metrics": metrics,
        "volatility_metrics": vol_metrics
    }


if __name__ == "__main__":
    # Example usage
    config = VolTargetConfig(
        target_volatility=0.15,
        vol_estimation_window=20,
        max_leverage=2.0,
        use_ewma=True,
        ewma_lambda=0.94
    )
    
    # Generate synthetic returns data for testing
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    
    # Generate returns with time-varying volatility
    vol_regime = np.sin(np.linspace(0, 4*np.pi, n_days)) * 0.1 + 0.15
    returns_data = pd.DataFrame(
        np.random.randn(n_days, n_assets) * vol_regime[:, np.newaxis] * 0.01,
        index=pd.date_range(start="2023-01-01", periods=n_days),
        columns=["ASSET1", "ASSET2", "ASSET3", "ASSET4", "ASSET5"]
    )
    
    print("Backtesting volatility targeting...")
    results = backtest_volatility_targeting(returns_data, config)
    
    print(f"\nPortfolio Metrics:")
    for key, value in results["portfolio_metrics"].items():
        print(f"  {key}: {value}")
    
    print(f"\nVolatility Metrics:")
    for key, value in results["volatility_metrics"].items():
        if isinstance(value, list):
            print(f"  {key}: {value[-5:]}")  # Show last 5 values
        else:
            print(f"  {key}: {value}")

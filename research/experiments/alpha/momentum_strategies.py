"""
Momentum Strategies

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#39)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Cross-sectional momentum
- Time-series momentum
- Factor momentum
- Risk-adjusted momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MomentumConfig:
    """Configuration for Momentum Strategies"""
    # Cross-sectional momentum parameters
    lookback_period: int = 252  # 1 year lookback
    holding_period: int = 20  # 1 month holding
    n_stocks_long: int = 10  # Number of stocks to long
    n_stocks_short: int = 10  # Number of stocks to short
    
    # Time-series momentum parameters
    ts_lookback: int = 126  # 6 months lookback
    ts_threshold: float = 0.02  # 2% threshold
    
    # Risk parameters
    volatility_adjustment: bool = True  # Adjust for volatility
    max_position_pct: float = 0.05  # 5% max position per stock
    
    # Rebalancing
    rebalance_frequency: str = "monthly"


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum Strategy
    
    Ranks stocks by past returns and goes long winners,
    short losers.
    """
    
    def __init__(self, config: MomentumConfig):
        self.config = config
    
    def calculate_momentum_scores(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum scores for all stocks
        
        Args:
            returns: Asset returns
            
        Returns:
            Momentum scores
        """
        # Calculate cumulative returns over lookback period
        cumulative_returns = (1 + returns.tail(self.config.lookback_period)).prod() - 1
        
        return cumulative_returns
    
    def rank_stocks(self, momentum_scores: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Rank stocks by momentum
        
        Args:
            momentum_scores: Momentum scores
            
        Returns:
            Tuple of (long_list, short_list)
        """
        # Sort by momentum
        sorted_scores = momentum_scores.sort_values(ascending=False)
        
        # Select top and bottom
        long_stocks = sorted_scores.head(self.config.n_stocks_long).index.tolist()
        short_stocks = sorted_scores.tail(self.config.n_stocks_short).index.tolist()
        
        return long_stocks, short_stocks
    
    def calculate_positions(self, long_stocks: List[str], short_stocks: List[str], 
                          returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate positions from momentum ranking
        
        Args:
            long_stocks: Stocks to long
            short_stocks: Stocks to short
            returns: Asset returns
            
        Returns:
            Dictionary of asset -> position
        """
        positions = {}
        
        # Calculate volatility-adjusted weights
        if self.config.volatility_adjustment:
            vol = returns.tail(20).std()
            inv_vol = 1 / (vol + 1e-8)
            
            # Normalize
            inv_vol = inv_vol / inv_vol.sum()
            
            for stock in long_stocks:
                positions[stock] = inv_vol[stock] * self.config.max_position_pct
            
            for stock in short_stocks:
                positions[stock] = -inv_vol[stock] * self.config.max_position_pct
        else:
            # Equal weights
            long_weight = self.config.max_position_pct / len(long_stocks)
            short_weight = self.config.max_position_pct / len(short_stocks)
            
            for stock in long_stocks:
                positions[stock] = long_weight
            
            for stock in short_stocks:
                positions[stock] = -short_weight
        
        return positions


class TimeSeriesMomentum:
    """
    Time-Series Momentum Strategy
    
    Goes long assets with positive past returns,
    short assets with negative past returns.
    """
    
    def __init__(self, config: MomentumConfig):
        self.config = config
    
    def calculate_ts_momentum(self, returns: pd.Series) -> float:
        """
        Calculate time-series momentum score
        
        Args:
            returns: Asset returns
            
        Returns:
            Momentum score
        """
        cumulative_return = (1 + returns.tail(self.config.ts_lookback)).prod() - 1
        return cumulative_return
    
    def generate_signal(self, returns: pd.DataFrame) -> Dict[str, str]:
        """
        Generate time-series momentum signals
        
        Args:
            returns: Asset returns
            
        Returns:
            Dictionary of asset -> signal
        """
        signals = {}
        
        for asset in returns.columns:
            momentum = self.calculate_ts_momentum(returns[asset])
            
            if momentum > self.config.ts_threshold:
                signals[asset] = "long"
            elif momentum < -self.config.ts_threshold:
                signals[asset] = "short"
            else:
                signals[asset] = "hold"
        
        return signals


class MomentumStrategy:
    """
    Combined Momentum Strategy
    
    Combines cross-sectional and time-series momentum
    for robust performance.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: MomentumConfig):
        self.config = config
        
        self.cs_momentum = CrossSectionalMomentum(config)
        self.ts_momentum = TimeSeriesMomentum(config)
        
        # Current positions
        self.positions: Dict[str, float] = {}
    
    def generate_signals(self, returns: pd.DataFrame) -> Dict[str, str]:
        """
        Generate combined momentum signals
        
        Args:
            returns: Asset returns
            
        Returns:
            Dictionary of asset -> signal
        """
        # Get cross-sectional signals
        momentum_scores = self.cs_momentum.calculate_momentum_scores(returns)
        long_stocks, short_stocks = self.cs_momentum.rank_stocks(momentum_scores)
        
        # Get time-series signals
        ts_signals = self.ts_momentum.generate_signal(returns)
        
        # Combine signals
        combined_signals = {}
        
        for asset in returns.columns:
            if asset in long_stocks and ts_signals.get(asset) == "long":
                combined_signals[asset] = "strong_long"
            elif asset in long_stocks:
                combined_signals[asset] = "long"
            elif asset in short_stocks and ts_signals.get(asset) == "short":
                combined_signals[asset] = "strong_short"
            elif asset in short_stocks:
                combined_signals[asset] = "short"
            else:
                combined_signals[asset] = "hold"
        
        return combined_signals
    
    def calculate_positions(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate positions from signals
        
        Args:
            returns: Asset returns
            
        Returns:
            Dictionary of asset -> position
        """
        # Get cross-sectional positions
        momentum_scores = self.cs_momentum.calculate_momentum_scores(returns)
        long_stocks, short_stocks = self.cs_momentum.rank_stocks(momentum_scores)
        
        positions = self.cs_momentum.calculate_positions(long_stocks, short_stocks, returns)
        
        self.positions = positions
        return positions
    
    def backtest(self, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest momentum strategy
        
        Args:
            returns: Asset returns
            
        Returns:
            Strategy returns
        """
        strategy_returns = []
        
        for i in range(self.config.lookback_period, len(returns)):
            window_returns = returns.iloc[i-self.config.lookback_period:i]
            
            # Calculate positions
            positions = self.calculate_positions(window_returns)
            
            # Calculate return for next period
            next_return = returns.iloc[i]
            strategy_return = sum(positions.get(asset, 0) * next_return[asset] 
                                for asset in positions.keys())
            
            strategy_returns.append(strategy_return)
        
        return pd.Series(strategy_returns, index=returns.index[self.config.lookback_period:])
    
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


def simulate_momentum_data(n_assets: int = 50, n_days: int = 500) -> pd.DataFrame:
    """Simulate momentum data for testing"""
    np.random.seed(42)
    
    # Generate assets with different momentum characteristics
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    returns = pd.DataFrame(index=dates, columns=asset_names)
    
    for i in range(n_assets):
        # Some assets have positive momentum, some negative
        if i < 25:
            drift = 0.0002  # Positive drift
        else:
            drift = -0.0001  # Negative drift
        
        asset_returns = np.random.randn(n_days) * 0.02 + drift
        returns[asset_names[i]] = asset_returns
    
    return returns


if __name__ == "__main__":
    # Example usage
    config = MomentumConfig(
        lookback_period=252,
        holding_period=20,
        n_stocks_long=10,
        n_stocks_short=10,
        volatility_adjustment=True
    )
    
    strategy = MomentumStrategy(config)
    
    # Simulate data
    print("Simulating momentum data...")
    returns = simulate_momentum_data(50, 500)
    
    # Backtest
    print("\nBacktesting momentum strategy...")
    strategy_returns = strategy.backtest(returns)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = strategy.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Current signals
    print("\nCurrent Signals:")
    signals = strategy.generate_signals(returns)
    active_signals = {k: v for k, v in signals.items() if v != "hold"}
    print(f"  Active signals: {len(active_signals)}")
    for asset, signal in list(active_signals.items())[:5]:
        print(f"    {asset}: {signal}")

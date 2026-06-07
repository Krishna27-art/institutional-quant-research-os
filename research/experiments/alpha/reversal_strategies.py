"""
Reversal Strategies

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#40)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Short-term reversal
- Overnight reversal
- Intraday reversal
- Volume-weighted reversal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ReversalConfig:
    """Configuration for Reversal Strategies"""
    # Short-term reversal parameters
    reversal_window: int = 5  # 5-day reversal window
    reversal_threshold: float = 0.05  # 5% threshold
    
    # Overnight reversal parameters
    overnight_threshold: float = 0.02  # 2% overnight threshold
    
    # Volume parameters
    volume_window: int = 20
    volume_threshold: float = 1.5  # Volume multiplier
    
    # Position sizing
    position_size: float = 0.1  # Position size per signal
    
    # Risk parameters
    max_drawdown_pct: float = 0.10  # 10% max drawdown


class ShortTermReversal:
    """
    Short-Term Reversal Strategy
    
    Fades short-term price moves.
    """
    
    def __init__(self, config: ReversalConfig):
        self.config = config
    
    def calculate_reversal_signal(self, returns: pd.Series) -> str:
        """
        Calculate reversal signal
        
        Args:
            returns: Return series
            
        Returns:
            Signal: "buy", "sell", or "hold"
        """
        # Calculate cumulative return over window
        cum_return = (1 + returns.tail(self.config.reversal_window)).prod() - 1
        
        if cum_return > self.config.reversal_threshold:
            return "sell"  # Fade the move
        elif cum_return < -self.config.reversal_threshold:
            return "buy"  # Fade the move
        else:
            return "hold"


class OvernightReversal:
    """
    Overnight Reversal Strategy
    
    Trades overnight price moves.
    """
    
    def __init__(self, config: ReversalConfig):
        self.config = config
    
    def calculate_overnight_return(self, prices: pd.DataFrame) -> pd.Series:
        """
        Calculate overnight returns
        
        Args:
            prices: Price data with OHLC
            
        Returns:
            Overnight returns
        """
        # Overnight return = (open - close_prev) / close_prev
        if "Open" in prices.columns and "Close" in prices.columns:
            overnight = (prices["Open"] - prices["Close"].shift(1)) / prices["Close"].shift(1)
            return overnight
        else:
            # Fallback: use close-to-close
            return prices.pct_change()
    
    def generate_signal(self, overnight_return: float) -> str:
        """
        Generate signal from overnight return
        
        Args:
            overnight_return: Overnight return
            
        Returns:
            Signal: "buy", "sell", or "hold"
        """
        if overnight_return > self.config.overnight_threshold:
            return "sell"  # Fade the overnight gap
        elif overnight_return < -self.config.overnight_threshold:
            return "buy"  # Fade the overnight gap
        else:
            return "hold"


class VolumeWeightedReversal:
    """
    Volume-Weighted Reversal Strategy
    
    Uses volume to confirm reversal signals.
    """
    
    def __init__(self, config: ReversalConfig):
        self.config = config
    
    def calculate_volume_signal(self, prices: pd.Series, volume: pd.Series) -> str:
        """
        Calculate volume-weighted reversal signal
        
        Args:
            prices: Price series
            volume: Volume series
            
        Returns:
            Signal: "buy", "sell", or "hold"
        """
        # Calculate price change
        price_change = prices.pct_change().iloc[-1]
        
        # Calculate volume ratio
        avg_volume = volume.rolling(self.config.volume_window).mean()
        volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1]
        
        # Only trade if volume is elevated
        if volume_ratio > self.config.volume_threshold:
            if price_change > self.config.reversal_threshold:
                return "sell"
            elif price_change < -self.config.reversal_threshold:
                return "buy"
        
        return "hold"


class ReversalStrategy:
    """
    Combined Reversal Strategy
    
    Combines multiple reversal signals for robust trading.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: ReversalConfig):
        self.config = config
        
        self.st_reversal = ShortTermReversal(config)
        self.overnight_reversal = OvernightReversal(config)
        self.volume_reversal = VolumeWeightedReversal(config)
        
        # Current positions
        self.positions: Dict[str, float] = {}
        
        # Entry prices
        self.entry_prices: Dict[str, float] = {}
    
    def generate_signals(self, prices: pd.DataFrame, volume: pd.DataFrame) -> Dict[str, str]:
        """
        Generate reversal signals for all assets
        
        Args:
            prices: Price data
            volume: Volume data
            
        Returns:
            Dictionary of asset -> signal
        """
        signals = {}
        
        for asset in prices.columns:
            # Get individual signals
            st_signal = self.st_reversal.calculate_reversal_signal(prices[asset].pct_change())
            
            # Overnight signal (if OHLC data available)
            if "Open" in prices.columns:
                overnight_return = self.overnight_reversal.calculate_overnight_return(prices)
                overnight_signal = self.overnight_reversal.generate_signal(overnight_return.iloc[-1])
            else:
                overnight_signal = "hold"
            
            # Volume signal
            volume_signal = self.volume_reversal.calculate_volume_signal(prices[asset], volume[asset])
            
            # Combine signals (majority vote)
            signals_list = [st_signal, overnight_signal, volume_signal]
            
            buy_votes = signals_list.count("buy")
            sell_votes = signals_list.count("sell")
            
            if buy_votes >= 2:
                signals[asset] = "buy"
            elif sell_votes >= 2:
                signals[asset] = "sell"
            else:
                signals[asset] = "hold"
        
        return signals
    
    def calculate_positions(self, signals: Dict[str, str], prices: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate positions from signals
        
        Args:
            signals: Trading signals
            prices: Current prices
            
        Returns:
            Dictionary of asset -> position
        """
        positions = {}
        
        for asset, signal in signals.items():
            if signal == "buy":
                positions[asset] = self.config.position_size
                self.entry_prices[asset] = prices[asset].iloc[-1]
            elif signal == "sell":
                positions[asset] = -self.config.position_size
                self.entry_prices[asset] = prices[asset].iloc[-1]
            else:
                positions[asset] = 0.0
                if asset in self.entry_prices:
                    del self.entry_prices[asset]
        
        self.positions = positions
        return positions
    
    def backtest(self, prices: pd.DataFrame, volume: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest reversal strategy
        
        Args:
            prices: Price data
            volume: Volume data
            returns: Return data
            
        Returns:
            Strategy returns
        """
        strategy_returns = []
        
        for i in range(self.config.reversal_window, len(prices)):
            window_prices = prices.iloc[i-self.config.reversal_window:i]
            window_volume = volume.iloc[i-self.config.reversal_window:i]
            
            # Generate signals
            signals = self.generate_signals(window_prices, window_volume)
            
            # Calculate positions
            positions = self.calculate_positions(signals, window_prices)
            
            # Calculate return for next period
            next_return = returns.iloc[i]
            strategy_return = sum(positions.get(asset, 0) * next_return[asset] 
                                for asset in positions.keys())
            
            strategy_returns.append(strategy_return)
        
        return pd.Series(strategy_returns, index=returns.index[self.config.reversal_window:])
    
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


def simulate_reversal_data(n_assets: int = 50, n_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate reversal data for testing"""
    np.random.seed(42)
    
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    prices = pd.DataFrame(index=dates, columns=asset_names)
    volume = pd.DataFrame(index=dates, columns=asset_names)
    
    for i in range(n_assets):
        # Generate prices with short-term reversals
        price = 100.0
        price_series = []
        volume_series = []
        
        for _ in range(n_days):
            # Add some short-term momentum that reverses
            momentum = np.random.randn() * 0.02
            reversal = -momentum * 0.3  # 30% reversal
            
            price = price * (1 + momentum + reversal)
            price_series.append(price)
            
            # Volume with occasional spikes
            vol = np.random.exponential(100000)
            if np.random.random() < 0.1:
                vol *= 2  # Volume spike
            volume_series.append(vol)
        
        prices[asset_names[i]] = price_series
        volume[asset_names[i]] = volume_series
    
    returns = prices.pct_change().dropna()
    
    return prices, volume, returns


if __name__ == "__main__":
    # Example usage
    config = ReversalConfig(
        reversal_window=5,
        reversal_threshold=0.05,
        volume_threshold=1.5,
        position_size=0.1
    )
    
    strategy = ReversalStrategy(config)
    
    # Simulate data
    print("Simulating reversal data...")
    prices, volume, returns = simulate_reversal_data(50, 500)
    
    # Backtest
    print("\nBacktesting reversal strategy...")
    strategy_returns = strategy.backtest(prices, volume, returns)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = strategy.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Current signals
    print("\nCurrent Signals:")
    signals = strategy.generate_signals(prices, volume)
    active_signals = {k: v for k, v in signals.items() if v != "hold"}
    print(f"  Active signals: {len(active_signals)}")
    for asset, signal in list(active_signals.items())[:5]:
        print(f"    {asset}: {signal}")

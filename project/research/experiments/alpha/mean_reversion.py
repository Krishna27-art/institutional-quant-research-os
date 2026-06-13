"""
Mean Reversion Strategies

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#38)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Bollinger Bands mean reversion
- RSI-based mean reversion
- Z-score mean reversion
- Statistical mean reversion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MeanReversionConfig:
    """Configuration for Mean Reversion Strategies"""
    # Bollinger Bands parameters
    bb_window: int = 20
    bb_std: float = 2.0
    
    # RSI parameters
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Z-score parameters
    z_window: int = 20
    z_entry_threshold: float = 2.0
    z_exit_threshold: float = 0.5
    
    # Position sizing
    position_size: float = 0.1  # Position size per signal
    
    # Stop loss
    stop_loss_pct: float = 0.05  # 5% stop loss


class BollingerBandsMeanReversion:
    """
    Bollinger Bands Mean Reversion Strategy
    
    Trades price deviations from moving mean.
    """
    
    def __init__(self, config: MeanReversionConfig):
        self.config = config
    
    def calculate_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = prices.rolling(self.config.bb_window).mean()
        std = prices.rolling(self.config.bb_window).std()
        
        upper_band = middle_band + self.config.bb_std * std
        lower_band = middle_band - self.config.bb_std * std
        
        return upper_band, middle_band, lower_band
    
    def generate_signal(self, prices: pd.Series) -> str:
        """
        Generate trading signal
        
        Args:
            prices: Price series
            
        Returns:
            Signal: "buy", "sell", or "hold"
        """
        upper_band, middle_band, lower_band = self.calculate_bands(prices)
        
        current_price = prices.iloc[-1]
        
        if current_price < lower_band.iloc[-1]:
            return "buy"  # Price below lower band - oversold
        elif current_price > upper_band.iloc[-1]:
            return "sell"  # Price above upper band - overbought
        else:
            return "hold"


class RSIMeanReversion:
    """
    RSI Mean Reversion Strategy
    
    Uses RSI indicator for mean reversion signals.
    """
    
    def __init__(self, config: MeanReversionConfig):
        self.config = config
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI
        
        Args:
            prices: Price series
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, prices: pd.Series) -> str:
        """
        Generate trading signal
        
        Args:
            prices: Price series
            
        Returns:
            Signal: "buy", "sell", or "hold"
        """
        rsi = self.calculate_rsi(prices)
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < self.config.rsi_oversold:
            return "buy"  # Oversold
        elif current_rsi > self.config.rsi_overbought:
            return "sell"  # Overbought
        else:
            return "hold"


class ZScoreMeanReversion:
    """
    Z-Score Mean Reversion Strategy
    
    Uses z-score of price deviations for mean reversion.
    """
    
    def __init__(self, config: MeanReversionConfig):
        self.config = config
    
    def calculate_z_score(self, prices: pd.Series) -> pd.Series:
        """
        Calculate z-score
        
        Args:
            prices: Price series
            
        Returns:
            Z-score series
        """
        mean = prices.rolling(self.config.z_window).mean()
        std = prices.rolling(self.config.z_window).std()
        
        z_score = (prices - mean) / std
        
        return z_score
    
    def generate_signal(self, prices: pd.Series) -> str:
        """
        Generate trading signal
        
        Args:
            prices: Price series
            
        Returns:
            Signal: "buy", "sell", or "hold"
        """
        z_score = self.calculate_z_score(prices)
        current_z = z_score.iloc[-1]
        
        if current_z < -self.config.z_entry_threshold:
            return "buy"  # Significantly below mean
        elif current_z > self.config.z_entry_threshold:
            return "sell"  # Significantly above mean
        else:
            return "hold"


class MeanReversionStrategy:
    """
    Combined Mean Reversion Strategy
    
    Combines multiple mean reversion signals for robust trading.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: MeanReversionConfig):
        self.config = config
        
        self.bb_strategy = BollingerBandsMeanReversion(config)
        self.rsi_strategy = RSIMeanReversion(config)
        self.z_strategy = ZScoreMeanReversion(config)
        
        # Current positions
        self.positions: Dict[str, float] = {}
        
        # Entry prices
        self.entry_prices: Dict[str, float] = {}
    
    def generate_signals(self, prices: pd.DataFrame) -> Dict[str, str]:
        """
        Generate signals for all assets
        
        Args:
            prices: Price DataFrame
            
        Returns:
            Dictionary of asset -> signal
        """
        signals = {}
        
        for asset in prices.columns:
            # Get individual signals
            bb_signal = self.bb_strategy.generate_signal(prices[asset])
            rsi_signal = self.rsi_strategy.generate_signal(prices[asset])
            z_signal = self.z_strategy.generate_signal(prices[asset])
            
            # Combine signals (majority vote)
            signals_list = [bb_signal, rsi_signal, z_signal]
            
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
    
    def check_stop_loss(self, prices: pd.DataFrame) -> Dict[str, str]:
        """
        Check stop-loss conditions
        
        Args:
            prices: Current prices
            
        Returns:
            Dictionary of asset -> action
        """
        actions = {}
        
        for asset, entry_price in self.entry_prices.items():
            current_price = prices[asset].iloc[-1]
            pnl_pct = (current_price - entry_price) / entry_price
            
            if abs(pnl_pct) > self.config.stop_loss_pct:
                actions[asset] = "close"
        
        return actions
    
    def backtest(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest mean reversion strategy
        
        Args:
            prices: Price data
            returns: Return data
            
        Returns:
            Strategy returns
        """
        strategy_returns = []
        
        for i in range(self.config.bb_window, len(prices)):
            window_prices = prices.iloc[i-self.config.bb_window:i]
            
            # Generate signals
            signals = self.generate_signals(window_prices)
            
            # Calculate positions
            positions = self.calculate_positions(signals, window_prices)
            
            # Calculate return for next period
            next_return = returns.iloc[i]
            strategy_return = sum(positions.get(asset, 0) * next_return[asset] 
                                for asset in positions.keys())
            
            strategy_returns.append(strategy_return)
        
        return pd.Series(strategy_returns, index=returns.index[self.config.bb_window:])
    
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


def simulate_mean_reverting_data(n_assets: int = 50, n_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate mean-reverting data for testing"""
    np.random.seed(42)
    
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    prices = pd.DataFrame(index=dates, columns=asset_names)
    
    for i in range(n_assets):
        # Mean-reverting process (Ornstein-Uhlenbeck)
        theta = 0.1  # Mean reversion speed
        mu = 100.0  # Long-term mean
        sigma = 2.0  # Volatility
        
        price = mu
        price_series = []
        
        for _ in range(n_days):
            price = price + theta * (mu - price) + sigma * np.random.randn()
            price_series.append(price)
        
        prices[asset_names[i]] = price_series
    
    returns = prices.pct_change().dropna()
    
    return prices, returns


if __name__ == "__main__":
    # Example usage
    config = MeanReversionConfig(
        bb_window=20,
        bb_std=2.0,
        rsi_window=14,
        z_window=20,
        z_entry_threshold=2.0,
        position_size=0.1
    )
    
    strategy = MeanReversionStrategy(config)
    
    # Simulate data
    print("Simulating mean-reverting data...")
    prices, returns = simulate_mean_reverting_data(50, 500)
    
    # Backtest
    print("\nBacktesting mean reversion strategy...")
    strategy_returns = strategy.backtest(prices, returns)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = strategy.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Current signals
    print("\nCurrent Signals:")
    signals = strategy.generate_signals(prices)
    active_signals = {k: v for k, v in signals.items() if v != "hold"}
    print(f"  Active signals: {len(active_signals)}")
    for asset, signal in list(active_signals.items())[:5]:
        print(f"    {asset}: {signal}")

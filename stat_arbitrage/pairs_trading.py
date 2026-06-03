"""
Pairs Trading

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#36)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Cointegration-based pair selection
- Spread trading
- Market-neutral pairs
- Statistical arbitrage on correlated assets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import coint
    from sklearn.linear_model import LinearRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")


@dataclass
class PairsTradingConfig:
    """Configuration for Pairs Trading"""
    # Pair selection parameters
    lookback_window: int = 252  # Lookback for cointegration test
    cointegration_threshold: float = 0.05  # P-value threshold
    
    # Trading parameters
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5  # Z-score threshold for exit
    stop_loss_threshold: float = 4.0  # Stop-loss threshold
    
    # Position sizing
    position_size: float = 0.5  # Position size per leg
    
    # Risk parameters
    max_pairs: int = 10  # Maximum number of pairs to trade
    max_correlation: float = 0.9  # Maximum correlation for pair selection


class PairsTrader:
    """
    Pairs Trading Strategy
    
    Identifies cointegrated asset pairs and trades the spread
    for market-neutral returns.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: PairsTradingConfig):
        self.config = config
        
        # Selected pairs
        self.selected_pairs: List[Tuple[str, str]] = []
        self.hedge_ratios: Dict[Tuple[str, str], float] = {}
        
        # Spread z-scores
        self.spread_z_scores: Dict[Tuple[str, str], pd.Series] = None
        
        # Current positions
        self.positions: Dict[str, float] = {}
        
        # Performance tracking
        self.pnl_history: List[float] = []
    
    def find_cointegrated_pairs(self, returns: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find cointegrated asset pairs
        
        Args:
            returns: Asset returns
            
        Returns:
            List of cointegrated pairs
        """
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available, using correlation")
            return self._find_correlated_pairs(returns)
        
        pairs = []
        assets = returns.columns.tolist()
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1 = assets[i]
                asset2 = assets[j]
                
                # Test for cointegration
                try:
                    score, pvalue, _ = coint(returns[asset1], returns[asset2])
                    
                    if pvalue < self.config.cointegration_threshold:
                        pairs.append((asset1, asset2))
                except:
                    continue
        
        # Sort by p-value and select top pairs
        pairs = sorted(pairs, key=lambda x: self._get_cointegration_pvalue(returns, x[0], x[1]))
        
        self.selected_pairs = pairs[:self.config.max_pairs]
        
        return self.selected_pairs
    
    def _find_correlated_pairs(self, returns: pd.DataFrame) -> List[Tuple[str, str]]:
        """Fallback: find correlated pairs"""
        corr_matrix = returns.corr()
        
        pairs = []
        assets = returns.columns.tolist()
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1 = assets[i]
                asset2 = assets[j]
                corr = corr_matrix.loc[asset1, asset2]
                
                if corr > 0.8:  # High correlation
                    pairs.append((asset1, asset2))
        
        return pairs[:self.config.max_pairs]
    
    def _get_cointegration_pvalue(self, returns: pd.DataFrame, asset1: str, asset2: str) -> float:
        """Get cointegration p-value for a pair"""
        try:
            _, pvalue, _ = coint(returns[asset1], returns[asset2])
            return pvalue
        except:
            return 1.0
    
    def calculate_hedge_ratios(self, returns: pd.DataFrame) -> None:
        """
        Calculate hedge ratios for selected pairs
        
        Args:
            returns: Asset returns
        """
        for asset1, asset2 in self.selected_pairs:
            # Use OLS to find hedge ratio
            model = LinearRegression()
            model.fit(returns[asset2].values.reshape(-1, 1), returns[asset1].values)
            
            hedge_ratio = model.coef_[0]
            self.hedge_ratios[(asset1, asset2)] = hedge_ratio
    
    def calculate_spread(self, prices: pd.DataFrame, asset1: str, asset2: str) -> pd.Series:
        """
        Calculate spread for a pair
        
        Args:
            prices: Price data
            asset1: First asset
            asset2: Second asset
            
        Returns:
            Spread series
        """
        hedge_ratio = self.hedge_ratios.get((asset1, asset2), 1.0)
        
        spread = prices[asset1] - hedge_ratio * prices[asset2]
        
        return spread
    
    def calculate_z_score(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate z-score of spread
        
        Args:
            spread: Spread series
            window: Rolling window
            
        Returns:
            Z-score series
        """
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        
        z_score = (spread - mean) / std
        
        return z_score
    
    def generate_signals(self, prices: pd.DataFrame) -> Dict[Tuple[str, str], str]:
        """
        Generate trading signals for all pairs
        
        Args:
            prices: Price data
            
        Returns:
            Dictionary of pair -> signal
        """
        signals = {}
        
        for asset1, asset2 in self.selected_pairs:
            # Calculate spread
            spread = self.calculate_spread(prices, asset1, asset2)
            
            # Calculate z-score
            z_score = self.calculate_z_score(spread)
            current_z = z_score.iloc[-1]
            
            # Generate signal
            if current_z > self.config.entry_threshold:
                signals[(asset1, asset2)] = "short_spread"  # Short asset1, long asset2
            elif current_z < -self.config.entry_threshold:
                signals[(asset1, asset2)] = "long_spread"  # Long asset1, short asset2
            elif abs(current_z) < self.config.exit_threshold:
                signals[(asset1, asset2)] = "close"
            else:
                signals[(asset1, asset2)] = "hold"
        
        return signals
    
    def calculate_positions(self, signals: Dict[Tuple[str, str], str]) -> Dict[str, float]:
        """
        Calculate positions from signals
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary of asset -> position
        """
        positions = {}
        
        for (asset1, asset2), signal in signals.items():
            if signal == "long_spread":
                positions[asset1] = self.config.position_size
                positions[asset2] = -self.config.position_size * self.hedge_ratios.get((asset1, asset2), 1.0)
            elif signal == "short_spread":
                positions[asset1] = -self.config.position_size
                positions[asset2] = self.config.position_size * self.hedge_ratios.get((asset1, asset2), 1.0)
            elif signal == "close":
                positions[asset1] = 0.0
                positions[asset2] = 0.0
        
        self.positions = positions
        return positions
    
    def backtest(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest pairs trading strategy
        
        Args:
            prices: Price data
            returns: Return data
            
        Returns:
            Strategy returns
        """
        # Find cointegrated pairs
        self.find_cointegrated_pairs(returns)
        
        if not self.selected_pairs:
            return pd.Series(0, index=returns.index)
        
        # Calculate hedge ratios
        self.calculate_hedge_ratios(returns)
        
        # Generate signals over time
        strategy_returns = []
        
        for i in range(self.config.lookback_window, len(prices)):
            window_prices = prices.iloc[i-self.config.lookback_window:i]
            window_returns = returns.iloc[i-self.config.lookback_window:i]
            
            # Recalculate pairs periodically
            if i % 20 == 0:
                self.find_cointegrated_pairs(window_returns)
                self.calculate_hedge_ratios(window_returns)
            
            # Generate signals
            signals = self.generate_signals(window_prices)
            
            # Calculate positions
            positions = self.calculate_positions(signals)
            
            # Calculate return for next period
            next_return = returns.iloc[i]
            strategy_return = sum(positions.get(asset, 0) * next_return[asset] 
                                for asset in positions.keys())
            
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
            "num_pairs": len(self.selected_pairs)
        }


def simulate_pair_data(n_assets: int = 50, n_days: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate cointegrated pair data for testing"""
    np.random.seed(42)
    
    # Generate common factor
    common_factor = np.cumsum(np.random.randn(n_days) * 0.01)
    
    # Generate asset prices
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    prices = pd.DataFrame(index=dates, columns=asset_names)
    
    for i in range(n_assets):
        # Some assets are cointegrated with the common factor
        if i < 20:
            prices[asset_names[i]] = 100 + common_factor * 0.5 + np.random.randn(n_days) * 0.5
        else:
            prices[asset_names[i]] = 100 + np.random.randn(n_days) * 2
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    return prices, returns


if __name__ == "__main__":
    # Example usage
    config = PairsTradingConfig(
        lookback_window=252,
        cointegration_threshold=0.05,
        entry_threshold=2.0,
        max_pairs=10
    )
    
    pairs_trader = PairsTrader(config)
    
    # Simulate data
    print("Simulating pair data...")
    prices, returns = simulate_pair_data(50, 500)
    
    # Find cointegrated pairs
    print("\nFinding cointegrated pairs...")
    pairs = pairs_trader.find_cointegrated_pairs(returns)
    print(f"  Found {len(pairs)} cointegrated pairs")
    
    if pairs:
        print(f"  Sample pairs: {pairs[:5]}")
    
    # Backtest
    print("\nBacktesting pairs trading...")
    strategy_returns = pairs_trader.backtest(prices, returns)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = pairs_trader.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Current signals
    print("\nCurrent Signals:")
    signals = pairs_trader.generate_signals(prices)
    active_signals = {k: v for k, v in signals.items() if v in ["long_spread", "short_spread"]}
    print(f"  Active signals: {len(active_signals)}")
    for pair, signal in list(active_signals.items())[:3]:
        print(f"    {pair}: {signal}")

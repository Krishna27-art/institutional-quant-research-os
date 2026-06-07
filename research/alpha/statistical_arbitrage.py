"""
Statistical Arbitrage

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#35)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Multivariate statistical arbitrage
- Factor model-based arbitrage
- Residual trading
- Market-neutral strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class StatArbConfig:
    """Configuration for Statistical Arbitrage"""
    # Factor model parameters
    n_factors: int = 5  # Number of factors
    lookback_window: int = 252  # Lookback window for factor estimation
    
    # Trading parameters
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5  # Z-score threshold for exit
    max_position_pct: float = 0.02  # Maximum position per stock (2%)
    
    # Risk parameters
    max_gross_exposure: float = 2.0  # Maximum gross exposure
    max_net_exposure: float = 0.1  # Maximum net exposure
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # "daily", "weekly"


class StatisticalArbitrage:
    """
    Statistical Arbitrage Strategy
    
    Uses factor models to identify mispriced securities.
    Trades residuals from factor model for market-neutral returns.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: StatArbConfig):
        self.config = config
        
        # Factor model
        self.factor_model = None
        self.factor_loadings: Optional[pd.DataFrame] = None
        
        # Residuals
        self.residuals: Optional[pd.DataFrame] = None
        self.z_scores: Optional[pd.DataFrame] = None
        
        # Current positions
        self.positions: Dict[str, float] = {}
        
        # Performance tracking
        self.pnl_history: List[float] = []
    
    def fit_factor_model(self, returns: pd.DataFrame) -> None:
        """
        Fit factor model to returns
        
        Args:
            returns: Asset returns
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available, using simple returns")
            self.residuals = returns
            self.z_scores = returns / returns.rolling(20).std()
            return
        
        # Use PCA for factor extraction
        pca = PCA(n_components=self.config.n_factors)
        pca.fit(returns.dropna())
        
        # Get factor loadings
        self.factor_loadings = pd.DataFrame(
            pca.components_.T,
            index=returns.columns,
            columns=[f"Factor_{i}" for i in range(self.config.n_factors)]
        )
        
        # Calculate residuals
        factor_returns = pca.transform(returns.dropna())
        reconstructed = factor_returns @ pca.components_
        
        self.residuals = returns.dropna() - pd.DataFrame(
            reconstructed,
            index=returns.dropna().index,
            columns=returns.columns
        )
        
        # Calculate z-scores
        self.z_scores = self.residuals / self.residuals.rolling(20).std()
    
    def generate_signals(self) -> Dict[str, float]:
        """
        Generate trading signals from z-scores
        
        Returns:
            Dictionary of asset -> signal
        """
        if self.z_scores is None:
            return {}
        
        # Get latest z-scores
        latest_z = self.z_scores.iloc[-1]
        
        signals = {}
        for asset in latest_z.index:
            z = latest_z[asset]
            
            if z > self.config.entry_threshold:
                signals[asset] = -1.0  # Short overvalued
            elif z < -self.config.entry_threshold:
                signals[asset] = 1.0  # Buy undervalued
            else:
                signals[asset] = 0.0
        
        return signals
    
    def calculate_positions(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate positions from signals
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary of asset -> position size
        """
        positions = {}
        
        for asset, signal in signals.items():
            if signal != 0:
                # Scale position by signal strength and max position
                positions[asset] = signal * self.config.max_position_pct
        
        # Normalize to respect gross exposure limit
        total_gross = sum(abs(p) for p in positions.values())
        if total_gross > self.config.max_gross_exposure:
            scale = self.config.max_gross_exposure / total_gross
            positions = {k: v * scale for k, v in positions.items()}
        
        self.positions = positions
        return positions
    
    def backtest(self, returns: pd.DataFrame) -> pd.Series:
        """
        Backtest statistical arbitrage strategy
        
        Args:
            returns: Asset returns
            
        Returns:
            Strategy returns
        """
        # Fit factor model
        self.fit_factor_model(returns)
        
        # Generate signals over time
        strategy_returns = []
        
        for i in range(self.config.lookback_window, len(returns)):
            window_returns = returns.iloc[i-self.config.lookback_window:i]
            
            # Fit factor model on window
            self.fit_factor_model(window_returns)
            
            # Generate signals
            signals = self.generate_signals()
            
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
            "calmar_ratio": total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }


def simulate_returns(n_assets: int = 50, n_days: int = 500) -> pd.DataFrame:
    """Simulate asset returns for testing"""
    np.random.seed(42)
    
    # Generate factor returns
    factor_returns = np.random.randn(n_days, 5) * 0.01
    
    # Generate factor loadings
    factor_loadings = np.random.randn(n_assets, 5) * 0.5
    
    # Generate idiosyncratic returns
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.02
    
    # Generate asset returns
    asset_returns = factor_returns @ factor_loadings.T + idiosyncratic
    
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    return pd.DataFrame(asset_returns, index=dates, columns=asset_names)


if __name__ == "__main__":
    # Example usage
    config = StatArbConfig(
        n_factors=5,
        lookback_window=252,
        entry_threshold=2.0,
        max_gross_exposure=2.0
    )
    
    stat_arb = StatisticalArbitrage(config)
    
    # Simulate returns
    print("Simulating asset returns...")
    returns = simulate_returns(50, 500)
    
    # Backtest
    print("\nBacktesting statistical arbitrage...")
    strategy_returns = stat_arb.backtest(returns)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = stat_arb.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Current signals
    print("\nCurrent Signals:")
    signals = stat_arb.generate_signals()
    active_signals = {k: v for k, v in signals.items() if v != 0}
    print(f"  Active signals: {len(active_signals)}")
    for asset, signal in list(active_signals.items())[:5]:
        print(f"    {asset}: {signal:.2f}")

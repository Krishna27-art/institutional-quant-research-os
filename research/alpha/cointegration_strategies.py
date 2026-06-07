"""
Cointegration-Based Strategies

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#37)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Engle-Granger cointegration test
- Johansen cointegration test for multiple assets
- VECM (Vector Error Correction Model)
- Long-term equilibrium trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import coint, johansen
    from statsmodels.tsa.vector_ar.vecm import VECM
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Install with: pip install statsmodels")


@dataclass
class CointegrationConfig:
    """Configuration for Cointegration Strategies"""
    # Cointegration test parameters
    lookback_window: int = 252  # Lookback for cointegration test
    significance_level: float = 0.05  # Significance level
    
    # Johansen parameters
    det_order: int = 0  # Deterministic order
    k_ar_diff: int = 1  # Number of lags
    
    # Trading parameters
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5  # Z-score threshold for exit
    
    # VECM parameters
    vecm_lags: int = 2  # VECM lags


class CointegrationStrategy:
    """
    Cointegration-Based Strategy
    
    Uses cointegration relationships to identify long-term
    equilibrium relationships and trade deviations.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: CointegrationConfig):
        self.config = config
        
        # Cointegration vectors
        self.cointegration_vectors: Optional[np.ndarray] = None
        
        # Error correction terms
        self.error_correction_terms: Optional[pd.DataFrame] = None
        
        # VECM model
        self.vecm_model = None
        
        # Current positions
        self.positions: Dict[str, float] = {}
    
    def test_engle_granger(self, prices: pd.DataFrame, asset1: str, asset2: str) -> Dict:
        """
        Test for cointegration using Engle-Granger test
        
        Args:
            prices: Price data
            asset1: First asset
            asset2: Second asset
            
        Returns:
            Test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            score, pvalue, crit_values = coint(prices[asset1], prices[asset2])
            
            return {
                "statistic": score,
                "pvalue": pvalue,
                "critical_values": crit_values,
                "is_cointegrated": pvalue < self.config.significance_level
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_johansen(self, prices: pd.DataFrame) -> Dict:
        """
        Test for cointegration using Johansen test
        
        Args:
            prices: Price data
            
        Returns:
            Test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            result = johansen(
                prices.values,
                det_order=self.config.det_order,
                k_ar_diff=self.config.k_ar_diff
            )
            
            eigenvalues = result.eig
            eigenvectors = result.evec
            r = result.r  # Number of cointegration relationships
            
            return {
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "cointegration_rank": r,
                "cointegration_vectors": eigenvectors[:, :r]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def fit_vecm(self, prices: pd.DataFrame) -> None:
        """
        Fit VECM model
        
        Args:
            prices: Price data
        """
        if not STATSMODELS_AVAILABLE:
            print("Statsmodels not available")
            return
        
        try:
            self.vecm_model = VECM(prices, k_ar_diff=self.config.k_ar_diff, coint_rank=1)
            self.vecm_model.fit()
        except Exception as e:
            print(f"Failed to fit VECM: {e}")
    
    def calculate_error_correction_term(self, prices: pd.DataFrame, eigenvector: np.ndarray) -> pd.Series:
        """
        Calculate error correction term
        
        Args:
            prices: Price data
            eigenvector: Cointegration vector
            
        Returns:
            Error correction term series
        """
        ect = prices @ eigenvector
        return ect
    
    def generate_signals(self, prices: pd.DataFrame, eigenvector: np.ndarray) -> Dict[str, float]:
        """
        Generate trading signals from cointegration relationship
        
        Args:
            prices: Price data
            eigenvector: Cointegration vector
            
        Returns:
            Dictionary of asset -> signal
        """
        # Calculate error correction term
        ect = self.calculate_error_correction_term(prices, eigenvector)
        
        # Calculate z-score
        ect_mean = ect.rolling(self.config.lookback_window // 10).mean()
        ect_std = ect.rolling(self.config.lookback_window // 10).std()
        z_score = (ect - ect_mean) / ect_std
        
        current_z = z_score.iloc[-1]
        
        # Generate signals based on eigenvector
        signals = {}
        for i, asset in enumerate(prices.columns):
            weight = eigenvector[i]
            
            if current_z > self.config.entry_threshold:
                # ECT is positive - short the portfolio
                signals[asset] = -weight
            elif current_z < -self.config.entry_threshold:
                # ECT is negative - long the portfolio
                signals[asset] = weight
            else:
                signals[asset] = 0.0
        
        return signals
    
    def backtest(self, prices: pd.DataFrame) -> pd.Series:
        """
        Backtest cointegration strategy
        
        Args:
            prices: Price data
            
        Returns:
            Strategy returns
        """
        # Test for cointegration
        johansen_result = self.test_johansen(prices)
        
        if "error" in johansen_result or johansen_result["cointegration_rank"] == 0:
            return pd.Series(0, index=prices.index)
        
        # Get cointegration vector
        eigenvector = johansen_result["cointegration_vectors"][:, 0]
        self.cointegration_vectors = eigenvector
        
        # Generate signals over time
        strategy_returns = []
        
        for i in range(self.config.lookback_window, len(prices)):
            window_prices = prices.iloc[i-self.config.lookback_window:i]
            
            # Generate signals
            signals = self.generate_signals(window_prices, eigenvector)
            
            # Calculate return for next period
            next_returns = prices.pct_change().iloc[i]
            strategy_return = sum(signals.get(asset, 0) * next_returns[asset] 
                                for asset in signals.keys())
            
            strategy_returns.append(strategy_return)
        
        return pd.Series(strategy_returns, index=prices.index[self.config.lookback_window:])
    
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


def simulate_cointegrated_data(n_assets: int = 5, n_days: int = 500) -> pd.DataFrame:
    """Simulate cointegrated data for testing"""
    np.random.seed(42)
    
    # Generate common stochastic trend
    common_trend = np.cumsum(np.random.randn(n_days) * 0.01)
    
    # Generate cointegrated assets
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    prices = pd.DataFrame(index=dates, columns=asset_names)
    
    for i in range(n_assets):
        # Each asset is a linear combination of the common trend plus noise
        weight = np.random.uniform(0.5, 1.5)
        noise = np.random.randn(n_days) * 0.5
        prices[asset_names[i]] = 100 + weight * common_trend + noise
    
    return prices


if __name__ == "__main__":
    # Example usage
    config = CointegrationConfig(
        lookback_window=252,
        significance_level=0.05,
        entry_threshold=2.0
    )
    
    strategy = CointegrationStrategy(config)
    
    # Simulate data
    print("Simulating cointegrated data...")
    prices = simulate_cointegrated_data(5, 500)
    
    # Test Engle-Granger
    print("\nTesting Engle-Granger cointegration...")
    eg_result = strategy.test_engle_granger(prices, "ASSET_0", "ASSET_1")
    print(f"  Statistic: {eg_result.get('statistic', 0):.4f}")
    print(f"  P-value: {eg_result.get('pvalue', 1):.4f}")
    print(f"  Cointegrated: {eg_result.get('is_cointegrated', False)}")
    
    # Test Johansen
    print("\nTesting Johansen cointegration...")
    johansen_result = strategy.test_johansen(prices)
    if "error" not in johansen_result:
        print(f"  Cointegration rank: {johansen_result['cointegration_rank']}")
        print(f"  Eigenvalues: {johansen_result['eigenvalues']}")
    
    # Backtest
    print("\nBacktesting cointegration strategy...")
    strategy_returns = strategy.backtest(prices)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = strategy.get_performance_metrics(strategy_returns)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Current signals
    print("\nCurrent Signals:")
    if strategy.cointegration_vectors is not None:
        signals = strategy.generate_signals(prices, strategy.cointegration_vectors)
        for asset, signal in signals.items():
            if signal != 0:
                print(f"  {asset}: {signal:.4f}")

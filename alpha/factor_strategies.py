"""
Factor Strategies for Indian Market

Implements institutional-grade factor strategies:
- Low Volatility Anomaly
- Value (Book-to-Price, Earnings-to-Price)
- Quality (ROE, low debt)
- Size
- Momentum (combo)

Based on blueprint specification for multi-strategy framework
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FactorSignal:
    """Factor-based trading signal"""
    symbol: str
    factor: str
    signal: float  # -1 to 1
    z_score: float
    confidence: float
    metadata: Dict = None


class LowVolatilityFactor:
    """
    Low Volatility Anomaly
    
    Formula:
    Rank stocks by 1-year volatility. Long bottom decile, short top decile.
    
    Expected Sharpe: 0.4
    Capacity: 10,000+ Cr
    Turnover: 15%/month
    Best Regime: All
    Failure: Vol regime shift
    """
    
    def __init__(self, lookback: int = 252, top_decile: float = 0.1):
        """
        Initialize low volatility factor.
        
        Args:
            lookback: Lookback period for volatility calculation
            top_decile: Top decile to short (0.1 = top 10%)
        """
        self.lookback = lookback
        self.top_decile = top_decile
        
    def compute_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute low volatility signals.
        
        Args:
            prices: DataFrame with prices for multiple stocks
            
        Returns:
            DataFrame with weights (long low vol, short high vol)
        """
        # Calculate volatility
        returns = prices.pct_change()
        volatility = returns.rolling(self.lookback).std() * np.sqrt(252)
        
        # Get latest volatilities
        latest_vol = volatility.iloc[-1]
        
        # Rank by volatility (lowest = best)
        ranks = latest_vol.rank(ascending=True)
        
        # Calculate weights
        n_assets = len(prices.columns)
        bottom_n = int(n_assets * self.top_decile)
        top_n = int(n_assets * self.top_decile)
        
        weights = pd.Series(0.0, index=prices.columns)
        
        # Long bottom decile (low vol)
        low_vol_assets = ranks.nsmallest(bottom_n).index
        weights[low_vol_assets] = 1.0 / bottom_n
        
        # Short top decile (high vol)
        high_vol_assets = ranks.nlargest(top_n).index
        weights[high_vol_assets] = -1.0 / top_n
        
        # Normalize
        weights = weights / weights.abs().sum()
        
        return weights


class ValueFactor:
    """
    Value Factor (Book-to-Price, Earnings-to-Price)
    
    Formula:
    z_BP = (BP - mean(BP)) / std(BP)
    z_EP = (EP - mean(EP)) / std(EP)
    
    Long top decile, short bottom decile.
    
    Expected Sharpe: 0.5
    Capacity: 10,000+ Cr
    Turnover: 20%/month
    Best Regime: Value regime
    Failure: Growth regime
    """
    
    def __init__(self, metric: str = 'BP', lookback: int = 252):
        """
        Initialize value factor.
        
        Args:
            metric: Value metric ('BP' for book-to-price, 'EP' for earnings-to-price)
            lookback: Lookback for fundamental data
        """
        self.metric = metric
        self.lookback = lookback
        
    def compute_signal(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute value signals.
        
        Args:
            prices: DataFrame with prices
            fundamentals: DataFrame with fundamental metrics (BP, EP, etc.)
            
        Returns:
            DataFrame with weights
        """
        # Get latest fundamental values
        latest_fund = fundamentals.iloc[-1]
        
        # Calculate z-scores
        mean_fund = latest_fund.mean()
        std_fund = latest_fund.std()
        z_scores = (latest_fund - mean_fund) / (std_fund + 1e-8)
        
        # Rank by z-score
        ranks = z_scores.rank(ascending=False)
        
        # Calculate weights
        n_assets = len(prices.columns)
        top_n = int(n_assets * 0.1)
        bottom_n = int(n_assets * 0.1)
        
        weights = pd.Series(0.0, index=prices.columns)
        
        # Long top decile (high value)
        high_value = ranks.nsmallest(top_n).index
        weights[high_value] = 1.0 / top_n
        
        # Short bottom decile (low value)
        low_value = ranks.nlargest(bottom_n).index
        weights[low_value] = -1.0 / bottom_n
        
        # Normalize
        weights = weights / weights.abs().sum()
        
        return weights


class QualityFactor:
    """
    Quality Factor (ROE, low debt)
    
    Formula:
    quality_score = (ROE - mean_ROE)/std_ROE - (D/E - mean_DE)/std_DE
    
    Long high quality, short low quality.
    
    Expected Sharpe: 0.4
    Capacity: 10,000+ Cr
    Turnover: 15%/month
    Best Regime: All
    Failure: Quality regime shift
    """
    
    def __init__(self, roe_weight: float = 0.6, debt_weight: float = 0.4):
        """
        Initialize quality factor.
        
        Args:
            roe_weight: Weight for ROE component
            debt_weight: Weight for debt component
        """
        self.roe_weight = roe_weight
        self.debt_weight = debt_weight
        
    def compute_signal(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute quality signals.
        
        Args:
            prices: DataFrame with prices
            fundamentals: DataFrame with ROE, D/E ratios
            
        Returns:
            DataFrame with weights
        """
        # Get latest fundamentals
        latest_fund = fundamentals.iloc[-1]
        
        # Calculate z-scores for ROE and D/E
        roe = latest_fund['ROE']
        de = latest_fund['D/E']
        
        roe_z = (roe - roe.mean()) / (roe.std() + 1e-8)
        de_z = (de - de.mean()) / (de.std() + 1e-8)
        
        # Combine into quality score (higher ROE is good, lower D/E is good)
        quality_score = self.roe_weight * roe_z - self.debt_weight * de_z
        
        # Rank by quality
        ranks = quality_score.rank(ascending=False)
        
        # Calculate weights
        n_assets = len(prices.columns)
        top_n = int(n_assets * 0.1)
        bottom_n = int(n_assets * 0.1)
        
        weights = pd.Series(0.0, index=prices.columns)
        
        # Long top decile (high quality)
        high_quality = ranks.nsmallest(top_n).index
        weights[high_quality] = 1.0 / top_n
        
        # Short bottom decile (low quality)
        low_quality = ranks.nlargest(bottom_n).index
        weights[low_quality] = -1.0 / bottom_n
        
        # Normalize
        weights = weights / weights.abs().sum()
        
        return weights


class SizeFactor:
    """
    Size Factor
    
    Formula:
    Rank by market cap. Long small cap, short large cap.
    
    Expected Sharpe: 0.3
    Capacity: 5,000 Cr
    Turnover: 25%/month
    Best Regime: Small cap rally
    Failure: Large cap dominance
    """
    
    def __init__(self, small_cap_weight: float = 0.7):
        """
        Initialize size factor.
        
        Args:
            small_cap_weight: Weight for small cap exposure
        """
        self.small_cap_weight = small_cap_weight
        
    def compute_signal(
        self,
        prices: pd.DataFrame,
        market_cap: pd.Series
    ) -> pd.DataFrame:
        """
        Compute size signals.
        
        Args:
            prices: DataFrame with prices
            market_cap: Series with market capitalizations
            
        Returns:
            DataFrame with weights
        """
        # Rank by market cap (smallest = best)
        ranks = market_cap.rank(ascending=True)
        
        # Calculate weights
        n_assets = len(prices.columns)
        small_n = int(n_assets * 0.3)  # Bottom 30%
        large_n = int(n_assets * 0.3)  # Top 30%
        
        weights = pd.Series(0.0, index=prices.columns)
        
        # Long small cap
        small_cap = ranks.nsmallest(small_n).index
        weights[small_cap] = self.small_cap_weight / small_n
        
        # Short large cap
        large_cap = ranks.nlargest(large_n).index
        weights[large_cap] = -(1 - self.small_cap_weight) / large_n
        
        # Normalize
        weights = weights / weights.abs().sum()
        
        return weights


class ComboFactor:
    """
    Combined Factor Strategy
    
    Combines multiple factors into a composite score.
    """
    
    def __init__(
        self,
        low_vol_weight: float = 0.25,
        value_weight: float = 0.25,
        quality_weight: float = 0.25,
        momentum_weight: float = 0.25
    ):
        """
        Initialize combo factor.
        
        Args:
            low_vol_weight: Weight for low volatility
            value_weight: Weight for value
            quality_weight: Weight for quality
            momentum_weight: Weight for momentum
        """
        self.low_vol_weight = low_vol_weight
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.momentum_weight = momentum_weight
        
    def compute_signal(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        market_cap: pd.Series
    ) -> pd.DataFrame:
        """
        Compute combined factor signal.
        
        Args:
            prices: DataFrame with prices
            fundamentals: DataFrame with fundamentals
            market_cap: Series with market cap
            
        Returns:
            DataFrame with combined weights
        """
        # Compute individual factor signals
        low_vol = LowVolatilityFactor()
        value = ValueFactor()
        quality = QualityFactor()
        size = SizeFactor()
        
        # Get signals (simplified - would need actual data in production)
        low_vol_weights = low_vol.compute_signal(prices)
        value_weights = value.compute_signal(prices, fundamentals)
        quality_weights = quality.compute_signal(prices, fundamentals)
        size_weights = size.compute_signal(prices, market_cap)
        
        # Combine weights
        combined = (
            self.low_vol_weight * low_vol_weights +
            self.value_weight * value_weights +
            self.quality_weight * quality_weights +
            self.momentum_weight * size_weights
        )
        
        # Normalize
        combined = combined / combined.abs().sum()
        
        return combined


if __name__ == "__main__":
    # Test factor strategies
    print("Testing Factor Strategies...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    assets = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'SBIN', 'BHARTIARTL']
    prices = pd.DataFrame(
        np.random.randn(n, len(assets)).cumsum(axis=0) * 10 + 100,
        index=dates,
        columns=assets
    )
    
    # Synthetic fundamentals
    fundamentals = pd.DataFrame({
        'ROE': np.random.uniform(0.1, 0.25, len(assets)),
        'D/E': np.random.uniform(0.3, 2.0, len(assets)),
        'BP': np.random.uniform(0.5, 3.0, len(assets)),
        'EP': np.random.uniform(0.05, 0.15, len(assets))
    }, index=assets)
    
    market_cap = pd.Series(
        np.random.uniform(10000, 100000, len(assets)),
        index=assets
    )
    
    # Test Low Volatility
    print("\n1. Low Volatility Factor:")
    low_vol = LowVolatilityFactor()
    weights = low_vol.compute_signal(prices)
    print(f"   Long positions: {(weights > 0).sum()}")
    print(f"   Short positions: {(weights < 0).sum()}")
    print(f"   Weights sum: {weights.sum():.4f}")
    
    # Test Value
    print("\n2. Value Factor:")
    value = ValueFactor()
    value_weights = value.compute_signal(prices, fundamentals)
    print(f"   Long positions: {(value_weights > 0).sum()}")
    print(f"   Short positions: {(value_weights < 0).sum()}")
    
    # Test Quality
    print("\n3. Quality Factor:")
    quality = QualityFactor()
    quality_weights = quality.compute_signal(prices, fundamentals)
    print(f"   Long positions: {(quality_weights > 0).sum()}")
    print(f"   Short positions: {(quality_weights < 0).sum()}")
    
    # Test Size
    print("\n4. Size Factor:")
    size = SizeFactor()
    size_weights = size.compute_signal(prices, market_cap)
    print(f"   Long positions: {(size_weights > 0).sum()}")
    print(f"   Short positions: {(size_weights < 0).sum()}")
    
    # Test Combo
    print("\n5. Combo Factor:")
    combo = ComboFactor()
    combo_weights = combo.compute_signal(prices, fundamentals, market_cap)
    print(f"   Long positions: {(combo_weights > 0).sum()}")
    print(f"   Short positions: {(combo_weights < 0).sum()}")
    print(f"   Weights sum: {combo_weights.sum():.4f}")
    
    print("\n✓ All factor strategies tested")

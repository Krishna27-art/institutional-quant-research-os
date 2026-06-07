"""
Trend Following Strategies

Implements institutional-grade trend following strategies:
- Time Series Momentum (TSMOM) - Moskowitz, Ooi, Pedersen (2012)
- Dual Momentum (Antonacci 2014)
- Sector Momentum Rotation (Moskowitz & Grinblatt 1999)

Based on blueprint specification for multi-strategy framework
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TSMOMResult:
    """Result of TSMOM strategy"""
    signal: pd.Series
    momentum: pd.Series
    realized_vol: pd.Series
    scaled_position: pd.Series


class TSMOM:
    """
    Time Series Momentum (TSMOM)
    
    Moskowitz, Ooi, Pedersen (2012)
    
    Formula:
    Signal_t = sign(r_{t-1,t-1-m}) * (σ_target / σ_{t-1,t-1-m})
    
    where:
    - r = return over past m months (typically 12)
    - σ_target = target volatility (e.g., 40%)
    - σ = realized volatility
    
    Expected Sharpe: 0.6-0.8
    Capacity: Very high
    Decay: 3-6 months
    Failure: Mean-reverting markets, V-shaped reversals
    """
    
    def __init__(
        self,
        lookback: int = 252,  # 12 months of trading days
        vol_lookback: int = 21,  # 1 month for vol estimation
        target_vol: float = 0.4,  # 40% annualized vol target
        max_leverage: float = 2.0
    ):
        """
        Initialize TSMOM strategy.
        
        Args:
            lookback: Lookback period for momentum calculation
            vol_lookback: Lookback period for volatility estimation
            target_vol: Target annualized volatility
            max_leverage: Maximum leverage cap
        """
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        
    def compute(self, prices: pd.Series) -> TSMOMResult:
        """
        Compute TSMOM signal.
        
        Args:
            prices: Price series
            
        Returns:
            TSMOMResult with signals and components
        """
        returns = prices.pct_change()
        
        # Momentum: annualized return over lookback
        momentum = returns.rolling(self.lookback).mean() * 252
        
        # Realized volatility: annualized std over vol_lookback
        realized_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Signal: sign of momentum scaled by vol target
        signal = np.sign(momentum) * (self.target_vol / (realized_vol + 1e-8))
        
        # Cap leverage
        signal = signal.clip(-self.max_leverage, self.max_leverage)
        
        # Scaled position
        scaled_position = signal
        
        return TSMOMResult(
            signal=signal,
            momentum=momentum,
            realized_vol=realized_vol,
            scaled_position=scaled_position
        )
    
    def backtest(self, prices: pd.Series) -> pd.DataFrame:
        """
        Simple backtest of TSMOM strategy.
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with returns and performance metrics
        """
        result = self.compute(prices)
        
        # Calculate strategy returns
        returns = prices.pct_change()
        strategy_returns = result.signal.shift(1) * returns
        
        # Calculate cumulative returns
        cumulative = (1 + strategy_returns).cumprod()
        
        df = pd.DataFrame({
            'price': prices,
            'signal': result.signal,
            'momentum': result.momentum,
            'realized_vol': result.realized_vol,
            'strategy_return': strategy_returns,
            'cumulative_return': cumulative
        })
        
        return df


class DualMomentum:
    """
    Dual Momentum (Antonacci 2014)
    
    Formula:
    Signal = I(Price > SMA(200)) * rank_cross_sectional
    
    Only take long if absolute momentum (price > 200d SMA) is positive.
    Combines absolute momentum (trend filter) with relative momentum.
    
    Expected Sharpe: 0.8
    Capacity: Very high
    Decay: 3-6 months
    Failure: V-shaped reversals
    """
    
    def __init__(
        self,
        lookback: int = 252,  # 12 months
        sma_period: int = 200
    ):
        """
        Initialize Dual Momentum strategy.
        
        Args:
            lookback: Lookback for relative momentum
            sma_period: SMA period for absolute momentum filter
        """
        self.lookback = lookback
        self.sma_period = sma_period
        
    def compute(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.Series
    ) -> pd.DataFrame:
        """
        Compute Dual Momentum signals for multiple assets.
        
        Args:
            prices: DataFrame with prices for multiple assets
            benchmark_prices: Benchmark price series (e.g., NIFTY)
            
        Returns:
            DataFrame with signals for each asset
        """
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        # Absolute momentum: price > SMA(200)
        sma = prices.rolling(self.sma_period).mean()
        abs_mom = prices > sma
        
        # Relative momentum: outperformance vs benchmark
        asset_returns = prices.pct_change(self.lookback)
        benchmark_returns = benchmark_prices.pct_change(self.lookback)
        rel_mom = asset_returns.sub(benchmark_returns, axis=0)
        
        # Combined signal
        signals = abs_mom.astype(float) * np.sign(rel_mom)
        
        return signals
    
    def select_top_assets(
        self,
        prices: pd.DataFrame,
        benchmark_prices: pd.Series,
        top_n: int = 3
    ) -> pd.Series:
        """
        Select top N assets based on dual momentum.
        
        Args:
            prices: DataFrame with prices
            benchmark_prices: Benchmark prices
            top_n: Number of top assets to select
            
        Returns:
            Series with weights (1/top_n for selected, 0 otherwise)
        """
        signals = self.compute(prices, benchmark_prices)
        
        # Rank by relative momentum
        asset_returns = prices.pct_change(self.lookback)
        benchmark_returns = benchmark_prices.pct_change(self.lookback)
        rel_mom = asset_returns.sub(benchmark_returns, axis=0)
        
        # Select top N at each point in time
        weights = pd.DataFrame(0, index=signals.index, columns=signals.columns)
        
        for date in signals.index:
            # Get assets with positive absolute momentum
            eligible = signals.loc[date] > 0
            if eligible.sum() == 0:
                continue
            
            # Rank eligible by relative momentum
            eligible_assets = eligible[eligible].index
            ranks = rel_mom.loc[date, eligible_assets].rank(ascending=False)
            
            # Select top N
            top_assets = ranks.nsmallest(top_n).index
            weights.loc[date, top_assets] = 1.0 / len(top_assets)
        
        return weights


class SectorMomentum:
    """
    Sector Momentum Rotation (Moskowitz & Grinblatt 1999)
    
    Formula:
    Rank sectors by 6-month return, long top 3, short bottom 3.
    
    Expected Sharpe: 0.5
    Capacity: 5,000 Cr
    Turnover: 50%/month
    Best Regime: Sector dispersion
    Failure: Crisis (correlations → 1)
    """
    
    def __init__(
        self,
        lookback: int = 126,  # 6 months
        top_n: int = 3,
        bottom_n: int = 3
    ):
        """
        Initialize Sector Momentum strategy.
        
        Args:
            lookback: Lookback period for momentum calculation
            top_n: Number of top sectors to long
            bottom_n: Number of bottom sectors to short
        """
        self.lookback = lookback
        self.top_n = top_n
        self.bottom_n = bottom_n
        
    def compute(self, sector_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sector momentum signals.
        
        Args:
            sector_returns: DataFrame with sector returns
            
        Returns:
            DataFrame with weights for each sector
        """
        # Calculate momentum (average return over lookback)
        momentum = sector_returns.rolling(self.lookback).mean()
        
        # Rank sectors by momentum
        ranks = momentum.rank(axis=1, ascending=False)
        
        # Create weights: long top N, short bottom N
        weights = pd.DataFrame(0, index=momentum.index, columns=momentum.columns)
        
        for date in momentum.index:
            if date < momentum.index[self.lookback]:
                continue
            
            # Long top N
            top_sectors = ranks.loc[date].nsmallest(self.top_n).index
            weights.loc[date, top_sectors] = 1.0 / self.top_n
            
            # Short bottom N
            bottom_sectors = ranks.loc[date].nlargest(self.bottom_n).index
            weights.loc[date, bottom_sectors] = -1.0 / self.bottom_n
        
        # Normalize to dollar-neutral
        weights = weights.div(weights.abs().sum(axis=1), axis=0)
        
        return weights
    
    def get_sector_signals(
        self,
        sector_prices: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get sector signals from prices.
        
        Args:
            sector_prices: DataFrame with sector prices
            
        Returns:
            Tuple of (weights, momentum)
        """
        # Calculate returns
        sector_returns = sector_prices.pct_change()
        
        # Compute momentum
        momentum = sector_returns.rolling(self.lookback).mean()
        
        # Compute weights
        weights = self.compute(sector_returns)
        
        return weights, momentum


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum
    
    Ranks assets by past returns and goes long top decile, short bottom decile.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        top_decile: float = 0.1,
        bottom_decile: float = 0.1
    ):
        """
        Initialize Cross-Sectional Momentum.
        
        Args:
            lookback: Lookback period
            top_decile: Top decile to long (0.1 = top 10%)
            bottom_decile: Bottom decile to short
        """
        self.lookback = lookback
        self.top_decile = top_decile
        self.bottom_decile = bottom_decile
        
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional momentum signals.
        
        Args:
            prices: DataFrame with prices
            
        Returns:
            DataFrame with weights
        """
        # Calculate returns
        returns = prices.pct_change(self.lookback)
        
        # Rank by returns
        ranks = returns.rank(axis=1, ascending=False)
        
        # Calculate weights
        n_assets = len(prices.columns)
        top_n = int(n_assets * self.top_decile)
        bottom_n = int(n_assets * self.bottom_decile)
        
        weights = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        for date in returns.index:
            if pd.isna(ranks.loc[date]).any():
                continue
            
            # Long top decile
            top_assets = ranks.loc[date].nsmallest(top_n).index
            weights.loc[date, top_assets] = 1.0 / top_n
            
            # Short bottom decile
            bottom_assets = ranks.loc[date].nlargest(bottom_n).index
            weights.loc[date, bottom_assets] = -1.0 / bottom_n
        
        # Normalize
        weights = weights.div(weights.abs().sum(axis=1), axis=0)
        
        return weights


if __name__ == "__main__":
    # Test trend following strategies
    print("Testing Trend Following Strategies...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Single asset for TSMOM
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(n) * 0.5),
        index=dates
    )
    
    # Test TSMOM
    print("\n1. TSMOM Strategy:")
    tsmom = TSMOM()
    result = tsmom.compute(prices)
    print(f"   Signal range: [{result.signal.min():.3f}, {result.signal.max():.3f}]")
    print(f"   Momentum range: [{result.momentum.min():.3f}, {result.momentum.max():.3f}]")
    print(f"   Vol range: [{result.realized_vol.min():.3f}, {result.realized_vol.max():.3f}]")
    
    # Multiple assets for Dual Momentum
    assets = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    multi_prices = pd.DataFrame(
        np.random.randn(n, 5).cumsum(axis=0) + 100,
        index=dates,
        columns=assets
    )
    benchmark = pd.Series(
        100 + np.cumsum(np.random.randn(n) * 0.3),
        index=dates
    )
    
    # Test Dual Momentum
    print("\n2. Dual Momentum:")
    dual_mom = DualMomentum()
    signals = dual_mom.compute(multi_prices, benchmark)
    print(f"   Signals shape: {signals.shape}")
    print(f"   Non-zero signals: {(signals != 0).sum().sum()}")
    
    # Test Sector Momentum
    print("\n3. Sector Momentum:")
    sector_returns = pd.DataFrame(
        np.random.randn(n, 5) * 0.01,
        index=dates,
        columns=['IT', 'BANK', 'PHARMA', 'AUTO', 'FMCG']
    )
    sector_mom = SectorMomentum()
    weights = sector_mom.compute(sector_returns)
    print(f"   Weights shape: {weights.shape}")
    print(f"   Long positions: {(weights > 0).sum().sum()}")
    print(f"   Short positions: {(weights < 0).sum().sum()}")
    
    # Test Cross-Sectional Momentum
    print("\n4. Cross-Sectional Momentum:")
    xsmom = CrossSectionalMomentum()
    xsmom_weights = xsmom.compute(multi_prices)
    print(f"   Weights shape: {xsmom_weights.shape}")
    print(f"   Non-zero weights: {(xsmom_weights != 0).sum().sum()}")
    
    print("\n✓ All trend following strategies tested")

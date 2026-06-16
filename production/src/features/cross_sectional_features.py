"""
Cross-Sectional Features & Engines

This module implements institutional-grade cross-sectional models:
- CorrelationEngine: Rolling pairwise asset correlation matrices & average correlation.
- CovarianceEngine: Rolling annualized covariance matrices.
- RelativeStrengthEngine: Rolling 3m, 6m, and 12m momentum relative strength ranking.
- MarketBreadthEngine: Rolling percentage of stocks above moving averages (20, 50, 200 EMA)
  and rolling percentage of stocks marking new highs/lows.

These engines process the multi-asset universe to generate signals used by the
feature pipeline, regime engine, and portfolio optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class CorrelationEngine:
    """
    Computes rolling cross-sectional asset correlations.
    
    Used for:
    - Estimating portfolio risk concentration
    - Systemic risk and regime change signals (correlation spikes during market crashes)
    - Weight penalties in portfolio optimization
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute_rolling_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling average pairwise correlation across the universe.
        
        Args:
            returns: DataFrame of asset returns (dates as index, symbols as columns)
            
        Returns:
            Series of average pairwise correlation over time
        """
        # Compute rolling correlation matrices
        rolling_corr = returns.rolling(window=self.window).corr()
        
        # Calculate average pairwise correlation at each timestamp
        avg_corrs = []
        timestamps = returns.index[self.window - 1:]
        
        for date in timestamps:
            corr_matrix = rolling_corr.loc[date]
            # Strip diagonal and compute mean of lower/upper triangle
            n = corr_matrix.shape[0]
            if n <= 1:
                avg_corrs.append(0.0)
                continue
            
            # Sum of all elements minus diagonal (which are 1.0)
            sum_off_diag = corr_matrix.sum().sum() - n
            # Number of off-diagonal elements
            num_off_diag = n * (n - 1)
            
            avg_corr = sum_off_diag / num_off_diag if num_off_diag > 0 else 0.0
            avg_corrs.append(float(avg_corr))
            
        return pd.Series(avg_corrs, index=timestamps).reindex(returns.index).ffill().fillna(0.0)

    def compute_pairwise_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the latest pairwise correlation matrix.
        
        Args:
            returns: DataFrame of asset returns (trailing self.window days)
            
        Returns:
            Correlation DataFrame
        """
        return returns.tail(self.window).corr().fillna(0.0)


class CovarianceEngine:
    """
    Computes rolling annualized covariance matrices.
    
    Used for:
    - Markowitz Mean-Variance optimization
    - Risk Parity allocations (equal risk contribution)
    - Intraday and daily volatility targeting
    """

    def __init__(self, window: int = 60, annualization_factor: int = 252):
        self.window = window
        self.annualization_factor = annualization_factor

    def compute_covariance_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the annualized covariance matrix over the lookback window.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Annualized covariance matrix
        """
        sample = returns.tail(self.window)
        # Handle small samples
        if len(sample) < 2:
            return pd.DataFrame(0.0, index=returns.columns, columns=returns.columns)
            
        cov = sample.cov() * self.annualization_factor
        return cov.fillna(0.0)

    def compute_rolling_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling annualized covariance matrices over the entire history.
        
        Returns:
            MultiIndex DataFrame (Date, Ticker) containing covariance matrices
        """
        return returns.rolling(window=self.window).cov() * self.annualization_factor


class RelativeStrengthEngine:
    """
    Computes cross-sectional relative strength momentum.
    
    Instead of simple absolute returns, Relative Strength ranks stocks relative to
    others, filtering out beta-driven market movements to isolate true alpha.
    """

    def __init__(self, windows: List[int] = None):
        # Default to 3-month (63 days), 6-month (126 days), 12-month (252 days)
        self.windows = windows or [63, 126, 252]

    def compute_relative_strength(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling relative strength momentum scores and ranks.
        
        Args:
            prices: DataFrame of asset prices (dates as index, symbols as columns)
            
        Returns:
            Dict mapping window name (e.g. 'rs_3m') to a DataFrame of cross-sectional ranks
        """
        results = {}
        
        for window in self.windows:
            window_name = f"rs_{window // 21}m" if window % 21 == 0 else f"rs_{window}d"
            
            # 1. Compute rolling cumulative returns
            rolling_returns = prices.pct_change(periods=window)
            
            # 2. Compute cross-sectional ranks (0.0 to 1.0 percentile)
            ranks = rolling_returns.rank(axis=1, pct=True)
            
            results[window_name] = rolling_returns
            results[f"{window_name}_rank"] = ranks
            
        return results


class MarketBreadthEngine:
    """
    Computes market breadth features (regime markers) over the stock universe.
    
    Used for:
    - Identifying market tops/bottoms (breadth divergence)
    - HMM regime classification
    - Risk mitigation triggers
    """

    def __init__(self, ema_periods: List[int] = None, high_low_window: int = 20):
        self.ema_periods = ema_periods or [20, 50, 200]
        self.high_low_window = high_low_window

    def compute_breadth(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentage above EMAs and % making new highs/lows.
        
        Args:
            prices: DataFrame of asset prices (dates as index, symbols as columns)
            
        Returns:
            DataFrame of market breadth features over time
        """
        breadth = pd.DataFrame(index=prices.index)
        
        # 1. Percentage of stocks above EMAs
        for period in self.ema_periods:
            emas = prices.ewm(span=period, adjust=False).mean()
            # Calculate % of active stocks above their EMA
            above_ema = prices > emas
            pct_above = above_ema.sum(axis=1) / prices.notna().sum(axis=1)
            breadth[f"pct_above_{period}ema"] = pct_above.fillna(0.5)
            
        # 2. Percentage of stocks making new highs/lows
        highs = prices.rolling(window=self.high_low_window).max()
        lows = prices.rolling(window=self.high_low_window).min()
        
        is_new_high = (prices >= highs - 1e-5) & (prices.notna())
        is_new_low = (prices <= lows + 1e-5) & (prices.notna())
        
        breadth[f"new_highs_{self.high_low_window}d"] = (is_new_high.sum(axis=1) / prices.notna().sum(axis=1)).fillna(0.0)
        breadth[f"new_lows_{self.high_low_window}d"] = (is_new_low.sum(axis=1) / prices.notna().sum(axis=1)).fillna(0.0)
        
        return breadth

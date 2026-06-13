"""
Long Memory Features - Fractional Differencing and Hurst Exponent

This module implements advanced statistical features for detecting and exploiting
long memory in financial time series, as described in:
- López de Prado (2018) - Advances in Financial Machine Learning
- Gatheral (2018) - Rough Volatility
- Deep et al. - Fractional calculus for financial time series

Key Features:
- Fractional differencing (López de Prado)
- Hurst exponent estimation (Gatheral)
- Roughness regime detection
- Long memory volatility modeling

Based on Blueprint Week 3-4: Mathematical & Statistical Toolkit
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)


class LongMemoryFeatures:
    """
    Long memory features for financial time series.
    
    Implements fractional differencing to retain long memory while ensuring
    stationarity, and Hurst exponent estimation for regime detection.
    """
    
    @staticmethod
    def frac_diff(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
        """
        Fractional differencing (López de Prado).
        
        Computes fractional differences of order d, which can retain long memory
        while making the series stationary. This is superior to integer differencing
        which removes too much memory.
        
        Args:
            series: Time series to difference
            d: Fractional differencing order (0 < d < 1)
            threshold: Minimum weight threshold to stop convolution
            
        Returns:
            Fractionally differenced series
        """
        if not 0 < d < 1:
            raise ValueError("Fractional differencing order d must be in (0, 1)")
        
        # Compute weights using binomial expansion
        weights = [1.0]
        for k in range(1, len(series)):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < threshold:
                break
            weights.append(w)
        
        weights = np.array(weights[::-1])
        
        # Convolve with series
        diffed = np.convolve(series.values, weights, mode='valid')
        
        return pd.Series(diffed, index=series.index[len(weights)-1:])
    
    @staticmethod
    def get_optimal_d(series: pd.Series, max_d: float = 0.9, threshold: float = 1e-5) -> float:
        """
        Find optimal fractional differencing order d.
        
        Uses the method from López de Prado: find the smallest d such that
        the fractionally differenced series passes the Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to analyze
            max_d: Maximum d to try
            threshold: Weight threshold for frac_diff
            
        Returns:
            Optimal d value
        """
        from statsmodels.tsa.stattools import adfuller
        
        def adf_pvalue(d):
            try:
                diffed = LongMemoryFeatures.frac_diff(series, d, threshold)
                if len(diffed) < 20:
                    return 1.0  # High p-value if not enough data
                result = adfuller(diffed, maxlag=1)
                return result[1]  # p-value
            except:
                return 1.0
        
        # Find smallest d that makes series stationary (p-value < 0.05)
        for d in np.arange(0.01, max_d + 0.01, 0.01):
            pval = adf_pvalue(d)
            if pval < 0.05:
                return d
        
        return max_d  # Return max_d if no d found
    
    @staticmethod
    def hurst(ts: pd.Series, max_lag: int = 100) -> float:
        """
        Hurst exponent estimation (Gatheral).
        
        The Hurst exponent H measures long-range dependence:
        - H = 0.5: Random walk (no memory)
        - H > 0.5: Persistent trend (momentum)
        - H < 0.5: Mean-reverting (anti-persistent)
        
        Uses R/S analysis (rescaled range method).
        
        Args:
            ts: Time series
            max_lag: Maximum lag to consider
            
        Returns:
            Hurst exponent H
        """
        if len(ts) < 20:
            return 0.5  # Default to random walk if insufficient data
        
        lags = range(2, min(max_lag, len(ts) // 2))
        
        # Calculate range of cumulative deviations
        tau = []
        for lag in lags:
            # Calculate cumulative deviations
            cumsum = np.cumsum(ts.values - np.mean(ts.values))
            range_val = np.max(cumsum[:lag]) - np.min(cumsum[:lag])
            
            # Standard deviation
            std_val = np.std(ts.values[:lag])
            
            if std_val > 0:
                tau.append(range_val / std_val)
            else:
                tau.append(0)
        
        if len(tau) == 0:
            return 0.5
        
        # Fit log-log regression
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        # Remove zeros
        valid_idx = log_tau > -np.inf
        if np.sum(valid_idx) < 2:
            return 0.5
        
        slope, _ = np.polyfit(log_lags[valid_idx], log_tau[valid_idx], 1)
        
        return slope
    
    @staticmethod
    def hurst_generalized(ts: pd.Series, method: str = 'rs') -> float:
        """
        Generalized Hurst exponent estimation with multiple methods.
        
        Args:
            ts: Time series
            method: Method to use ('rs' for R/S, 'dma' for detrended moving average)
            
        Returns:
            Hurst exponent H
        """
        if method == 'rs':
            return LongMemoryFeatures.hurst(ts)
        elif method == 'dma':
            return LongMemoryFeatures.hurst_dma(ts)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def hurst_dma(ts: pd.Series, min_q: int = 2, max_q: int = 20) -> float:
        """
        Hurst exponent using Detrended Moving Average (DMA) method.
        
        DMA is more robust to non-stationarity than R/S analysis.
        
        Args:
            ts: Time series
            min_q: Minimum window size
            max_q: Maximum window size
            
        Returns:
            Hurst exponent H
        """
        if len(ts) < max_q * 2:
            return 0.5
        
        q_values = range(min_q, max_q + 1)
        F_q = []
        
        for q in q_values:
            # Calculate moving average
            ma = ts.rolling(window=q).mean()
            
            # Detrended series
            detrended = ts - ma
            
            # Fluctuation function
            F_q.append(np.sqrt(np.mean(detrended**2)))
        
        # Fit log-log regression
        log_q = np.log(q_values)
        log_F = np.log(F_q)
        
        slope, _ = np.polyfit(log_q, log_F, 1)
        
        return slope
    
    @staticmethod
    def classify_regime(hurst: float) -> str:
        """
        Classify market regime based on Hurst exponent.
        
        Args:
            hurst: Hurst exponent
            
        Returns:
            Regime classification
        """
        if hurst < 0.45:
            return 'MEAN_REVERTING'
        elif hurst > 0.55:
            return 'TRENDING'
        else:
            return 'RANDOM_WALK'
    
    @staticmethod
    def rough_volatility_estimation(returns: pd.Series, window: int = 252) -> dict:
        """
        Rough volatility estimation following Gatheral (2018).
        
        Estimates the roughness parameter H for volatility using realized
        volatility and the relationship:
        log(RV_t) ~ log(RV_{t-1}) + noise
        
        Args:
            returns: Return series
            window: Window for realized volatility
            
        Returns:
            Dictionary with roughness estimates
        """
        if len(returns) < window:
            return {'H': 0.5, 'roughness': 0.5, 'regime': 'UNKNOWN'}
        
        # Calculate realized volatility
        rv = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Log realized volatility
        log_rv = np.log(rv.dropna())
        
        if len(log_rv) < 20:
            return {'H': 0.5, 'roughness': 0.5, 'regime': 'UNKNOWN'}
        
        # Calculate autocorrelation of log RV
        acf = [log_rv.autocorr(lag) for lag in range(1, 11)]
        
        # Roughness H is related to decay of autocorrelation
        # For rough volatility, H < 0.5
        H = 0.5 - 0.5 * np.mean(acf)
        H = np.clip(H, 0.0, 0.5)
        
        regime = 'ROUGH' if H < 0.3 else 'SMOOTH'
        
        return {
            'H': H,
            'roughness': H,
            'regime': regime,
            'rv_mean': rv.mean(),
            'rv_std': rv.std()
        }


if __name__ == "__main__":
    # Test long memory features
    print("Testing Long Memory Features...")
    
    # Create sample data
    np.random.seed(42)
    n = 500
    
    # Random walk
    rw = np.cumsum(np.random.normal(0, 1, n))
    ts_rw = pd.Series(rw)
    
    # Fractional Brownian motion approximation
    fbm = np.cumsum(np.random.normal(0, 1, n))
    for i in range(1, n):
        fbm[i] = 0.7 * fbm[i-1] + fbm[i]
    ts_fbm = pd.Series(fbm)
    
    # Test Hurst exponent
    print(f"\nRandom Walk Hurst: {LongMemoryFeatures.hurst(ts_rw):.3f}")
    print(f"Fractional BM Hurst: {LongMemoryFeatures.hurst(ts_fbm):.3f}")
    
    # Test fractional differencing
    d_opt = LongMemoryFeatures.get_optimal_d(ts_fbm)
    print(f"\nOptimal d for fractional BM: {d_opt:.3f}")
    
    diffed = LongMemoryFeatures.frac_diff(ts_fbm, d_opt)
    print(f"Fractionally differenced series length: {len(diffed)}")
    
    # Test regime classification
    print(f"\nRandom Walk Regime: {LongMemoryFeatures.classify_regime(LongMemoryFeatures.hurst(ts_rw))}")
    print(f"Fractional BM Regime: {LongMemoryFeatures.classify_regime(LongMemoryFeatures.hurst(ts_fbm))}")
    
    print("\nLong Memory Features test completed.")

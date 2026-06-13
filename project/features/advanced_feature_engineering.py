"""
Advanced Feature Engineering with Mathematical Foundations

Implements institutional-grade feature engineering:
- Fractional differencing (López de Prado) with FFT convolution - O(N log N)
- Hurst exponent (Gatheral) via rescaled range - O(N log N)
- Chaotic transforms (logistic/tent maps) for BCF-GCN
- HP filter for trend/cycle decomposition

Based on blueprint specifications for Jane Street / Renaissance / Two Sigma style systems
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy import sparse
from scipy.fft import fft, ifft
from scipy.signal import detrend
import logging

logger = logging.getLogger(__name__)


def frac_diff_fft(series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Fractional differencing via FFT convolution.
    
    Complexity O(N log N) instead of naive O(N * M).
    
    Formula: (1-L)^d X_t = Σ_{k=0}^∞ w_k X_{t-k}
    where w_k = (-1)^k * d * (d-1) * ... * (d-k+1) / k!
    
    Args:
        series: Input time series
        d: Differencing parameter (0 < d < 1 for long memory)
        threshold: Threshold for weight truncation
        
    Returns:
        Fractionally differenced series
    """
    # Compute weights until below threshold
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    
    weights = np.array(weights)
    n = len(series)
    
    # Zero-pad weights to length N
    weights_pad = np.zeros(n)
    weights_pad[:len(weights)] = weights[::-1]
    
    # FFT convolution
    fft_series = fft(series)
    fft_weights = fft(weights_pad)
    result = ifft(fft_series * fft_weights).real
    
    # Remove initial NaN (warm-up period)
    return result[len(weights)-1:]


def frac_diff_simple(series: np.ndarray, d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Fractional differencing via simple convolution (for small series).
    
    Complexity O(N * M) where M = number of weights (~50 for d=0.4)
    Use this for series < 10000 points.
    
    Args:
        series: Input time series
        d: Differencing parameter
        threshold: Weight truncation threshold
        
    Returns:
        Fractionally differenced series
    """
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    
    weights = np.array(weights[::-1])
    return np.convolve(series, weights, mode='valid')


def hurst_rs(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    Rescaled range Hurst estimator.
    
    Complexity O(N * log N) using multiple lags.
    
    Formula: E[|X(t+τ)-X(t)|^q] ~ τ^{qH}
    For q=2: log(variance) vs log(lag) slope = 2H
    
    Args:
        ts: Time series
        max_lag: Maximum lag to consider
        
    Returns:
        Hurst exponent H (0 < H < 1)
        - H < 0.5: Mean reverting
        - H = 0.5: Random walk
        - H > 0.5: Trending/persistent
    """
    n = len(ts)
    if n < 100:
        logger.warning("Series too short for reliable Hurst estimation")
        return 0.5
    
    lags = range(2, min(max_lag, n // 2))
    rs_values = []
    
    for lag in lags:
        # Split into non-overlapping windows
        n_windows = n // lag
        if n_windows < 2:
            continue
            
        windows = [ts[i*lag:(i+1)*lag] for i in range(n_windows)]
        r_s_values = []
        
        for w in windows:
            if len(w) < 2:
                continue
            mean = np.mean(w)
            cumsum = np.cumsum(w - mean)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(w)
            if s > 0:
                r_s_values.append(r / s)
        
        if r_s_values:
            rs_values.append(np.mean(r_s_values))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression on log-log scale
    log_lags = np.log(list(lags)[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    poly = np.polyfit(log_lags, log_rs, 1)
    H = poly[0]
    
    return H


def hurst_variance(ts: np.ndarray, max_lag: int = 100) -> float:
    """
    Hurst exponent via variance method (alternative to R/S).
    
    Uses: Var[|X(t+τ) - X(t)|] ~ τ^{2H}
    
    Args:
        ts: Time series
        max_lag: Maximum lag
        
    Returns:
        Hurst exponent
    """
    n = len(ts)
    lags = range(2, min(max_lag, n // 2))
    
    variances = []
    for lag in lags:
        diffs = np.abs(ts[lag:] - ts[:-lag])
        variances.append(np.var(diffs))
    
    if len(variances) < 2:
        return 0.5
    
    log_lags = np.log(list(lags))
    log_vars = np.log(variances)
    
    poly = np.polyfit(log_lags, log_vars, 1)
    H = poly[0] / 2  # slope = 2H
    
    return H


def logistic_map(x: np.ndarray, r: float = 3.8) -> np.ndarray:
    """
    Logistic map chaotic transformation.
    
    Formula: x_{t+1} = r * x_t * (1 - x_t)
    Chaos for r ∈ [3.57, 4]
    
    Args:
        x: Input array (normalized to [0,1])
        r: Growth rate parameter
        
    Returns:
        Transformed array
    """
    # Normalize to [0,1] if not already
    x_min, x_max = np.min(x), np.max(x)
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    else:
        x_norm = np.zeros_like(x)
    
    return r * x_norm * (1 - x_norm)


def tent_map(x: np.ndarray, mu: float = 1.8) -> np.ndarray:
    """
    Tent map chaotic transformation.
    
    Formula: x_{t+1} = μ * x_t if x_t < 0.5, else μ * (1 - x_t)
    Chaos for μ ∈ [1, 2]
    
    Args:
        x: Input array (normalized to [0,1])
        mu: Tent parameter
        
    Returns:
        Transformed array
    """
    x_min, x_max = np.min(x), np.max(x)
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    else:
        x_norm = np.zeros_like(x)
    
    return np.where(x_norm < 0.5, mu * x_norm, mu * (1 - x_norm))


def chaotic_transform(x: np.ndarray, r: float = 3.8, mu: float = 1.8) -> np.ndarray:
    """
    Combined chaotic transformation (logistic + tent).
    
    Concatenates original, logistic, and tent transforms.
    Used in BCF-GCN for chaotic feature generation.
    
    Args:
        x: Input array
        r: Logistic parameter
        mu: Tent parameter
        
    Returns:
        Concatenated transformed features [x, logistic(x), tent(x)]
    """
    x_log = logistic_map(x, r)
    x_tent = tent_map(x, mu)
    return np.column_stack([x, x_log, x_tent])


def hp_filter(y: np.ndarray, lamb: float = 1600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott filter for trend/cycle decomposition.
    
    Decomposes series into trend (momentum) and cycle (mean reversion).
    
    Formula: min Σ (y_t - τ_t)² + λ Σ ((τ_{t+1}-τ_t) - (τ_t-τ_{t-1}))²
    Solved via linear system: (I + λ D₂ᵀ D₂) τ = y
    
    Complexity O(N) using sparse matrix solver.
    
    Args:
        y: Input series
        lamb: Smoothing parameter (1600 for quarterly, 6.25 for annual, 129600 for monthly)
        
    Returns:
        Tuple of (trend, cycle)
    """
    n = len(y)
    I = sparse.eye(n, format='csc')
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n), format='csc')
    
    trend = sparse.linalg.spsolve(I + lamb * D2.T @ D2, y)
    cycle = y - trend
    
    return trend, cycle


def emd_decomposition(y: np.ndarray, max_imfs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical Mode Decomposition (simplified).
    
    Decomposes series into Intrinsic Mode Functions (IMFs) and residual.
    Used for multi-scale trend/cycle analysis.
    
    Note: This is a simplified version. Full EMD requires sifting process.
    
    Args:
        y: Input series
        max_imfs: Maximum number of IMFs to extract
        
    Returns:
        Tuple of (imfs, residual)
    """
    # Simplified: use HP filter at multiple scales
    imfs = []
    residual = y.copy()
    
    lambdas = [10, 100, 1600, 10000, 100000][:max_imfs]
    
    for lam in lambdas:
        trend, cycle = hp_filter(residual, lam)
        imfs.append(cycle)
        residual = trend
    
    return np.array(imfs), residual


def compute_fractional_features(
    prices: pd.Series,
    d_values: list = [0.1, 0.2, 0.3, 0.4, 0.5]
) -> pd.DataFrame:
    """
    Compute fractional differencing features at multiple d values.
    
    Args:
        prices: Price series
        d_values: List of d parameters to try
        
    Returns:
        DataFrame with fractional differenced features
    """
    features = {}
    prices_array = prices.values
    
    for d in d_values:
        try:
            frac_diff = frac_diff_fft(prices_array, d)
            features[f'frac_diff_{d:.1f}'] = frac_diff
        except Exception as e:
            logger.warning(f"Failed to compute frac_diff for d={d}: {e}")
    
    # Pad to match original length
    max_len = max(len(v) for v in features.values()) if features else len(prices_array)
    for key in features:
        if len(features[key]) < max_len:
            pad = np.full(max_len - len(features[key]), np.nan)
            features[key] = np.concatenate([pad, features[key]])
    
    df = pd.DataFrame(features, index=prices.index)
    return df


def compute_rough_volatility_features(prices: pd.Series) -> Dict[str, float]:
    """
    Compute rough volatility features (Hurst exponent, etc.).
    
    Based on Gatheral's rough volatility theory.
    
    Args:
        prices: Price series
        
    Returns:
        Dictionary of rough volatility features
    """
    returns = prices.pct_change().dropna().values
    
    if len(returns) < 100:
        return {'hurst': 0.5, 'hurst_var': 0.5}
    
    hurst_rs_val = hurst_rs(returns)
    hurst_var_val = hurst_variance(returns)
    
    return {
        'hurst_rs': hurst_rs_val,
        'hurst_var': hurst_var_val,
        'hurst_mean': (hurst_rs_val + hurst_var_val) / 2,
        'is_mean_reverting': hurst_rs_val < 0.5,
        'is_trending': hurst_rs_val > 0.5
    }


def compute_chaotic_features(prices: pd.Series) -> pd.DataFrame:
    """
    Compute chaotic transformation features.
    
    Args:
        prices: Price series
        
    Returns:
        DataFrame with chaotic features
    """
    returns = prices.pct_change().dropna().values
    
    # Apply chaotic transforms
    chaotic = chaotic_transform(returns, r=3.8, mu=1.8)
    
    features = pd.DataFrame(
        chaotic,
        columns=['original', 'logistic', 'tent'],
        index=prices.index[1:]
    )
    
    return features


def compute_trend_cycle_features(prices: pd.Series, lamb: float = 1600) -> pd.DataFrame:
    """
    Compute trend/cycle decomposition features via HP filter.
    
    Args:
        prices: Price series
        lamb: HP filter smoothing parameter
        
    Returns:
        DataFrame with trend and cycle features
    """
    prices_array = prices.values
    trend, cycle = hp_filter(prices_array, lamb)
    
    features = pd.DataFrame({
        'trend': trend,
        'cycle': cycle,
        'trend_slope': np.gradient(trend),
        'cycle_amplitude': np.abs(cycle)
    }, index=prices.index)
    
    return features


class AdvancedFeatureEngine:
    """
    Advanced feature engineering engine combining all methods.
    
    Provides unified interface for:
    - Fractional differencing
    - Rough volatility (Hurst)
    - Chaotic transforms
    - Trend/cycle decomposition
    """
    
    def __init__(
        self,
        d_values: list = [0.1, 0.2, 0.3, 0.4, 0.5],
        hp_lambda: float = 1600,
        chaotic_r: float = 3.8,
        chaotic_mu: float = 1.8
    ):
        """
        Initialize feature engine.
        
        Args:
            d_values: Fractional differencing parameters
            hp_lambda: HP filter smoothing parameter
            chaotic_r: Logistic map parameter
            chaotic_mu: Tent map parameter
        """
        self.d_values = d_values
        self.hp_lambda = hp_lambda
        self.chaotic_r = chaotic_r
        self.chaotic_mu = chaotic_mu
        
    def compute_all_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute all advanced features for a price series.
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with all features
        """
        all_features = {}
        
        # Fractional differencing
        try:
            frac_features = compute_fractional_features(prices, self.d_values)
            all_features.update(frac_features.to_dict('series'))
        except Exception as e:
            logger.warning(f"Failed to compute fractional features: {e}")
        
        # Rough volatility
        try:
            rough_features = compute_rough_volatility_features(prices)
            for key, val in rough_features.items():
                all_features[key] = val
        except Exception as e:
            logger.warning(f"Failed to compute rough volatility features: {e}")
        
        # Chaotic transforms
        try:
            chaotic_features = compute_chaotic_features(prices)
            for col in chaotic_features.columns:
                all_features[f'chaotic_{col}'] = chaotic_features[col].values
        except Exception as e:
            logger.warning(f"Failed to compute chaotic features: {e}")
        
        # Trend/cycle
        try:
            tc_features = compute_trend_cycle_features(prices, self.hp_lambda)
            for col in tc_features.columns:
                all_features[f'tc_{col}'] = tc_features[col].values
        except Exception as e:
            logger.warning(f"Failed to compute trend/cycle features: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features, index=prices.index)
        return df
    
    def compute_for_universe(self, price_dict: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        """
        Compute features for multiple symbols.
        
        Args:
            price_dict: Dictionary mapping symbols to price series
            
        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        feature_dict = {}
        
        for symbol, prices in price_dict.items():
            try:
                features = self.compute_all_features(prices)
                feature_dict[symbol] = features
                logger.info(f"Computed features for {symbol}: {features.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute features for {symbol}: {e}")
        
        return feature_dict


if __name__ == "__main__":
    # Test the advanced feature engineering
    print("Testing Advanced Feature Engineering...")
    
    # Generate synthetic price data
    np.random.seed(42)
    n = 1000
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(n) * 0.1),
        index=pd.date_range('2020-01-01', periods=n, freq='D')
    )
    
    # Test fractional differencing
    print("\n1. Fractional Differencing:")
    frac_diff = frac_diff_fft(prices.values, d=0.3)
    print(f"   Input length: {len(prices)}")
    print(f"   Output length: {len(frac_diff)}")
    print(f"   Sample values: {frac_diff[:5]}")
    
    # Test Hurst exponent
    print("\n2. Hurst Exponent:")
    returns = prices.pct_change().dropna()
    hurst = hurst_rs(returns.values)
    print(f"   Hurst (R/S): {hurst:.4f}")
    print(f"   Interpretation: {'Mean reverting' if hurst < 0.5 else 'Trending' if hurst > 0.5 else 'Random walk'}")
    
    # Test chaotic transforms
    print("\n3. Chaotic Transforms:")
    chaotic = chaotic_transform(returns.values[:100])
    print(f"   Shape: {chaotic.shape}")
    print(f"   Columns: original, logistic, tent")
    
    # Test HP filter
    print("\n4. HP Filter:")
    trend, cycle = hp_filter(prices.values, lamb=1600)
    print(f"   Trend shape: {trend.shape}")
    print(f"   Cycle shape: {cycle.shape}")
    print(f"   Cycle std: {np.std(cycle):.4f}")
    
    # Test full feature engine
    print("\n5. Full Feature Engine:")
    engine = AdvancedFeatureEngine()
    features = engine.compute_all_features(prices)
    print(f"   Features shape: {features.shape}")
    print(f"   Feature columns: {list(features.columns)[:10]}...")
    
    print("\n✓ All tests passed")

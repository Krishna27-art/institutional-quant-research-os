"""
Long Memory Volatility Model
Based on Deep et al. (2026) methodology

Key findings from research:
- d = 0.226 (GPH), 0.440 (local Whittle)
- H = 0.063 (rough volatility)
- Cross-sectional mean d rises 68% in crisis
- COVID d spike: +86% from baseline
- Slow uncertainty propagation

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize


@dataclass
class LongMemoryParams:
    """Long memory volatility parameters"""
    d: float  # Fractional integration parameter
    H: float  # Hurst exponent
    sigma: float  # Volatility parameter
    mu: float  # Mean parameter


@dataclass
class VolatilityForecast:
    """Volatility forecast results"""
    forecast: np.ndarray
    confidence_interval: Tuple[np.ndarray, np.ndarray]
    regime: str  # "low_vol", "normal", "high_vol", "crisis"
    d_estimate: float
    H_estimate: float


class LongMemoryVolatility:
    """
    Long Memory Volatility Model based on Deep et al. (2026).
    
    Key findings:
    - Volatility exhibits long memory (d = 0.226)
    - Hurst exponent H = 0.063 (rough volatility)
    - Cross-sectional mean d rises 68% in crisis
    - COVID d spike: +86% from baseline
    
    Methods:
    - GPH (Geweke-Porter-Hudak) estimator for d
    - Local Whittle estimator for d
    - Hurst exponent estimation
    - Regime classification based on d
    """
    
    def __init__(self):
        self.params: Optional[LongMemoryParams] = None
        self.is_fitted = False
        
        # Regime thresholds (based on Deep et al. findings)
        self.regime_thresholds = {
            "low_vol": 0.15,      # d < 0.15: Low volatility
            "normal": 0.25,       # 0.15 <= d < 0.25: Normal
            "high_vol": 0.35,     # 0.25 <= d < 0.35: High volatility
            "crisis": 0.35        # d >= 0.35: Crisis
        }
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns."""
        return np.log(prices).diff().dropna()
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate realized volatility."""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def gph_estimator(
        self,
        series: np.ndarray,
        m: Optional[int] = None
    ) -> float:
        """
        Geweke-Porter-Hudak (GPH) estimator for fractional integration parameter d.
        
        Args:
            series: Time series data
            m: Number of low frequencies to use (default: int(n^0.5))
            
        Returns:
            Estimated d parameter
        """
        n = len(series)
        if m is None:
            m = int(n ** 0.5)
        
        # Calculate periodogram
        fft_result = np.fft.fft(series)
        periodogram = np.abs(fft_result[:n//2]) ** 2 / n
        
        # Low frequencies
        frequencies = np.arange(1, m+1) / n
        low_freq_periodogram = periodogram[1:m+1]
        
        # GPH regression
        log_periodogram = np.log(low_freq_periodogram)
        log_frequencies = np.log(frequencies)
        
        # Regression: log(I(w)) = c - d * log(w) + error
        X = np.column_stack([np.ones(m), log_frequencies])
        y = log_periodogram
        
        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        d_estimate = -beta[1]
        
        return d_estimate
    
    def local_whittle_estimator(
        self,
        series: np.ndarray,
        m: Optional[int] = None
    ) -> float:
        """
        Local Whittle estimator for fractional integration parameter d.
        
        Args:
            series: Time series data
            m: Number of low frequencies to use
            
        Returns:
            Estimated d parameter
        """
        n = len(series)
        if m is None:
            m = int(n ** 0.5)
        
        # Calculate periodogram
        fft_result = np.fft.fft(series)
        periodogram = np.abs(fft_result[:n//2]) ** 2 / n
        
        # Low frequencies
        frequencies = np.arange(1, m+1) / n
        low_freq_periodogram = periodogram[1:m+1]
        
        def objective(d):
            """Objective function to minimize."""
            # Theoretical spectral density for fractionally integrated process
            theoretical = 4 * np.sin(np.pi * frequencies / 2) ** (2 * d)
            theoretical = theoretical * (np.sin(np.pi * frequencies) / (np.pi * frequencies)) ** (2 * d)
            
            # Local Whittle likelihood
            return np.sum(np.log(low_freq_periodogram / theoretical))
        
        # Optimize
        result = minimize(objective, x0=0.2, bounds=[(0, 0.5)])
        d_estimate = result.x[0]
        
        return d_estimate
    
    def hurst_exponent(
        self,
        series: np.ndarray,
        max_lag: int = 20
    ) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        Args:
            series: Time series data
            max_lag: Maximum lag to consider
            
        Returns:
            Hurst exponent H
        """
        lags = range(2, max_lag)
        
        # Calculate R/S for each lag
        rs_values = []
        
        for lag in lags:
            # Split into subseries
            n_subseries = len(series) // lag
            subseries = series[:n_subseries * lag].reshape(n_subseries, lag)
            
            # Calculate R/S for each subseries
            rs_sub = []
            for sub in subseries:
                # Cumulative deviations
                cumdev = np.cumsum(sub - np.mean(sub))
                
                # Range
                R = np.max(cumdev) - np.min(cumdev)
                
                # Standard deviation
                S = np.std(sub)
                
                if S > 0:
                    rs_sub.append(R / S)
            
            if rs_sub:
                rs_values.append(np.mean(rs_sub))
        
        # Regress log(R/S) on log(lag)
        log_lags = np.log(lags)
        log_rs = np.log(rs_values)
        
        # OLS regression
        X = np.column_stack([np.ones(len(lags)), log_lags])
        y = log_rs
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        H = beta[1]
        
        return H
    
    def fit(
        self,
        prices: pd.Series,
        method: str = "gph"
    ) -> LongMemoryParams:
        """
        Fit long memory volatility model.
        
        Args:
            prices: Price series
            method: Estimation method ("gph" or "whittle")
            
        Returns:
            LongMemoryParams with estimated parameters
        """
        print(f"Fitting Long Memory Volatility model using {method.upper()} estimator...")
        
        # Calculate returns and realized volatility
        returns = self.calculate_returns(prices)
        realized_vol = self.calculate_realized_volatility(returns)
        
        # Estimate d parameter
        if method == "gph":
            d = self.gph_estimator(realized_vol.dropna().values)
        elif method == "whittle":
            d = self.local_whittle_estimator(realized_vol.dropna().values)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Estimate Hurst exponent
        H = self.hurst_exponent(realized_vol.dropna().values)
        
        # Estimate sigma and mu
        sigma = np.std(realized_vol.dropna())
        mu = np.mean(realized_vol.dropna())
        
        self.params = LongMemoryParams(
            d=d,
            H=H,
            sigma=sigma,
            mu=mu
        )
        
        self.is_fitted = True
        print(f"Model fitted: d={d:.4f}, H={H:.4f}, sigma={sigma:.4f}")
        
        return self.params
    
    def classify_regime(self, d: float) -> str:
        """Classify volatility regime based on d parameter."""
        if d < self.regime_thresholds["low_vol"]:
            return "low_vol"
        elif d < self.regime_thresholds["normal"]:
            return "normal"
        elif d < self.regime_thresholds["high_vol"]:
            return "high_vol"
        else:
            return "crisis"
    
    def forecast(
        self,
        horizon: int = 20,
        confidence_level: float = 0.95
    ) -> VolatilityForecast:
        """
        Forecast volatility using long memory model.
        
        Args:
            horizon: Forecast horizon in days
            confidence_level: Confidence level for prediction interval
            
        Returns:
            VolatilityForecast with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Simple ARFIMA-like forecast
        # In production, would use proper ARFIMA model
        forecasts = np.zeros(horizon)
        
        # Mean reversion to long-run mean
        for h in range(horizon):
            # Decay factor based on d
            decay = (h + 1) ** (-self.params.d)
            forecasts[h] = self.params.mu + (self.params.sigma - self.params.mu) * decay
        
        # Calculate confidence intervals
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        std_errors = self.params.sigma * np.sqrt(np.arange(1, horizon + 1) ** self.params.d)
        
        lower = forecasts - z * std_errors
        upper = forecasts + z * std_errors
        
        # Classify regime
        regime = self.classify_regime(self.params.d)
        
        return VolatilityForecast(
            forecast=forecasts,
            confidence_interval=(lower, upper),
            regime=regime,
            d_estimate=self.params.d,
            H_estimate=self.params.H
        )
    
    def calculate_cross_sectional_d(
        self,
        volatilities: Dict[str, pd.Series]
    ) -> Dict[str, float]:
        """
        Calculate cross-sectional d estimates for multiple assets.
        
        Args:
            volatilities: Dictionary mapping asset names to volatility series
            
        Returns:
            Dictionary mapping asset names to d estimates
        """
        d_estimates = {}
        
        for symbol, vol_series in volatilities.items():
            try:
                d = self.gph_estimator(vol_series.dropna().values)
                d_estimates[symbol] = d
            except:
                d_estimates[symbol] = 0.0
        
        return d_estimates
    
    def print_model_summary(self) -> None:
        """Print model summary."""
        if not self.is_fitted:
            print("Model not fitted")
            return
        
        print("\n" + "="*60)
        print("LONG MEMORY VOLATILITY MODEL SUMMARY")
        print("="*60)
        print(f"Fractional Integration (d): {self.params.d:.4f}")
        print(f"Hurst Exponent (H): {self.params.H:.4f}")
        print(f"Volatility (σ): {self.params.sigma:.4f}")
        print(f"Mean (μ): {self.params.mu:.4f}")
        
        regime = self.classify_regime(self.params.d)
        print(f"\nRegime Classification: {regime.upper()}")
        
        print("\nInterpretation:")
        if self.params.d < 0.1:
            print("  - Very short memory (anti-persistent)")
        elif self.params.d < 0.3:
            print("  - Moderate long memory (typical for volatility)")
        elif self.params.d < 0.5:
            print("  - Strong long memory (persistent volatility)")
        else:
            print("  - Very strong long memory (crisis conditions)")
        
        if self.params.H < 0.5:
            print("  - Rough volatility (anti-persistent)")
        elif self.params.H < 0.5:
            print("  - Smooth volatility (persistent)")
        else:
            print("  - Random walk volatility")
        
        print("\nDeep et al. (2026) Benchmarks:")
        print(f"  - GPH d: 0.226 (baseline)")
        print(f"  - Local Whittle d: 0.440 (baseline)")
        print(f"  - Hurst H: 0.063 (baseline)")
        print(f"  - COVID d spike: +86% from baseline")
        print("="*60)


def run_sample_analysis():
    """Run sample long memory volatility analysis."""
    # Create synthetic price data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    
    np.random.seed(42)
    
    # Simulate prices with volatility clustering
    prices = []
    vol = 0.15
    
    for i in range(len(dates)):
        # Volatility with long memory
        vol = 0.95 * vol + 0.05 * 0.20 + np.random.normal(0, 0.02)
        vol = max(vol, 0.05)
        
        if i == 0:
            price = 20000
        else:
            ret = np.random.normal(0.0005, vol / np.sqrt(252))
            price = prices[-1] * (1 + ret)
        
        prices.append(price)
    
    price_series = pd.Series(prices, index=dates)
    
    # Initialize and fit model
    lmv = LongMemoryVolatility()
    params = lmv.fit(price_series, method="gph")
    
    # Print summary
    lmv.print_model_summary()
    
    # Forecast
    forecast = lmv.forecast(horizon=20)
    
    print(f"\n20-Day Volatility Forecast:")
    print(f"  Mean: {np.mean(forecast.forecast):.4f}")
    print(f"  Regime: {forecast.regime}")
    
    return params


if __name__ == "__main__":
    run_sample_analysis()

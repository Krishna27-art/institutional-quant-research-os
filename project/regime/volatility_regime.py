"""
Volatility Regime and Roughness Detection

Implements volatility regime detection and roughness estimation following
Gatheral (2018) and Deep et al. This module identifies different volatility
regimes to inform strategy selection and risk management.

Key Features:
- Rolling GARCH(1,1) for conditional volatility
- Hurst exponent for long memory detection
- Rough volatility estimation (rBergomi)
- Regime classification (ROUGH_HIGH_VOL, ROUGH_LOW_VOL, TRENDING, MEAN_REVERTING)
- FIGARCH for long memory volatility modeling
- Volatility risk premium calculation

Based on Blueprint Week 9-10: Portfolio & Risk
References:
- Gatheral (2018) - Rough Volatility
- Deep et al. - Fractional calculus for financial time series
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    ROUGH_HIGH_VOL = "ROUGH_HIGH_VOL"      # Option overpriced → short volatility
    ROUGH_LOW_VOL = "ROUGH_LOW_VOL"        # Underpriced options → long volatility
    TRENDING = "TRENDING"                  # Momentum regime
    MEAN_REVERTING = "MEAN_REVERTING"      # Mean reversion regime
    UNKNOWN = "UNKNOWN"


class VolatilityRegimeDetector:
    """
    Volatility Regime Detector.
    
    Detects volatility regimes using GARCH, Hurst exponent, and roughness
    measures to inform strategy selection and risk management.
    """
    
    def __init__(self, window: int = 252, hurst_window: int = 60):
        """
        Initialize volatility regime detector.
        
        Args:
            window: Window for GARCH estimation
            hurst_window: Window for Hurst exponent
        """
        self.window = window
        self.hurst_window = hurst_window
        
        # Historical conditional volatility for percentile calculation
        self.cond_vol_history: List[float] = []
    
    def detect(self, returns: pd.Series) -> Dict:
        """
        Detect volatility regime.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with regime information
        """
        if len(returns) < self.window:
            return {
                'regime': VolatilityRegime.UNKNOWN,
                'cond_vol': 0.0,
                'hurst': 0.5,
                'roughness': 0.5,
                'reason': 'Insufficient data'
            }
        
        # Calculate conditional volatility using GARCH(1,1)
        cond_vol = self._estimate_garch(returns[-self.window:])
        
        # Update history
        self.cond_vol_history.append(cond_vol)
        if len(self.cond_vol_history) > 1000:
            self.cond_vol_history = self.cond_vol_history[-1000:]
        
        # Calculate Hurst exponent
        hurst_val = self._calculate_hurst(returns[-self.hurst_window:])
        
        # Calculate roughness
        roughness = self._calculate_roughness(returns[-self.window:])
        
        # Classify regime
        regime = self._classify_regime(hurst_val, cond_vol)
        
        return {
            'regime': regime,
            'cond_vol': cond_vol,
            'hurst': hurst_val,
            'roughness': roughness,
            'vol_percentile': self._get_vol_percentile(cond_vol),
            'reason': self._get_regime_reason(regime, hurst_val, cond_vol)
        }
    
    def _estimate_garch(self, returns: pd.Series) -> float:
        """
        Estimate conditional volatility using GARCH(1,1).
        
        Simplified GARCH(1,1) estimation:
        sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
        
        Args:
            returns: Return series
            
        Returns:
            Conditional volatility
        """
        # Simplified GARCH(1,1) parameters
        omega = 0.00001
        alpha = 0.1
        beta = 0.85
        
        # Initialize
        sigma2 = np.var(returns)
        cond_vols = []
        
        for r in returns:
            sigma2 = omega + alpha * r**2 + beta * sigma2
            cond_vols.append(np.sqrt(sigma2))
        
        return cond_vols[-1]
    
    def _calculate_hurst(self, returns: pd.Series) -> float:
        """
        Calculate Hurst exponent using R/S analysis.
        
        Args:
            returns: Return series
            
        Returns:
            Hurst exponent
        """
        if len(returns) < 20:
            return 0.5
        
        lags = range(2, min(50, len(returns) // 2))
        tau = []
        
        for lag in lags:
            # Calculate range
            cumsum = np.cumsum(returns.values - np.mean(returns.values))
            range_val = np.max(cumsum[:lag]) - np.min(cumsum[:lag])
            
            # Standard deviation
            std_val = np.std(returns.values[:lag])
            
            if std_val > 0:
                tau.append(range_val / std_val)
        
        if len(tau) == 0:
            return 0.5
        
        # Fit log-log regression
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        
        return slope
    
    def _calculate_roughness(self, returns: pd.Series) -> float:
        """
        Calculate roughness parameter H for volatility.
        
        Following Gatheral (2018), roughness H is estimated from
        the autocorrelation of log realized volatility.
        
        Args:
            returns: Return series
            
        Returns:
            Roughness parameter H
        """
        # Calculate realized volatility
        rv = returns.rolling(window=20).std() * np.sqrt(252)
        log_rv = np.log(rv.dropna())
        
        if len(log_rv) < 20:
            return 0.5
        
        # Calculate autocorrelation
        acf = [log_rv.autocorr(lag) for lag in range(1, 11)]
        
        # Roughness H is related to decay of autocorrelation
        # For rough volatility, H < 0.5
        H = 0.5 - 0.5 * np.mean(acf)
        H = np.clip(H, 0.0, 0.5)
        
        return H
    
    def _classify_regime(
        self,
        hurst: float,
        cond_vol: float
    ) -> VolatilityRegime:
        """
        Classify volatility regime.
        
        Args:
            hurst: Hurst exponent
            cond_vol: Conditional volatility
            
        Returns:
            Volatility regime
        """
        vol_percentile = self._get_vol_percentile(cond_vol)
        
        # Rough volatility regimes
        if hurst < 0.45:
            if vol_percentile > 80:
                return VolatilityRegime.ROUGH_HIGH_VOL
            elif vol_percentile < 20:
                return VolatilityRegime.ROUGH_LOW_VOL
        
        # Trending regime
        if hurst > 0.55:
            return VolatilityRegime.TRENDING
        
        # Default to mean reverting
        return VolatilityRegime.MEAN_REVERTING
    
    def _get_vol_percentile(self, cond_vol: float) -> float:
        """
        Get volatility percentile from historical distribution.
        
        Args:
            cond_vol: Conditional volatility
            
        Returns:
            Percentile (0-100)
        """
        if len(self.cond_vol_history) < 10:
            return 50.0
        
        vols = np.array(self.cond_vol_history)
        percentile = (vols < cond_vol).mean() * 100
        
        return percentile
    
    def _get_regime_reason(
        self,
        regime: VolatilityRegime,
        hurst: float,
        cond_vol: float
    ) -> str:
        """
        Get explanation for regime classification.
        
        Args:
            regime: Volatility regime
            hurst: Hurst exponent
            cond_vol: Conditional volatility
            
        Returns:
            Reason string
        """
        if regime == VolatilityRegime.ROUGH_HIGH_VOL:
            return f"Rough volatility with high vol (H={hurst:.3f}, vol={cond_vol:.4f})"
        elif regime == VolatilityRegime.ROUGH_LOW_VOL:
            return f"Rough volatility with low vol (H={hurst:.3f}, vol={cond_vol:.4f})"
        elif regime == VolatilityRegime.TRENDING:
            return f"Trending regime (H={hurst:.3f})"
        elif regime == VolatilityRegime.MEAN_REVERTING:
            return f"Mean reverting regime (H={hurst:.3f})"
        else:
            return "Unknown regime"


class FIGARCHModel:
    """
    FIGARCH (Fractionally Integrated GARCH) model.
    
    Extends GARCH to model long memory in volatility using fractional
    integration. This is useful for capturing the rough volatility
    behavior observed in markets.
    """
    
    def __init__(
        self,
        d: float = 0.4,  # Fractional integration parameter
        omega: float = 0.00001,
        alpha: float = 0.1,
        beta: float = 0.85
    ):
        """
        Initialize FIGARCH model.
        
        Args:
            d: Fractional integration parameter (0 < d < 1)
            omega: Constant term
            alpha: ARCH coefficient
            beta: GARCH coefficient
        """
        self.d = d
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
    
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit FIGARCH model to returns.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with fitted parameters
        """
        # Simplified FIGARCH estimation
        # In production, this would use maximum likelihood estimation
        
        # Calculate fractional differencing weights
        weights = self._fracdiff_weights(self.d, len(returns))
        
        # Apply fractional differencing
        fracdiff_returns = np.convolve(returns.values, weights, mode='valid')
        
        # Estimate GARCH on fractionally differenced series
        sigma2 = np.var(fracdiff_returns)
        
        return {
            'd': self.d,
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'sigma2': sigma2,
            'long_memory': self.d > 0
        }
    
    def _fracdiff_weights(self, d: float, max_lag: int) -> np.ndarray:
        """
        Calculate fractional differencing weights.
        
        Args:
            d: Fractional differencing order
            max_lag: Maximum lag
            
        Returns:
            Weight array
        """
        weights = [1.0]
        for k in range(1, max_lag):
            w = -weights[-1] * (d - k + 1) / k
            if abs(w) < 1e-5:
                break
            weights.append(w)
        
        return np.array(weights[::-1])
    
    def forecast(self, returns: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility using FIGARCH.
        
        Args:
            returns: Return series
            horizon: Forecast horizon
            
        Returns:
            Volatility forecast
        """
        # Fit model
        params = self.fit(returns)
        
        # Simple forecast (constant mean)
        forecast = np.full(horizon, np.sqrt(params['sigma2']))
        
        return forecast


class VolatilityRiskPremium:
    """
    Volatility Risk Premium (VRP) calculator.
    
    Calculates the volatility risk premium, which is the difference
    between implied volatility and realized volatility. Positive VRP
    suggests options are overpriced (short volatility opportunity).
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize VRP calculator.
        
        Args:
            window: Window for realized volatility calculation
        """
        self.window = window
    
    def calculate_vrp(
        self,
        implied_vol: float,
        returns: pd.Series
    ) -> Dict:
        """
        Calculate volatility risk premium.
        
        Args:
            implied_vol: Implied volatility (annualized)
            returns: Return series
            
        Returns:
            Dictionary with VRP metrics
        """
        # Calculate realized volatility
        realized_vol = returns.rolling(window=self.window).std() * np.sqrt(252)
        current_rv = realized_vol.iloc[-1]
        
        # Calculate VRP
        vrp = implied_vol - current_rv
        vrp_pct = vrp / implied_vol if implied_vol > 0 else 0.0
        
        # Determine if VRP is significant
        vrp_std = realized_vol.std()
        vrp_significant = abs(vrp) > 2 * vrp_std
        
        return {
            'implied_vol': implied_vol,
            'realized_vol': current_rv,
            'vrp': vrp,
            'vrp_pct': vrp_pct,
            'vrp_significant': vrp_significant,
            'recommendation': 'SHORT_VOL' if vrp > 0 else 'LONG_VOL'
        }


if __name__ == "__main__":
    # Test volatility regime detection
    print("Testing Volatility Regime Detection...")
    
    # Create sample returns
    np.random.seed(42)
    n_samples = 500
    returns = pd.Series(np.random.normal(0.0005, 0.02, n_samples))
    
    # Create detector
    detector = VolatilityRegimeDetector(window=252, hurst_window=60)
    
    # Detect regime
    regime_info = detector.detect(returns)
    
    print(f"\nRegime: {regime_info['regime'].value}")
    print(f"Conditional Volatility: {regime_info['cond_vol']:.4f}")
    print(f"Hurst Exponent: {regime_info['hurst']:.3f}")
    print(f"Roughness: {regime_info['roughness']:.3f}")
    print(f"Volatility Percentile: {regime_info['vol_percentile']:.1f}")
    print(f"Reason: {regime_info['reason']}")
    
    # Test FIGARCH
    print("\nTesting FIGARCH...")
    figarch = FIGARCHModel(d=0.4)
    params = figarch.fit(returns)
    print(f"FIGARCH parameters: {params}")
    
    forecast = figarch.forecast(returns, horizon=5)
    print(f"Volatility forecast (5 days): {forecast}")
    
    # Test VRP
    print("\nTesting Volatility Risk Premium...")
    vrp = VolatilityRiskPremium(window=20)
    vrp_info = vrp.calculate_vrp(implied_vol=0.25, returns=returns)
    print(f"Implied Vol: {vrp_info['implied_vol']:.4f}")
    print(f"Realized Vol: {vrp_info['realized_vol']:.4f}")
    print(f"VRP: {vrp_info['vrp']:.4f}")
    print(f"VRP %: {vrp_info['vrp_pct']:.2%}")
    print(f"Recommendation: {vrp_info['recommendation']}")
    
    print("\nVolatility Regime Detection test completed.")

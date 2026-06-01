"""
GARCH(1,1) Volatility Modeling
Integrated from quant_probability_engine folder

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters"""
    omega: float  # Constant term
    alpha: float  # ARCH term (lagged squared shock)
    beta: float   # GARCH term (lagged variance)
    mu: float    # Mean return
    
    def validate(self) -> bool:
        """Validate GARCH parameters."""
        return (
            self.omega > 0 and
            self.alpha >= 0 and
            self.beta >= 0 and
            self.alpha + self.beta < 1
        )


class GARCHModel:
    """
    GARCH(1,1) volatility model with expiry segmentation.
    
    Fits two separate GARCH models:
    - GARCH_Regular for non-expiry sessions
    - GARCH_Expiry for expiry week sessions
    
    Model:
    ε_t = r_t - μ
    σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
    """
    
    def __init__(self):
        self.regular_params: Optional[GARCHParams] = None
        self.expiry_params: Optional[GARCHParams] = None
        self.is_fitted = False
    
    def _garch_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Negative log-likelihood for GARCH(1,1).
        
        Args:
            params: [omega, alpha, beta, mu]
            returns: Return series
            
        Returns:
            Negative log-likelihood
        """
        omega, alpha, beta, mu = params
        
        # Validate parameters
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(returns)
        sigma2 = np.ones(n) * np.var(returns)
        
        # Initialize with unconditional variance
        sigma2[0] = omega / (1 - alpha - beta)
        
        # Compute variance path
        for t in range(1, n):
            epsilon = returns[t-1] - mu
            sigma2[t] = omega + alpha * epsilon**2 + beta * sigma2[t-1]
        
        # Compute log-likelihood
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * sigma2) + (returns - mu)**2 / sigma2
        )
        
        return -log_likelihood
    
    def fit(self, returns: pd.Series, is_expiry_week: bool = False) -> GARCHParams:
        """
        Fit GARCH(1,1) model using MLE.
        
        Args:
            returns: Return series
            is_expiry_week: Whether this is expiry week data
            
        Returns:
            Fitted GARCH parameters
        """
        # Initial parameter guess
        initial_params = [
            np.var(returns) * 0.1,  # omega
            0.1,  # alpha
            0.85,  # beta
            np.mean(returns)  # mu
        ]
        
        # Parameter bounds
        bounds = [
            (1e-6, None),  # omega > 0
            (0, 0.99),     # alpha in [0, 0.99]
            (0, 0.99),     # beta in [0, 0.99]
            (None, None)   # mu unconstrained
        ]
        
        # Optimize
        result = minimize(
            self._garch_log_likelihood,
            initial_params,
            args=(returns.values,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            params = GARCHParams(*result.x)
            
            if is_expiry_week:
                self.expiry_params = params
            else:
                self.regular_params = params
            
            self.is_fitted = True
            return params
        else:
            raise RuntimeError(f"GARCH fitting failed: {result.message}")
    
    def forecast_volatility(
        self,
        current_variance: float,
        last_return: float,
        horizon: int = 1,
        is_expiry_week: bool = False
    ) -> np.ndarray:
        """
        Forecast volatility for given horizon.
        
        Args:
            current_variance: Current variance
            last_return: Last return
            horizon: Forecast horizon
            is_expiry_week: Use expiry week model
            
        Returns:
            Array of forecasted variances
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        params = self.expiry_params if is_expiry_week else self.regular_params
        
        forecasts = np.zeros(horizon)
        sigma2 = current_variance
        epsilon = last_return - params.mu
        
        for h in range(horizon):
            sigma2 = params.omega + params.alpha * epsilon**2 + params.beta * sigma2
            forecasts[h] = sigma2
            epsilon = 0  # For multi-step forecast, assume zero shock
        
        return forecasts
    
    def get_conditional_volatility(
        self,
        returns: pd.Series,
        is_expiry_week: bool = False
    ) -> pd.Series:
        """
        Get conditional volatility series from fitted model.
        
        Args:
            returns: Return series
            is_expiry_week: Use expiry week model
            
        Returns:
            Series of conditional volatilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        params = self.expiry_params if is_expiry_week else self.regular_params
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = params.omega / (1 - params.alpha - params.beta)
        
        for t in range(1, n):
            epsilon = returns.iloc[t-1] - params.mu
            sigma2[t] = params.omega + params.alpha * epsilon**2 + params.beta * sigma2[t-1]
        
        return pd.Series(np.sqrt(sigma2), index=returns.index)


class RegimeGARCHManager:
    """
    Manages GARCH models for different regimes.
    
    Maintains separate models for:
    - Regular sessions
    - Expiry week sessions
    - High volatility regime
    - Low volatility regime
    """
    
    def __init__(self):
        self.regular_model = GARCHModel()
        self.expiry_model = GARCHModel()
        self.high_vol_model = GARCHModel()
        self.low_vol_model = GARCHModel()
    
    def fit_all(
        self,
        returns: pd.Series,
        expiry_mask: pd.Series,
        vol_threshold: float = 0.20
    ) -> None:
        """
        Fit all GARCH models.
        
        Args:
            returns: Return series
            expiry_mask: Boolean series indicating expiry weeks
            vol_threshold: Volatility threshold for regime classification
        """
        # Fit regular model
        regular_returns = returns[~expiry_mask]
        if len(regular_returns) > 100:
            self.regular_model.fit(regular_returns, is_expiry_week=False)
        
        # Fit expiry model
        expiry_returns = returns[expiry_mask]
        if len(expiry_returns) > 50:
            self.expiry_model.fit(expiry_returns, is_expiry_week=True)
        
        # Fit high volatility model
        high_vol_mask = returns.rolling(20).std() > vol_threshold
        high_vol_returns = returns[high_vol_mask]
        if len(high_vol_returns) > 100:
            self.high_vol_model.fit(high_vol_returns, is_expiry_week=False)
        
        # Fit low volatility model
        low_vol_mask = returns.rolling(20).std() <= vol_threshold
        low_vol_returns = returns[low_vol_mask]
        if len(low_vol_returns) > 100:
            self.low_vol_model.fit(low_vol_returns, is_expiry_week=False)
    
    def get_model_for_regime(
        self,
        is_expiry_week: bool,
        current_volatility: float,
        vol_threshold: float = 0.20
    ) -> GARCHModel:
        """
        Get appropriate GARCH model for current regime.
        
        Args:
            is_expiry_week: Whether currently in expiry week
            current_volatility: Current volatility level
            vol_threshold: Volatility threshold
            
        Returns:
            Appropriate GARCH model
        """
        if is_expiry_week and self.expiry_model.is_fitted:
            return self.expiry_model
        
        if current_volatility > vol_threshold and self.high_vol_model.is_fitted:
            return self.high_vol_model
        
        if current_volatility <= vol_threshold and self.low_vol_model.is_fitted:
            return self.low_vol_model
        
        # Default to regular model
        return self.regular_model

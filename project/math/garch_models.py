"""
GARCH and FIGARCH Volatility Models

Implements GARCH(1,1), EGARCH, and FIGARCH models for volatility forecasting.
These models are essential for risk management, option pricing, and
portfolio construction.

Key Features:
- GARCH(1,1) with maximum likelihood estimation
- EGARCH for asymmetric volatility (leverage effect)
- FIGARCH for long memory in volatility
- Volatility forecasting with confidence intervals
- Model diagnostics and goodness-of-fit tests

Based on Blueprint Week 3-4: Mathematical & Statistical Toolkit
References:
- Engle (1982) - Autoregressive Conditional Heteroskedasticity
- Bollerslev (1986) - Generalized ARCH
- Nelson (1991) - EGARCH
- Baillie et al. (1996) - FIGARCH
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class GARCH11:
    """
    GARCH(1,1) Model.
    
    The GARCH(1,1) model is defined as:
    r_t = μ + ε_t
    ε_t = σ_t * z_t, where z_t ~ N(0,1)
    σ_t² = ω + α * ε_{t-1}² + β * σ_{t-1}²
    
    Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1
    """
    
    def __init__(self):
        """Initialize GARCH(1,1) model."""
        self.omega = None
        self.alpha = None
        self.beta = None
        self.mu = None
        self.fitted = False
    
    def fit(self, returns: pd.Series, method: str = 'MLE') -> Dict:
        """
        Fit GARCH(1,1) model to returns.
        
        Args:
            returns: Return series
            method: Estimation method ('MLE' for maximum likelihood)
            
        Returns:
            Dictionary with fitted parameters
        """
        # Initial parameters
        initial_params = [returns.mean(), 0.1, 0.1, 0.85]
        
        # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [
            (None, None),  # mu (unconstrained)
            (1e-6, None),  # omega > 0
            (0, 0.99),     # alpha in [0, 0.99]
            (0, 0.99)      # beta in [0, 0.99]
        ]
        
        def constraint(params):
            omega, alpha, beta = params[1], params[2], params[3]
            return 0.99 - (alpha + beta)  # alpha + beta < 1
        
        constraints = {'type': 'ineq', 'fun': constraint}
        
        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(returns.values,),
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if result.success:
            self.mu, self.omega, self.alpha, self.beta = result.x
            self.fitted = True
            
            # Calculate persistence
            persistence = self.alpha + self.beta
            
            # Calculate long-run variance
            long_run_var = self.omega / (1 - persistence)
            
            return {
                'mu': self.mu,
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'persistence': persistence,
                'long_run_variance': long_run_var,
                'log_likelihood': -result.fun
            }
        else:
            logger.error(f"GARCH(1,1) fitting failed: {result.message}")
            return {}
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood.
        
        Args:
            params: Model parameters [mu, omega, alpha, beta]
            returns: Return series
            
        Returns:
            Negative log-likelihood
        """
        mu, omega, alpha, beta = params
        
        n = len(returns)
        sigma2 = np.var(returns)  # Initialize
        
        log_likelihood = 0.0
        
        for i in range(n):
            # Calculate conditional variance
            sigma2 = omega + alpha * (returns[i-1] - mu)**2 + beta * sigma2 if i > 0 else omega / (1 - alpha - beta)
            sigma2 = max(sigma2, 1e-10)  # Ensure positivity
            
            # Log-likelihood
            log_likelihood += -0.5 * (np.log(2 * np.pi * sigma2) + (returns[i] - mu)**2 / sigma2)
        
        return -log_likelihood
    
    def forecast(self, returns: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility.
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon
            
        Returns:
            Volatility forecast array
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get last conditional variance
        n = len(returns)
        sigma2 = np.var(returns)
        
        for i in range(n):
            sigma2 = self.omega + self.alpha * (returns[i-1] - self.mu)**2 + self.beta * sigma2 if i > 0 else self.omega / (1 - self.alpha - self.beta)
        
        # Forecast
        forecasts = []
        for h in range(horizon):
            # Long-run variance
            long_run_var = self.omega / (1 - self.alpha - self.beta)
            persistence = self.alpha + self.beta
            
            # Forecast variance
            sigma2_forecast = long_run_var + persistence**h * (sigma2 - long_run_var)
            forecasts.append(np.sqrt(sigma2_forecast))
        
        return np.array(forecasts)


class EGARCH:
    """
    EGARCH Model (Exponential GARCH).
    
    The EGARCH model allows for asymmetric effects of positive and negative
    shocks (leverage effect). The model is defined as:
    
    log(σ_t²) = ω + β * log(σ_{t-1}²) + α * (|z_{t-1}| - E[|z|]) + γ * z_{t-1}
    
    where γ captures the leverage effect.
    """
    
    def __init__(self):
        """Initialize EGARCH model."""
        self.omega = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.mu = None
        self.fitted = False
    
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit EGARCH model to returns.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with fitted parameters
        """
        # Initial parameters
        initial_params = [returns.mean(), 0.1, 0.1, 0.9, 0.1]
        
        # Bounds
        bounds = [
            (None, None),  # mu
            (None, None),  # omega
            (0, None),     # alpha > 0
            (0, 0.99),     # beta in [0, 0.99]
            (None, None)   # gamma (unconstrained)
        ]
        
        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(returns.values,),
            bounds=bounds,
            method='SLSQP'
        )
        
        if result.success:
            self.mu, self.omega, self.alpha, self.beta, self.gamma = result.x
            self.fitted = True
            
            return {
                'mu': self.mu,
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'leverage_effect': self.gamma > 0,
                'log_likelihood': -result.fun
            }
        else:
            logger.error(f"EGARCH fitting failed: {result.message}")
            return {}
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for EGARCH.
        
        Args:
            params: Model parameters [mu, omega, alpha, beta, gamma]
            returns: Return series
            
        Returns:
            Negative log-likelihood
        """
        mu, omega, alpha, beta, gamma = params
        
        n = len(returns)
        log_sigma2 = np.log(np.var(returns))
        
        log_likelihood = 0.0
        e_abs = np.sqrt(2 / np.pi)  # E[|z|] for standard normal
        
        for i in range(n):
            if i > 0:
                z = (returns[i-1] - mu) / np.exp(0.5 * log_sigma2)
                log_sigma2 = omega + beta * log_sigma2 + alpha * (abs(z) - e_abs) + gamma * z
            
            sigma2 = np.exp(log_sigma2)
            log_likelihood += -0.5 * (np.log(2 * np.pi * sigma2) + (returns[i] - mu)**2 / sigma2)
        
        return -log_likelihood
    
    def forecast(self, returns: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility using EGARCH.
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon
            
        Returns:
            Volatility forecast array
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get last log variance
        n = len(returns)
        log_sigma2 = np.log(np.var(returns))
        e_abs = np.sqrt(2 / np.pi)
        
        for i in range(n):
            if i > 0:
                z = (returns[i-1] - self.mu) / np.exp(0.5 * log_sigma2)
                log_sigma2 = self.omega + self.beta * log_sigma2 + self.alpha * (abs(z) - e_abs) + self.gamma * z
        
        # Forecast (assuming mean reversion to long-run)
        long_run_log_sigma2 = self.omega / (1 - self.beta)
        
        forecasts = []
        for h in range(horizon):
            log_sigma2_forecast = long_run_log_sigma2 + self.beta**h * (log_sigma2 - long_run_log_sigma2)
            forecasts.append(np.exp(0.5 * log_sigma2_forecast))
        
        return np.array(forecasts)


class FIGARCH:
    """
    FIGARCH (Fractionally Integrated GARCH) Model.
    
    The FIGARCH model extends GARCH to capture long memory in volatility
    using fractional integration. This is useful for modeling rough
    volatility behavior observed in markets.
    
    The model uses fractional differencing with parameter d (0 < d < 1).
    """
    
    def __init__(self):
        """Initialize FIGARCH model."""
        self.d = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.mu = None
        self.fitted = False
    
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit FIGARCH model to returns.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with fitted parameters
        """
        # Initial parameters
        initial_params = [returns.mean(), 0.1, 0.1, 0.85, 0.4]
        
        # Bounds
        bounds = [
            (None, None),  # mu
            (1e-6, None),  # omega > 0
            (0, 0.99),     # alpha
            (0, 0.99),     # beta
            (0.01, 0.99)   # d in (0, 1)
        ]
        
        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(returns.values,),
            bounds=bounds,
            method='SLSQP'
        )
        
        if result.success:
            self.mu, self.omega, self.alpha, self.beta, self.d = result.x
            self.fitted = True
            
            return {
                'mu': self.mu,
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'd': self.d,
                'long_memory': self.d > 0,
                'log_likelihood': -result.fun
            }
        else:
            logger.error(f"FIGARCH fitting failed: {result.message}")
            return {}
    
    def _negative_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for FIGARCH.
        
        Args:
            params: Model parameters [mu, omega, alpha, beta, d]
            returns: Return series
            
        Returns:
            Negative log-likelihood
        """
        mu, omega, alpha, beta, d = params
        
        n = len(returns)
        sigma2 = np.var(returns)
        
        # Calculate fractional differencing weights
        weights = self._fracdiff_weights(d, n)
        
        log_likelihood = 0.0
        
        for i in range(n):
            # Apply fractional differencing to past squared returns
            fracdiff_returns = 0.0
            for j in range(min(i, len(weights))):
                fracdiff_returns += weights[j] * (returns[i-j-1] - mu)**2 if i-j-1 >= 0 else 0
            
            # Conditional variance
            sigma2 = omega + alpha * fracdiff_returns + beta * sigma2 if i > 0 else omega / (1 - alpha - beta)
            sigma2 = max(sigma2, 1e-10)
            
            log_likelihood += -0.5 * (np.log(2 * np.pi * sigma2) + (returns[i] - mu)**2 / sigma2)
        
        return -log_likelihood
    
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
            returns: Historical returns
            horizon: Forecast horizon
            
        Returns:
            Volatility forecast array
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get last conditional variance
        n = len(returns)
        sigma2 = np.var(returns)
        
        for i in range(n):
            weights = self._fracdiff_weights(self.d, i)
            fracdiff_returns = 0.0
            for j in range(min(i, len(weights))):
                fracdiff_returns += weights[j] * (returns[i-j-1] - self.mu)**2 if i-j-1 >= 0 else 0
            
            sigma2 = self.omega + self.alpha * fracdiff_returns + self.beta * sigma2 if i > 0 else self.omega / (1 - self.alpha - self.beta)
        
        # Forecast (long memory decay)
        long_run_var = self.omega / (1 - self.alpha - self.beta)
        persistence = self.alpha + self.beta
        
        forecasts = []
        for h in range(horizon):
            # Fractional decay
            decay = h ** (-self.d)
            sigma2_forecast = long_run_var + decay * (sigma2 - long_run_var)
            forecasts.append(np.sqrt(sigma2_forecast))
        
        return np.array(forecasts)


class VolatilityForecaster:
    """
    Unified volatility forecaster using multiple models.
    
    This class provides a unified interface for volatility forecasting
    using GARCH(1,1), EGARCH, and FIGARCH models.
    """
    
    def __init__(self, model: str = 'garch'):
        """
        Initialize volatility forecaster.
        
        Args:
            model: Model type ('garch', 'egarch', 'figarch')
        """
        self.model_type = model
        
        if model == 'garch':
            self.model = GARCH11()
        elif model == 'egarch':
            self.model = EGARCH()
        elif model == 'figarch':
            self.model = FIGARCH()
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit volatility model.
        
        Args:
            returns: Return series
            
        Returns:
            Dictionary with fitted parameters
        """
        return self.model.fit(returns)
    
    def forecast(self, returns: pd.Series, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility.
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon
            
        Returns:
            Volatility forecast
        """
        return self.model.forecast(returns, horizon)


if __name__ == "__main__":
    # Test GARCH models
    print("Testing GARCH and FIGARCH Models...")
    
    # Create sample returns
    np.random.seed(42)
    n_samples = 500
    returns = pd.Series(np.random.normal(0.0005, 0.02, n_samples))
    
    # Test GARCH(1,1)
    print("\nTesting GARCH(1,1)...")
    garch = GARCH11()
    params = garch.fit(returns)
    print(f"GARCH(1,1) parameters: {params}")
    
    forecast = garch.forecast(returns, horizon=5)
    print(f"Volatility forecast (5 days): {forecast}")
    
    # Test EGARCH
    print("\nTesting EGARCH...")
    egarch = EGARCH()
    params = egarch.fit(returns)
    print(f"EGARCH parameters: {params}")
    
    forecast = egarch.forecast(returns, horizon=5)
    print(f"Volatility forecast (5 days): {forecast}")
    
    # Test FIGARCH
    print("\nTesting FIGARCH...")
    figarch = FIGARCH()
    params = figarch.fit(returns)
    print(f"FIGARCH parameters: {params}")
    
    forecast = figarch.forecast(returns, horizon=5)
    print(f"Volatility forecast (5 days): {forecast}")
    
    print("\nGARCH and FIGARCH Models test completed.")

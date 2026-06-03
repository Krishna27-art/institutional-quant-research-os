"""
Volatility Surface Modeling

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#25)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- SABR model for volatility surface
- Heston stochastic volatility model
- Volatility smile/skew modeling
- Used for options pricing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Install with: pip install scipy")


@dataclass
class VolatilitySurfaceConfig:
    """Configuration for Volatility Surface"""
    # SABR parameters
    initial_alpha: float = 0.2  # Initial volatility
    initial_beta: float = 0.5  # Elasticity parameter
    initial_rho: float = -0.2  # Correlation
    initial_nu: float = 0.3  # Vol of vol
    
    # Heston parameters
    initial_v0: float = 0.04  # Initial variance
    initial_kappa: float = 2.0  # Mean reversion speed
    initial_theta: float = 0.04  # Long-term variance
    initial_sigma_v: float = 0.3  # Vol of vol
    initial_rho_h: float = -0.5  # Correlation
    
    # Calibration parameters
    max_iterations: int = 100
    tolerance: float = 1e-6
    
    # Surface parameters
    min_strike: float = 0.5  # Minimum moneyness
    max_strike: float = 1.5  # Maximum moneyness
    min_expiry: float = 0.08  # Minimum expiry (years)
    max_expiry: float = 1.0  # Maximum expiry (years)


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) Volatility Model
    
    Models the volatility smile using stochastic volatility.
    Widely used in interest rate and equity options markets.
    """
    
    def __init__(self, config: VolatilitySurfaceConfig):
        self.config = config
        
        # SABR parameters
        self.alpha = config.initial_alpha
        self.beta = config.initial_beta
        self.rho = config.initial_rho
        self.nu = config.initial_nu
    
    def sabr_volatility(self, F: float, K: float, T: float) -> float:
        """
        Calculate SABR implied volatility
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry (years)
            
        Returns:
            Implied volatility
        """
        if T <= 0:
            return self.alpha
        
        z = (self.nu / self.alpha) * (F * K) ** ((1 - self.beta) / 2) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))
        
        if abs(F - K) < 1e-6:
            # ATM formula
            sigma = self.alpha / (F ** (1 - self.beta)) * (
                1 + ((1 - self.beta) ** 2 * self.alpha ** 2 / (24 * F ** (2 - 2 * self.beta)) +
                     self.rho * self.beta * self.nu * self.alpha / (4 * F ** (1 - self.beta)) +
                     (2 - 3 * self.rho) * self.nu ** 2 / 24) * T
            )
        else:
            # General formula
            sigma = self.alpha / ((F * K) ** ((1 - self.beta) / 2) * (1 + ((1 - self.beta) ** 2 * self.alpha ** 2 / (24 * (F * K) ** (1 - self.beta)) +
                     self.rho * self.beta * self.nu * self.alpha / (4 * (F * K) ** ((1 - self.beta) / 2)) +
                     (2 - 3 * self.rho) * self.nu ** 2 / 24) * T))
        
        return sigma
    
    def calibrate(self, market_data: pd.DataFrame) -> Dict:
        """
        Calibrate SABR parameters to market data
        
        Args:
            market_data: DataFrame with columns [strike, expiry, iv, forward]
            
        Returns:
            Calibrated parameters
        """
        if not SCIPY_AVAILABLE:
            return {"error": "SciPy not available"}
        
        def objective(params):
            alpha, beta, rho, nu = params
            
            # Update parameters
            self.alpha = alpha
            self.beta = beta
            self.rho = rho
            self.nu = nu
            
            # Calculate error
            errors = []
            for _, row in market_data.iterrows():
                F = row['forward']
                K = row['strike']
                T = row['expiry']
                market_iv = row['iv']
                
                model_iv = self.sabr_volatility(F, K, T)
                errors.append((model_iv - market_iv) ** 2)
            
            return np.sum(errors)
        
        # Optimize
        initial_params = [self.alpha, self.beta, self.rho, self.nu]
        bounds = [(0.001, 2.0), (0.0, 1.0), (-0.999, 0.999), (0.001, 2.0)]
        
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
        )
        
        if result.success:
            self.alpha, self.beta, self.rho, self.nu = result.x
            return {
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "nu": self.nu,
                "success": True,
                "error": result.fun
            }
        else:
            return {"success": False, "error": result.message}
    
    def generate_surface(self, 
                       forward: float, 
                       strikes: np.ndarray, 
                       expiries: np.ndarray) -> pd.DataFrame:
        """
        Generate volatility surface
        
        Args:
            forward: Forward price
            strikes: Array of strike prices
            expiries: Array of expiry times (years)
            
        Returns:
            DataFrame with volatility surface
        """
        surface_data = []
        
        for T in expiries:
            for K in strikes:
                iv = self.sabr_volatility(forward, K, T)
                surface_data.append({
                    "strike": K,
                    "expiry": T,
                    "iv": iv,
                    "moneyness": K / forward
                })
        
        return pd.DataFrame(surface_data)


class HestonModel:
    """
    Heston Stochastic Volatility Model
    
    Models volatility as a mean-reverting process.
    Used for pricing European options.
    """
    
    def __init__(self, config: VolatilitySurfaceConfig):
        self.config = config
        
        # Heston parameters
        self.v0 = config.initial_v0
        self.kappa = config.initial_kappa
        self.theta = config.initial_theta
        self.sigma_v = config.initial_sigma_v
        self.rho_h = config.initial_rho_h
    
    def characteristic_function(self, phi: float, S: float, K: float, T: float, r: float) -> complex:
        """
        Heston characteristic function
        
        Args:
            phi: Integration variable
            S: Spot price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            
        Returns:
            Characteristic function value
        """
        # Simplified implementation
        # Full implementation requires complex integration
        i = 1j
        
        d = np.sqrt((self.rho_h * self.sigma_v * i * phi - self.kappa) ** 2 + 
                    self.sigma_v ** 2 * (i * phi + phi ** 2))
        
        g = (self.kappa - self.rho_h * self.sigma_v * i * phi - d) / (self.kappa - self.rho_h * self.sigma_v * i * phi + d)
        
        C = r * i * phi * T + (self.kappa * self.theta / self.sigma_v ** 2) * (
            (self.kappa - self.rho_h * self.sigma_v * i * phi - d) * T - 
            2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        
        D = (self.kappa - self.rho_h * self.sigma_v * i * phi - d) / self.sigma_v ** 2 * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        )
        
        return np.exp(C + D * self.v0 + i * phi * np.log(S))
    
    def option_price(self, S: float, K: float, T: float, r: float, option_type: str = "call") -> float:
        """
        Calculate option price using Heston model
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        if not SCIPY_AVAILABLE:
            # Fallback to Black-Scholes
            sigma = np.sqrt(self.v0)
            from alpha.options_skew_carry import BlackScholes, OptionType
            opt_type = OptionType.CALL if option_type == "call" else OptionType.PUT
            if option_type == "call":
                return BlackScholes.call_price(S, K, T, r, sigma)
            else:
                return BlackScholes.put_price(S, K, T, r, sigma)
        
        # Simplified Heston pricing (full implementation requires numerical integration)
        # Using Fourier transform method
        k = np.log(K / S)
        
        def integrand(phi):
            return (np.exp(-1j * phi * k) * self.characteristic_function(phi - 1j, S, K, T, r) / 
                   (1j * phi * self.characteristic_function(-1j, S, K, T, r))).real
        
        # Simple numerical integration
        phi_values = np.linspace(0.01, 100, 1000)
        integral = np.trapz([integrand(phi) for phi in phi_values], phi_values)
        
        if option_type == "call":
            price = S - K * np.exp(-r * T) + 0.5 * integral / np.pi
        else:
            price = K * np.exp(-r * T) - S + 0.5 * integral / np.pi
        
        return max(price, 0.0)


class VolatilitySurface:
    """
    Volatility Surface Manager
    
    Manages SABR and Heston models for volatility surface modeling.
    """
    
    def __init__(self, config: VolatilitySurfaceConfig):
        self.config = config
        
        self.sabr_model = SABRModel(config)
        self.heston_model = HestonModel(config)
        
        # Surface data
        self.surface_data: Optional[pd.DataFrame] = None
    
    def calibrate_sabr(self, market_data: pd.DataFrame) -> Dict:
        """Calibrate SABR model"""
        return self.sabr_model.calibrate(market_data)
    
    def generate_sabr_surface(self, forward: float) -> pd.DataFrame:
        """Generate SABR volatility surface"""
        strikes = np.linspace(forward * self.config.min_strike, 
                             forward * self.config.max_strike, 20)
        expiries = np.linspace(self.config.min_expiry, self.config.max_expiry, 10)
        
        surface = self.sabr_model.generate_surface(forward, strikes, expiries)
        self.surface_data = surface
        
        return surface
    
    def calculate_skew(self, forward: float, expiry: float) -> float:
        """
        Calculate volatility skew at given expiry
        
        Args:
            forward: Forward price
            expiry: Time to expiry (years)
            
        Returns:
            Skew value
        """
        # Calculate IV at 25-delta put and call
        strike_put = forward * 0.9
        strike_call = forward * 1.1
        
        iv_put = self.sabr_model.sabr_volatility(forward, strike_put, expiry)
        iv_call = self.sabr_model.sabr_volatility(forward, strike_call, expiry)
        
        skew = (iv_put - iv_call) / ((iv_put + iv_call) / 2)
        return skew
    
    def get_surface_summary(self) -> Dict:
        """Get surface summary statistics"""
        if self.surface_data is None:
            return {}
        
        return {
            "mean_iv": self.surface_data['iv'].mean(),
            "std_iv": self.surface_data['iv'].std(),
            "min_iv": self.surface_data['iv'].min(),
            "max_iv": self.surface_data['iv'].max(),
            "num_points": len(self.surface_data)
        }


def simulate_market_data(n_samples: int = 100, forward: float = 100.0) -> pd.DataFrame:
    """Simulate market option data for testing"""
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        strike = np.random.uniform(forward * 0.8, forward * 1.2)
        expiry = np.random.uniform(0.08, 1.0)
        
        # Simulate IV with skew
        moneyness = strike / forward
        if moneyness < 1.0:
            iv = 0.25 + (1.0 - moneyness) * 0.15 + np.random.randn() * 0.02
        else:
            iv = 0.25 + (moneyness - 1.0) * 0.05 + np.random.randn() * 0.02
        
        data.append({
            "strike": strike,
            "expiry": expiry,
            "iv": max(iv, 0.05),
            "forward": forward
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    config = VolatilitySurfaceConfig(
        initial_alpha=0.2,
        initial_beta=0.5,
        max_iterations=50
    )
    
    surface = VolatilitySurface(config)
    
    # Simulate market data
    print("Simulating market data...")
    market_data = simulate_market_data(100, forward=100.0)
    
    # Calibrate SABR
    print("\nCalibrating SABR model...")
    if SCIPY_AVAILABLE:
        sabr_params = surface.calibrate_sabr(market_data)
        print(f"\nSABR Parameters:")
        for key, value in sabr_params.items():
            print(f"  {key}: {value}")
    else:
        print("Skipping calibration (SciPy not available)")
    
    # Generate surface
    print("\nGenerating volatility surface...")
    surface_data = surface.generate_sabr_surface(forward=100.0)
    
    print(f"\nSurface Data (first 10 rows):")
    print(surface_data.head(10).to_string())
    
    # Calculate skew
    print("\nCalculating skew...")
    skew = surface.calculate_skew(forward=100.0, expiry=0.25)
    print(f"  Skew (25d): {skew:.4f}")
    
    # Surface summary
    print("\nSurface Summary:")
    summary = surface.get_surface_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

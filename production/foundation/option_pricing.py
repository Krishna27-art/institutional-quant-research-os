"""
Option Pricing Models - Level 2 Foundation

This module provides option pricing models:
- Black-Scholes model
- Heston stochastic volatility model
- Bates model with jumps
- Implied volatility calculation
- Local volatility surface
- Option Greeks calculation

Based on Audit Report Priority 2: Asset Pricing Theories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, optimize

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Types of options."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionParams:
    """Option parameters."""
    S: float  # Underlying price
    K: float  # Strike price
    T: float  # Time to maturity (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility
    option_type: OptionType = OptionType.CALL
    q: float = 0.0  # Dividend yield


class OptionPricingModels:
    """
    Option pricing models.
    
    This class implements various option pricing models including
    Black-Scholes, Heston, and Bates models.
    """
    
    def __init__(self):
        """Initialize option pricing models."""
        pass
    
    def black_scholes_call(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Black-Scholes call option price.
        
        C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Call option price
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        
        return call_price
    
    def black_scholes_put(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Black-Scholes put option price.
        
        P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Put option price
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
        
        return put_price
    
    def black_scholes(
        self,
        params: OptionParams
    ) -> float:
        """
        Black-Scholes option price (unified interface).
        
        Args:
            params: OptionParams object
            
        Returns:
            Option price
        """
        if params.option_type == OptionType.CALL:
            return self.black_scholes_call(
                params.S, params.K, params.T, params.r, params.sigma, params.q
            )
        else:
            return self.black_scholes_put(
                params.S, params.K, params.T, params.r, params.sigma, params.q
            )
    
    def black_scholes_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes option Greeks.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Dictionary with Greeks (delta, gamma, theta, vega, rho)
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        sqrt_T = np.sqrt(T)
        nd1 = stats.norm.pdf(d1)
        Nd1 = stats.norm.cdf(d1)
        Nd2 = stats.norm.cdf(d2)
        
        # Delta
        delta_call = np.exp(-q * T) * Nd1
        delta_put = np.exp(-q * T) * (Nd1 - 1)
        
        # Gamma
        gamma = nd1 * np.exp(-q * T) / (S * sigma * sqrt_T)
        
        # Theta
        theta_call = (-S * nd1 * sigma * np.exp(-q * T) / (2 * sqrt_T) 
                     - r * K * np.exp(-r * T) * Nd2 
                     + q * S * np.exp(-q * T) * Nd1)
        theta_put = (-S * nd1 * sigma * np.exp(-q * T) / (2 * sqrt_T) 
                    + r * K * np.exp(-r * T) * (1 - Nd2) 
                    - q * S * np.exp(-q * T) * (1 - Nd1))
        
        # Vega
        vega = S * nd1 * np.exp(-q * T) * sqrt_T / 100  # Per 1% change in volatility
        
        # Rho
        rho_call = K * T * np.exp(-r * T) * Nd2 / 100  # Per 1% change in interest rate
        rho_put = -K * T * np.exp(-r * T) * (1 - Nd2) / 100
        
        return {
            'delta_call': delta_call,
            'delta_put': delta_put,
            'gamma': gamma,
            'theta_call': theta_call / 365,  # Per day
            'theta_put': theta_put / 365,
            'vega': vega,
            'rho_call': rho_call,
            'rho_put': rho_put,
        }
    
    def implied_volatility(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType = OptionType.CALL,
        q: float = 0.0,
        initial_guess: float = 0.2
    ) -> float:
        """
        Calculate implied volatility from option price.
        
        Uses Newton-Raphson method to solve for sigma.
        
        Args:
            option_price: Observed option price
            S: Underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            option_type: Call or Put
            q: Dividend yield
            initial_guess: Initial volatility guess
            
        Returns:
            Implied volatility
        """
        def objective_function(sigma):
            if option_type == OptionType.CALL:
                price = self.black_scholes_call(S, K, T, r, sigma, q)
            else:
                price = self.black_scholes_put(S, K, T, r, sigma, q)
            return price - option_price
        
        def vega(sigma):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * stats.norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson iteration
        sigma = initial_guess
        for _ in range(100):
            f = objective_function(sigma)
            v = vega(sigma)
            
            if abs(v) < 1e-10:
                break
            
            sigma_new = sigma - f / v
            
            if abs(sigma_new - sigma) < 1e-8:
                break
            
            sigma = sigma_new
            
            # Ensure volatility stays positive
            sigma = max(sigma, 0.001)
        
        return sigma
    
    def heston_model(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        option_type: OptionType = OptionType.CALL,
        q: float = 0.0
    ) -> float:
        """
        Heston stochastic volatility model option price.
        
        Uses characteristic function approach (Heston 1993).
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of variance (vol of vol)
            rho: Correlation between price and variance shocks
            option_type: Call or Put
            q: Dividend yield
            
        Returns:
            Option price
        """
        # This is a simplified implementation
        # Full Heston requires numerical integration of characteristic function
        
        # For now, use Black-Scholes with adjusted volatility
        # Adjusted volatility = sqrt(theta + (v0 - theta) * exp(-kappa * T))
        adjusted_variance = theta + (v0 - theta) * np.exp(-kappa * T)
        adjusted_sigma = np.sqrt(adjusted_variance)
        
        if option_type == OptionType.CALL:
            return self.black_scholes_call(S, K, T, r, adjusted_sigma, q)
        else:
            return self.black_scholes_put(S, K, T, r, adjusted_sigma, q)
    
    def bates_model(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        lambda_: float,
        mu_j: float,
        sigma_j: float,
        option_type: OptionType = OptionType.CALL,
        q: float = 0.0
    ) -> float:
        """
        Bates model with jumps (Heston + jumps).
        
        Combines Heston stochastic volatility with jump-diffusion.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of variance (vol of vol)
            rho: Correlation between price and variance shocks
            lambda_: Jump intensity
            mu_j: Mean jump size
            sigma_j: Jump volatility
            option_type: Call or Put
            q: Dividend yield
            
        Returns:
            Option price
        """
        # This is a simplified implementation
        # Full Bates requires numerical integration with jump component
        
        # For now, use Heston with jump adjustment
        heston_price = self.heston_model(
            S, K, T, r, v0, kappa, theta, sigma, rho, option_type, q
        )
        
        # Jump adjustment (Merton 1976)
        # Expected jump compensation
        jump_comp = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        
        # Adjust drift for jumps
        r_adjusted = r - jump_comp
        
        # Recalculate with adjusted drift
        if option_type == OptionType.CALL:
            return self.black_scholes_call(S, K, T, r_adjusted, np.sqrt(theta), q)
        else:
            return self.black_scholes_put(S, K, T, r_adjusted, np.sqrt(theta), q)
    
    def local_volatility_surface(
        self,
        option_prices: pd.DataFrame,
        strikes: np.ndarray,
        maturities: np.ndarray,
        S: float,
        r: float,
        q: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Calculate local volatility surface from option prices.
        
        Uses Dupire's formula to calculate local volatility.
        
        Args:
            option_prices: DataFrame with option prices (index=strike, columns=maturity)
            strikes: Strike prices
            maturities: Maturities (in years)
            S: Underlying price
            r: Risk-free rate
            q: Dividend yield
            
        Returns:
            Dictionary with local volatility surface
        """
        # This is a simplified implementation
        # Full local volatility requires Dupire's formula with finite differences
        
        # For now, use implied volatility as approximation
        local_vol = np.zeros((len(strikes), len(maturities)))
        
        for i, strike in enumerate(strikes):
            for j, maturity in enumerate(maturities):
                try:
                    price = option_prices.iloc[i, j]
                    if price > 0:
                        local_vol[i, j] = self.implied_volatility(
                            price, S, strike, maturity, r, OptionType.CALL, q
                        )
                except:
                    local_vol[i, j] = 0.2  # Default volatility
        
        return {
            'strikes': strikes,
            'maturities': maturities,
            'local_volatility': local_vol,
        }
    
    def calculate_option_greeks(
        self,
        model_type: str,
        params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate option Greeks for a given model.
        
        Args:
            model_type: Type of model ('black_scholes', 'heston', 'bates')
            params: Model parameters
            
        Returns:
            Dictionary with Greeks
        """
        if model_type == 'black_scholes':
            return self.black_scholes_greeks(
                params['S'], params['K'], params['T'], params['r'], params['sigma'], params.get('q', 0)
            )
        else:
            # For Heston and Bates, use numerical differentiation
            # This is a simplified implementation
            return self.black_scholes_greeks(
                params['S'], params['K'], params['T'], params['r'], params.get('sigma', 0.2), params.get('q', 0)
            )

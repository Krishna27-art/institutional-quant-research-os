"""
Stochastic Calculus for Pricing

Based on Comprehensive Upgrade Analysis - Tier 3 Upgrade (#28)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Ito's Lemma for stochastic processes
- Geometric Brownian Motion (GBM)
- Monte Carlo simulation for pricing
- Stochastic differential equations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class StochasticProcessConfig:
    """Configuration for Stochastic Processes"""
    # GBM parameters
    mu: float = 0.08  # Drift
    sigma: float = 0.2  # Volatility
    S0: float = 100.0  # Initial price
    
    # Simulation parameters
    T: float = 1.0  # Time horizon (years)
    n_steps: int = 252  # Number of time steps
    n_paths: int = 10000  # Number of Monte Carlo paths
    
    # Numerical methods
    method: str = "euler"  # "euler", "milstein"
    antithetic: bool = True  # Use antithetic variates
    control_variate: bool = False  # Use control variates


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM)
    
    dS = mu * S * dt + sigma * S * dW
    """
    
    def __init__(self, config: StochasticProcessConfig):
        self.config = config
    
    def simulate(self) -> np.ndarray:
        """
        Simulate GBM paths using Euler method
        
        Returns:
            Array of shape (n_paths, n_steps + 1)
        """
        dt = self.config.T / self.config.n_steps
        n_paths = self.config.n_paths
        n_steps = self.config.n_steps
        
        if self.config.antithetic:
            n_paths = n_paths // 2
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.config.S0
        
        # Simulate
        for t in range(n_steps):
            # Random increments
            dW = np.random.randn(n_paths) * np.sqrt(dt)
            
            if self.config.method == "euler":
                # Euler-Maruyama
                paths[:, t + 1] = paths[:, t] * (1 + self.config.mu * dt + self.config.sigma * dW)
            elif self.config.method == "milstein":
                # Milstein scheme (higher order)
                paths[:, t + 1] = paths[:, t] * (1 + self.config.mu * dt + self.config.sigma * dW + 
                                                0.5 * self.config.sigma ** 2 * (dW ** 2 - dt))
        
        # Antithetic variates
        if self.config.antithetic:
            antithetic_paths = np.zeros((n_paths, n_steps + 1))
            antithetic_paths[:, 0] = self.config.S0
            
            for t in range(n_steps):
                dW = np.random.randn(n_paths) * np.sqrt(dt)
                antithetic_paths[:, t + 1] = antithetic_paths[:, t] * (1 + self.config.mu * dt - self.config.sigma * dW)
            
            paths = np.vstack([paths, antithetic_paths])
        
        return paths
    
    def calculate_option_price(self, K: float, T: float, r: float, option_type: str = "call") -> float:
        """
        Calculate option price using Monte Carlo
        
        Args:
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        # Update config for this option
        self.config.T = T
        paths = self.simulate()
        
        # Calculate payoffs
        S_T = paths[:, -1]
        
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount
        price = np.exp(-r * T) * payoffs.mean()
        
        return price


class ItoProcess:
    """
    General Ito Process
    
    dX = mu(X, t) * dt + sigma(X, t) * dW
    """
    
    def __init__(self, mu_func: Callable, sigma_func: Callable, config: StochasticProcessConfig):
        self.mu_func = mu_func
        self.sigma_func = sigma_func
        self.config = config
    
    def simulate(self, X0: float) -> np.ndarray:
        """
        Simulate Ito process
        
        Args:
            X0: Initial value
            
        Returns:
            Array of shape (n_paths, n_steps + 1)
        """
        dt = self.config.T / self.config.n_steps
        n_paths = self.config.n_paths
        n_steps = self.config.n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = X0
        
        for t in range(n_steps):
            current_time = t * dt
            X = paths[:, t]
            
            # Calculate drift and diffusion
            mu = self.mu_func(X, current_time)
            sigma = self.sigma_func(X, current_time)
            
            # Random increments
            dW = np.random.randn(n_paths) * np.sqrt(dt)
            
            # Euler-Maruyama
            paths[:, t + 1] = X + mu * dt + sigma * dW
        
        return paths


class StochasticCalculus:
    """
    Stochastic Calculus Tools
    
    Implements Ito's Lemma and related concepts.
    """
    
    @staticmethod
    def itos_lemma(f: Callable, X: np.ndarray, mu: float, sigma: float, dt: float) -> np.ndarray:
        """
        Apply Ito's Lemma: df = f'(X) * dX + 0.5 * f''(X) * sigma^2 * dt
        
        Args:
            f: Function to apply
            X: Current value
            mu: Drift
            sigma: Volatility
            dt: Time step
            
        Returns:
            Change in f(X)
        """
        # Numerical derivatives
        epsilon = 1e-6
        f_prime = (f(X + epsilon) - f(X - epsilon)) / (2 * epsilon)
        f_double_prime = (f(X + epsilon) - 2 * f(X) + f(X - epsilon)) / (epsilon ** 2)
        
        # Ito's Lemma
        dX = mu * X * dt + sigma * X * np.random.randn() * np.sqrt(dt)
        df = f_prime * dX + 0.5 * f_double_prime * (sigma * X) ** 2 * dt
        
        return df
    
    @staticmethod
    def girsanov_theorem(mu_P: float, mu_Q: float, sigma: float, T: float, n_steps: int) -> np.ndarray:
        """
        Change of measure using Girsanov's theorem
        
        Args:
            mu_P: Drift under P (physical measure)
            mu_Q: Drift under Q (risk-neutral measure)
            sigma: Volatility
            T: Time horizon
            n_steps: Number of steps
            
        Returns:
            Radon-Nikodym derivative path
        """
        dt = T / n_steps
        dW = np.random.randn(n_steps) * np.sqrt(dt)
        
        # Calculate Radon-Nikodym derivative
        # dZ/dW = (mu_P - mu_Q) / sigma
        lambda_val = (mu_P - mu_Q) / sigma
        
        # Z_T = exp(-0.5 * lambda^2 * T + lambda * W_T)
        W_T = np.sum(dW)
        Z_T = np.exp(-0.5 * lambda_val ** 2 * T + lambda_val * W_T)
        
        return Z_T
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict:
        """
        Calculate option Greeks using Black-Scholes
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            
        Returns:
            Dictionary with Greeks
        """
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)
        else:
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega
        }


class MonteCarloPricer:
    """
    Monte Carlo Option Pricer
    
    Uses Monte Carlo simulation to price derivatives.
    """
    
    def __init__(self, config: StochasticProcessConfig):
        self.config = config
        self.gbm = GeometricBrownianMotion(config)
    
    def price_european_option(self, K: float, T: float, r: float, option_type: str = "call") -> Dict:
        """
        Price European option using Monte Carlo
        
        Args:
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            option_type: "call" or "put"
            
        Returns:
            Dictionary with price and confidence interval
        """
        paths = self.gbm.simulate()
        S_T = paths[:, -1]
        
        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate price and confidence interval
        price = discounted_payoffs.mean()
        std_error = discounted_payoffs.std() / np.sqrt(len(discounted_payoffs))
        confidence_interval = 1.96 * std_error
        
        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (price - confidence_interval, price + confidence_interval)
        }
    
    def price_asian_option(self, K: float, T: float, r: float, option_type: str = "call") -> float:
        """
        Price Asian option (arithmetic average)
        
        Args:
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        paths = self.gbm.simulate()
        
        # Calculate arithmetic average
        avg_prices = paths[:, 1:].mean(axis=1)
        
        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
        
        # Discount
        price = np.exp(-r * T) * payoffs.mean()
        
        return price
    
    def price_barrier_option(self, K: float, B: float, T: float, r: float, 
                           barrier_type: str = "up-and-out", option_type: str = "call") -> float:
        """
        Price barrier option
        
        Args:
            K: Strike price
            B: Barrier level
            T: Time to expiry
            r: Risk-free rate
            barrier_type: "up-and-out", "down-and-out", "up-and-in", "down-and-in"
            option_type: "call" or "put"
            
        Returns:
            Option price
        """
        paths = self.gbm.simulate()
        
        # Check barrier condition
        if barrier_type == "up-and-out":
            barrier_hit = (paths.max(axis=1) >= B)
            active_paths = ~barrier_hit
        elif barrier_type == "down-and-out":
            barrier_hit = (paths.min(axis=1) <= B)
            active_paths = ~barrier_hit
        elif barrier_type == "up-and-in":
            barrier_hit = (paths.max(axis=1) >= B)
            active_paths = barrier_hit
        else:  # down-and-in
            barrier_hit = (paths.min(axis=1) <= B)
            active_paths = barrier_hit
        
        # Calculate payoffs for active paths
        S_T = paths[active_paths, -1]
        
        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount (payoff is 0 for inactive paths)
        all_payoffs = np.zeros(len(paths))
        all_payoffs[active_paths] = payoffs
        
        price = np.exp(-r * T) * all_payoffs.mean()
        
        return price


def simulate_gbm_paths(config: StochasticProcessConfig) -> np.ndarray:
    """Simulate GBM paths for visualization"""
    gbm = GeometricBrownianMotion(config)
    return gbm.simulate()


if __name__ == "__main__":
    # Example usage
    config = StochasticProcessConfig(
        mu=0.08,
        sigma=0.2,
        S0=100.0,
        T=1.0,
        n_steps=252,
        n_paths=10000,
        antithetic=True
    )
    
    # Simulate GBM
    print("Simulating GBM paths...")
    gbm = GeometricBrownianMotion(config)
    paths = gbm.simulate()
    
    print(f"\nGBM Simulation Results:")
    print(f"  Number of paths: {len(paths)}")
    print(f"  Number of steps: {config.n_steps}")
    print(f"  Final price mean: {paths[:, -1].mean():.2f}")
    print(f"  Final price std: {paths[:, -1].std():.2f}")
    print(f"  Final price min: {paths[:, -1].min():.2f}")
    print(f"  Final price max: {paths[:, -1].max():.2f}")
    
    # Price European option
    print("\nPricing European option...")
    pricer = MonteCarloPricer(config)
    result = pricer.price_european_option(K=100, T=1.0, r=0.05, option_type="call")
    
    print(f"\nEuropean Call Option:")
    print(f"  Price: {result['price']:.4f}")
    print(f"  Std Error: {result['std_error']:.4f}")
    print(f"  95% CI: ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
    
    # Price Asian option
    print("\nPricing Asian option...")
    asian_price = pricer.price_asian_option(K=100, T=1.0, r=0.05, option_type="call")
    print(f"  Asian Call Price: {asian_price:.4f}")
    
    # Price barrier option
    print("\nPricing barrier option...")
    barrier_price = pricer.price_barrier_option(K=100, B=120, T=1.0, r=0.05, 
                                                barrier_type="up-and-out", option_type="call")
    print(f"  Up-and-Out Call Price: {barrier_price:.4f}")
    
    # Calculate Greeks
    print("\nCalculating Greeks...")
    greeks = StochasticCalculus.calculate_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    for key, value in greeks.items():
        print(f"  {key}: {value:.4f}")
    
    # Ito's Lemma example
    print("\nApplying Ito's Lemma...")
    def f(x):
        return x ** 2  # f(S) = S^2
    
    dS = 100.0
    df = StochasticCalculus.itos_lemma(f, np.array([dS]), 0.08, 0.2, 1/252)
    print(f"  Change in S^2: {df[0]:.4f}")

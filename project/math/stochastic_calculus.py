"""
Stochastic Calculus Library

Implements fundamental stochastic calculus models and methods for
quantitative finance, including Ito's Lemma, Geometric Brownian Motion,
Ornstein-Uhlenbeck process, and Monte Carlo simulation.

Key Features:
- Ito's Lemma for stochastic differential equations
- Geometric Brownian Motion (GBM) for asset price modeling
- Ornstein-Uhlenbeck (OU) for mean-reverting processes
- Monte Carlo simulation with variance reduction
- Option pricing using stochastic models
- Numerical methods for SDEs (Euler-Maruyama, Milstein)

Based on Blueprint Week 3-4: Mathematical & Statistical Toolkit
References:
- Ito (1944) - Stochastic Integrals
- Black-Scholes (1973) - Option Pricing
- Vasicek (1977) - Interest Rate Modeling
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple, List
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class ItoLemma:
    """
    Ito's Lemma for stochastic differential equations.
    
    Ito's Lemma provides the differential of a function of a stochastic process.
    For a process dX = μ dt + σ dW, and function f(X,t):
    df = (∂f/∂t + μ ∂f/∂X + 0.5 σ² ∂²f/∂X²) dt + σ ∂f/∂X dW
    """
    
    @staticmethod
    def apply(
        f: Callable,
        x: float,
        t: float,
        mu: float,
        sigma: float,
        dt: float = 0.01
    ) -> Tuple[float, float]:
        """
        Apply Ito's Lemma to function f(X,t).
        
        Args:
            f: Function of X and t
            x: Current value of X
            t: Current time
            mu: Drift coefficient
            sigma: Diffusion coefficient
            dt: Time step
            
        Returns:
            Tuple of (drift, diffusion) for df
        """
        # Numerical derivatives
        h = 1e-5
        
        # ∂f/∂t
        df_dt = (f(x, t + h) - f(x, t - h)) / (2 * h)
        
        # ∂f/∂X
        df_dx = (f(x + h, t) - f(x - h, t)) / (2 * h)
        
        # ∂²f/∂X²
        d2f_dx2 = (f(x + h, t) - 2 * f(x, t) + f(x - h, t)) / (h ** 2)
        
        # Ito's Lemma
        drift = df_dt + mu * df_dx + 0.5 * sigma ** 2 * d2f_dx2
        diffusion = sigma * df_dx
        
        return drift, diffusion


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM) model.
    
    The GBM is defined by the SDE:
    dS = μ S dt + σ S dW
    
    Solution: S_t = S_0 exp((μ - 0.5σ²)t + σ W_t)
    
    Used for modeling asset prices in the Black-Scholes framework.
    """
    
    def __init__(self, mu: float, sigma: float, S0: float = 100.0):
        """
        Initialize GBM model.
        
        Args:
            mu: Drift (expected return)
            sigma: Volatility
            S0: Initial price
        """
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
    
    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        method: str = 'euler'
    ) -> np.ndarray:
        """
        Simulate GBM paths.
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            method: Numerical method ('euler' or 'exact')
            
        Returns:
            Array of simulated paths [n_paths, n_steps+1]
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        if method == 'exact':
            # Exact solution
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, i] = paths[:, 0] * np.exp(
                    (self.mu - 0.5 * self.sigma ** 2) * i * dt +
                    self.sigma * np.sum(dW[:i], axis=0)
                )
        else:
            # Euler-Maruyama method
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, i] = paths[:, i-1] * (1 + self.mu * dt + self.sigma * dW)
        
        return paths
    
    def analytical_mean(self, t: float) -> float:
        """
        Calculate analytical mean at time t.
        
        Args:
            t: Time
            
        Returns:
            Expected value E[S_t]
        """
        return self.S0 * np.exp(self.mu * t)
    
    def analytical_variance(self, t: float) -> float:
        """
        Calculate analytical variance at time t.
        
        Args:
            t: Time
            
        Returns:
            Variance Var[S_t]
        """
        return self.S0 ** 2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma ** 2 * t) - 1)
    
    def option_price(
        self,
        K: float,
        T: float,
        option_type: str = 'call',
        r: float = 0.05
    ) -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            r: Risk-free rate
            
        Returns:
            Option price
        """
        d1 = (np.log(self.S0 / K) + (r + 0.5 * self.sigma ** 2) * T) / (self.sigma * np.sqrt(T))
        d2 = d1 - self.sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = self.S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        
        return price


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck (OU) process.
    
    The OU process is defined by the SDE:
    dX = θ(μ - X) dt + σ dW
    
    Solution: X_t = μ + (X_0 - μ) exp(-θ t) + σ ∫_0^t exp(-θ(t-s)) dW_s
    
    Used for modeling mean-reverting processes like interest rates,
    commodity prices, and volatility.
    """
    
    def __init__(self, theta: float, mu: float, sigma: float, X0: float = 0.0):
        """
        Initialize OU process.
        
        Args:
            theta: Mean reversion speed (θ > 0)
            mu: Long-term mean
            sigma: Volatility
            X0: Initial value
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X0 = X0
    
    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        method: str = 'euler'
    ) -> np.ndarray:
        """
        Simulate OU process paths.
        
        Args:
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            method: Numerical method ('euler' or 'exact')
            
        Returns:
            Array of simulated paths [n_paths, n_steps+1]
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.X0
        
        if method == 'exact':
            # Exact solution
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                # Cumulative integral
                integral = 0.0
                for j in range(i):
                    integral += np.exp(-self.theta * (i - j) * dt) * dW[j]
                
                paths[:, i] = self.mu + (self.X0 - self.mu) * np.exp(-self.theta * i * dt) + \
                             self.sigma * integral
        else:
            # Euler-Maruyama method
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, i] = paths[:, i-1] + self.theta * (self.mu - paths[:, i-1]) * dt + \
                             self.sigma * dW
        
        return paths
    
    def analytical_mean(self, t: float) -> float:
        """
        Calculate analytical mean at time t.
        
        Args:
            t: Time
            
        Returns:
            Expected value E[X_t]
        """
        return self.mu + (self.X0 - self.mu) * np.exp(-self.theta * t)
    
    def analytical_variance(self, t: float) -> float:
        """
        Calculate analytical variance at time t.
        
        Args:
            t: Time
            
        Returns:
            Variance Var[X_t]
        """
        return self.sigma ** 2 / (2 * self.theta) * (1 - np.exp(-2 * self.theta * t))
    
    def half_life(self) -> float:
        """
        Calculate half-life of mean reversion.
        
        Returns:
            Time to revert half the distance to mean
        """
        return np.log(2) / self.theta


class MonteCarloSimulator:
    """
    Monte Carlo simulator for stochastic processes.
    
    Provides variance reduction techniques and efficient simulation
    for complex derivatives and risk metrics.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_gbm(
        self,
        mu: float,
        sigma: float,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate GBM with optional antithetic variates.
        
        Args:
            mu: Drift
            sigma: Volatility
            S0: Initial price
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            antithetic: Use antithetic variates for variance reduction
            
        Returns:
            Simulated paths
        """
        gbm = GeometricBrownianMotion(mu, sigma, S0)
        
        if antithetic:
            # Simulate half the paths and use antithetic
            paths = gbm.simulate(T, n_steps, n_paths // 2)
            antithetic_paths = gbm.simulate(T, n_steps, n_paths // 2)
            # Flip the sign of Brownian increments
            antithetic_paths[:, 1:] = S0 + (S0 - antithetic_paths[:, 1:])
            paths = np.vstack([paths, antithetic_paths])
        else:
            paths = gbm.simulate(T, n_steps, n_paths)
        
        return paths
    
    def asian_option_price(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        option_type: str = 'call'
    ) -> float:
        """
        Price Asian option using Monte Carlo.
        
        Args:
            S0: Initial price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        gbm = GeometricBrownianMotion(r, sigma, S0)
        paths = gbm.simulate(T, n_steps, n_paths)
        
        # Calculate arithmetic average
        averages = np.mean(paths[:, 1:], axis=1)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(averages - K, 0)
        else:
            payoffs = np.maximum(K - averages, 0)
        
        # Discount to present
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price
    
    def barrier_option_price(
        self,
        S0: float,
        K: float,
        B: float,
        r: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int,
        barrier_type: str = 'up-and-out',
        option_type: str = 'call'
    ) -> float:
        """
        Price barrier option using Monte Carlo.
        
        Args:
            S0: Initial price
            K: Strike price
            B: Barrier level
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            barrier_type: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        gbm = GeometricBrownianMotion(r, sigma, S0)
        paths = gbm.simulate(T, n_steps, n_paths)
        
        # Check barrier condition
        if barrier_type == 'up-and-out':
            barrier_hit = np.any(paths >= B, axis=1)
            active = ~barrier_hit
        elif barrier_type == 'down-and-out':
            barrier_hit = np.any(paths <= B, axis=1)
            active = ~barrier_hit
        elif barrier_type == 'up-and-in':
            barrier_hit = np.any(paths >= B, axis=1)
            active = barrier_hit
        elif barrier_type == 'down-and-in':
            barrier_hit = np.any(paths <= B, axis=1)
            active = barrier_hit
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")
        
        # Calculate payoffs for active paths
        final_prices = paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        payoffs = payoffs * active
        
        # Discount to present
        price = np.exp(-r * T) * np.mean(payoffs)
        
        return price


class NumericalSDE:
    """
    Numerical methods for solving stochastic differential equations.
    
    Implements Euler-Maruyama and Milstein schemes for SDEs.
    """
    
    @staticmethod
    def euler_maruyama(
        drift: Callable,
        diffusion: Callable,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> np.ndarray:
        """
        Euler-Maruyama scheme for SDEs.
        
        Args:
            drift: Drift function μ(X,t)
            diffusion: Diffusion function σ(X,t)
            x0: Initial value
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            
        Returns:
            Simulated paths
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, i] = paths[:, i-1] + drift(paths[:, i-1], (i-1) * dt) * dt + \
                         diffusion(paths[:, i-1], (i-1) * dt) * dW
        
        return paths
    
    @staticmethod
    def milstein(
        drift: Callable,
        diffusion: Callable,
        diffusion_derivative: Callable,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> np.ndarray:
        """
        Milstein scheme for SDEs (higher order).
        
        Args:
            drift: Drift function μ(X,t)
            diffusion: Diffusion function σ(X,t)
            diffusion_derivative: Derivative of diffusion ∂σ/∂X
            x0: Initial value
            T: Time horizon
            n_steps: Number of steps
            n_paths: Number of paths
            
        Returns:
            Simulated paths
        """
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            x_prev = paths[:, i-1]
            t_prev = (i-1) * dt
            
            paths[:, i] = x_prev + drift(x_prev, t_prev) * dt + \
                         diffusion(x_prev, t_prev) * dW + \
                         0.5 * diffusion(x_prev, t_prev) * \
                         diffusion_derivative(x_prev, t_prev) * (dW ** 2 - dt)
        
        return paths


if __name__ == "__main__":
    # Test stochastic calculus library
    print("Testing Stochastic Calculus Library...")
    
    # Test Ito's Lemma
    print("\nTesting Ito's Lemma...")
    ito = ItoLemma()
    f = lambda x, t: x ** 2  # f(X) = X^2
    drift, diffusion = ito.apply(f, x=1.0, t=0.0, mu=0.1, sigma=0.2)
    print(f"Drift: {drift:.4f}, Diffusion: {diffusion:.4f}")
    
    # Test GBM
    print("\nTesting Geometric Brownian Motion...")
    gbm = GeometricBrownianMotion(mu=0.1, sigma=0.2, S0=100.0)
    paths = gbm.simulate(T=1.0, n_steps=252, n_paths=10)
    print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} steps")
    print(f"Analytical mean at T=1: {gbm.analytical_mean(1.0):.2f}")
    print(f"Simulated mean at T=1: {paths[:, -1].mean():.2f}")
    
    # Test option pricing
    option_price = gbm.option_price(K=100, T=1.0, option_type='call', r=0.05)
    print(f"Call option price: {option_price:.2f}")
    
    # Test OU process
    print("\nTesting Ornstein-Uhlenbeck Process...")
    ou = OrnsteinUhlenbeck(theta=0.5, mu=0.0, sigma=0.1, X0=1.0)
    paths = ou.simulate(T=1.0, n_steps=252, n_paths=10)
    print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} steps")
    print(f"Analytical mean at T=1: {ou.analytical_mean(1.0):.4f}")
    print(f"Simulated mean at T=1: {paths[:, -1].mean():.4f}")
    print(f"Half-life: {ou.half_life():.2f}")
    
    # Test Monte Carlo
    print("\nTesting Monte Carlo Simulator...")
    mc = MonteCarloSimulator(seed=42)
    asian_price = mc.asian_option_price(
        S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
        n_steps=252, n_paths=10000, option_type='call'
    )
    print(f"Asian call option price: {asian_price:.2f}")
    
    barrier_price = mc.barrier_option_price(
        S0=100, K=100, B=120, r=0.05, sigma=0.2, T=1.0,
        n_steps=252, n_paths=10000, barrier_type='up-and-out', option_type='call'
    )
    print(f"Up-and-out barrier call option price: {barrier_price:.2f}")
    
    # Test numerical SDE methods
    print("\nTesting Numerical SDE Methods...")
    drift = lambda x, t: 0.1 * x
    diffusion = lambda x, t: 0.2 * x
    diffusion_derivative = lambda x, t: 0.2
    
    euler_paths = NumericalSDE.euler_maruyama(
        drift, diffusion, x0=100.0, T=1.0, n_steps=252, n_paths=10
    )
    print(f"Euler-Maruyama: {euler_paths.shape}")
    
    milstein_paths = NumericalSDE.milstein(
        drift, diffusion, diffusion_derivative, x0=100.0, T=1.0, n_steps=252, n_paths=10
    )
    print(f"Milstein: {milstein_paths.shape}")
    
    print("\nStochastic Calculus Library test completed.")

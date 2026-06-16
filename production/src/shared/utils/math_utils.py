"""
Math Utilities - Online algorithms for streaming statistics and stochastic calculus models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable, Tuple, List
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class WelfordOnline:
    """Welford's online algorithm for computing mean and variance in O(1)"""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from mean
    
    def update(self, value: float) -> None:
        """Update with new value in O(1)"""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def get_mean(self) -> float:
        """Get current mean"""
        return self.mean
    
    def get_variance(self) -> float:
        """Get current variance"""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)
    
    def get_std(self) -> float:
        """Get current standard deviation"""
        return np.sqrt(self.get_variance())
    
    def reset(self) -> None:
        """Reset statistics"""
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0


def welford_online(values: np.ndarray) -> tuple:
    """
    Compute mean and std using Welford's online algorithm
    
    Args:
        values: Array of values
        
    Returns:
        (mean, std)
    """
    algo = WelfordOnline()
    for v in values:
        algo.update(v)
    return algo.get_mean(), algo.get_std()


def exponential_moving_average(values: np.ndarray, span: int) -> np.ndarray:
    """
    Compute exponential moving average
    
    Args:
        values: Array of values
        span: Span parameter (similar to pandas)
        
    Returns:
        EMA array
    """
    alpha = 2 / (span + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    
    return ema


class ItoLemma:
    """
    Ito's Lemma for stochastic differential equations.
    
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
        """Apply Ito's Lemma to function f(X,t)."""
        h = 1e-5
        df_dt = (f(x, t + h) - f(x, t - h)) / (2 * h)
        df_dx = (f(x + h, t) - f(x - h, t)) / (2 * h)
        d2f_dx2 = (f(x + h, t) - 2 * f(x, t) + f(x - h, t)) / (h ** 2)
        drift = df_dt + mu * df_dx + 0.5 * sigma ** 2 * d2f_dx2
        diffusion = sigma * df_dx
        return drift, diffusion


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion (GBM) model.
    dS = μ S dt + σ S dW
    """
    
    def __init__(self, mu: float, sigma: float, S0: float = 100.0):
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
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        if method == 'exact':
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, i] = paths[:, 0] * np.exp(
                    (self.mu - 0.5 * self.sigma ** 2) * i * dt +
                    self.sigma * np.sum(dW[:i], axis=0)
                )
        else:
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, i] = paths[:, i-1] * (1 + self.mu * dt + self.sigma * dW)
        
        return paths
    
    def analytical_mean(self, t: float) -> float:
        return self.S0 * np.exp(self.mu * t)
    
    def analytical_variance(self, t: float) -> float:
        return self.S0 ** 2 * np.exp(2 * self.mu * t) * (np.exp(self.sigma ** 2 * t) - 1)
    
    def option_price(
        self,
        K: float,
        T: float,
        option_type: str = 'call',
        r: float = 0.05
    ) -> float:
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
    dX = θ(μ - X) dt + σ dW
    """
    
    def __init__(self, theta: float, mu: float, sigma: float, X0: float = 0.0):
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
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.X0
        
        if method == 'exact':
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                integral = 0.0
                for j in range(i):
                    integral += np.exp(-self.theta * (i - j) * dt) * dW[j]
                paths[:, i] = self.mu + (self.X0 - self.mu) * np.exp(-self.theta * i * dt) + \
                             self.sigma * integral
        else:
            for i in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt), n_paths)
                paths[:, i] = paths[:, i-1] + self.theta * (self.mu - paths[:, i-1]) * dt + \
                             self.sigma * dW
        
        return paths
    
    def analytical_mean(self, t: float) -> float:
        return self.mu + (self.X0 - self.mu) * np.exp(-self.theta * t)
    
    def analytical_variance(self, t: float) -> float:
        return self.sigma ** 2 / (2 * self.theta) * (1 - np.exp(-2 * self.theta * t))
    
    def half_life(self) -> float:
        return np.log(2) / self.theta


class MonteCarloSimulator:
    """
    Monte Carlo simulator for stochastic processes.
    """
    
    def __init__(self, seed: Optional[int] = None):
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
        gbm = GeometricBrownianMotion(mu, sigma, S0)
        if antithetic:
            paths = gbm.simulate(T, n_steps, n_paths // 2)
            antithetic_paths = gbm.simulate(T, n_steps, n_paths // 2)
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
        gbm = GeometricBrownianMotion(r, sigma, S0)
        paths = gbm.simulate(T, n_steps, n_paths)
        averages = np.mean(paths[:, 1:], axis=1)
        if option_type == 'call':
            payoffs = np.maximum(averages - K, 0)
        else:
            payoffs = np.maximum(K - averages, 0)
        return float(np.exp(-r * T) * np.mean(payoffs))
    
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
        gbm = GeometricBrownianMotion(r, sigma, S0)
        paths = gbm.simulate(T, n_steps, n_paths)
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
        final_prices = paths[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        payoffs = payoffs * active
        return float(np.exp(-r * T) * np.mean(payoffs))


class NumericalSDE:
    """
    Numerical methods for solving stochastic differential equations.
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

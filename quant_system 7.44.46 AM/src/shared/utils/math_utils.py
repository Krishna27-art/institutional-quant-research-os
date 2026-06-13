"""
Math Utilities - Online algorithms for streaming statistics
"""

import numpy as np
from typing import Optional


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

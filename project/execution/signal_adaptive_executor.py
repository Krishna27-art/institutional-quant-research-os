"""
Signal-Adaptive Quoting Executor (Yu 2026)
Implements closed-form solution for optimal market making with inventory risk.

Solves the triangular ODE system using divided differences.
Complexity O(Q_max^2) per update, where Q_max ~ 100.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimalQuote:
    """Optimal quote parameters"""
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    inventory: int
    time_left: float


class OptimalQuoteExecutor:
    """
    Signal-Adaptive Quoting Executor (Yu 2026)
    
    Implements the closed-form solution from Yu (2026) for optimal
    market making with inventory risk and signal-adaptive quoting.
    
    Solves the triangular ODE system using divided differences.
    Complexity O(Q_max^2) per update, where Q_max ~ 100.
    """
    
    def __init__(
        self,
        kappa: float = 0.1,
        a: float = 0.001,
        b: float = 0.001,
        gamma: float = 0.1,
        sigma: float = 0.02,
        lambda_intensity: float = 10.0,
        q_max: int = 100
    ):
        """
        Args:
            kappa: Intensity of Poisson process for order arrivals
            a: Base spread parameter
            b: Inventory aversion parameter
            gamma: Risk aversion parameter
            sigma: Volatility
            lambda_intensity: Order arrival intensity
            q_max: Maximum inventory level
        """
        self.kappa = kappa
        self.a = a
        self.b = b
        self.gamma = gamma
        self.sigma = sigma
        self.lam = lambda_intensity
        self.q_max = q_max
        
    def optimal_delta(
        self,
        inventory: int,
        signal: float,
        time_left: float
    ) -> float:
        """
        Calculate optimal bid-ask spread delta.
        
        Solves for w(t,q) recursively using backward induction.
        
        Args:
            inventory: Current inventory
            signal: Market signal (positive = bullish)
            time_left: Time until market close
            
        Returns:
            Optimal delta (half-spread)
        """
        # Solve for w(t,q) recursively
        q = min(abs(int(inventory)), self.q_max)
        w = self._solve_ode(q, max(time_left, 1e-6), signal=signal)
        
        # Calculate delta using closed-form solution
        delta = (1 / (self.b * self.gamma)) * np.log(
            (self.kappa + self.b * self.gamma) / self.kappa
        )
        delta += self.a / self.b
        
        # Add inventory adjustment
        if q > 0:
            ratio = w[q] / max(w[q - 1], 1e-12)
            inventory_term = (1 / self.kappa) * np.log(max(ratio, 1e-12))
            delta += np.sign(inventory) * inventory_term
        
        # Add signal adjustment
        delta -= signal / self.kappa
        
        return float(np.clip(delta, 0.0001, 0.05))
    
    def _solve_ode(self, q_max: int, T: float, signal: float = 0.0) -> np.ndarray:
        """
        Solve the triangular ODE system using backward induction.
        
        Args:
            q_max: Maximum inventory level
            T: Time horizon
            
        Returns:
            w(t,q) array
        """
        n_steps = max(1, int(T * 100))
        dt = T / n_steps
        w = np.ones((n_steps + 1, q_max + 1))
        C = self.lam * (
            self.kappa / (self.kappa + self.b * self.gamma)
        )**(self.kappa / self.gamma + 1) * np.exp(-self.kappa * self.a / self.b)
        
        for t in reversed(range(n_steps)):
            for q in range(1, q_max + 1):
                A = (self.kappa / self.b) * (
                    signal * q - 0.5 * self.sigma**2 * self.gamma * q**2
                )
                w[t, q] = w[t + 1, q] + dt * (A * w[t + 1, q] + C * w[t + 1, q - 1])
                w[t, q] = max(w[t, q], 1e-12)
        
        return w[0]
    
    def get_optimal_quotes(
        self,
        mid_price: float,
        inventory: int,
        signal: float,
        time_left: float,
        base_size: int = 100
    ) -> OptimalQuote:
        """
        Get optimal bid and ask quotes.
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory
            signal: Market signal
            time_left: Time until market close
            base_size: Base order size
            
        Returns:
            OptimalQuote with bid/ask prices and sizes
        """
        delta = self.optimal_delta(inventory, signal, time_left)
        
        # Calculate bid and ask prices
        bid_price = mid_price - delta
        ask_price = mid_price + delta
        
        # Adjust sizes based on inventory
        if inventory > 0:
            # Long inventory: reduce bid size, increase ask size
            bid_size = max(1, base_size // 2)
            ask_size = base_size
        elif inventory < 0:
            # Short inventory: increase bid size, reduce ask size
            bid_size = base_size
            ask_size = max(1, base_size // 2)
        else:
            # Neutral inventory: equal sizes
            bid_size = base_size
            ask_size = base_size
        
        return OptimalQuote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            inventory=inventory,
            time_left=time_left
        )
    
    def update_parameters(
        self,
        kappa: Optional[float] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        gamma: Optional[float] = None,
        sigma: Optional[float] = None,
        lambda_intensity: Optional[float] = None
    ):
        """
        Update model parameters dynamically.
        
        Args:
            kappa: New kappa value
            a: New a value
            b: New b value
            gamma: New gamma value
            sigma: New sigma value
            lambda_intensity: New lambda value
        """
        if kappa is not None:
            self.kappa = kappa
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if gamma is not None:
            self.gamma = gamma
        if sigma is not None:
            self.sigma = sigma
        if lambda_intensity is not None:
            self.lam = lambda_intensity

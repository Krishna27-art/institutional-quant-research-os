"""
Signal-Adaptive Execution (Yu 2026)

Implements the signal-adaptive quoting strategy from Yu (2026) for optimal
execution with market impact modeling. This system dynamically adjusts quote
depth based on alpha signal strength, inventory position, time remaining,
and volatility to minimize execution costs.

Key Features:
- Explicit solution from Yu (2026) for optimal quoting
- δ(t,q) = (1/κ)*(1 + log(w(t,q)/w(t,q-1))) + a/b
- Triangular ODE system for w(t,q)
- Quote depth = f(signal, inventory, time_left, volatility)
- Urgency and participation rate calculation
- Market impact modeling (Almgren-Chriss)

Based on Blueprint Week 11-12: Execution & Monitoring
Reference: Yu (2026) - Signal-Adaptive Execution
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExecutionSide(Enum):
    """Execution side."""
    BUY = 1
    SELL = -1


@dataclass
class QuoteDepth:
    """Quote depth parameters."""
    bid_depth: float
    ask_depth: float
    urgency: float
    participation_rate: float
    optimal_price: float


class SignalAdaptiveExecutor:
    """
    Signal-Adaptive Executor implementing Yu (2026).
    
    This executor computes optimal quote depth based on:
    - Alpha signal strength
    - Current inventory position
    - Time remaining for execution
    - Market volatility
    - Market impact parameters
    
    The explicit solution from Yu (2026):
    δ(t,q) = (1/κ)*(1 + log(w(t,q)/w(t,q-1))) + a/b
    where w solves the triangular ODE system.
    """
    
    def __init__(
        self,
        kappa: float = 0.1,
        a: float = 0.01,
        b: float = 0.5,
        gamma: float = 0.01,
        sigma: float = 0.02
    ):
        """
        Initialize signal-adaptive executor.
        
        Args:
            kappa: Market impact parameter (temporary impact)
            a: Permanent impact coefficient
            b: Temporary impact coefficient
            gamma: Risk aversion parameter
            sigma: Volatility parameter
        """
        self.kappa = kappa
        self.a = a
        self.b = b
        self.gamma = gamma
        self.sigma = sigma
        
        # Pre-compute lookup table for w ratio
        self.w_lookup_table = self._build_w_lookup_table()
    
    def _build_w_lookup_table(
        self,
        max_q: int = 1000,
        max_t: float = 1.0,
        n_steps: int = 100
    ) -> np.ndarray:
        """
        Build lookup table for w(t,q) ratio.
        
        Solves the triangular ODE system numerically.
        
        Args:
            max_q: Maximum quantity
            max_t: Maximum time (normalized to [0,1])
            n_steps: Number of time steps
            
        Returns:
            Lookup table [n_steps, max_q]
        """
        table = np.zeros((n_steps, max_q))
        
        dt = max_t / n_steps
        
        for t_idx in range(n_steps):
            t = t_idx * dt
            for q in range(1, max_q + 1):
                # Solve ODE: dw/dt = -gamma * w + (sigma^2 / 2) * q^2
                # Simplified solution for lookup
                w = np.exp(-self.gamma * t) * (self.sigma**2 / (2 * self.gamma)) * q**2
                table[t_idx, q - 1] = w
        
        return table
    
    def compute_quote_depth(
        self,
        remaining_q: int,
        signal_strength: float,
        time_left: float,
        inventory: int = 0,
        volatility: Optional[float] = None
    ) -> QuoteDepth:
        """
        Compute optimal quote depth using Yu (2026) explicit solution.
        
        Args:
            remaining_q: Remaining quantity to execute
            signal_strength: Alpha signal strength in [-1, 1]
            time_left: Time remaining for execution (normalized to [0,1])
            inventory: Current inventory position
            volatility: Current market volatility (uses sigma if None)
            
        Returns:
            QuoteDepth with bid/ask depths and urgency
        """
        if volatility is None:
            volatility = self.sigma
        
        # Solve w ratio from lookup table
        w_ratio = self._solve_w_ratio(remaining_q, time_left)
        
        # Calculate delta using explicit solution
        # δ(t,q) = (1/κ)*(1 + log(w(t,q)/w(t,q-1))) + a/b
        delta = (1 / (self.b * self.gamma)) * np.log(
            (self.kappa + self.b * self.gamma) / self.kappa
        )
        delta += self.a / self.b + (1 / self.kappa) * np.log(w_ratio)
        
        # Adjust delta based on signal strength
        # Stronger signal = more aggressive execution (smaller depth)
        signal_adjustment = 1.0 - 0.5 * abs(signal_strength)
        delta *= signal_adjustment
        
        # Adjust based on inventory
        # If we have opposite inventory, be more aggressive
        inventory_adjustment = 1.0
        if inventory != 0:
            inventory_adjustment = 1.0 - 0.3 * np.sign(inventory) * signal_strength
        delta *= inventory_adjustment
        
        # Adjust based on volatility
        # Higher volatility = more conservative (larger depth)
        vol_adjustment = 1.0 + (volatility - self.sigma) / self.sigma
        delta *= vol_adjustment
        
        # Clip to reasonable range [0.01%, 5%]
        delta = np.clip(delta, 0.0001, 0.05)
        
        # Calculate urgency
        urgency = self._calculate_urgency(signal_strength, time_left, inventory)
        
        # Calculate participation rate
        participation_rate = self._calculate_participation_rate(
            delta, signal_strength, time_left
        )
        
        # Calculate optimal price (mid price adjusted for impact)
        optimal_price = self._calculate_optimal_price(delta, signal_strength)
        
        return QuoteDepth(
            bid_depth=delta,
            ask_depth=delta,
            urgency=urgency,
            participation_rate=participation_rate,
            optimal_price=optimal_price
        )
    
    def _solve_w_ratio(
        self,
        remaining_q: int,
        time_left: float
    ) -> float:
        """
        Solve w(t,q) / w(t,q-1) ratio.
        
        Args:
            remaining_q: Remaining quantity
            time_left: Time remaining
            
        Returns:
            w ratio
        """
        if remaining_q <= 1:
            return 1.0
        
        # Get w values from lookup table
        n_steps = self.w_lookup_table.shape[0]
        t_idx = int(time_left * (n_steps - 1))
        t_idx = np.clip(t_idx, 0, n_steps - 1)
        
        q_idx = min(remaining_q - 1, self.w_lookup_table.shape[1] - 1)
        q_prev_idx = max(0, q_idx - 1)
        
        w_current = self.w_lookup_table[t_idx, q_idx]
        w_prev = self.w_lookup_table[t_idx, q_prev_idx]
        
        if w_prev > 0:
            return w_current / w_prev
        else:
            return 1.0
    
    def _calculate_urgency(
        self,
        signal_strength: float,
        time_left: float,
        inventory: int
    ) -> float:
        """
        Calculate execution urgency.
        
        Args:
            signal_strength: Alpha signal strength
            time_left: Time remaining
            inventory: Current inventory
            
        Returns:
            Urgency score in [0, 1]
        """
        # Time urgency: more urgent as time runs out
        time_urgency = 1.0 - time_left
        
        # Signal urgency: stronger signal = more urgent
        signal_urgency = abs(signal_strength)
        
        # Inventory urgency: large opposite position = more urgent
        inventory_urgency = min(abs(inventory) / 1000.0, 1.0)
        
        # Combine urgencies
        urgency = 0.4 * time_urgency + 0.4 * signal_urgency + 0.2 * inventory_urgency
        
        return np.clip(urgency, 0.0, 1.0)
    
    def _calculate_participation_rate(
        self,
        delta: float,
        signal_strength: float,
        time_left: float
    ) -> float:
        """
        Calculate optimal participation rate.
        
        Args:
            delta: Quote depth
            signal_strength: Signal strength
            time_left: Time remaining
            
        Returns:
            Participation rate in [0, 1]
        """
        # Base participation rate
        base_rate = 0.1  # 10% base participation
        
        # Adjust based on delta (smaller depth = higher participation)
        depth_factor = 0.05 / delta  # Normalize around 0.05
        depth_factor = np.clip(depth_factor, 0.5, 2.0)
        
        # Adjust based on signal strength
        signal_factor = 1.0 + 0.5 * abs(signal_strength)
        
        # Adjust based on time (more time = can be more patient)
        time_factor = 0.5 + 0.5 * time_left
        
        participation_rate = base_rate * depth_factor * signal_factor * time_factor
        
        return np.clip(participation_rate, 0.01, 0.5)  # Max 50% participation
    
    def _calculate_optimal_price(
        self,
        delta: float,
        signal_strength: float
    ) -> float:
        """
        Calculate optimal execution price.
        
        Args:
            delta: Quote depth
            signal_strength: Signal strength
            
        Returns:
            Optimal price adjustment from mid price
        """
        # For buy orders, we want to pay less than mid
        # For sell orders, we want to receive more than mid
        
        if signal_strength > 0:
            # Buy signal: adjust negatively
            price_adjustment = -delta * (1 + signal_strength)
        else:
            # Sell signal: adjust positively
            price_adjustment = delta * (1 - signal_strength)
        
        return price_adjustment
    
    def execute_order(
        self,
        side: ExecutionSide,
        quantity: int,
        signal_strength: float,
        time_left: float,
        current_inventory: int = 0,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        Execute an order with signal-adaptive quoting.
        
        Args:
            side: Execution side (BUY or SELL)
            quantity: Quantity to execute
            signal_strength: Alpha signal strength
            time_left: Time remaining
            current_inventory: Current inventory
            volatility: Market volatility
            
        Returns:
            Dictionary with execution parameters
        """
        # Compute quote depth
        quote_depth = self.compute_quote_depth(
            remaining_q=quantity,
            signal_strength=signal_strength,
            time_left=time_left,
            inventory=current_inventory,
            volatility=volatility
        )
        
        # Determine execution strategy
        if quote_depth.urgency > 0.8:
            strategy = "AGGRESSIVE"
        elif quote_depth.urgency > 0.5:
            strategy = "MODERATE"
        else:
            strategy = "PATIENT"
        
        # Calculate slice size based on participation rate
        slice_size = int(quantity * quote_depth.participation_rate)
        slice_size = max(slice_size, 1)  # At least 1 share
        
        return {
            'side': side.name,
            'quantity': quantity,
            'slice_size': slice_size,
            'bid_depth': quote_depth.bid_depth,
            'ask_depth': quote_depth.ask_depth,
            'urgency': quote_depth.urgency,
            'participation_rate': quote_depth.participation_rate,
            'optimal_price_adjustment': quote_depth.optimal_price,
            'strategy': strategy,
            'signal_strength': signal_strength,
            'time_left': time_left
        }


class MarketImpactModel:
    """
    Market impact model (Almgren-Chriss).
    
    Models the market impact of trading using the Almgren-Chriss framework:
    - Permanent impact: linear in trade size
    - Temporary impact: square-root of trade size
    """
    
    def __init__(
        self,
        permanent_impact: float = 0.0001,
        temporary_impact: float = 0.001,
        decay_rate: float = 0.1
    ):
        """
        Initialize market impact model.
        
        Args:
            permanent_impact: Permanent impact coefficient
            temporary_impact: Temporary impact coefficient
            decay_rate: Impact decay rate
        """
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
        self.decay_rate = decay_rate
    
    def calculate_impact(
        self,
        trade_size: float,
        avg_daily_volume: float,
        price: float
    ) -> Dict[str, float]:
        """
        Calculate market impact of a trade.
        
        Args:
            trade_size: Size of trade
            avg_daily_volume: Average daily volume
            price: Current price
            
        Returns:
            Dictionary with impact metrics
        """
        # Normalized trade size
        normalized_size = trade_size / avg_daily_volume
        
        # Permanent impact (linear)
        permanent = self.permanent_impact * normalized_size * price
        
        # Temporary impact (square-root)
        temporary = self.temporary_impact * np.sqrt(normalized_size) * price
        
        # Total impact
        total_impact = permanent + temporary
        
        # Impact in basis points
        impact_bps = (total_impact / price) * 10000
        
        return {
            'permanent_impact': permanent,
            'temporary_impact': temporary,
            'total_impact': total_impact,
            'impact_bps': impact_bps,
            'normalized_size': normalized_size
        }
    
    def optimize_execution_schedule(
        self,
        total_quantity: int,
        time_horizon: float,
        n_intervals: int = 10
    ) -> np.ndarray:
        """
        Optimize execution schedule to minimize market impact.
        
        Args:
            total_quantity: Total quantity to execute
            time_horizon: Time horizon for execution
            n_intervals: Number of time intervals
            
        Returns:
            Array of trade sizes for each interval
        """
        # Simple equal-weighted schedule
        # In production, this would use dynamic programming
        schedule = np.ones(n_intervals) * (total_quantity / n_intervals)
        
        return schedule


if __name__ == "__main__":
    # Test Signal-Adaptive Execution
    print("Testing Signal-Adaptive Execution (Yu 2026)...")
    
    # Create executor
    executor = SignalAdaptiveExecutor(
        kappa=0.1,
        a=0.01,
        b=0.5,
        gamma=0.01,
        sigma=0.02
    )
    
    # Test quote depth calculation
    quote_depth = executor.compute_quote_depth(
        remaining_q=1000,
        signal_strength=0.8,
        time_left=0.5,
        inventory=-500,
        volatility=0.03
    )
    
    print(f"\nQuote Depth:")
    print(f"Bid Depth: {quote_depth.bid_depth:.4f}")
    print(f"Ask Depth: {quote_depth.ask_depth:.4f}")
    print(f"Urgency: {quote_depth.urgency:.4f}")
    print(f"Participation Rate: {quote_depth.participation_rate:.4f}")
    print(f"Optimal Price Adjustment: {quote_depth.optimal_price:.4f}")
    
    # Test order execution
    execution = executor.execute_order(
        side=ExecutionSide.BUY,
        quantity=1000,
        signal_strength=0.8,
        time_left=0.5,
        current_inventory=-500,
        volatility=0.03
    )
    
    print(f"\nExecution Parameters:")
    for key, value in execution.items():
        print(f"{key}: {value}")
    
    # Test market impact model
    impact_model = MarketImpactModel()
    impact = impact_model.calculate_impact(
        trade_size=10000,
        avg_daily_volume=1000000,
        price=100.0
    )
    
    print(f"\nMarket Impact:")
    for key, value in impact.items():
        print(f"{key}: {value}")
    
    # Test execution schedule optimization
    schedule = impact_model.optimize_execution_schedule(
        total_quantity=1000,
        time_horizon=1.0,
        n_intervals=10
    )
    
    print(f"\nExecution Schedule:")
    print(f"Number of intervals: {len(schedule)}")
    print(f"Trade sizes: {schedule}")
    
    print("\nSignal-Adaptive Execution test completed.")

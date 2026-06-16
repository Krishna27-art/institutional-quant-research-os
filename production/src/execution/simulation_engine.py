"""
Capacity & Execution Simulator (Reality Simulator)
Models partial fills, latency, and queue constraints to ensure alphas can survive real capital.
"""

import numpy as np
import random
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from .fill_simulator.fill_simulator import FillSimulator
from .brokers.broker_adapter import Order, Fill, OrderSide, OrderType

@dataclass
class SimulationConfig:
    network_latency_ms: int = 25
    processing_latency_ms: int = 15
    market_impact_alpha: float = 0.1  # For square root impact model
    queue_position_penalty: float = 0.5  # Probability of partial fill on touch
    auction_spread_multiplier: float = 2.0
    random_seed: int = 42

class SimulationEngine(FillSimulator):
    """
    Advanced execution simulator that overrides the basic FillSimulator.
    Models realistic institutional trading frictions.
    """
    def __init__(self, config: SimulationConfig = None):
        super().__init__()
        self.config = config or SimulationConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        
    def simulate_fill(self, order: Order, market_state: Dict[str, Any]) -> Fill:
        """
        Simulate an order fill with latency and partial fill logic.
        """
        base_price = market_state.get('price', 1000.0)
        volume_at_price = market_state.get('volume', 10000)
        
        # 1. Latency Modeling
        # Price could drift during the latency window
        drift_volatility = market_state.get('volatility', 0.001)
        latency_seconds = (self.config.network_latency_ms + self.config.processing_latency_ms) / 1000.0
        drift = self.rng.normal(0, drift_volatility * np.sqrt(latency_seconds))
        
        effective_price = base_price * (1 + drift)
        
        # 2. Capacity & Market Impact Modeling (Square Root Law)
        # Impact = alpha * volatility * sqrt(order_size / ADV)
        adv = market_state.get('adv', 1000000)
        participation_rate = order.quantity / adv if adv > 0 else 1.0
        
        impact = self.config.market_impact_alpha * drift_volatility * np.sqrt(participation_rate)
        
        if order.side == OrderSide.BUY:
            execution_price = effective_price * (1 + impact)
        else:
            execution_price = effective_price * (1 - impact)
            
        # 3. Partial Fills (Queue Constraints)
        # If order size is large relative to current L1 volume, partial fill occurs
        fill_quantity = order.quantity
        if order.quantity > volume_at_price * 0.1: # Max 10% of L1 volume
            if self.rng.random() < self.config.queue_position_penalty:
                # Partial fill
                fill_quantity = max(1, int(volume_at_price * 0.1))
                
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=execution_price,
            timestamp=market_state.get('timestamp') or datetime.now(),
            commission=order.quantity * execution_price * 0.0001,  # 1 bps commission
            exchange_order_id=f"sim_{order.order_id}"
        )

"""
Partial Fill Model
Models partial fills based on order size, queue position, and volatility.

Critical for realistic execution simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class FillModel(Enum):
    """Fill probability models"""
    QUEUE_BASED = "queue_based"
    VOLUME_BASED = "volume_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"


@dataclass
class PartialFillResult:
    """Result of partial fill simulation"""
    order_id: str
    total_quantity: int
    filled_quantity: int
    unfilled_quantity: int
    fill_ratio: float
    fill_price: float
    avg_fill_price: float
    slippage_bps: float
    num_fills: int
    fill_times: List[datetime]


class PartialFillModel:
    """
    Partial Fill Model
    
    Simulates partial fills based on:
    - Order size relative to available volume
    - Queue position in order book
    - Market volatility
    - Time of day
    
    Fill probability decreases with:
    - Larger order size
    - Worse queue position
    - Higher volatility (more adverse selection)
    """
    
    def __init__(self, fill_model: FillModel = FillModel.QUEUE_BASED):
        self.fill_model = fill_model
        self.fill_history: List[PartialFillResult] = []
    
    def simulate_fill(self, order_id: str, order_quantity: int, order_price: float,
                    available_volume: int, queue_position: int = 0,
                    volatility: float = 0.2, time_of_day: float = 0.5) -> PartialFillResult:
        """
        Simulate partial fill for an order.
        
        Args:
            order_id: Order identifier
            order_quantity: Total order quantity
            order_price: Limit price
            available_volume: Available volume at price level
            queue_position: Position in queue (0 = front)
            volatility: Market volatility (annualized)
            time_of_day: Time of day (0 = open, 1 = close)
        
        Returns:
            PartialFillResult
        """
        # Calculate fill probability
        fill_prob = self._calculate_fill_probability(
            order_quantity, available_volume, queue_position, volatility, time_of_day
        )
        
        # Simulate fill
        filled_quantity = int(order_quantity * fill_prob)
        unfilled_quantity = order_quantity - filled_quantity
        
        # Simulate price impact (slippage)
        slippage_bps = self._calculate_slippage(
            filled_quantity, available_volume, volatility
        )
        
        # Calculate average fill price
        avg_fill_price = order_price * (1 + slippage_bps / 10000)
        
        # Simulate number of fills (partial fills happen in chunks)
        num_fills = self._simulate_num_fills(filled_quantity, queue_position)
        
        # Generate fill times
        fill_times = self._generate_fill_times(num_fills)
        
        result = PartialFillResult(
            order_id=order_id,
            total_quantity=order_quantity,
            filled_quantity=filled_quantity,
            unfilled_quantity=unfilled_quantity,
            fill_ratio=filled_quantity / order_quantity if order_quantity > 0 else 0,
            fill_price=avg_fill_price,
            avg_fill_price=avg_fill_price,
            slippage_bps=slippage_bps,
            num_fills=num_fills,
            fill_times=fill_times
        )
        
        self.fill_history.append(result)
        
        return result
    
    def _calculate_fill_probability(self, order_quantity: int, available_volume: int,
                                   queue_position: int, volatility: float,
                                   time_of_day: float) -> float:
        """Calculate probability of fill"""
        if self.fill_model == FillModel.QUEUE_BASED:
            return self._queue_based_fill(order_quantity, available_volume, queue_position)
        elif self.fill_model == FillModel.VOLUME_BASED:
            return self._volume_based_fill(order_quantity, available_volume)
        elif self.fill_model == FillModel.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_fill(order_quantity, available_volume,
                                                  queue_position, volatility, time_of_day)
        else:
            return 1.0
    
    def _queue_based_fill(self, order_quantity: int, available_volume: int,
                          queue_position: int) -> float:
        """Queue-based fill model"""
        # Base fill probability from volume ratio
        volume_ratio = min(order_quantity / (available_volume + 1), 1.0)
        base_prob = 1.0 - volume_ratio
        
        # Adjust for queue position (worse position = lower probability)
        queue_penalty = queue_position * 0.1
        queue_prob = base_prob * (1.0 - min(queue_penalty, 0.9))
        
        return max(0.0, min(1.0, queue_prob))
    
    def _volume_based_fill(self, order_quantity: int, available_volume: int) -> float:
        """Volume-based fill model"""
        # Fill probability decreases with order size
        participation_rate = order_quantity / (available_volume + 1)
        
        # Square-root model (common in market impact)
        fill_prob = 1.0 - np.sqrt(participation_rate)
        
        return max(0.0, min(1.0, fill_prob))
    
    def _volatility_adjusted_fill(self, order_quantity: int, available_volume: int,
                                 queue_position: int, volatility: float,
                                 time_of_day: float) -> float:
        """Volatility-adjusted fill model"""
        # Base fill from queue model
        base_fill = self._queue_based_fill(order_quantity, available_volume, queue_position)
        
        # Volatility penalty (higher vol = more adverse selection = lower fill)
        vol_penalty = min(volatility / 0.5, 0.5)  # Cap at 50% penalty
        vol_adjusted_fill = base_fill * (1.0 - vol_penalty)
        
        # Time of day adjustment (better fills mid-day)
        time_adjustment = 1.0 - 0.2 * abs(time_of_day - 0.5)  # Peak at 0.5
        final_fill = vol_adjusted_fill * time_adjustment
        
        return max(0.0, min(1.0, final_fill))
    
    def _calculate_slippage(self, filled_quantity: int, available_volume: int,
                          volatility: float) -> float:
        """Calculate slippage in bps"""
        if available_volume == 0:
            return 0.0
        
        participation = filled_quantity / available_volume
        
        # Square-root impact model
        impact = 0.05 * np.sqrt(participation)
        
        # Volatility adjustment
        vol_adjustment = volatility * 0.5
        
        total_slippage = (impact + vol_adjustment) * 100  # Convert to bps
        
        return total_slippage
    
    def _simulate_num_fills(self, filled_quantity: int, queue_position: int) -> int:
        """Simulate number of partial fills"""
        if filled_quantity == 0:
            return 0
        
        # More fills for larger orders and worse queue positions
        base_fills = max(1, int(filled_quantity / 100))  # 1 fill per 100 shares
        queue_penalty = queue_position * 0.1
        
        num_fills = int(base_fills * (1 + queue_penalty))
        
        return num_fills
    
    def _generate_fill_times(self, num_fills: int) -> List[datetime]:
        """Generate fill times"""
        if num_fills == 0:
            return []
        
        base_time = datetime.now()
        fill_times = []
        
        for i in range(num_fills):
            # Spaced by 100ms to 1s
            delay_ms = 100 + i * 200
            fill_time = base_time + pd.Timedelta(milliseconds=delay_ms)
            fill_times.append(fill_time)
        
        return fill_times
    
    def get_average_fill_ratio(self, n_recent: int = 100) -> Optional[float]:
        """Get average fill ratio"""
        if not self.fill_history:
            return None
        
        recent = self.fill_history[-n_recent:]
        return np.mean([r.fill_ratio for r in recent])
    
    def get_average_slippage(self, n_recent: int = 100) -> Optional[float]:
        """Get average slippage"""
        if not self.fill_history:
            return None
        
        recent = self.fill_history[-n_recent:]
        return np.mean([r.slippage_bps for r in recent])
    
    def generate_report(self) -> str:
        """Generate fill model report"""
        if not self.fill_history:
            return "No fill history available"
        
        avg_fill_ratio = self.get_average_fill_ratio()
        avg_slippage = self.get_average_slippage()
        
        report = f"""
Partial Fill Model Report
{'=' * 50}
Fill Model: {self.fill_model.value}
Total Orders: {len(self.fill_history)}
Average Fill Ratio: {avg_fill_ratio:.2%}
Average Slippage: {avg_slippage:.2f} bps

Recent Fills:
{'-' * 50}
"""
        
        for result in self.fill_history[-10:]:
            report += f"Order {result.order_id}: {result.fill_ratio:.2%} filled, "
            report += f"{result.slippage_bps:.2f} bps slippage\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    model = PartialFillModel(fill_model=FillModel.VOLATILITY_ADJUSTED)
    
    # Simulate fills
    print("Simulating partial fills...")
    for i in range(10):
        order_qty = int(np.random.exponential(1000))
        available_vol = int(np.random.exponential(5000))
        queue_pos = int(np.random.exponential(5))
        vol = np.random.uniform(0.1, 0.4)
        
        result = model.simulate_fill(
            order_id=f"order_{i}",
            order_quantity=order_qty,
            order_price=100.0,
            available_volume=available_vol,
            queue_position=queue_pos,
            volatility=vol
        )
        
        print(f"Order {i}: {result.fill_ratio:.2%} filled, {result.slippage_bps:.2f} bps slippage")
    
    print(model.generate_report())

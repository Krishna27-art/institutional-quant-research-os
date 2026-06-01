"""
VWAP + POV Hybrid Execution Engine
Based on research recommendations for Indian markets

Key findings from research:
- VWAP: Medium urgency, medium size, Medium market impact
- POV: Low-Medium market impact, Hiding large orders
- Hybrid: Best of both for Indian markets
- NSE: 2-5 lakhs per minute liquidity
- BSE: Lower liquidity than NSE

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from enum import Enum
import heapq


class Venue(Enum):
    """Trading venues for Indian markets"""
    NSE = "NSE"
    BSE = "BSE"


class OrderSide(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: int
    price: float
    timestamp: datetime
    venue: Venue


@dataclass
class ExecutionSlice:
    """Execution slice"""
    slice_id: str
    order_id: str
    quantity: int
    target_time: datetime
    venue: Venue
    max_participation_rate: float


@dataclass
class ExecutionResult:
    """Execution result"""
    order_id: str
    total_quantity: int
    filled_quantity: int
    avg_fill_price: float
    vwap_price: float
    arrival_price: float
    slippage_bps: float
    implementation_shortfall: float
    slices: List[ExecutionSlice]


class VWAPPOVExecutionEngine:
    """
    VWAP + POV Hybrid Execution Engine for Indian Markets.
    
    Architecture:
    - VWAP schedule based on historical volume profile
    - POV (Percentage of Volume) to hide large orders
    - Venue selection (NSE vs BSE)
    - Adaptive participation rate
    - Real-time market impact model
    
    Indian Market Specifics:
    - NSE: 2-5 lakhs per minute liquidity
    - BSE: 2 lakhs per minute liquidity
    - Higher transaction costs than US
    - Circuit breakers on extreme moves
    """
    
    def __init__(
        self,
        nse_liquidity: float = 500000,
        bse_liquidity: float = 200000,
        default_pov: float = 0.10
    ):
        self.nse_liquidity = nse_liquidity  # ₹ per minute
        self.bse_liquidity = bse_liquidity  # ₹ per minute
        self.default_pov = default_pov  # 10% of market volume
        
        # Historical volume profiles (typical Indian market)
        self.volume_profile = self._get_indian_volume_profile()
        
        # Active orders
        self.active_orders: Dict[str, Order] = {}
        self.execution_slices: Dict[str, List[ExecutionSlice]] = {}
    
    def _get_indian_volume_profile(self) -> Dict[int, float]:
        """
        Get typical Indian market volume profile.
        
        Returns:
            Dictionary mapping minute (0-390) to volume percentage
        """
        # Typical Indian market volume profile (390 minutes = 6.5 hours)
        profile = {}
        
        # First hour (9:15-10:15): High volume (35% of daily)
        for i in range(60):
            profile[i] = 0.005 + 0.003 * np.sin(i / 60 * np.pi)  # Peak at open
        
        # Midday (10:15-14:30): Lower volume (40% of daily)
        for i in range(60, 255):
            profile[i] = 0.002  # Steady midday volume
        
        # Last hour (14:30-15:30): High volume (25% of daily)
        for i in range(255, 390):
            profile[i] = 0.003 + 0.002 * np.sin((i - 255) / 135 * np.pi)  # Peak at close
        
        # Normalize
        total = sum(profile.values())
        profile = {k: v/total for k, v in profile.items()}
        
        return profile
    
    def select_venue(
        self,
        order: Order,
        current_volume: Dict[str, float]
    ) -> Venue:
        """
        Select optimal venue (NSE vs BSE).
        
        Args:
            order: Order to execute
            current_volume: Current volume at each venue
            
        Returns:
            Selected venue
        """
        # Calculate liquidity at each venue
        nse_liq = current_volume.get('NSE', self.nse_liquidity)
        bse_liq = current_volume.get('BSE', self.bse_liquidity)
        
        # Choose venue with higher liquidity
        if nse_liq > bse_liq:
            return Venue.NSE
        else:
            return Venue.BSE
    
    def calculate_vwap_schedule(
        self,
        order: Order,
        start_time: datetime,
        end_time: datetime
    ) -> List[Tuple[datetime, int]]:
        """
        Calculate VWAP execution schedule.
        
        Args:
            order: Order to execute
            start_time: Execution start time
            end_time: Execution end time
            
        Returns:
            List of (time, quantity) tuples
        """
        # Calculate total minutes
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        
        # Get volume profile for execution window
        schedule = []
        
        for minute in range(total_minutes):
            # Get volume percentage for this minute
            minute_of_day = (start_time + timedelta(minutes=minute)).hour * 60 + \
                           (start_time + timedelta(minutes=minute)).minute - 9*60 - 15
            
            if minute_of_day in self.volume_profile:
                vol_pct = self.volume_profile[minute_of_day]
            else:
                vol_pct = 0.002  # Default midday volume
            
            # Calculate slice quantity
            slice_qty = int(order.total_quantity * vol_pct)
            
            if slice_qty > 0:
                slice_time = start_time + timedelta(minutes=minute)
                schedule.append((slice_time, slice_qty))
        
        # Adjust for rounding errors
        scheduled_qty = sum(q for _, q in schedule)
        if scheduled_qty < order.total_quantity:
            # Add remaining to last slice
            if schedule:
                last_time, last_qty = schedule[-1]
                schedule[-1] = (last_time, last_qty + (order.total_quantity - scheduled_qty))
            else:
                schedule.append((end_time, order.total_quantity - scheduled_qty))
        
        return schedule
    
    def calculate_pov_schedule(
        self,
        order: Order,
        start_time: datetime,
        end_time: datetime,
        current_volume: Dict[str, float]
    ) -> List[Tuple[datetime, int]]:
        """
        Calculate POV (Percentage of Volume) execution schedule.
        
        Args:
            order: Order to execute
            start_time: Execution start time
            end_time: Execution end time
            current_volume: Current volume at each venue
            
        Returns:
            List of (time, quantity) tuples
        """
        # Select venue
        venue = self.select_venue(order, current_volume)
        
        # Get venue liquidity
        if venue == Venue.NSE:
            venue_liquidity = self.nse_liquidity
        else:
            venue_liquidity = self.bse_liquidity
        
        # Calculate total minutes
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        
        # Calculate slice quantity based on POV
        slice_qty_per_minute = int(venue_liquidity * self.default_pov)
        
        schedule = []
        for minute in range(total_minutes):
            slice_time = start_time + timedelta(minutes=minute)
            
            # Adjust for current volume
            vol_adjustment = current_volume.get(venue.name, venue_liquidity) / venue_liquidity
            adjusted_qty = int(slice_qty_per_minute * vol_adjustment)
            
            schedule.append((slice_time, adjusted_qty))
        
        # Adjust for rounding
        scheduled_qty = sum(q for _, q in schedule)
        if scheduled_qty < order.total_quantity:
            # Add remaining to last slice
            if schedule:
                last_time, last_qty = schedule[-1]
                schedule[-1] = (last_time, last_qty + (order.total_quantity - scheduled_qty))
            else:
                schedule.append((end_time, order.total_quantity - scheduled_qty))
        
        return schedule
    
    def execute_order(
        self,
        order: Order,
        execution_type: str = "vwap",
        duration_minutes: int = 30,
        current_volume: Optional[Dict[str, float]] = None
    ) -> ExecutionResult:
        """
        Execute order using VWAP or POV schedule.
        
        Args:
            order: Order to execute
            execution_type: "vwap" or "pov" or "hybrid"
            duration_minutes: Execution duration in minutes
            current_volume: Current volume at each venue
            
        Returns:
            ExecutionResult with execution details
        """
        if current_volume is None:
            current_volume = {'NSE': self.nse_liquidity, 'BSE': self.bse_liquidity}
        
        # Calculate schedule
        start_time = order.timestamp
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        if execution_type == "vwap":
            schedule = self.calculate_vwap_schedule(order, start_time, end_time)
        elif execution_type == "pov":
            schedule = self.calculate_pov_schedule(order, start_time, end_time, current_volume)
        else:  # hybrid
            vwap_schedule = self.calculate_vwap_schedule(order, start_time, end_time)
            pov_schedule = self.calculate_pov_schedule(order, start_time, end_time, current_volume)
            
            # Average both schedules
            schedule = []
            for (vwap_time, vwap_qty), (pov_time, pov_qty) in zip(vwap_schedule, pov_schedule):
                avg_qty = (vwap_qty + pov_qty) // 2
                schedule.append((vwap_time, avg_qty))
        
        # Simulate execution
        arrival_price = order.price
        fills = []
        total_filled = 0
        fill_prices = []
        
        for slice_time, slice_qty in schedule:
            if total_filled >= order.total_quantity:
                break
            
            # Adjust for remaining quantity
            remaining = order.total_quantity - total_filled
            actual_qty = min(slice_qty, remaining)
            
            # Simulate fill price with market impact
            market_impact = self._calculate_market_impact(actual_qty, current_volume)
            
            if order.side == OrderSide.BUY:
                fill_price = arrival_price * (1 + market_impact)
            else:
                fill_price = arrival_price * (1 - market_impact)
            
            fills.append((slice_time, actual_qty, fill_price))
            fill_prices.append(fill_price)
            total_filled += actual_qty
        
        # Calculate metrics
        avg_fill_price = np.average(fill_prices, weights=[q for _, q, _ in fills])
        vwap_price = np.average(fill_prices, weights=[q for _, q, _ in fills])
        
        # Calculate slippage
        if order.side == OrderSide.BUY:
            slippage_bps = (avg_fill_price - arrival_price) / arrival_price * 10000
        else:
            slippage_bps = (arrival_price - avg_fill_price) / arrival_price * 10000
        
        # Calculate implementation shortfall
        implementation_shortfall = (vwap_price - arrival_price) / arrival_price * 10000
        
        # Create execution slices
        slices = []
        for i, (slice_time, slice_qty, fill_price) in enumerate(fills):
            slices.append(ExecutionSlice(
                slice_id=f"{order.order_id}_slice_{i}",
                order_id=order.order_id,
                quantity=slice_qty,
                target_time=slice_time,
                venue=self.select_venue(order, current_volume),
                max_participation_rate=self.default_pov
            ))
        
        return ExecutionResult(
            order_id=order.order_id,
            total_quantity=order.total_quantity,
            filled_quantity=total_filled,
            avg_fill_price=avg_fill_price,
            vwap_price=vwap_price,
            arrival_price=arrival_price,
            slippage_bps=slippage_bps,
            implementation_shortfall=implementation_shortfall,
            slices=slices
        )
    
    def _calculate_market_impact(
        self,
        quantity: int,
        current_volume: Dict[str, float]
    ) -> float:
        """
        Calculate market impact using square-root model.
        
        Args:
            quantity: Order quantity
            current_volume: Current volume at each venue
            
        Returns:
            Market impact as percentage
        """
        # Get total market volume
        total_volume = sum(current_volume.values())
        
        # Square-root impact model
        # Impact = k * sqrt(Quantity / Volume)
        k = 0.01  # Impact coefficient
        impact = k * np.sqrt(quantity / total_volume)
        
        return impact
    
    def print_execution_report(self, result: ExecutionResult) -> None:
        """Print execution report."""
        print("\n" + "="*60)
        print("VWAP + POV EXECUTION REPORT")
        print("="*60)
        print(f"Order ID: {result.order_id}")
        print(f"Total Quantity: {result.total_quantity}")
        print(f"Filled Quantity: {result.filled_quantity}")
        print(f"Fill Rate: {result.filled_quantity/result.total_quantity:.2%}")
        print(f"Arrival Price: ₹{result.arrival_price:.2f}")
        print(f"Average Fill Price: ₹{result.avg_fill_price:.2f}")
        print(f"VWAP Price: ₹{result.vwap_price:.2f}")
        print(f"Slippage: {result.slippage_bps:.2f} bps")
        print(f"Implementation Shortfall: {result.implementation_shortfall:.2f} bps")
        
        print(f"\nExecution Slices: {len(result.slices)}")
        for i, slice in enumerate(result.slices[:5]):  # Show first 5
            print(f"  Slice {i+1}: {slice.quantity} @ {slice.target_time} on {slice.venue.name}")
        
        if len(result.slices) > 5:
            print(f"  ... and {len(result.slices) - 5} more slices")
        
        print("="*60)


def run_sample_execution():
    """Run sample execution."""
    engine = VWAPPOVExecutionEngine()
    
    # Create sample order
    order = Order(
        order_id="ORD001",
        symbol="RELIANCE",
        side=OrderSide.BUY,
        total_quantity=10000,
        price=2500.0,
        timestamp=datetime.now(),
        venue=Venue.NSE
    )
    
    # Execute with VWAP
    print("Executing with VWAP...")
    vwap_result = engine.execute_order(order, execution_type="vwap", duration_minutes=30)
    engine.print_execution_report(vwap_result)
    
    # Execute with POV
    print("\nExecuting with POV...")
    pov_result = engine.execute_order(order, execution_type="pov", duration_minutes=30)
    engine.print_execution_report(pov_result)
    
    # Execute with Hybrid
    print("\nExecuting with Hybrid...")
    hybrid_result = engine.execute_order(order, execution_type="hybrid", duration_minutes=30)
    engine.print_execution_report(hybrid_result)


if __name__ == "__main__":
    run_sample_execution()

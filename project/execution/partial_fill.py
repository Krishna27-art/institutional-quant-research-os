"""
Partial Fill Simulation for Backtesting
Implements realistic partial fill logic based on order size relative to average daily volume.

This module provides partial fill simulation that can be integrated into backtesters
to match paper trading simulator assumptions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
import pandas as pd


@dataclass
class PartialFillConfig:
    """Configuration for partial fill simulation"""
    # Volume participation thresholds
    max_participation_rate: float = 0.05  # 5% of ADV
    partial_fill_threshold: float = 0.01  # 1% of ADV triggers partial fills
    
    # Fill rate curve parameters
    # Fill rate decreases as participation increases
    fill_rate_at_max_participation: float = 0.5  # 50% fill at max participation
    
    # Time-based fill parameters
    fill_time_window_minutes: int = 30  # Time window to fill order
    fill_rate_per_minute: float = 0.1  # 10% of remaining fills per minute


class PartialFillSimulator:
    """
    Simulates partial fills based on order size relative to average daily volume.
    
    Matches paper trading simulator assumptions for realistic execution.
    """
    
    def __init__(self, config: PartialFillConfig = None):
        self.config = config or PartialFillConfig()
    
    def calculate_fill_rate(
        self,
        order_quantity: int,
        avg_daily_volume: int
    ) -> float:
        """
        Calculate fill rate based on order size relative to ADV.
        
        Args:
            order_quantity: Order quantity
            avg_daily_volume: Average daily volume
            
        Returns:
            Fill rate (0.0 to 1.0)
        """
        if avg_daily_volume == 0:
            return 0.0
        
        participation_rate = order_quantity / avg_daily_volume
        
        # If participation is very small, full fill
        if participation_rate < self.config.partial_fill_threshold:
            return 1.0
        
        # If participation exceeds max, reject order
        if participation_rate > self.config.max_participation_rate:
            return 0.0
        
        # Linear interpolation between thresholds
        # Fill rate decreases from 1.0 at partial_fill_threshold
        # to fill_rate_at_max_participation at max_participation
        range_size = self.config.max_participation_rate - self.config.partial_fill_threshold
        if range_size == 0:
            return self.config.fill_rate_at_max_participation
        
        position_in_range = (participation_rate - self.config.partial_fill_threshold) / range_size
        fill_rate = 1.0 - position_in_range * (1.0 - self.config.fill_rate_at_max_participation)
        
        return max(0.0, min(1.0, fill_rate))
    
    def simulate_partial_fill(
        self,
        order_quantity: int,
        avg_daily_volume: int,
        num_bars: int = 1
    ) -> Tuple[int, int]:
        """
        Simulate partial fill over multiple bars.
        
        Args:
            order_quantity: Order quantity
            avg_daily_volume: Average daily volume
            num_bars: Number of bars to attempt fill
            
        Returns:
            (filled_quantity, remaining_quantity)
        """
        overall_fill_rate = self.calculate_fill_rate(order_quantity, avg_daily_volume)
        
        if overall_fill_rate == 0.0:
            return 0, order_quantity
        
        if overall_fill_rate == 1.0:
            return order_quantity, 0
        
        # Simulate fill over multiple bars
        filled_quantity = 0
        remaining_quantity = order_quantity
        
        for bar in range(num_bars):
            if remaining_quantity == 0:
                break
            
            # Calculate fill rate for this bar
            # Earlier bars have higher fill probability
            bar_fill_rate = overall_fill_rate * (1.0 - bar * 0.1)
            bar_fill_rate = max(0.0, bar_fill_rate)
            
            # Calculate fill for this bar
            bar_fill = int(remaining_quantity * bar_fill_rate)
            filled_quantity += bar_fill
            remaining_quantity -= bar_fill
        
        return filled_quantity, remaining_quantity
    
    def simulate_time_based_fill(
        self,
        order_quantity: int,
        avg_daily_volume: int,
        minutes_elapsed: int
    ) -> Tuple[int, int]:
        """
        Simulate fill over time (minutes).
        
        Args:
            order_quantity: Order quantity
            avg_daily_volume: int
            minutes_elapsed: Minutes elapsed since order placement
            
        Returns:
            (filled_quantity, remaining_quantity)
        """
        overall_fill_rate = self.calculate_fill_rate(order_quantity, avg_daily_volume)
        
        if overall_fill_rate == 0.0:
            return 0, order_quantity
        
        if overall_fill_rate == 1.0:
            return order_quantity, 0
        
        # Calculate cumulative fill rate based on time
        # Fill rate increases over time up to the time window
        time_fraction = min(minutes_elapsed / self.config.fill_time_window_minutes, 1.0)
        cumulative_fill_rate = overall_fill_rate * time_fraction
        
        filled_quantity = int(order_quantity * cumulative_fill_rate)
        remaining_quantity = order_quantity - filled_quantity
        
        return filled_quantity, remaining_quantity
    
    def check_order_rejection(
        self,
        order_quantity: int,
        avg_daily_volume: int
    ) -> bool:
        """
        Check if order should be rejected due to size.
        
        Args:
            order_quantity: Order quantity
            avg_daily_volume: Average daily volume
            
        Returns:
            True if order should be rejected
        """
        if avg_daily_volume == 0:
            return True
        
        participation_rate = order_quantity / avg_daily_volume
        return participation_rate > self.config.max_participation_rate


def get_partial_fill_simulator() -> PartialFillSimulator:
    """Get the partial fill simulator instance."""
    return PartialFillSimulator()

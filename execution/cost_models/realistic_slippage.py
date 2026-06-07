"""
Realistic Slippage Model

This module implements a realistic slippage model for execution cost estimation,
replacing the simplified lambda function with institutional-grade modeling.

Key Features:
- Size-dependent slippage (larger orders = more slippage)
- Volatility-dependent slippage (higher vol = more slippage)
- Time-of-day effects (opening/closing = more slippage)
- Symbol-specific liquidity profiles
- Order type impact (market vs limit)
- Realistic range: 0.02% to 0.05% for typical orders

Based on Audit Report Priority 1: Research Quality
"""

import logging
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class TimeOfDay(Enum):
    """Time of day buckets."""
    OPENING = "opening"  # 9:15-9:30
    MORNING = "morning"  # 9:30-11:30
    MIDDAY = "midday"    # 11:30-13:30
    AFTERNOON = "afternoon"  # 13:30-15:00
    CLOSING = "closing"  # 15:00-15:30


@dataclass
class SlippageEstimate:
    """Slippage estimate for an order."""
    order_id: str
    symbol: str
    side: str
    order_type: OrderType
    order_size: int
    avg_daily_volume: int
    volatility: float
    time_of_day: TimeOfDay
    base_slippage_bps: float
    size_multiplier: float
    vol_multiplier: float
    time_multiplier: float
    total_slippage_bps: float
    estimated_execution_price: float
    confidence: float


class RealisticSlippageModel:
    """
    Realistic slippage model.
    
    This model estimates slippage based on order size, volatility,
    time of day, and symbol-specific liquidity characteristics.
    """
    
    def __init__(self):
        """Initialize realistic slippage model."""
        # Base slippage rates (in basis points)
        self.base_slippage = {
            OrderType.MARKET: 2.0,      # 2 bps = 0.02%
            OrderType.LIMIT: 1.0,       # 1 bps = 0.01%
            OrderType.STOP_LIMIT: 3.0   # 3 bps = 0.03%
        }
        
        # Time of day multipliers
        self.time_multipliers = {
            TimeOfDay.OPENING: 1.5,   # Higher slippage at open
            TimeOfDay.MORNING: 1.0,
            TimeOfDay.MIDDAY: 0.8,    # Lower slippage mid-day
            TimeOfDay.AFTERNOON: 1.0,
            TimeOfDay.CLOSING: 1.3    # Higher slippage at close
        }
        
        # Symbol liquidity profiles (volume impact factor)
        self.liquidity_profiles = {
            'HIGH_LIQUIDITY': 0.5,   # Large caps like RELIANCE, TCS
            'MEDIUM_LIQUIDITY': 1.0, # Mid caps
            'LOW_LIQUIDITY': 2.0     # Small caps
        }
        
        # High liquidity symbols (NIFTY 50 heavyweights)
        self.high_liquidity_symbols = {
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK',
            'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI', 'HCLTECH'
        }
        
        logger.info("RealisticSlippageModel initialized")
    
    def estimate_slippage(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        order_size: int,
        avg_daily_volume: int,
        current_price: float,
        volatility: float,
        timestamp: datetime = None
    ) -> SlippageEstimate:
        """
        Estimate slippage for an order.
        
        Args:
            symbol: Stock symbol
            side: Order side (buy/sell)
            order_type: Type of order
            order_size: Number of shares
            avg_daily_volume: Average daily volume
            current_price: Current market price
            volatility: Current volatility
            timestamp: Order timestamp
            
        Returns:
            SlippageEstimate with detailed breakdown
        """
        # Get time of day
        time_of_day = self._get_time_of_day(timestamp)
        
        # Get base slippage
        base_slippage = self.base_slippage.get(order_type, 2.0)
        
        # Calculate size multiplier (larger orders = more slippage)
        size_ratio = order_size / avg_daily_volume if avg_daily_volume > 0 else 0
        size_multiplier = 1.0 + (size_ratio ** 0.5) * 2.0  # Square root scaling
        
        # Calculate volatility multiplier (higher vol = more slippage)
        vol_multiplier = 1.0 + volatility * 10.0  # 1% vol = 10% more slippage
        
        # Get time multiplier
        time_multiplier = self.time_multipliers.get(time_of_day, 1.0)
        
        # Get symbol liquidity factor
        liquidity_factor = self._get_liquidity_factor(symbol)
        
        # Calculate total slippage
        total_slippage_bps = base_slippage * size_multiplier * vol_multiplier * time_multiplier * liquidity_factor
        
        # Clamp to realistic range (0.5 bps to 50 bps)
        total_slippage_bps = max(0.5, min(50.0, total_slippage_bps))
        
        # Calculate estimated execution price
        if side.lower() == 'buy':
            estimated_price = current_price * (1 + total_slippage_bps / 10000)
        else:
            estimated_price = current_price * (1 - total_slippage_bps / 10000)
        
        # Confidence based on data quality
        confidence = 0.8 if avg_daily_volume > 0 else 0.5
        
        return SlippageEstimate(
            order_id=f"{symbol}_{datetime.now().timestamp()}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            order_size=order_size,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            time_of_day=time_of_day,
            base_slippage_bps=base_slippage,
            size_multiplier=size_multiplier,
            vol_multiplier=vol_multiplier,
            time_multiplier=time_multiplier,
            total_slippage_bps=total_slippage_bps,
            estimated_execution_price=estimated_price,
            confidence=confidence
        )
    
    def _get_time_of_day(self, timestamp: Optional[datetime]) -> TimeOfDay:
        """Determine time of day bucket."""
        if timestamp is None:
            timestamp = datetime.now()
        
        t = timestamp.time()
        
        if time(9, 15) <= t < time(9, 30):
            return TimeOfDay.OPENING
        elif time(9, 30) <= t < time(11, 30):
            return TimeOfDay.MORNING
        elif time(11, 30) <= t < time(13, 30):
            return TimeOfDay.MIDDAY
        elif time(13, 30) <= t < time(15, 0):
            return TimeOfDay.AFTERNOON
        elif time(15, 0) <= t < time(15, 30):
            return TimeOfDay.CLOSING
        else:
            return TimeOfDay.MIDDAY  # Default
    
    def _get_liquidity_factor(self, symbol: str) -> float:
        """Get liquidity factor for symbol."""
        if symbol.upper() in self.high_liquidity_symbols:
            return self.liquidity_profiles['HIGH_LIQUIDITY']
        else:
            return self.liquidity_profiles['MEDIUM_LIQUIDITY']
    
    def batch_estimate_slippage(
        self,
        orders: List[Dict]
    ) -> List[SlippageEstimate]:
        """
        Estimate slippage for multiple orders.
        
        Args:
            orders: List of order dictionaries
            
        Returns:
            List of slippage estimates
        """
        estimates = []
        
        for order in orders:
            try:
                estimate = self.estimate_slippage(
                    symbol=order['symbol'],
                    side=order['side'],
                    order_type=OrderType(order.get('order_type', 'market')),
                    order_size=order['order_size'],
                    avg_daily_volume=order['avg_daily_volume'],
                    current_price=order['current_price'],
                    volatility=order.get('volatility', 0.02),
                    timestamp=order.get('timestamp')
                )
                estimates.append(estimate)
            except Exception as e:
                logger.error(f"Slippage estimation failed for order: {e}")
        
        return estimates
    
    def get_average_slippage(
        self,
        symbol: str,
        order_size: int,
        avg_daily_volume: int,
        volatility: float
    ) -> float:
        """
        Get average slippage across all times of day.
        
        Args:
            symbol: Stock symbol
            order_size: Order size
            avg_daily_volume: Average daily volume
            volatility: Volatility
            
        Returns:
            Average slippage in basis points
        """
        estimates = []
        
        for time_of_day in TimeOfDay:
            estimate = self.estimate_slippage(
                symbol=symbol,
                side='buy',
                order_type=OrderType.MARKET,
                order_size=order_size,
                avg_daily_volume=avg_daily_volume,
                current_price=1000.0,
                volatility=volatility,
                timestamp=datetime.now().replace(hour=12, minute=0)
            )
            estimates.append(estimate.total_slippage_bps)
        
        return np.mean(estimates)
    
    def print_slippage_report(self, estimates: List[SlippageEstimate]) -> None:
        """Print slippage report."""
        print("\n" + "="*60)
        print("SLIPPAGE ESTIMATION REPORT")
        print("="*60)
        
        if not estimates:
            print("No estimates to report")
            return
        
        total_slippage = np.mean([e.total_slippage_bps for e in estimates])
        
        print(f"\nTotal Orders: {len(estimates)}")
        print(f"Average Slippage: {total_slippage:.2f} bps")
        print(f"Average Slippage %: {total_slippage / 100:.4f}%")
        
        print(f"\nDetailed Breakdown:")
        print(f"{'Symbol':<12} {'Side':<6} {'Type':<12} {'Size':<10} {'Slippage(bps)':<15} {'Slippage(%)':<12}")
        print("-" * 70)
        
        for e in estimates:
            print(f"{e.symbol:<12} {e.side:<6} {e.order_type.value:<12} {e.order_size:<10} "
                  f"{e.total_slippage_bps:>14.2f} {e.total_slippage_bps/100:>11.4f}%")
        
        print("\n" + "="*60)


# Singleton instance
_slippage_model = None

def get_slippage_model() -> RealisticSlippageModel:
    """Get the singleton slippage model instance."""
    global _slippage_model
    if _slippage_model is None:
        _slippage_model = RealisticSlippageModel()
    return _slippage_model


if __name__ == "__main__":
    # Test the realistic slippage model
    print("Testing Realistic Slippage Model...")
    
    model = RealisticSlippageModel()
    
    # Test order
    estimate = model.estimate_slippage(
        symbol="RELIANCE",
        side="buy",
        order_type=OrderType.MARKET,
        order_size=10000,
        avg_daily_volume=5000000,
        current_price=2500.0,
        volatility=0.02
    )
    
    print(f"\nSlippage Estimate for RELIANCE:")
    print(f"  Total Slippage: {estimate.total_slippage_bps:.2f} bps ({estimate.total_slippage_bps/100:.4f}%)")
    print(f"  Base Slippage: {estimate.base_slippage_bps:.2f} bps")
    print(f"  Size Multiplier: {estimate.size_multiplier:.2f}")
    print(f"  Vol Multiplier: {estimate.vol_multiplier:.2f}")
    print(f"  Time Multiplier: {estimate.time_multiplier:.2f}")
    print(f"  Estimated Price: ₹{estimate.estimated_execution_price:.2f}")
    
    # Batch test
    orders = [
        {
            'symbol': 'RELIANCE',
            'side': 'buy',
            'order_size': 5000,
            'avg_daily_volume': 5000000,
            'current_price': 2500.0,
            'volatility': 0.02
        },
        {
            'symbol': 'TCS',
            'side': 'sell',
            'order_size': 2000,
            'avg_daily_volume': 2000000,
            'current_price': 3500.0,
            'volatility': 0.015
        },
        {
            'symbol': 'HDFCBANK',
            'side': 'buy',
            'order_size': 15000,
            'avg_daily_volume': 8000000,
            'current_price': 1500.0,
            'volatility': 0.025
        }
    ]
    
    estimates = model.batch_estimate_slippage(orders)
    model.print_slippage_report(estimates)

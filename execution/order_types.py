"""
Order Types for Backtesting
Implements limit orders, stop loss orders, and market orders for realistic execution simulation.

This module provides order type logic that can be integrated into backtesters
to match paper trading simulator assumptions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import pandas as pd
import numpy as np


class OrderType(Enum):
    """Order types supported by the system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]  # Required for limit/stop orders
    stop_price: Optional[float]  # Required for stop orders
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    timestamp: pd.Timestamp = None
    expiry_time: Optional[pd.Timestamp] = None


class OrderExecutionEngine:
    """
    Order execution engine for backtesting.
    
    Simulates order execution with realistic fill logic for different order types.
    Matches paper trading simulator assumptions.
    """
    
    def __init__(self, limit_order_patience_seconds: int = 30):
        self.limit_order_patience_seconds = limit_order_patience_seconds
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.DataFrame,
        current_time: pd.Timestamp
    ) -> Tuple[Order, Optional[pd.Timestamp], str]:
        """
        Execute an order based on market data.
        
        Args:
            order: Order to execute
            market_data: Market data (OHLCV) for the symbol
            current_time: Current timestamp
            
        Returns:
            (updated_order, fill_time, fill_reason)
        """
        if order.status != OrderStatus.PENDING:
            return order, None, "already_processed"
        
        # Check if order has expired
        if order.expiry_time and current_time >= order.expiry_time:
            order.status = OrderStatus.EXPIRED
            return order, None, "expired"
        
        # Get current market data
        if current_time not in market_data.index:
            return order, None, "no_market_data"
        
        current_bar = market_data.loc[current_time]
        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']
        
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, current_price, current_time)
        
        elif order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, high, low, current_time)
        
        elif order.order_type == OrderType.STOP_LOSS:
            return self._execute_stop_loss_order(order, current_price, current_time)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            return self._execute_stop_limit_order(order, high, low, current_time)
        
        else:
            order.status = OrderStatus.REJECTED
            return order, None, "unsupported_order_type"
    
    def _execute_market_order(
        self,
        order: Order,
        current_price: float,
        current_time: pd.Timestamp
    ) -> Tuple[Order, pd.Timestamp, str]:
        """Execute market order at current price."""
        order.filled_quantity = order.quantity
        order.filled_price = current_price
        order.status = OrderStatus.FILLED
        return order, current_time, "market_fill"
    
    def _execute_limit_order(
        self,
        order: Order,
        high: float,
        low: float,
        current_time: pd.Timestamp
    ) -> Tuple[Order, Optional[pd.Timestamp], str]:
        """
        Execute limit order.
        
        Buy limit: Fill if price <= limit price
        Sell limit: Fill if price >= limit price
        """
        if order.side == OrderSide.BUY:
            # Buy limit: fill if low <= limit price
            if low <= order.price:
                # Fill at the better price (between low and limit)
                fill_price = min(order.price, high)
                order.filled_quantity = order.quantity
                order.filled_price = fill_price
                order.status = OrderStatus.FILLED
                return order, current_time, "limit_fill"
        else:
            # Sell limit: fill if high >= limit price
            if high >= order.price:
                # Fill at the better price (between high and limit)
                fill_price = max(order.price, low)
                order.filled_quantity = order.quantity
                order.filled_price = fill_price
                order.status = OrderStatus.FILLED
                return order, current_time, "limit_fill"
        
        # Check if order has expired (patience exceeded)
        if order.expiry_time and current_time >= order.expiry_time:
            order.status = OrderStatus.EXPIRED
            return order, None, "expired"
        
        return order, None, "no_fill"
    
    def _execute_stop_loss_order(
        self,
        order: Order,
        current_price: float,
        current_time: pd.Timestamp
    ) -> Tuple[Order, Optional[pd.Timestamp], str]:
        """
        Execute stop loss order.
        
        Buy stop: Trigger if price >= stop price
        Sell stop: Trigger if price <= stop price
        """
        if order.side == OrderSide.BUY:
            # Buy stop: trigger if price >= stop price
            if current_price >= order.stop_price:
                order.filled_quantity = order.quantity
                order.filled_price = current_price
                order.status = OrderStatus.FILLED
                return order, current_time, "stop_trigger"
        else:
            # Sell stop: trigger if price <= stop price
            if current_price <= order.stop_price:
                order.filled_quantity = order.quantity
                order.filled_price = current_price
                order.status = OrderStatus.FILLED
                return order, current_time, "stop_trigger"
        
        return order, None, "no_trigger"
    
    def _execute_stop_limit_order(
        self,
        order: Order,
        high: float,
        low: float,
        current_time: pd.Timestamp
    ) -> Tuple[Order, Optional[pd.Timestamp], str]:
        """
        Execute stop limit order.
        
        First triggers like stop loss, then becomes a limit order.
        """
        if order.side == OrderSide.BUY:
            # Buy stop limit: trigger if price >= stop price
            if high >= order.stop_price:
                # Triggered, now act as limit order
                if low <= order.price:
                    fill_price = min(order.price, high)
                    order.filled_quantity = order.quantity
                    order.filled_price = fill_price
                    order.status = OrderStatus.FILLED
                    return order, current_time, "stop_limit_fill"
        else:
            # Sell stop limit: trigger if price <= stop price
            if low <= order.stop_price:
                # Triggered, now act as limit order
                if high >= order.price:
                    fill_price = max(order.price, low)
                    order.filled_quantity = order.quantity
                    order.filled_price = fill_price
                    order.status = OrderStatus.FILLED
                    return order, current_time, "stop_limit_fill"
        
        return order, None, "no_trigger"


def create_market_order(
    order_id: str,
    symbol: str,
    side: OrderSide,
    quantity: int,
    timestamp: pd.Timestamp
) -> Order:
    """Create a market order."""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=None,
        stop_price=None,
        status=OrderStatus.PENDING,
        timestamp=timestamp
    )


def create_limit_order(
    order_id: str,
    symbol: str,
    side: OrderSide,
    quantity: int,
    price: float,
    timestamp: pd.Timestamp,
    patience_seconds: int = 30
) -> Order:
    """Create a limit order."""
    expiry_time = timestamp + pd.Timedelta(seconds=patience_seconds)
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        stop_price=None,
        status=OrderStatus.PENDING,
        timestamp=timestamp,
        expiry_time=expiry_time
    )


def create_stop_loss_order(
    order_id: str,
    symbol: str,
    side: OrderSide,
    quantity: int,
    stop_price: float,
    timestamp: pd.Timestamp
) -> Order:
    """Create a stop loss order."""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.STOP_LOSS,
        quantity=quantity,
        price=None,
        stop_price=stop_price,
        status=OrderStatus.PENDING,
        timestamp=timestamp
    )

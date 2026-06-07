"""
L3 Order Book Simulator with Queue Dynamics
Full limit order book simulation for realistic backtesting.

Critical for institutional-grade execution modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import heapq


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class LimitOrder:
    """Limit order in the book"""
    order_id: str
    side: OrderSide
    price: float
    quantity: int
    timestamp: datetime
    queue_position: int  # Position in queue at this price level
    is_hidden: bool = False


@dataclass
class Trade:
    """Executed trade"""
    trade_id: str
    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    timestamp: datetime


@dataclass
class PriceLevel:
    """Price level in order book"""
    price: float
    orders: List[LimitOrder]
    total_quantity: int
    
    def add_order(self, order: LimitOrder):
        """Add order to this price level"""
        order.queue_position = len(self.orders)
        self.orders.append(order)
        self.total_quantity += order.quantity
    
    def remove_order(self, order_id: str):
        """Remove order from this price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                self.total_quantity -= order.quantity
                self.orders.pop(i)
                # Update queue positions
                for j in range(i, len(self.orders)):
                    self.orders[j].queue_position = j
                return True
        return False


class L3OrderBook:
    """
    L3 Order Book Simulator
    
    Simulates full limit order book with queue dynamics.
    
    Features:
    - Price-time priority
    - Queue position tracking
    - Hidden liquidity (iceberg orders)
    - Partial fills
    - Order cancellations
    - Market orders
    """
    
    def __init__(self, tick_size: float = 0.05):
        self.tick_size = tick_size
        
        # Order book: price -> PriceLevel
        self.bids: Dict[float, PriceLevel] = {}  # Sorted descending
        self.asks: Dict[float, PriceLevel] = {}  # Sorted ascending
        
        # Order tracking
        self.orders: Dict[str, LimitOrder] = {}
        self.trades: List[Trade] = []
        
        # Order ID counter
        self.order_counter = 0
        self.trade_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"order_{self.order_counter}"
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        return f"trade_{self.trade_counter}"
    
    def add_limit_order(self, side: OrderSide, price: float, quantity: int,
                       is_hidden: bool = False, timestamp: Optional[datetime] = None) -> str:
        """
        Add limit order to book.
        
        Returns:
            Order ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Round price to tick size
        price = round(price / self.tick_size) * self.tick_size
        
        order_id = self._generate_order_id()
        order = LimitOrder(
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
            queue_position=0,
            is_hidden=is_hidden
        )
        
        self.orders[order_id] = order
        
        # Add to appropriate side
        if side == OrderSide.BUY:
            if price not in self.bids:
                self.bids[price] = PriceLevel(price=price, orders=[], total_quantity=0)
            self.bids[price].add_order(order)
        else:
            if price not in self.asks:
                self.asks[price] = PriceLevel(price=price, orders=[], total_quantity=0)
            self.asks[price].add_order(order)
        
        return order_id
    
    def add_market_order(self, side: OrderSide, quantity: int,
                       timestamp: Optional[datetime] = None) -> List[Trade]:
        """
        Add market order (immediate execution).
        
        Returns:
            List of executed trades
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        trades = []
        remaining_quantity = quantity
        
        if side == OrderSide.BUY:
            # Buy: consume asks
            while remaining_quantity > 0 and self.asks:
                best_ask = min(self.asks.keys())
                price_level = self.asks[best_ask]
                
                # Match orders in queue
                for order in price_level.orders[:]:  # Copy to allow modification
                    if remaining_quantity <= 0:
                        break
                    
                    fill_quantity = min(order.quantity, remaining_quantity)
                    
                    # Create trade
                    trade = Trade(
                        trade_id=self._generate_trade_id(),
                        price=best_ask,
                        quantity=fill_quantity,
                        buy_order_id="market",
                        sell_order_id=order.order_id,
                        timestamp=timestamp
                    )
                    trades.append(trade)
                    self.trades.append(trade)
                    
                    # Update order
                    order.quantity -= fill_quantity
                    remaining_quantity -= fill_quantity
                    
                    # Remove filled order
                    if order.quantity == 0:
                        price_level.remove_order(order.order_id)
                        del self.orders[order.order_id]
                
                # Remove empty price level
                if price_level.total_quantity == 0:
                    del self.asks[best_ask]
        
        else:  # SELL
            # Sell: consume bids
            while remaining_quantity > 0 and self.bids:
                best_bid = max(self.bids.keys())
                price_level = self.bids[best_bid]
                
                # Match orders in queue
                for order in price_level.orders[:]:
                    if remaining_quantity <= 0:
                        break
                    
                    fill_quantity = min(order.quantity, remaining_quantity)
                    
                    # Create trade
                    trade = Trade(
                        trade_id=self._generate_trade_id(),
                        price=best_bid,
                        quantity=fill_quantity,
                        buy_order_id=order.order_id,
                        sell_order_id="market",
                        timestamp=timestamp
                    )
                    trades.append(trade)
                    self.trades.append(trade)
                    
                    # Update order
                    order.quantity -= fill_quantity
                    remaining_quantity -= fill_quantity
                    
                    # Remove filled order
                    if order.quantity == 0:
                        price_level.remove_order(order.order_id)
                        del self.orders[order.order_id]
                
                # Remove empty price level
                if price_level.total_quantity == 0:
                    del self.bids[best_bid]
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Returns:
            True if order was cancelled, False if not found
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Remove from price level
        if order.side == OrderSide.BUY:
            if order.price in self.bids:
                self.bids[order.price].remove_order(order_id)
                if self.bids[order.price].total_quantity == 0:
                    del self.bids[order.price]
        else:
            if order.price in self.asks:
                self.asks[order.price].remove_order(order_id)
                if self.asks[order.price].total_quantity == 0:
                    del self.asks[order.price]
        
        del self.orders[order_id]
        return True
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    def get_depth(self, n_levels: int = 5) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]]]:
        """
        Get order book depth.
        
        Returns:
            Tuple of (bids, asks) where each is list of (price, total_quantity)
        """
        # Get top N bid levels
        bid_prices = sorted(self.bids.keys(), reverse=True)[:n_levels]
        bids = [(p, self.bids[p].total_quantity) for p in bid_prices]
        
        # Get top N ask levels
        ask_prices = sorted(self.asks.keys())[:n_levels]
        asks = [(p, self.asks[p].total_quantity) for p in ask_prices]
        
        return bids, asks
    
    def get_queue_position(self, order_id: str) -> Optional[int]:
        """Get queue position for an order"""
        if order_id not in self.orders:
            return None
        return self.orders[order_id].queue_position
    
    def estimate_fill_probability(self, order_id: str, 
                                 expected_volume_per_level: float = 1000) -> float:
        """
        Estimate probability of order being filled.
        
        Simplified model: probability decreases with queue position
        """
        if order_id not in self.orders:
            return 0.0
        
        order = self.orders[order_id]
        queue_pos = order.queue_position
        
        # Simple model: probability = 1 / (queue_position + 1)
        # In production, would use historical fill rates by queue position
        probability = 1.0 / (queue_pos + 1)
        
        return probability
    
    def get_total_volume(self) -> Tuple[int, int]:
        """Get total bid and ask volume"""
        bid_volume = sum(level.total_quantity for level in self.bids.values())
        ask_volume = sum(level.total_quantity for level in self.asks.values())
        return bid_volume, ask_volume


if __name__ == "__main__":
    # Example usage
    book = L3OrderBook(tick_size=0.05)
    
    # Add limit orders
    print("Adding limit orders...")
    book.add_limit_order(OrderSide.BUY, 100.0, 100)
    book.add_limit_order(OrderSide.BUY, 99.95, 200)
    book.add_limit_order(OrderSide.SELL, 100.05, 150)
    book.add_limit_order(OrderSide.SELL, 100.10, 100)
    
    print(f"Best Bid: {book.get_best_bid()}")
    print(f"Best Ask: {book.get_best_ask()}")
    print(f"Mid Price: {book.get_mid_price()}")
    print(f"Spread: {book.get_spread()}")
    
    # Get depth
    bids, asks = book.get_depth(3)
    print(f"\nBids (top 3): {bids}")
    print(f"Asks (top 3): {asks}")
    
    # Market order
    print("\nExecuting market buy order for 200 shares...")
    trades = book.add_market_order(OrderSide.BUY, 200)
    print(f"Executed {len(trades)} trades")
    for trade in trades:
        print(f"  Trade: {trade.quantity} @ {trade.price}")
    
    print(f"\nBest Bid after trade: {book.get_best_bid()}")
    print(f"Best Ask after trade: {book.get_best_ask()}")

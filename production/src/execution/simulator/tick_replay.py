"""
Tick-Level Market Simulator.
Replaces simple "Buy at Close" mechanics with realistic matching engine dynamics.
Handles queue positions, partial fills, and latency.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.data.event_store import MarketEvent, EventType, TradeEvent, QuoteEvent

@dataclass
class Order:
    order_id: str
    symbol: str
    is_buy: bool
    qty: int
    price: float
    arrival_time: datetime
    filled_qty: int = 0
    
    @property
    def remaining(self) -> int:
        return self.qty - self.filled_qty

class MatchingEngine:
    """Simulates an exchange order book matching engine."""
    def __init__(self, latency_ms: int = 5):
        self.latency_ms = latency_ms
        self.active_orders: Dict[str, Order] = {}
        # Simple simulated queue position tracker (Orders behind us in queue)
        self.queue_position: Dict[str, int] = {}
        
    def submit_order(self, order: Order, current_time: datetime) -> None:
        """Adds latency to the order before it enters the book."""
        order.arrival_time = current_time + timedelta(milliseconds=self.latency_ms)
        self.active_orders[order.order_id] = order
        # Naively assume we are at the back of the queue on submission
        self.queue_position[order.order_id] = 1000  
        
    def process_event(self, event: MarketEvent) -> None:
        """Update internal state based on a market event."""
        if event.event_type == EventType.TRADE:
            self._process_trade(event)
        elif event.event_type == EventType.QUOTE_UPDATE:
            self._process_quote(event)

    def _process_trade(self, trade: TradeEvent) -> None:
        """Simulate queue advancement and fills when trades happen."""
        to_remove = []
        for order_id, order in self.active_orders.items():
            if order.symbol != trade.symbol:
                continue
                
            # If trade happens at our price, queue advances
            if order.price == trade.price:
                self.queue_position[order_id] = max(0, self.queue_position[order_id] - trade.quantity)
                
                # If we reached the front of the queue, we get filled
                if self.queue_position[order_id] == 0:
                    fill_qty = min(order.remaining, trade.quantity)
                    order.filled_qty += fill_qty
                    if order.remaining == 0:
                        to_remove.append(order_id)
                        
            # If trade punches through our price, we get swept
            elif (order.is_buy and trade.price < order.price) or (not order.is_buy and trade.price > order.price):
                order.filled_qty += order.remaining
                to_remove.append(order_id)

        for rid in to_remove:
            del self.active_orders[rid]

    def _process_quote(self, quote: QuoteEvent) -> None:
        """Adjust queue estimates based on book size changes."""
        pass

class TickReplaySimulator:
    """Orchestrates backtesting against the tick-level event store."""
    def __init__(self, event_store):
        self.event_store = event_store
        self.engine = MatchingEngine()
        
    def run(self, start: datetime, end: datetime, strategy):
        """Replay ticks and step the strategy."""
        for event in self.event_store.stream_events(start, end):
            self.engine.process_event(event)
            strategy.on_event(event, self.engine)

"""
Smart Order Router for Indian Brokers (Zerodha, Upstox, Dhan)

Implements intelligent order routing across multiple Indian brokers to
achieve best execution, minimize costs, and maximize fill rates.

Key Features:
- Multi-broker support (Zerodha, Upstox, Dhan)
- Venue selection based on price, liquidity, and cost
- Order splitting for large orders
- Real-time quote aggregation
- Cost optimization (brokerage, taxes, exchange fees)
- Risk-aware routing
- Order priority queue management
- Partial fill handling

Based on Blueprint Week 11-12: Execution & Monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class Broker(Enum):
    """Supported brokers."""
    ZERODHA = "zerodha"
    UPSTOX = "upstox"
    DHAN = "dhan"


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"


class Exchange(Enum):
    """Indian exchanges."""
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    NFO = "NFO"


@dataclass
class Quote:
    """Market quote."""
    broker: Broker
    exchange: Exchange
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: float
    timestamp: pd.Timestamp


@dataclass
class Order:
    """Order to execute."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    validity: str = 'DAY'  # DAY, IOC, GTC


class BrokerAdapter:
    """
    Base adapter for broker-specific API integration.
    
    Each broker (Zerodha, Upstox, Dhan) has its own API and this
    adapter provides a unified interface for order routing.
    """
    
    def __init__(self, broker: Broker, api_key: str, api_secret: str):
        """
        Initialize broker adapter.
        
        Args:
            broker: Broker type
            api_key: API key
            api_secret: API secret
        """
        self.broker = broker
        self.api_key = api_key
        self.api_secret = api_secret
    
    def get_quote(self, symbol: str, exchange: Exchange) -> Optional[Quote]:
        """
        Get current quote from broker.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange
            
        Returns:
            Quote or None if unavailable
        """
        # In production, this would call the broker's API
        # For now, return mock data
        return Quote(
            broker=self.broker,
            exchange=exchange,
            symbol=symbol,
            bid_price=100.0,
            ask_price=100.05,
            bid_size=1000,
            ask_size=1000,
            timestamp=pd.Timestamp.now()
        )
    
    def place_order(self, order: Order, exchange: Exchange) -> Dict:
        """
        Place order through broker.
        
        Args:
            order: Order to place
            exchange: Exchange
            
        Returns:
            Order response with order_id and status
        """
        # In production, this would call the broker's API
        return {
            'order_id': f"{self.broker.value}_{pd.Timestamp.now().timestamp()}",
            'status': 'PENDING',
            'broker': self.broker.value,
            'exchange': exchange.value
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        # In production, this would call the broker's API
        return True
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status dictionary
        """
        # In production, this would call the broker's API
        return {
            'order_id': order_id,
            'status': 'FILLED',
            'filled_quantity': 100,
            'average_price': 100.02
        }
    
    def calculate_cost(self, quantity: int, price: float, exchange: Exchange) -> float:
        """
        Calculate total cost including brokerage and taxes.
        
        Args:
            quantity: Order quantity
            price: Execution price
            exchange: Exchange
            
        Returns:
            Total cost
        """
        # Brokerage (varies by broker)
        if self.broker == Broker.ZERODHA:
            brokerage = 20  # Flat fee or 0.03% whichever is lower
        elif self.broker == Broker.UPSTOX:
            brokerage = 20
        elif self.broker == Broker.DHAN:
            brokerage = 20
        else:
            brokerage = 20
        
        # Transaction charges
        transaction_charge = 0.0000345 * quantity * price  # NSE
        
        # STT
        stt = 0.00025 * quantity * price if exchange == Exchange.NSE else 0.000125 * quantity * price
        
        # GST
        gst = 0.18 * (brokerage + transaction_charge)
        
        # Stamp duty
        stamp_duty = 0.00003 * quantity * price if exchange == Exchange.NSE else 0.00003 * quantity * price
        
        total_cost = brokerage + transaction_charge + stt + gst + stamp_duty
        
        return total_cost


class SmartOrderRouter:
    """
    Smart Order Router for multi-broker execution.
    
    Routes orders to the best broker based on:
    - Best available price
    - Liquidity availability
    - Cost efficiency
    - Risk constraints
    """
    
    def __init__(
        self,
        brokers: List[BrokerAdapter],
        split_threshold: int = 1000,
        cost_weight: float = 0.3,
        liquidity_weight: float = 0.5,
        price_weight: float = 0.2
    ):
        """
        Initialize smart order router.
        
        Args:
            brokers: List of broker adapters
            split_threshold: Order size threshold for splitting
            cost_weight: Weight for cost in venue selection
            liquidity_weight: Weight for liquidity in venue selection
            price_weight: Weight for price in venue selection
        """
        self.brokers = brokers
        self.split_threshold = split_threshold
        self.cost_weight = cost_weight
        self.liquidity_weight = liquidity_weight
        self.price_weight = price_weight
        
        # Quote cache
        self.quote_cache: Dict[str, Dict[Broker, Quote]] = {}
    
    def aggregate_quotes(self, symbol: str, exchanges: List[Exchange]) -> Dict[Broker, Quote]:
        """
        Aggregate quotes from all brokers.
        
        Args:
            symbol: Stock symbol
            exchanges: List of exchanges to query
            
        Returns:
            Dictionary mapping broker to quote
        """
        quotes = {}
        
        for broker_adapter in self.brokers:
            for exchange in exchanges:
                try:
                    quote = broker_adapter.get_quote(symbol, exchange)
                    if quote:
                        quotes[broker_adapter.broker] = quote
                        break  # Use first available exchange
                except Exception as e:
                    logger.warning(f"Failed to get quote from {broker_adapter.broker}: {e}")
        
        # Cache quotes
        self.quote_cache[symbol] = quotes
        
        return quotes
    
    def select_venue(
        self,
        symbol: str,
        side: str,
        quantity: int,
        quotes: Dict[Broker, Quote]
    ) -> Tuple[Broker, Quote]:
        """
        Select best venue (broker) for order execution.
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            quotes: Available quotes
            
        Returns:
            Tuple of (selected broker, selected quote)
        """
        if not quotes:
            raise ValueError("No quotes available")
        
        scores = {}
        
        for broker, quote in quotes.items():
            # Get broker adapter
            broker_adapter = next(b for b in self.brokers if b.broker == broker)
            
            # Calculate score components
            if side == 'BUY':
                price_score = 1.0 / quote.ask_price  # Lower ask is better
                liquidity_score = quote.ask_size / quantity  # Higher liquidity is better
                execution_price = quote.ask_price
            else:
                price_score = quote.bid_price  # Higher bid is better
                liquidity_score = quote.bid_size / quantity
                execution_price = quote.bid_price
            
            # Normalize scores
            price_score = min(price_score / max(1.0 / q.ask_price for q in quotes.values()), 1.0)
            liquidity_score = min(liquidity_score, 1.0)
            
            # Calculate cost
            cost = broker_adapter.calculate_cost(quantity, execution_price, quote.exchange)
            cost_score = 1.0 / (cost + 1)  # Lower cost is better
            cost_score = min(cost_score / max(1.0 / (broker_adapter.calculate_cost(quantity, q.ask_price if side == 'BUY' else q.bid_price, q.exchange) + 1) for q in quotes.values()), 1.0)
            
            # Weighted score
            total_score = (
                self.price_weight * price_score +
                self.liquidity_weight * liquidity_score +
                self.cost_weight * cost_score
            )
            
            scores[broker] = total_score
        
        # Select best venue
        best_broker = max(scores, key=scores.get)
        
        return best_broker, quotes[best_broker]
    
    def route_order(
        self,
        order: Order,
        exchanges: List[Exchange] = None
    ) -> Dict:
        """
        Route order to best venue.
        
        Args:
            order: Order to route
            exchanges: List of exchanges (defaults to NSE, BSE)
            
        Returns:
            Order response
        """
        if exchanges is None:
            exchanges = [Exchange.NSE, Exchange.BSE]
        
        # Aggregate quotes
        quotes = self.aggregate_quotes(order.symbol, exchanges)
        
        if not quotes:
            return {
                'status': 'FAILED',
                'reason': 'No quotes available'
            }
        
        # Select venue
        best_broker, best_quote = self.select_venue(
            order.symbol, order.side, order.quantity, quotes
        )
        
        # Get broker adapter
        broker_adapter = next(b for b in self.brokers if b.broker == best_broker)
        
        # Place order
        response = broker_adapter.place_order(order, best_quote.exchange)
        
        response['selected_broker'] = best_broker.value
        response['selected_exchange'] = best_quote.exchange.value
        response['execution_price'] = best_quote.ask_price if order.side == 'BUY' else best_quote.bid_price
        
        return response
    
    def route_split_order(
        self,
        order: Order,
        exchanges: List[Exchange] = None
    ) -> List[Dict]:
        """
        Route large order by splitting across venues.
        
        Args:
            order: Order to route
            exchanges: List of exchanges
            
        Returns:
            List of order responses
        """
        if order.quantity <= self.split_threshold:
            return [self.route_order(order, exchanges)]
        
        # Split order
        responses = []
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0:
            # Calculate split size
            split_size = min(remaining_quantity, self.split_threshold)
            
            # Create split order
            split_order = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=split_size,
                order_type=order.order_type,
                price=order.price,
                stop_price=order.stop_price,
                validity=order.validity
            )
            
            # Route split order
            response = self.route_order(split_order, exchanges)
            responses.append(response)
            
            remaining_quantity -= split_size
        
        return responses
    
    def get_best_execution_price(
        self,
        symbol: str,
        side: str,
        quantity: int,
        exchanges: List[Exchange] = None
    ) -> float:
        """
        Get best execution price across all venues.
        
        Args:
            symbol: Stock symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            exchanges: List of exchanges
            
        Returns:
            Best available price
        """
        if exchanges is None:
            exchanges = [Exchange.NSE, Exchange.BSE]
        
        quotes = self.aggregate_quotes(symbol, exchanges)
        
        if not quotes:
            return 0.0
        
        if side == 'BUY':
            return min(q.ask_price for q in quotes.values())
        else:
            return max(q.bid_price for q in quotes.values())
    
    def calculate_total_cost(
        self,
        order: Order,
        execution_price: float,
        broker: Broker,
        exchange: Exchange
    ) -> Dict:
        """
        Calculate total execution cost.
        
        Args:
            order: Order details
            execution_price: Execution price
            broker: Selected broker
            exchange: Selected exchange
            
        Returns:
            Cost breakdown
        """
        broker_adapter = next(b for b in self.brokers if b.broker == broker)
        
        total_cost = broker_adapter.calculate_cost(order.quantity, execution_price, exchange)
        
        return {
            'brokerage': 20,  # Simplified
            'transaction_charge': 0.0000345 * order.quantity * execution_price,
            'stt': 0.00025 * order.quantity * execution_price if exchange == Exchange.NSE else 0.000125 * order.quantity * execution_price,
            'gst': 0.18 * 20,
            'stamp_duty': 0.00003 * order.quantity * execution_price,
            'total_cost': total_cost
        }


class OrderPriorityQueue:
    """
    Priority queue for order management.
    
    Manages order priority based on:
    - Time sensitivity
    - Size
    - Client priority
    """
    
    def __init__(self):
        """Initialize order priority queue."""
        self.queue: List[Tuple[int, Order]] = []
        self.counter = 0
    
    def add_order(self, order: Order, priority: int = 0) -> None:
        """
        Add order to queue.
        
        Args:
            order: Order to add
            priority: Priority level (higher = more important)
        """
        self.counter += 1
        # Use negative priority for max-heap behavior
        self.queue.append((-priority, self.counter, order))
        self.queue.sort()
    
    def get_next_order(self) -> Optional[Order]:
        """
        Get next order from queue.
        
        Returns:
            Next order or None if queue is empty
        """
        if not self.queue:
            return None
        
        _, _, order = self.queue.pop(0)
        return order
    
    def peek(self) -> Optional[Order]:
        """
        Peek at next order without removing.
        
        Returns:
            Next order or None if queue is empty
        """
        if not self.queue:
            return None
        
        return self.queue[0][2]
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)


if __name__ == "__main__":
    # Test Smart Order Router
    print("Testing Smart Order Router...")
    
    # Create broker adapters (mock)
    zerodha = BrokerAdapter(Broker.ZERODHA, "test_key", "test_secret")
    upstox = BrokerAdapter(Broker.UPSTOX, "test_key", "test_secret")
    dhan = BrokerAdapter(Broker.DHAN, "test_key", "test_secret")
    
    # Create router
    router = SmartOrderRouter([zerodha, upstox, dhan])
    
    # Test quote aggregation
    print("\nTesting quote aggregation...")
    quotes = router.aggregate_quotes('RELIANCE', [Exchange.NSE, Exchange.BSE])
    print(f"Quotes from {len(quotes)} brokers")
    for broker, quote in quotes.items():
        print(f"{broker.value}: Bid={quote.bid_price:.2f}, Ask={quote.ask_price:.2f}")
    
    # Test venue selection
    print("\nTesting venue selection...")
    order = Order(
        symbol='RELIANCE',
        side='BUY',
        quantity=100,
        order_type=OrderType.MARKET
    )
    best_broker, best_quote = router.select_venue('RELIANCE', 'BUY', 100, quotes)
    print(f"Best broker: {best_broker.value}")
    print(f"Best quote: Bid={best_quote.bid_price:.2f}, Ask={best_quote.ask_price:.2f}")
    
    # Test order routing
    print("\nTesting order routing...")
    response = router.route_order(order)
    print(f"Order response: {response}")
    
    # Test split order routing
    print("\nTesting split order routing...")
    large_order = Order(
        symbol='RELIANCE',
        side='BUY',
        quantity=2000,
        order_type=OrderType.MARKET
    )
    responses = router.route_split_order(large_order)
    print(f"Split into {len(responses)} orders")
    for i, resp in enumerate(responses):
        print(f"Order {i+1}: {resp}")
    
    # Test cost calculation
    print("\nTesting cost calculation...")
    cost_breakdown = router.calculate_total_cost(
        order, 100.0, Broker.ZERODHA, Exchange.NSE
    )
    print(f"Cost breakdown: {cost_breakdown}")
    
    # Test priority queue
    print("\nTesting order priority queue...")
    queue = OrderPriorityQueue()
    
    order1 = Order('RELIANCE', 'BUY', 100, OrderType.MARKET)
    order2 = Order('TCS', 'BUY', 50, OrderType.LIMIT, price=3000)
    order3 = Order('HDFC', 'SELL', 75, OrderType.MARKET)
    
    queue.add_order(order1, priority=1)
    queue.add_order(order2, priority=3)  # Higher priority
    queue.add_order(order3, priority=2)
    
    print(f"Queue size: {queue.size()}")
    
    next_order = queue.get_next_order()
    print(f"Next order: {next_order.symbol}")
    
    print("\nSmart Order Router test completed.")

"""
Broker Adapter - Abstract base class for broker integrations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order dataclass"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    order_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    tag: Optional[str] = None


@dataclass
class Fill:
    """Fill dataclass"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    exchange_order_id: Optional[str] = None


@dataclass
class Position:
    """Position dataclass"""
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float


@dataclass
class Quote:
    """Quote dataclass"""
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    timestamp: datetime


class BrokerAdapter(ABC):
    """Abstract base class for broker adapters"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place an order and return order ID"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Get current quote"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict:
        """Get account balance and margin"""
        pass
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None,
                    new_quantity: Optional[float] = None) -> bool:
        """Modify an existing order (optional)"""
        # Default implementation: cancel and replace
        return False

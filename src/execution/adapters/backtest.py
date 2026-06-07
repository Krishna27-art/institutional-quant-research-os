"""
Backtesting execution adapter
"""

from typing import Dict, List, Optional
from datetime import datetime
from ..brokers.broker_adapter import BrokerAdapter, Order, Fill, Position, Quote, OrderSide, OrderType, OrderStatus


class BacktestAdapter(BrokerAdapter):
    """Backtesting adapter simulating instant/delayed fills on historical data."""
    
    def __init__(self) -> None:
        self.is_connected = False
        self.orders: Dict[str, Order] = {}

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def place_order(self, order: Order) -> str:
        if not self.is_connected:
            raise RuntimeError("Not connected to backtest broker")
        order_id = order.order_id or f"BACKTEST_{datetime.now().timestamp()}"
        self.orders[order_id] = order
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id in self.orders:
            return OrderStatus.FILLED
        return OrderStatus.REJECTED

    def get_positions(self) -> List[Position]:
        return []

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=100.0,
            ask=100.1,
            bid_size=100,
            ask_size=100,
            last=100.05,
            timestamp=datetime.now()
        )

    def get_account_balance(self) -> Dict:
        return {
            'available_cash': 10000000.0,
            'used_margin': 0.0,
            'total_margin': 10000000.0
        }

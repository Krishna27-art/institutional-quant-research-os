"""
Execution Engine — Entry point orchestrating order lifecycle, routing, and broker adapters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

from .order_manager.order_manager import OrderManager
from .smart_order_router.smart_order_router import SmartOrderRouter
from .brokers.broker_adapter import BrokerAdapter, Order, Fill, OrderStatus, OrderSide, OrderType


logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Consolidated Execution Engine for managing order lifecycle,
    routing to multiple broker adapters, and calculating costs.
    """

    def __init__(self, brokers: List[BrokerAdapter]) -> None:
        self.router = SmartOrderRouter(brokers)
        self.manager = OrderManager(self.router)
        self.brokers = brokers

    def execute_signal(self, symbol: str, side: str, quantity: float, price: Optional[float] = None) -> str:
        """Create and submit an order from a signal."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        order_type = OrderType.LIMIT if price is not None else OrderType.MARKET
        
        order_id = self.manager.create_order(
            symbol=symbol,
            side=order_side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            tag="execution_engine"
        )
        self.manager.submit_order(order_id)
        return order_id

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        return self.manager.get_order_status(order_id)

    def get_fills(self, order_id: str) -> List[Fill]:
        return self.manager.get_fills(order_id)

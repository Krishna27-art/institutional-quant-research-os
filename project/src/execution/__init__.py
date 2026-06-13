"""
Execution System - Order placement, routing, and broker adapters
"""

from .brokers.broker_adapter import BrokerAdapter, Order, Fill, Position, Quote, OrderSide, OrderType, OrderStatus
from .smart_order_router.smart_order_router import SmartOrderRouter
from .order_manager.order_manager import OrderManager
from .signal_adaptive import QuoteDecision, SignalAdaptiveExecutor
from .engine import ExecutionEngine
from .adapters.zerodha import ZerodhaAdapter
from .adapters.paper import PaperAdapter
from .adapters.backtest import BacktestAdapter
from .cost_model import NSETransactionCostModel

__all__ = [
    'BrokerAdapter',
    'Order',
    'Fill',
    'Position',
    'Quote',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'SmartOrderRouter',
    'OrderManager',
    'QuoteDecision',
    'SignalAdaptiveExecutor',
    'ExecutionEngine',
    'ZerodhaAdapter',
    'PaperAdapter',
    'BacktestAdapter',
    'NSETransactionCostModel',
]

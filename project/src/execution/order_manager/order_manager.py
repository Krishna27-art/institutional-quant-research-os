"""
Order Manager - Manage order lifecycle
"""

import uuid
from typing import Dict, List, Optional
from datetime import datetime
from ..brokers.broker_adapter import Order, Fill, OrderStatus, OrderSide, OrderType
from ..smart_order_router.smart_order_router import SmartOrderRouter


class OrderManager:
    """Manage order lifecycle and tracking"""
    
    def __init__(self, router: SmartOrderRouter):
        self.router = router
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, List[Fill]] = {}
        self.order_status: Dict[str, OrderStatus] = {}
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                   quantity: float, price: Optional[float] = None,
                   trigger_price: Optional[float] = None,
                   tag: Optional[str] = None) -> str:
        """
        Create a new order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            trigger_price: Trigger price (for stop orders)
            tag: Order tag for identification
            
        Returns:
            Order ID
        """
        order_id = str(uuid.uuid4())
        
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            order_id=order_id,
            timestamp=datetime.now(),
            tag=tag
        )
        
        self.orders[order_id] = order
        self.order_status[order_id] = OrderStatus.PENDING
        self.fills[order_id] = []
        
        return order_id
    
    def submit_order(self, order_id: str) -> bool:
        """
        Submit order to broker via router
        
        Args:
            order_id: Order ID
            
        Returns:
            Success status
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        try:
            broker, broker_order_id = self.router.route(order)
            self.order_status[order_id] = OrderStatus.PLACED
            return True
        except Exception as e:
            print(f"Failed to submit order: {e}")
            self.order_status[order_id] = OrderStatus.REJECTED
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            Success status
        """
        if order_id not in self.orders:
            return False
        
        if self.order_status[order_id] in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order = self.orders[order_id]
        
        try:
            # Cancel via broker (simplified)
            self.order_status[order_id] = OrderStatus.CANCELLED
            return True
        except Exception as e:
            print(f"Failed to cancel order: {e}")
            return False
    
    def add_fill(self, order_id: str, fill: Fill) -> None:
        """
        Add a fill to an order
        
        Args:
            order_id: Order ID
            fill: Fill object
        """
        if order_id not in self.fills:
            self.fills[order_id] = []
        
        self.fills[order_id].append(fill)
        
        # Update order status
        total_filled = sum(f.quantity for f in self.fills[order_id])
        order_qty = self.orders[order_id].quantity
        
        if total_filled >= order_qty:
            self.order_status[order_id] = OrderStatus.FILLED
        elif total_filled > 0:
            self.order_status[order_id] = OrderStatus.PARTIALLY_FILLED
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        return self.order_status.get(order_id)
    
    def get_fills(self, order_id: str) -> List[Fill]:
        """Get fills for an order"""
        return self.fills.get(order_id, [])
    
    def get_all_orders(self) -> List[Order]:
        """Get all orders"""
        return list(self.orders.values())
    
    def get_pending_orders(self) -> List[Order]:
        """Get pending orders"""
        return [
            order for order_id, order in self.orders.items()
            if self.order_status.get(order_id) == OrderStatus.PENDING
        ]
    
    def get_open_orders(self) -> List[Order]:
        """Get open orders (placed but not filled)"""
        return [
            order for order_id, order in self.orders.items()
            if self.order_status.get(order_id) in [OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED]
        ]

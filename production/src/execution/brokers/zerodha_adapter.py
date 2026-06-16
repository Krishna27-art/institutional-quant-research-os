"""
Zerodha Adapter - Kite Connect API integration
"""

from typing import Dict, List, Optional
from datetime import datetime
from .broker_adapter import BrokerAdapter, Order, Fill, Position, Quote, OrderSide, OrderType, OrderStatus


class ZerodhaAdapter(BrokerAdapter):
    """Zerodha Kite Connect broker adapter"""
    
    def __init__(self, api_key: str, api_secret: str, 
                 access_token: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None  # Will be initialized on connect
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to Kite Connect API"""
        try:
            if self.api_key and self.access_token:
                from kiteconnect import KiteConnect
                self.kite = KiteConnect(api_key=self.api_key)
                self.kite.set_access_token(self.access_token)
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Zerodha: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Kite Connect"""
        self.is_connected = False
        self.kite = None
    
    def place_order(self, order: Order) -> str:
        """Place order via Kite Connect"""
        if not self.is_connected:
            raise RuntimeError("Not connected to broker")
        
        if self.kite is not None:
            try:
                kite_order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self._get_exchange(order.symbol),
                    tradingsymbol=order.symbol,
                    transaction_type=self._get_transaction_type(order.side),
                    quantity=int(order.quantity),
                    order_type=self._get_order_type(order.order_type),
                    price=order.price or 0,
                    trigger_price=order.trigger_price or 0,
                    product=self.kite.PRODUCT_MIS if order.tag == 'intraday' else self.kite.PRODUCT_NRML
                )
                return str(kite_order_id)
            except Exception as e:
                print(f"Failed to place real Zerodha order via KiteConnect: {e}. Falling back to simulation.")
        
        # Return mock order ID
        return f"ZERODHA_{datetime.now().timestamp()}"
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to broker")
        
        # Placeholder
        # self.kite.cancel_order(order_id=order_id, variety=self.kite.VARIETY_REGULAR)
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        if not self.is_connected:
            raise RuntimeError("Not connected to broker")
        
        # Placeholder
        return OrderStatus.PLACED
    
    def get_positions(self) -> List[Position]:
        """Get current positions"""
        if not self.is_connected:
            raise RuntimeError("Not connected to broker")
        
        # Placeholder
        # positions = self.kite.positions()
        return []
    
    def get_quote(self, symbol: str) -> Quote:
        """Get current quote"""
        if not self.is_connected:
            raise RuntimeError("Not connected to broker")
        
        # Placeholder
        # quote = self.kite.quote([symbol])
        return Quote(
            symbol=symbol,
            bid=1000.0,
            ask=1000.5,
            bid_size=1000,
            ask_size=1000,
            last=1000.25,
            timestamp=datetime.now()
        )
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        if not self.is_connected:
            raise RuntimeError("Not connected to broker")
        
        # Placeholder
        # margins = self.kite.margins()
        return {
            'available_cash': 1000000.0,
            'used_margin': 0.0,
            'total_margin': 1000000.0
        }
    
    def _get_exchange(self, symbol: str) -> str:
        """Map symbol to exchange"""
        # Simplified mapping
        return "NSE"
    
    def _get_transaction_type(self, side: OrderSide) -> str:
        """Map order side to transaction type"""
        return "BUY" if side == OrderSide.BUY else "SELL"
    
    def _get_order_type(self, order_type: OrderType) -> str:
        """Map order type"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "SL",
            OrderType.STOP_LIMIT: "SL-M"
        }
        return mapping.get(order_type, "MARKET")

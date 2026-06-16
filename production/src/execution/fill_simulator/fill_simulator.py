"""
Fill Simulator - Simulate order fills for paper trading
"""

import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from ..brokers.broker_adapter import Order, Fill, OrderSide, OrderType


class FillSimulator:
    """Simulate order fills with realistic slippage and market impact"""
    
    def __init__(self, slippage_bps: float = 5.0, 
                 market_impact_bps: float = 5.0,
                 bid_ask_spread_bps: float = 2.0):
        self.slippage_bps = slippage_bps
        self.market_impact_bps = market_impact_bps
        self.bid_ask_spread_bps = bid_ask_spread_bps
        
        self.pending_orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
    
    def simulate_fill(self, order: Order, market_state: Dict) -> Fill:
        """
        Simulate order fill
        
        Args:
            order: Order to fill
            market_state: Dict with current market state (price, volume, bid, ask)
            
        Returns:
            Fill object
        """
        base_price = market_state.get('price', 1000.0)
        volume = market_state.get('volume', 1000000)
        bid = market_state.get('bid', base_price - 0.5)
        ask = market_state.get('bid', base_price + 0.5)
        
        # Calculate fill price based on order type
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                fill_price = ask
            else:
                fill_price = bid
        elif order.order_type == OrderType.LIMIT:
            if order.price:
                fill_price = order.price
            else:
                fill_price = base_price
        else:
            fill_price = base_price
        
        # Apply slippage
        slippage = fill_price * self.slippage_bps / 10000
        if order.side == OrderSide.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage
        
        # Apply market impact based on order size
        participation_rate = order.quantity / volume if volume > 0 else 0
        market_impact = fill_price * self.market_impact_bps / 10000 * np.sqrt(participation_rate)
        
        if order.side == OrderSide.BUY:
            fill_price += market_impact
        else:
            fill_price -= market_impact
        
        # Calculate commission (simplified)
        commission = max(20.0, fill_price * order.quantity * 0.0001)
        
        # Create fill
        fill = Fill(
            order_id=order.order_id or f"SIM_{datetime.now().timestamp()}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=commission
        )
        
        self.fills.append(fill)
        return fill
    
    def simulate_partial_fill(self, order: Order, market_state: Dict, 
                            fill_fraction: float = 0.5) -> Fill:
        """
        Simulate partial fill
        
        Args:
            order: Order to fill
            market_state: Market state
            fill_fraction: Fraction of order to fill
            
        Returns:
            Fill object
        """
        partial_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity * fill_fraction,
            price=order.price,
            trigger_price=order.trigger_price,
            order_id=order.order_id,
            timestamp=order.timestamp,
            tag=order.tag
        )
        
        return self.simulate_fill(partial_order, market_state)
    
    def get_average_slippage(self, symbol: Optional[str] = None) -> float:
        """Get average slippage for fills"""
        if not self.fills:
            return 0.0
        
        fills = self.fills if symbol is None else [f for f in self.fills if f.symbol == symbol]
        
        if not fills:
            return 0.0
        
        # Simplified slippage calculation
        return self.slippage_bps
    
    def get_fill_rate(self) -> float:
        """Get fill rate (fills / orders)"""
        if not self.pending_orders:
            return 1.0
        
        return len(self.fills) / len(self.pending_orders)
    
    def reset(self) -> None:
        """Reset simulator state"""
        self.pending_orders.clear()
        self.fills.clear()

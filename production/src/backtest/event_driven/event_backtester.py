"""
Event-Driven Backtester - High-fidelity backtest with realistic costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class EventType(Enum):
    BAR = "bar"
    FILL = "fill"
    SIGNAL = "signal"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Event:
    """Base event class"""
    type: EventType
    time: datetime
    symbol: str
    data: Dict


@dataclass
class Order:
    """Order class"""
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    time: Optional[datetime] = None
    order_id: Optional[str] = None


@dataclass
class Fill:
    """Fill class"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    time: datetime
    commission: float = 0.0


class EventDrivenBacktester:
    """Event-driven backtester with realistic costs"""
    
    def __init__(self, initial_capital: float = 1_000_000,
                 slippage_bps: float = 5.0,
                 commission_per_order: float = 20.0,
                 commission_pct: float = 0.01):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_per_order = commission_per_order
        self.commission_pct = commission_pct
        
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        self.pnl_history: List[float] = []
    
    def run(self, data: pd.DataFrame, signal_generator: Callable) -> Dict:
        """
        Run event-driven backtest
        
        Args:
            data: OHLCV data with datetime index
            signal_generator: Function that generates signals
            
        Returns:
            Dict with backtest results
        """
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.fills = []
        self.pnl_history = []
        
        # Generate events
        events = self._generate_events(data, signal_generator)
        
        # Process events
        for event in events:
            if event.type == EventType.BAR:
                self._on_bar(event, data)
            elif event.type == EventType.FILL:
                self._on_fill(event)
            elif event.type == EventType.SIGNAL:
                self._on_signal(event)
        
        # Compute final metrics
        return self._compute_results()
    
    def _generate_events(self, data: pd.DataFrame, 
                        signal_generator: Callable) -> List[Event]:
        """Generate event stream"""
        events = []
        
        for idx, row in data.iterrows():
            # BAR event
            events.append(Event(
                type=EventType.BAR,
                time=idx,
                symbol='default',
                data=row.to_dict()
            ))
            
            # SIGNAL event (at open)
            signal = signal_generator(data.loc[:idx])
            if signal != 0:
                events.append(Event(
                    type=EventType.SIGNAL,
                    time=idx,
                    symbol='default',
                    data={'signal': signal}
                ))
        
        return events
    
    def _on_bar(self, event: Event, data: pd.DataFrame) -> None:
        """Handle BAR event"""
        # Update position values
        position_value = 0.0
        for symbol, qty in self.positions.items():
            if symbol in data.columns:
                price = data.loc[event.time, 'close']
                position_value += qty * price
        
        # Compute total portfolio value
        total_value = self.cash + position_value
        
        # Record PnL
        if len(self.pnl_history) > 0:
            pnl = (total_value - self.pnl_history[-1]) / self.pnl_history[-1]
        else:
            pnl = 0.0
        
        self.pnl_history.append(total_value)
    
    def _on_signal(self, event: Event) -> None:
        """Handle SIGNAL event"""
        signal = event.data['signal']
        
        if signal > 0:
            # Buy signal
            self._place_order(Order(
                symbol=event.symbol,
                side=OrderSide.BUY,
                quantity=abs(signal),
                time=event.time
            ))
        elif signal < 0:
            # Sell signal
            self._place_order(Order(
                symbol=event.symbol,
                side=OrderSide.SELL,
                quantity=abs(signal),
                time=event.time
            ))
    
    def _place_order(self, order: Order) -> None:
        """Place an order"""
        order.order_id = f"order_{len(self.orders)}"
        self.orders.append(order)
        
        # Simulate immediate fill (simplified)
        self._fill_order(order, order.data.get('close', 100.0) if hasattr(order, 'data') else 100.0)
    
    def _fill_order(self, order: Order, price: float) -> None:
        """Fill an order with slippage and commission"""
        # Apply slippage
        slippage = price * self.slippage_bps / 10000
        if order.side == OrderSide.BUY:
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        # Compute commission
        trade_value = order.quantity * fill_price
        commission = max(self.commission_per_order, trade_value * self.commission_pct / 100)
        
        # Update positions and cash
        if order.side == OrderSide.BUY:
            if self.cash >= trade_value + commission:
                self.cash -= trade_value + commission
                self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        else:
            if order.symbol in self.positions and self.positions[order.symbol] >= order.quantity:
                self.cash += trade_value - commission
                self.positions[order.symbol] -= order.quantity
        
        # Record fill
        self.fills.append(Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            time=order.time or datetime.now(),
            commission=commission
        ))
    
    def _on_fill(self, event: Event) -> None:
        """Handle FILL event"""
        # Fill processing is done in _fill_order
        pass
    
    def _compute_results(self) -> Dict:
        """Compute backtest results"""
        if not self.pnl_history:
            return {}
        
        pnl_series = pd.Series(self.pnl_history)
        returns = pnl_series.pct_change().dropna()
        
        # Total return
        total_return = (pnl_series.iloc[-1] / self.initial_capital) - 1
        
        # Sharpe
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = pnl_series / self.initial_capital
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': pnl_series.iloc[-1],
            'num_trades': len(self.fills),
            'total_commission': sum(f.commission for f in self.fills)
        }

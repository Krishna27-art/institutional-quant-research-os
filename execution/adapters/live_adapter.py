"""
Live Trading Adapter - Single execution path for live trading

This adapter provides a unified interface for live trading execution,
consolidating the previously separate live/ and live_trading/ modules.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import logging

from execution.cost_models import SmartOrderRouter, VenueQuote
from execution.risk_checks import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    """Configuration for live trading execution"""
    broker_api_key: str
    broker_api_secret: str
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.30
    order_timeout_seconds: int = 30
    enable_circuit_breaker: bool = True
    daily_loss_limit_pct: float = 0.03


@dataclass
class LiveResult:
    """Results from live trading execution"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]


class LiveAdapter:
    """
    Unified live trading execution adapter
    
    This consolidates:
    - live/broker_api.py
    - live/market_stream.py
    - live/server.py
    - live_trading/api_server.py
    
    Into a single execution path.
    """
    
    def __init__(self, config: LiveConfig):
        self.config = config
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict[str, Any]] = {}
        self.equity_curve: List[float] = []
        self.start_date = datetime.now()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.smart_order_router = SmartOrderRouter()
        self.circuit_breaker = CircuitBreaker(enabled=config.enable_circuit_breaker)
        
    def execute_order(self, symbol: str, quantity: int, direction: str, 
                     price: Optional[float] = None, order_type: str = 'MARKET',
                     routing_strategy: str = 'best_price', **kwargs) -> Dict[str, Any]:
        """
        Execute an order in live trading mode
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            direction: 'BUY' or 'SELL'
            price: Limit price (for limit orders)
            order_type: 'MARKET' or 'LIMIT'
            routing_strategy: Smart order routing strategy
            **kwargs: Additional parameters
            
        Returns:
            Trade execution details
        """
        # Circuit breaker check
        if self.circuit_breaker.is_active():
            raise Exception(f"Circuit breaker is active: {self.circuit_breaker.trigger_reason}")
        
        if self.config.enable_circuit_breaker:
            if self.daily_pnl < -self.config.daily_loss_limit_pct * self.get_portfolio_value():
                self.circuit_breaker.trigger("Daily loss limit exceeded")
                raise Exception(f"Circuit breaker triggered: Daily loss limit exceeded")
        
        # Validate order
        if not self.validate_order(symbol, quantity):
            raise Exception(f"Order validation failed for {symbol}")
        
        # Smart order routing
        routing_result = self.smart_order_router.route_order(
            side=direction,
            quantity=quantity,
            strategy=routing_strategy
        )
        
        # Execute order via broker API
        # This would integrate with actual broker API (Zerodha, etc.)
        trade_id = self._submit_to_broker(symbol, quantity, direction, price, order_type)
        
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'direction': direction,
            'price': price,
            'order_type': order_type,
            'timestamp': datetime.now(),
            'trade_id': trade_id,
            'mode': 'live',
            'routing_result': routing_result
        }
        self.trades[trade_id] = trade
        self.daily_trades += 1
        
        return trade
    
    def _submit_to_broker(self, symbol: str, quantity: int, direction: str, 
                         price: Optional[float], order_type: str) -> str:
        """
        Submit order to broker API
        
        CRITICAL FIX: Implement actual Zerodha Kite Connect integration.
        """
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise ImportError("kiteconnect not installed. Run: pip install kiteconnect")
        
        # Initialize Kite Connect
        kite = KiteConnect(api_key=self.config.broker_api_key)
        kite.set_access_token(self.config.broker_api_secret)
        
        # Map direction to Kite format
        kite_direction = direction.upper()
        if kite_direction not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Prepare order parameters
        order_params = {
            'exchange': 'NSE',
            'tradingsymbol': symbol,
            'transaction_type': kite_direction,
            'quantity': quantity,
            'order_type': order_type.upper(),
            'product': 'NRML'  # Normal order (can be changed to MIS for intraday)
        }
        
        # Add price for limit orders
        if order_type.upper() == 'LIMIT' and price is not None:
            order_params['price'] = price
        else:
            order_params['price'] = None  # Market order
        
        try:
            # Place order
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=order_params['exchange'],
                tradingsymbol=order_params['tradingsymbol'],
                transaction_type=order_params['transaction_type'],
                quantity=order_params['quantity'],
                order_type=order_params['order_type'],
                price=order_params['price'],
                product=order_params['product'],
                validity='DAY'
            )
            
            logger.info(f"Order submitted to Kite: {order_id}")
            return str(order_id)
            
        except Exception as e:
            logger.error(f"Failed to submit order to Kite: {e}")
            raise Exception(f"Broker API error: {e}")
    
    def validate_order(self, symbol: str, quantity: int) -> bool:
        """
        Validate order before execution
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            
        Returns:
            True if order is valid
        """
        # Check position limits
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        portfolio_value = self.get_portfolio_value()
        
        if portfolio_value > 0:
            position_value = abs(new_position) * self._get_current_price(symbol)
            position_pct = position_value / portfolio_value
            if position_pct > self.config.max_position_pct:
                return False
                
        return True
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol (placeholder)"""
        # Placeholder: Would fetch from market data
        return 100.0
    
    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        position_value = sum(
            self.positions[symbol] * self._get_current_price(symbol)
            for symbol in self.positions
        )
        return position_value
    
    def update_venue_quote(self, quote: VenueQuote) -> None:
        """Update venue quote for smart order routing"""
        self.smart_order_router.update_quote(quote)
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """Update position after fill"""
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
    
    def trigger_circuit_breaker(self, reason: str) -> None:
        """Manually trigger circuit breaker"""
        self.circuit_breaker.trigger(reason)
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker"""
        self.circuit_breaker.reset()
        
    def get_performance_metrics(self) -> LiveResult:
        """
        Get current performance metrics
        
        Returns:
            LiveResult with performance metrics
        """
        # Calculate returns based on actual fills
        # This would be calculated from actual trade data
        total_return = 0.0  # Placeholder
        sharpe_ratio = 0.0  # Placeholder
        max_drawdown = 0.0  # Placeholder
        win_rate = 0.0  # Placeholder
        avg_trade_return = 0.0  # Placeholder
        
        return LiveResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            equity_curve=pd.Series(self.equity_curve),
            trades=list(self.trades.values())
        )

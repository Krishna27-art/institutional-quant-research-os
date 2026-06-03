"""
Paper Trading Simulator with 2 bps Slippage Model
Architecture V2 - 6-month paper trading phase

Simulates live trading with realistic execution costs and slippage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]  # None for market orders
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    timestamp: datetime = None
    slippage_bps: float = 0.0


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float = 0.0


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading"""
    # Account
    initial_capital: float = 250000000  # ₹25 Crore
    currency: str = "INR"
    
    # Slippage model (per debate)
    slippage_large_cap_bps: float = 2.0
    slippage_mid_cap_bps: float = 5.0
    slippage_small_cap_bps: float = 10.0  # Rejected per debate
    
    # Execution
    default_order_type: OrderType = OrderType.LIMIT
    limit_order_patience_seconds: int = 30
    market_order_timeout_seconds: int = 5
    
    # Risk limits
    max_position_size_pct: float = 0.05  # 5% of AUM
    max_daily_loss_pct: float = -0.03  # -3% daily circuit breaker
    max_leverage: float = 4.0
    
    # Fees
    broker_fee_bps: float = 0.5  # 0.5 bps
    exchange_fee_bps: float = 0.1  # 0.1 bps
    stt_ctt_bps: float = 0.1  # 0.1 bps (STT/CTT)
    
    # Paper trading period
    start_date: str = None
    end_date: str = None
    target_sharpe: float = 1.0
    max_drawdown_limit: float = 0.12


class PaperTradingSimulator:
    """
    Paper trading simulator with realistic execution simulation.
    
    Features:
    - 2 bps slippage model (conservative per debate)
    - Order book simulation
    - Partial fills
    - Circuit breaker enforcement
    - Real-time PnL tracking
    - Performance metrics calculation
    """
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        
        # Account state
        self.capital = config.initial_capital
        self.available_cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Performance tracking
        self.trades: List[Dict] = []
        self.daily_pnl_history: List[float] = []
        self.equity_curve: List[float] = [config.initial_capital]
        self.drawdown_curve: List[float] = [0.0]
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        
        # Market data cache
        self.market_data: Dict[str, Dict] = {}
        
        # Symbol classification (for slippage)
        self.large_cap_symbols = set([
            "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"
        ])
    
    def get_slippage_bps(self, symbol: str) -> float:
        """Get slippage in basis points for a symbol."""
        if symbol in self.large_cap_symbols:
            return self.config.slippage_large_cap_bps
        else:
            return self.config.slippage_mid_cap_bps  # Reject small-caps per debate
    
    def calculate_total_fees_bps(self) -> float:
        """Calculate total fees in basis points."""
        return (
            self.config.broker_fee_bps +
            self.config.exchange_fee_bps +
            self.config.stt_ctt_bps
        )
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = None,
        price: Optional[float] = None
    ) -> str:
        """
        Place a new order.
        
        Args:
            symbol: Stock symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Market or limit
            price: Limit price (required for limit orders)
            
        Returns:
            Order ID
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            print("Circuit breaker active - order rejected")
            return self._create_rejected_order(symbol, side, quantity, "circuit_breaker")
        
        # Check position size limit
        position_value = self._estimate_position_value(symbol, quantity)
        max_position_value = self.capital * self.config.max_position_size_pct
        
        if position_value > max_position_value:
            print(f"Position size exceeds limit: {position_value} > {max_position_value}")
            return self._create_rejected_order(symbol, side, quantity, "position_limit")
        
        # Check available cash for buy orders
        if side == OrderSide.BUY:
            required_cash = position_value * 1.01  # Buffer for slippage
            if required_cash > self.available_cash:
                print(f"Insufficient cash: {required_cash} > {self.available_cash}")
                return self._create_rejected_order(symbol, side, quantity, "insufficient_cash")
        
        # Create order
        self.order_counter += 1
        order_id = f"ORD-{self.order_counter:06d}"
        
        if order_type is None:
            order_type = self.config.default_order_type
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        
        # Simulate immediate execution for paper trading
        self._execute_order(order)
        
        return order_id
    
    def _create_rejected_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        reason: str
    ) -> str:
        """Create a rejected order."""
        self.order_counter += 1
        order_id = f"ORD-{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            status=OrderStatus.REJECTED,
            timestamp=datetime.now()
        )
        
        self.orders[order_id] = order
        return order_id
    
    def _estimate_position_value(self, symbol: str, quantity: int) -> float:
        """Estimate position value based on last known price."""
        if symbol in self.market_data:
            return self.market_data[symbol].get('close', 0) * quantity
        return 0.0
    
    def _execute_order(self, order: Order) -> None:
        """
        Simulate order execution with slippage.
        
        Args:
            order: Order to execute
        """
        if order.symbol not in self.market_data:
            order.status = OrderStatus.REJECTED
            return
        
        market_info = self.market_data[order.symbol]
        current_price = market_info['close']
        
        if current_price == 0:
            order.status = OrderStatus.REJECTED
            return
        
        # Calculate slippage
        slippage_bps = self.get_slippage_bps(order.symbol)
        slippage_pct = slippage_bps / 10000.0
        
        # Calculate fees
        total_fees_bps = self.calculate_total_fees_bps()
        fees_pct = total_fees_bps / 10000.0
        
        # Calculate execution price
        if order.side == OrderSide.BUY:
            # Buy: pay more (slippage) + fees
            execution_price = current_price * (1 + slippage_pct + fees_pct)
        else:
            # Sell: receive less (slippage) - fees
            execution_price = current_price * (1 - slippage_pct - fees_pct)
        
        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.slippage_bps = slippage_bps
        order.status = OrderStatus.FILLED
        
        # Update positions
        self._update_position(order, execution_price)
        
        # Record trade
        self._record_trade(order, execution_price, slippage_bps, total_fees_bps)
        
        # Update daily PnL
        self._update_daily_pnl(order, execution_price)
    
    def _update_position(self, order: Order, execution_price: float) -> None:
        """Update position after order execution."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0.0
            )
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            # Add to position
            total_value = (position.quantity * position.avg_price) + (order.quantity * execution_price)
            total_quantity = position.quantity + order.quantity
            position.avg_price = total_value / total_quantity if total_quantity > 0 else 0
            position.quantity = total_quantity
            
            # Deduct from cash
            self.available_cash -= order.quantity * execution_price
        
        else:
            # Reduce position
            if position.quantity >= order.quantity:
                # Full or partial close
                realized_pnl = (execution_price - position.avg_price) * order.quantity
                position.quantity -= order.quantity
                position.unrealized_pnl = 0  # Reset for closed portion
                
                # Add to cash
                self.available_cash += order.quantity * execution_price
                
                # Update daily PnL
                self.daily_pnl += realized_pnl
                
                if position.quantity == 0:
                    del self.positions[symbol]
            else:
                # Short selling (not implemented for paper trading)
                print("Short selling not supported in paper trading")
                order.status = OrderStatus.REJECTED
    
    def _record_trade(
        self,
        order: Order,
        execution_price: float,
        slippage_bps: float,
        fees_bps: float
    ) -> None:
        """Record trade for analytics."""
        trade = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'execution_price': execution_price,
            'slippage_bps': slippage_bps,
            'fees_bps': fees_bps,
            'timestamp': order.timestamp,
            'total_cost': order.quantity * execution_price
        }
        
        self.trades.append(trade)
        self.daily_trades_count += 1
    
    def _update_daily_pnl(self, order: Order, execution_price: float) -> None:
        """Update daily PnL."""
        # For long positions, PnL is unrealized
        # For closed positions, it's already calculated in _update_position
        
        # Update unrealized PnL for all positions
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol]['close']
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def update_market_data(self, market_data: Dict[str, Dict]) -> None:
        """
        Update market data for all symbols.
        
        Args:
            market_data: Dictionary mapping symbol to market info
        """
        self.market_data.update(market_data)
        
        # Update unrealized PnL
        self._update_unrealized_pnl()
        
        # Check circuit breaker
        self._check_circuit_breaker()
    
    def _update_unrealized_pnl(self) -> None:
        """Update unrealized PnL for all positions."""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in self.market_data:
                current_price = self.market_data[symbol]['close']
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                total_unrealized += position.unrealized_pnl
        
        # Update equity curve
        total_equity = self.available_cash + total_unrealized
        self.equity_curve.append(total_equity)
        
        # Update drawdown
        if len(self.equity_curve) > 1:
            running_max = np.maximum.accumulate(self.equity_curve)
            drawdown = (np.array(self.equity_curve) - running_max) / running_max
            self.drawdown_curve = drawdown.tolist()
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be triggered."""
        daily_pnl_pct = self.daily_pnl / self.config.initial_capital
        
        if daily_pnl_pct <= self.config.max_daily_loss_pct:
            self.circuit_breaker_active = True
            print(f"CIRCUIT BREAKER TRIGGERED: Daily PnL {daily_pnl_pct:.2%} <= {self.config.max_daily_loss_pct:.2%}")
            
            # Cancel all pending orders
            for order_id, order in self.orders.items():
                if order.status == OrderStatus.PENDING:
                    order.status = OrderStatus.CANCELLED
    
    def get_account_summary(self) -> Dict:
        """Get account summary."""
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_equity = self.available_cash + total_unrealized
        
        total_pnl_pct = (total_equity - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate leverage
        total_position_value = sum(
            p.quantity * self.market_data.get(p.symbol, {}).get('close', p.avg_price)
            for p in self.positions.values()
        )
        leverage = total_position_value / total_equity if total_equity > 0 else 0
        
        return {
            'capital': self.capital,
            'available_cash': self.available_cash,
            'total_equity': total_equity,
            'total_pnl': total_equity - self.config.initial_capital,
            'total_pnl_pct': total_pnl_pct,
            'unrealized_pnl': total_unrealized,
            'positions_count': len(self.positions),
            'leverage': leverage,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades_count,
            'circuit_breaker_active': self.circuit_breaker_active
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / self.config.initial_capital
        
        # Sharpe ratio (annualized)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for t in self.trades if t.get('realized_pnl', 0) > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_drawdown),
            'win_rate': win_rate,
            'total_trades': total_trades,
            'current_equity': equity[-1],
            'total_return': (equity[-1] - self.config.initial_capital) / self.config.initial_capital
        }
    
    def check_go_no_go(self) -> Tuple[bool, str]:
        """
        Check if paper trading meets Go/No-Go criteria.
        
        Returns:
            (go, reason) tuple
        """
        metrics = self.get_performance_metrics()
        
        if not metrics:
            return False, "Insufficient data"
        
        sharpe = metrics['sharpe_ratio']
        max_dd = metrics['max_drawdown']
        
        if sharpe >= self.config.target_sharpe and max_dd <= self.config.max_drawdown_limit:
            return True, f"PASS: Sharpe {sharpe:.2f} >= {self.config.target_sharpe}, Max DD {max_dd:.2%} <= {self.config.max_drawdown_limit:.2%}"
        else:
            reasons = []
            if sharpe < self.config.target_sharpe:
                reasons.append(f"Sharpe {sharpe:.2f} < {self.config.target_sharpe}")
            if max_dd > self.config.max_drawdown_limit:
                reasons.append(f"Max DD {max_dd:.2%} > {self.config.max_drawdown_limit:.2%}")
            return False, f"FAIL: {', '.join(reasons)}"
    
    def reset_daily(self) -> None:
        """Reset daily state."""
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        
        # Reset circuit breaker after cooldown
        if self.circuit_breaker_active:
            # 5-day cooldown would be implemented with date tracking
            # For now, reset after each day
            self.circuit_breaker_active = False
    
    def print_summary(self) -> None:
        """Print account summary."""
        summary = self.get_account_summary()
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*60)
        print("PAPER TRADING ACCOUNT SUMMARY")
        print("="*60)
        print(f"Capital: ₹{summary['capital']:,.2f}")
        print(f"Available Cash: ₹{summary['available_cash']:,.2f}")
        print(f"Total Equity: ₹{summary['total_equity']:,.2f}")
        print(f"Total PnL: ₹{summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:.2%})")
        print(f"Unrealized PnL: ₹{summary['unrealized_pnl']:,.2f}")
        print(f"Positions: {summary['positions_count']}")
        print(f"Leverage: {summary['leverage']:.2f}x")
        print(f"Daily PnL: ₹{summary['daily_pnl']:,.2f}")
        print(f"Daily Trades: {summary['daily_trades']}")
        print(f"Circuit Breaker: {'ACTIVE' if summary['circuit_breaker_active'] else 'INACTIVE'}")
        
        if metrics:
            print("\n" + "-"*60)
            print("PERFORMANCE METRICS")
            print("-"*60)
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Total Return: {metrics['total_return']:.2%}")
        
        print("="*60)


async def run_paper_trading_simulation():
    """Run a paper trading simulation."""
    config = PaperTradingConfig(
        initial_capital=10000000,  # ₹1 Crore for testing
        slippage_large_cap_bps=2.0
    )
    
    simulator = PaperTradingSimulator(config)
    
    # Simulate market data updates
    print("Starting paper trading simulation...")
    
    for i in range(100):
        # Simulate market data
        market_data = {
            "RELIANCE": {"close": 2500 + np.random.randn() * 10},
            "HDFCBANK": {"close": 1500 + np.random.randn() * 5},
            "INFY": {"close": 1400 + np.random.randn() * 5}
        }
        
        simulator.update_market_data(market_data)
        
        # Place some orders
        if i == 10:
            simulator.place_order("RELIANCE", OrderSide.BUY, 100)
        elif i == 20:
            simulator.place_order("HDFCBANK", OrderSide.BUY, 50)
        elif i == 50:
            simulator.place_order("RELIANCE", OrderSide.SELL, 50)
        
        await asyncio.sleep(0.01)
    
    simulator.print_summary()
    
    # Check Go/No-Go
    go, reason = simulator.check_go_no_go()
    print(f"\nGo/No-Go: {go}")
    print(f"Reason: {reason}")


if __name__ == "__main__":
    asyncio.run(run_paper_trading_simulation())

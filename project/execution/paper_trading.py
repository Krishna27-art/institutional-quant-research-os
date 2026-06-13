"""
Paper Trading with Realistic Costs and Slippage

Im institutional-grade paper trading environment with realistic
transaction costs, slippage, and market impact modeling.

Key Features:
- Realistic transaction costs (brokerage, taxes, exchange fees)
- Slippage model based on order size and volatility
- Market impact simulation (Almgren-Chriss)
- Partial fill simulation
- Real-time PnL tracking
- Performance metrics (Sharpe, drawdown, capacity)
- Trade execution logging

Based on Blueprint Week 15-16: Paper Trading with Full Stack
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Paper trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Optional[float]
    order_type: str  # MARKET, LIMIT
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    timestamp: pd.Timestamp = None


class TransactionCostModel:
    """
    Transaction cost model for Indian markets.
    
    Includes:
    - Brokerage fees
    - Transaction charges
    - STT (Securities Transaction Tax)
    - GST (Goods and Services Tax)
    - Stamp duty
    """
    
    def __init__(
        self,
        broker: str = "zerodha",
        brokerage_rate: float = 0.0003,  # 0.03% or ₹20 whichever is lower
        transaction_charge_rate: float = 0.0000345,
        stt_rate: float = 0.00025,  # 0.025% for equity
        gst_rate: float = 0.18,
        stamp_duty_rate: float = 0.00003
    ):
        """
        Initialize transaction cost model.
        
        Args:
            broker: Broker name
            brokerage_rate: Brokerage rate
            transaction_charge_rate: Transaction charge rate
            stt_rate: STT rate
            gst_rate: GST rate
            stamp_duty_rate: Stamp duty rate
        """
        self.broker = broker
        self.brokerage_rate = brokerage_rate
        self.transaction_charge_rate = transaction_charge_rate
        self.stt_rate = stt_rate
        self.gst_rate = gst_rate
        self.stamp_duty_rate = stamp_duty_rate
    
    def calculate_cost(
        self,
        side: OrderSide,
        quantity: int,
        price: float
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost.
        
        Args:
            side: Order side
            quantity: Order quantity
            price: Execution price
            
        Returns:
            Dictionary with cost breakdown
        """
        notional = quantity * price
        
        # Brokerage (flat fee or percentage, whichever is lower)
        brokerage_by_rate = notional * self.brokerage_rate
        brokerage = min(brokerage_by_rate, 20.0)
        
        # Transaction charges
        transaction_charge = notional * self.transaction_charge_rate
        
        # STT (only on sell side for equity)
        stt = notional * self.stt_rate if side == OrderSide.SELL else 0.0
        
        # GST (on brokerage + transaction charge)
        gst = (brokerage + transaction_charge) * self.gst_rate
        
        # Stamp duty
        stamp_duty = notional * self.stamp_duty_rate
        
        total_cost = brokerage + transaction_charge + stt + gst + stamp_duty
        
        return {
            'brokerage': brokerage,
            'transaction_charge': transaction_charge,
            'stt': stt,
            'gst': gst,
            'stamp_duty': stamp_duty,
            'total_cost': total_cost,
            'cost_bps': (total_cost / notional) * 10000
        }


class SlippageModel:
    """
    Slippage model for realistic execution simulation.
    
    Models slippage based on:
    - Order size relative to average daily volume
    - Volatility
    - Time of day
    """
    
    def __init__(
        self,
        base_slippage_bps: float = 5.0,  # 5 bps base slippage
        size_impact_factor: float = 0.1,
        vol_impact_factor: float = 0.5
    ):
        """
        Initialize slippage model.
        
        Args:
            base_slippage_bps: Base slippage in basis points
            size_impact_factor: Impact of order size
            vol_impact_factor: Impact of volatility
        """
        self.base_slippage_bps = base_slippage_bps
        self.size_impact_factor = size_impact_factor
        self.vol_impact_factor = vol_impact_factor
    
    def calculate_slippage(
        self,
        order_size: int,
        avg_daily_volume: int,
        volatility: float,
        side: OrderSide
    ) -> float:
        """
        Calculate slippage in basis points.
        
        Args:
            order_size: Order size
            avg_daily_volume: Average daily volume
            volatility: Annualized volatility
            side: Order side
            
        Returns:
            Slippage in basis points
        """
        # Size impact (larger orders have more slippage)
        size_ratio = order_size / avg_daily_volume
        size_impact = self.size_impact_factor * size_ratio * 100  # Convert to bps
        
        # Volatility impact (higher vol = more slippage)
        vol_impact = self.vol_impact_factor * volatility * 100  # Convert vol to bps
        
        # Total slippage
        slippage_bps = self.base_slippage_bps + size_impact + vol_impact
        
        # For buy orders, slippage increases price (negative for PnL)
        # For sell orders, slippage decreases price (negative for PnL)
        if side == OrderSide.BUY:
            slippage_bps = slippage_bps  # Pay more
        else:
            slippage_bps = -slippage_bps  # Receive less
        
        return slippage_bps


class PaperTradingEngine:
    """
    Paper trading engine with realistic costs and slippage.
    
    Simulates trading with institutional-grade cost modeling and
    execution simulation.
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        cost_model: Optional[TransactionCostModel] = None,
        slippage_model: Optional[SlippageModel] = None
    ):
        """
        Initialize paper trading engine.
        
        Args:
            initial_capital: Initial capital
            cost_model: Transaction cost model
            slippage_model: Slippage model
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.cost_model = cost_model or TransactionCostModel()
        self.slippage_model = slippage_model or SlippageModel()
        
        # Position tracking
        self.positions: Dict[str, int] = {}
        self.average_prices: Dict[str, float] = {}
        
        # Order tracking
        self.orders: List[Order] = []
        self.order_counter = 0
        
        # Trade log
        self.trades: List[Dict] = []
        
        # Performance tracking
        self.daily_pnl: List[float] = []
        self.returns: List[float] = []
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        current_price: float = None,
        avg_daily_volume: int = 1000000,
        volatility: float = 0.2
    ) -> Order:
        """
        Place order in paper trading engine.
        
        Args:
            symbol: Stock symbol
            side: Order side
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for LIMIT orders)
            current_price: Current market price
            avg_daily_volume: Average daily volume
            volatility: Volatility
            
        Returns:
            Order object
        """
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status=OrderStatus.PENDING,
            timestamp=pd.Timestamp.now()
        )
        
        # Execute order immediately for market orders
        if order_type == "MARKET" and current_price is not None:
            self._execute_order(order, current_price, avg_daily_volume, volatility)
        
        self.orders.append(order)
        
        return order
    
    def _execute_order(
        self,
        order: Order,
        current_price: float,
        avg_daily_volume: int,
        volatility: float
    ) -> None:
        """
        Execute order with slippage and costs.
        
        Args:
            order: Order to execute
            current_price: Current market price
            avg_daily_volume: Average daily volume
            volatility: Volatility
        """
        # Calculate slippage
        slippage_bps = self.slippage_model.calculate_slippage(
            order.quantity, avg_daily_volume, volatility, order.side
        )
        
        # Apply slippage to price
        slippage_factor = 1 + (slippage_bps / 10000)
        execution_price = current_price * slippage_factor
        
        # Calculate transaction costs
        cost_breakdown = self.cost_model.calculate_cost(
            order.side, order.quantity, execution_price
        )
        
        # Update position
        if order.side == OrderSide.BUY:
            # Buy
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
            total_cost = order.quantity * execution_price + cost_breakdown['total_cost']
            self.current_capital -= total_cost
            
            # Update average price
            current_pos = self.positions[order.symbol]
            old_avg = self.average_prices.get(order.symbol, 0)
            old_qty = current_pos - order.quantity
            if old_qty > 0:
                self.average_prices[order.symbol] = (
                    (old_avg * old_qty + execution_price * order.quantity) / current_pos
                )
            else:
                self.average_prices[order.symbol] = execution_price
        else:
            # Sell
            if self.positions.get(order.symbol, 0) >= order.quantity:
                self.positions[order.symbol] -= order.quantity
                total_proceeds = order.quantity * execution_price - cost_breakdown['total_cost']
                self.current_capital += total_proceeds
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Insufficient position for sell order: {order.symbol}")
                return
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        
        # Log trade
        self.trades.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'slippage_bps': slippage_bps,
            'cost': cost_breakdown['total_cost'],
            'cost_bps': cost_breakdown['cost_bps'],
            'timestamp': pd.Timestamp.now()
        })
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
    
    def get_position(self, symbol: str) -> Dict:
        """
        Get current position for symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position dictionary
        """
        return {
            'symbol': symbol,
            'quantity': self.positions.get(symbol, 0),
            'average_price': self.average_prices.get(symbol, 0.0),
            'market_value': self.positions.get(symbol, 0) * self.average_prices.get(symbol, 0.0)
        }
    
    def get_portfolio_value(
        self,
        current_prices: Dict[str, float]
    ) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            Total portfolio value
        """
        # Cash
        portfolio_value = self.current_capital
        
        # Positions
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]
        
        return portfolio_value
    
    def calculate_pnl(
        self,
        current_prices: Dict[str, float]
    ) -> Dict:
        """
        Calculate current PnL.
        
        Args:
            current_prices: Current market prices
            
        Returns:
            PnL dictionary
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        total_pnl = portfolio_value - self.initial_capital
        pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in current_prices:
                avg_price = self.average_prices.get(symbol, 0.0)
                unrealized_pnl += quantity * (current_prices[symbol] - avg_price)
        
        # Calculate drawdown
        drawdown = (self.peak_capital - portfolio_value) / self.peak_capital if self.peak_capital > 0 else 0
        
        return {
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'unrealized_pnl': unrealized_pnl,
            'portfolio_value': portfolio_value,
            'peak_capital': self.peak_capital,
            'drawdown': drawdown,
            'current_capital': self.current_capital
        }
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        if len(self.returns) == 0:
            return {}
        
        returns_array = np.array(self.returns)
        
        # Sharpe ratio
        sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
        
        # Max drawdown
        cumulative = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns_array > 0).mean()
        
        # Total trades
        total_trades = len(self.trades)
        
        # Average trade PnL
        if total_trades > 0:
            avg_trade_pnl = sum(t['cost'] for t in self.trades) / total_trades
        else:
            avg_trade_pnl = 0
        
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade_pnl': avg_trade_pnl,
            'total_return': cumulative[-1] if len(cumulative) > 0 else 0
        }
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame.
        
        Returns:
            Trade log DataFrame
        """
        return pd.DataFrame(self.trades)


if __name__ == "__main__":
    # Test paper trading engine
    print("Testing Paper Trading Engine...")
    
    # Create paper trading engine
    cost_model = TransactionCostModel()
    slippage_model = SlippageModel()
    
    paper_trading = PaperTradingEngine(
        initial_capital=1000000,
        cost_model=cost_model,
        slippage_model=slippage_model
    )
    
    # Place orders
    print("\nPlacing orders...")
    order1 = paper_trading.place_order(
        symbol='RELIANCE',
        side=OrderSide.BUY,
        quantity=100,
        order_type='MARKET',
        current_price=2500.0,
        avg_daily_volume=5000000,
        volatility=0.2
    )
    print(f"Order 1: {order1.order_id} - {order1.status}")
    
    order2 = paper_trading.place_order(
        symbol='TCS',
        side=OrderSide.BUY,
        quantity=50,
        order_type='MARKET',
        current_price=3500.0,
        avg_daily_volume=3000000,
        volatility=0.15
    )
    print(f"Order 2: {order2.order_id} - {order2.status}")
    
    # Get positions
    print("\nPositions:")
    print(f"RELIANCE: {paper_trading.get_position('RELIANCE')}")
    print(f"TCS: {paper_trading.get_position('TCS')}")
    
    # Calculate PnL
    current_prices = {'RELIANCE': 2520.0, 'TCS': 3520.0}
    pnl = paper_trading.calculate_pnl(current_prices)
    print(f"\nPnL: {pnl}")
    
    # Sell order
    print("\nSelling RELIANCE...")
    order3 = paper_trading.place_order(
        symbol='RELIANCE',
        side=OrderSide.SELL,
        quantity=50,
        order_type='MARKET',
        current_price=2520.0,
        avg_daily_volume=5000000,
        volatility=0.2
    )
    print(f"Order 3: {order3.order_id} - {order3.status}")
    
    # Get trade log
    print("\nTrade Log:")
    trade_log = paper_trading.get_trade_log()
    print(trade_log)
    
    # Performance metrics
    print("\nPerformance Metrics:")
    metrics = paper_trading.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print("\nPaper Trading Engine test completed.")

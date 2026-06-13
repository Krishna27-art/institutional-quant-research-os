"""
Backtest Adapter - Single execution path for backtesting

This adapter provides a unified interface for backtesting execution,
consolidating the previously separate backtest/ and backtesting/ modules.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

from execution.cost_models import MarketImpactModel, TransactionCostModel

# Import trade logger
try:
    from portfolio.trade_logger import get_trade_logger, Trade, TradeSide, TradeStatus
    trade_logger = get_trade_logger()
except Exception:
    trade_logger = None

# Import model registry
try:
    from models.model_registry import get_model_registry, ModelType, ModelStage
    model_registry = get_model_registry()
except Exception:
    model_registry = None


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.0005
    slippage_bps: float = 2.0
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.30
    use_market_impact: bool = True
    enable_smart_routing: bool = False


@dataclass
class BacktestResult:
    """Results from backtest execution"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_market_impact: float = 0.0


class BacktestAdapter:
    """
    Unified backtest execution adapter
    
    This consolidates:
    - backtest/backtester.py
    - backtesting/ (strategy-specific backtests)
    
    Into a single execution path.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.cost_model = TransactionCostModel(
            commission_rate=config.commission_rate,
            slippage_bps=config.slippage_bps
        )
        self.market_data_cache: Dict[str, Dict] = {}
        
    def execute_order(self, symbol: str, quantity: int, price: float, 
                     direction: str, timestamp: datetime, cost_estimate: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute an order in backtest mode
        
        CRITICAL FIX: Add realistic fill simulation with partial fills and time-based execution.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Execution price
            direction: 'BUY' or 'SELL'
            timestamp: Order timestamp
            cost_estimate: Pre-calculated cost estimate
            **kwargs: Additional parameters
            
        Returns:
            Trade execution details
        """
        # CRITICAL FIX: Simulate realistic fill behavior
        # Large orders may not fill completely at once
        fill_quantity = self._simulate_fill(symbol, quantity, price, direction, timestamp)
        
        if fill_quantity == 0:
            # Order did not fill
            return {
                'symbol': symbol,
                'quantity': 0,
                'price': price,
                'direction': direction,
                'timestamp': timestamp,
                'filled': False,
                'reason': 'No liquidity'
            }
        
        # Use cost estimate if provided, otherwise calculate
        if cost_estimate:
            commission = cost_estimate['commission'] * (fill_quantity / quantity)
            slippage = cost_estimate['slippage'] * (fill_quantity / quantity)
            market_impact = cost_estimate.get('market_impact', 0) * (fill_quantity / quantity)
            total_cost = cost_estimate['total_cost'] * (fill_quantity / quantity)
        else:
            # Calculate transaction costs with market impact if enabled
            market_data = self.market_data_cache.get(symbol, {})
            cost_estimate = self.cost_model.estimate_cost(
                symbol=symbol,
                quantity=fill_quantity,
                price=price,
                direction=direction,
                market_data=market_data
            )
            commission = cost_estimate['commission']
            slippage = cost_estimate['slippage']
            market_impact = cost_estimate['market_impact']
            total_cost = cost_estimate['total_cost']
        
        # Execute trade
        if direction == 'BUY':
            cost = fill_quantity * price + total_cost
            self.current_capital -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + fill_quantity
        else:  # SELL
            proceeds = fill_quantity * price - total_cost
            self.current_capital += proceeds
            self.positions[symbol] = self.positions.get(symbol, 0) - fill_quantity
            
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': fill_quantity,
            'requested_quantity': quantity,
            'fill_rate': fill_quantity / quantity if quantity > 0 else 0,
            'price': price,
            'direction': direction,
            'timestamp': timestamp,
            'commission': commission,
            'slippage': slippage,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'capital_after': self.current_capital,
            'cost_breakdown': cost_estimate if cost_estimate else None,
            'filled': True
        }
        
        # Log trade to trade logger
        if trade_logger:
            try:
                from portfolio.trade_logger import TradeSide, TradeStatus
                trade_record = Trade(
                    trade_id=f"{symbol}_{timestamp.timestamp()}",
                    symbol=symbol,
                    side=TradeSide.BUY if direction == 'BUY' else TradeSide.SELL,
                    quantity=int(fill_quantity),
                    entry_price=price,
                    entry_time=timestamp,
                    status=TradeStatus.FILLED,
                    commission=commission,
                    slippage_bps=slippage / price * 10000 if price > 0 else 0,
                    metadata={
                        'fill_rate': fill_quantity / quantity if quantity > 0 else 0,
                        'market_impact': market_impact
                    }
                )
                trade_logger.log_trade(trade_record)
            except Exception as e:
                pass  # Don't fail if logging fails
        
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        return trade
    
    def _simulate_fill(self, symbol: str, quantity: int, price: float, 
                      direction: str, timestamp: datetime) -> int:
        """
        Simulate realistic fill behavior
        
        CRITICAL FIX: Add fill simulation based on order size and market conditions.
        """
        # Get market data for this symbol
        market_data = self.market_data_cache.get(symbol, {})
        adv = market_data.get('adv', 1_000_000)  # Average daily volume
        
        # Calculate order size as percentage of ADV
        order_pct_adv = quantity / adv if adv > 0 else 0
        
        # Fill rate decreases with larger orders
        # Small orders (<1% ADV) fill 100%
        # Medium orders (1-5% ADV) fill 80-95%
        # Large orders (>5% ADV) fill 50-80%
        if order_pct_adv < 0.01:
            fill_rate = 1.0
        elif order_pct_adv < 0.05:
            fill_rate = 0.95 - (order_pct_adv - 0.01) * 2.5  # 0.95 to 0.85
        else:
            fill_rate = max(0.5, 0.85 - (order_pct_adv - 0.05) * 7)  # 0.85 to 0.5
        
        # Add randomness to fill simulation
        import random
        fill_rate = min(1.0, max(0.0, fill_rate + random.uniform(-0.05, 0.05)))
        
        fill_quantity = int(quantity * fill_rate)
        return fill_quantity
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        position_value = sum(
            self.positions[symbol] * prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.current_capital + position_value
    
    def run_backtest(self, signals: pd.DataFrame, price_data: pd.DataFrame,
                   market_data: Optional[Dict[str, Dict]] = None) -> BacktestResult:
        """
        Run complete backtest
        
        Args:
            signals: DataFrame with trading signals
            price_data: DataFrame with price data
            market_data: Optional dict with ADV, volatility, etc. for each symbol
            
        Returns:
            BacktestResult with performance metrics
        """
        # Cache market data for cost modeling
        if market_data:
            self.market_data_cache = market_data
        
        # Filter signals by date range
        signals = signals[
            (signals.index >= self.config.start_date) & 
            (signals.index <= self.config.end_date)
        ]
        
        # Execute signals
        for timestamp, signal in signals.iterrows():
            symbol = signal.get('symbol')
            direction = signal.get('direction')
            quantity = signal.get('quantity', 0)
            
            if symbol and direction and quantity > 0:
                price = price_data.loc[timestamp, symbol]
                self.execute_order(symbol, quantity, price, direction, timestamp)
        
        # Calculate results
        final_value = self.current_capital
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate Sharpe ratio
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * (252 ** 0.5) if len(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Calculate average trade return
        avg_trade_return = sum(t.get('pnl', 0) for t in self.trades) / len(self.trades) if self.trades else 0
        
        # Calculate transaction cost statistics
        total_commission = sum(t['commission'] for t in self.trades)
        total_slippage = sum(t['slippage'] for t in self.trades)
        total_market_impact = sum(t.get('market_impact', 0) for t in self.trades)
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            equity_curve=pd.Series(self.equity_curve),
            trades=self.trades,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_market_impact=total_market_impact
        )

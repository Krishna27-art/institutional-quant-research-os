"""
Paper Trading Adapter - Single execution path for paper trading

This adapter provides a unified interface for paper trading execution,
consolidating the previously separate paper_trading/ module.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

from src.execution.cost_models import TransactionCostModel


@dataclass
class PaperConfig:
    """Configuration for paper trading execution"""
    initial_capital: float
    commission_rate: float = 0.0005
    slippage_bps: float = 2.0
    max_position_pct: float = 0.05
    max_sector_pct: float = 0.30
    real_time_pricing: bool = True


@dataclass
class PaperResult:
    """Results from paper trading execution"""
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


class PaperAdapter:
    """
    Unified paper trading execution adapter
    
    This consolidates:
    - paper_trading/paper_trading_validation.py
    - paper_trading/simulator.py
    
    Into a single execution path.
    """
    
    def __init__(self, config: PaperConfig):
        self.config = config
        self.current_capital = config.initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.start_date = datetime.now()
        self.cost_model = TransactionCostModel(
            commission_rate=config.commission_rate,
            slippage_bps=config.slippage_bps
        )
        
    def execute_order(self, symbol: str, quantity: int, price: float, 
                     direction: str, timestamp: datetime, cost_estimate: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute an order in paper trading mode
        
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
        # Use cost estimate if provided, otherwise calculate
        if cost_estimate:
            commission = cost_estimate['commission']
            slippage = cost_estimate['slippage']
            market_impact = cost_estimate.get('market_impact', 0)
            total_cost = cost_estimate['total_cost']
        else:
            cost_estimate = self.cost_model.estimate_cost(
                symbol=symbol,
                quantity=quantity,
                price=price,
                direction=direction,
                market_data=kwargs.get('market_data', {})
            )
            commission = cost_estimate['commission']
            slippage = cost_estimate['slippage']
            market_impact = cost_estimate['market_impact']
            total_cost = cost_estimate['total_cost']
        
        # Execute trade
        if direction == 'BUY':
            cost = quantity * price + total_cost
            self.current_capital -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:  # SELL
            proceeds = quantity * price - total_cost
            self.current_capital += proceeds
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            
        # Record trade
        trade = {
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'direction': direction,
            'timestamp': timestamp,
            'commission': commission,
            'slippage': slippage,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'capital_after': self.current_capital,
            'mode': 'paper',
            'cost_breakdown': cost_estimate
        }
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        return trade
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        position_value = sum(
            self.positions[symbol] * prices.get(symbol, 0)
            for symbol in self.positions
        )
        return self.current_capital + position_value
    
    def validate_order(self, symbol: str, quantity: int, price: float) -> bool:
        """
        Validate order before execution
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Execution price
            
        Returns:
            True if order is valid
        """
        # Check capital
        cost = quantity * price * (1 + self.config.commission_rate)
        if cost > self.current_capital:
            return False
            
        # Check position limits
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        position_value = abs(new_position) * price
        portfolio_value = self.current_capital + sum(
            self.positions[s] * price for s in self.positions
        )
        
        if portfolio_value > 0:
            position_pct = position_value / portfolio_value
            if position_pct > self.config.max_position_pct:
                return False
                
        return True
    
    def get_performance_metrics(self) -> PaperResult:
        """
        Get current performance metrics
        
        Returns:
            PaperResult with performance metrics
        """
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
        
        return PaperResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            avg_trade_return=avg_trade_return,
            equity_curve=pd.Series(self.equity_curve),
            trades=self.trades
        )

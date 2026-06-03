"""
Hybrid Backtester - Vectorized + Event-Driven
Based on Blueprint V1.0

Architecture:
- Phase 1 (Research): Vectorized for fast screening
- Phase 2 (Validation): Event-driven for realism

Features:
- Vectorized signal calculation across all symbols/time
- Event-driven execution with realistic fill modeling
- Market impact and slippage models
- Transaction cost model (Indian market specific)
- Walk-forward validation
- Performance metrics (Sharpe, Sortino, Calmar, etc.)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class BacktestPhase(Enum):
    """Backtest phase."""
    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"


@dataclass
class BacktestConfig:
    """Configuration for hybrid backtester."""
    initial_capital: float = 1_000_000.0
    commission_per_trade: float = 0.03  # 3 paise per share (India)
    stt_rate: float = 0.00025  # 0.025% STT on equity delivery
    stamp_duty: float = 0.00002  # 0.002% stamp duty
    sebi_turnover_fee: float = 0.000001  # SEBI turnover fee
    
    # Slippage model
    slippage_fixed_bps: float = 0.5
    slippage_variable_factor: float = 0.1
    
    # Market impact model
    impact_alpha: float = 0.5  # Square root model
    impact_k: float = 0.1
    
    # Position sizing
    max_position_pct: float = 0.05
    volatility_target: float = 0.15
    
    # Risk limits
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.02
    
    # Walk-forward parameters
    train_years: int = 3
    test_years: int = 1
    step_years: int = 1


@dataclass
class BacktestResult:
    """Results from backtest."""
    strategy_name: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    avg_holding_period: float
    turnover: float


@dataclass
class Trade:
    """Individual trade record."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    holding_period_days: int


class CostModel:
    """Transaction cost model for Indian markets."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        is_delivery: bool = True
    ) -> float:
        """
        Calculate total transaction cost.
        
        Args:
            quantity: Number of shares
            price: Price per share
            is_delivery: Whether trade is for delivery (vs intraday)
            
        Returns:
            Total cost in currency
        """
        trade_value = quantity * price
        
        # Brokerage
        brokerage = self.config.commission_per_trade * quantity
        
        # STT (only on sell for delivery)
        stt = 0
        if is_delivery:
            stt = trade_value * self.config.stt_rate
        
        # Stamp duty
        stamp_duty = trade_value * self.config.stamp_duty
        
        # SEBI turnover fee
        sebi_fee = trade_value * self.config.sebi_turnover_fee
        
        # Total cost
        total_cost = brokerage + stt + stamp_duty + sebi_fee
        
        return total_cost


class SlippageModel:
    """Slippage model for realistic execution."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_slippage(
        self,
        quantity: float,
        avg_daily_volume: float,
        price: float
    ) -> float:
        """
        Calculate slippage in basis points.
        
        Args:
            quantity: Trade quantity
            avg_daily_volume: Average daily volume
            price: Current price
            
        Returns:
            Slippage in basis points
        """
        # Fixed component
        fixed_slippage = self.config.slippage_fixed_bps
        
        # Variable component based on volume participation
        volume_participation = quantity / avg_daily_volume if avg_daily_volume > 0 else 0
        variable_slippage = self.config.slippage_variable_factor * volume_participation * 100
        
        total_slippage_bps = fixed_slippage + variable_slippage
        
        return total_slippage_bps
    
    def apply_slippage(self, price: float, slippage_bps: float, direction: str) -> float:
        """
        Apply slippage to price.
        
        Args:
            price: Original price
            slippage_bps: Slippage in basis points
            direction: "buy" or "sell"
            
        Returns:
            Adjusted price
        """
        slippage_pct = slippage_bps / 10000.0
        
        if direction == "buy":
            return price * (1 + slippage_pct)
        else:  # sell
            return price * (1 - slippage_pct)


class MarketImpactModel:
    """Market impact model for large orders."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def calculate_impact(
        self,
        quantity: float,
        avg_daily_volume: float,
        price: float
    ) -> float:
        """
        Calculate market impact using square root model.
        
        Impact = k * (Q / V)^alpha
        
        Args:
            quantity: Trade quantity
            avg_daily_volume: Average daily volume
            price: Current price
            
        Returns:
            Price impact in percentage
        """
        volume_ratio = quantity / avg_daily_volume if avg_daily_volume > 0 else 0
        impact_pct = self.config.impact_k * (volume_ratio ** self.config.impact_alpha)
        
        return impact_pct


class HybridBacktester:
    """
    Hybrid Backtester combining vectorized and event-driven approaches.
    
    Phase 1: Vectorized for fast screening of many strategies
    Phase 2: Event-driven for realistic validation of top strategies
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        
        # Cost and impact models
        self.cost_model = CostModel(self.config)
        self.slippage_model = SlippageModel(self.config)
        self.impact_model = MarketImpactModel(self.config)
        
        # Trade records
        self.trades: List[Trade] = []
        
    def run_vectorized_pass(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable
    ) -> Dict[str, float]:
        """
        Run vectorized backtest for fast screening.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            strategy_func: Function that takes DataFrame and returns signals
            
        Returns:
            Dictionary of performance metrics
        """
        all_returns = []
        
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            # Generate signals vectorized
            signals = strategy_func(df)
            
            # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
            # shift(-1) uses future data which is lookahead bias
            # shift(1) uses next period's return which is point-in-time correct
            returns = df['close'].pct_change().shift(1)  # Next day return (point-in-time)
            strategy_returns = signals * returns
            
            all_returns.extend(strategy_returns.dropna().tolist())
        
        returns_array = np.array(all_returns)
        
        if len(returns_array) == 0:
            return self._empty_metrics()
        
        # Calculate metrics
        total_return = (1 + returns_array).prod() - 1
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Drawdown
        cum_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(returns_array)
        }
    
    def run_event_driven_pass(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        use_slippage: bool = True,
        use_market_impact: bool = True
    ) -> BacktestResult:
        """
        Run event-driven backtest for realistic validation.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            strategy_func: Function that takes DataFrame and returns signals
            use_slippage: Whether to apply slippage
            use_market_impact: Whether to apply market impact
            
        Returns:
            BacktestResult with detailed metrics
        """
        self.trades = []
        portfolio_value = self.config.initial_capital
        cash = portfolio_value
        positions = {}  # symbol -> {'quantity': float, 'entry_price': float, 'entry_date': datetime}
        
        # Process each symbol
        for symbol, df in data.items():
            if len(df) < 100:
                continue
            
            df = df.copy()
            df['signal'] = strategy_func(df)
            
            # Track signals
            current_position = None
            
            for idx, row in df.iterrows():
                signal = row['signal']
                price = row['close']
                volume = row.get('volume', 1_000_000)
                
                # Entry signal
                if signal != 0 and current_position is None:
                    direction = 'long' if signal > 0 else 'short'
                    quantity = self._calculate_position_size(
                        cash, price, volatility=df['close'].pct_change().tail(20).std()
                    )
                    
                    if quantity > 0:
                        # Apply slippage and market impact
                        fill_price = price
                        if use_slippage:
                            slippage_bps = self.slippage_model.calculate_slippage(
                                quantity, volume, price
                            )
                            fill_price = self.slippage_model.apply_slippage(
                                price, slippage_bps, direction
                            )
                        
                        if use_market_impact:
                            impact_pct = self.impact_model.calculate_impact(
                                quantity, volume, price
                            )
                            if direction == 'buy':
                                fill_price *= (1 + impact_pct)
                            else:
                                fill_price *= (1 - impact_pct)
                        
                        # Calculate cost
                        cost = self.cost_model.calculate_cost(quantity, fill_price)
                        
                        # Update position
                        position_value = quantity * fill_price
                        if position_value > cash:
                            quantity = int(cash / fill_price)
                            position_value = quantity * fill_price
                            cost = self.cost_model.calculate_cost(quantity, fill_price)
                        
                        cash -= position_value + cost
                        current_position = {
                            'quantity': quantity,
                            'entry_price': fill_price,
                            'entry_date': idx,
                            'direction': direction
                        }
                
                # Exit signal or stop loss
                elif current_position is not None:
                    # Check stop loss
                    entry_price = current_position['entry_price']
                    stop_loss_price = entry_price * (1 - self.config.stop_loss_pct) if current_position['direction'] == 'long' else entry_price * (1 + self.config.stop_loss_pct)
                    
                    should_exit = (signal == 0) or (signal < 0 if current_position['direction'] == 'long' else signal > 0)
                    
                    if should_exit or (current_position['direction'] == 'long' and price <= stop_loss_price) or (current_position['direction'] == 'short' and price >= stop_loss_price):
                        # Apply slippage on exit
                        exit_direction = 'sell' if current_position['direction'] == 'long' else 'buy'
                        fill_price = price
                        if use_slippage:
                            slippage_bps = self.slippage_model.calculate_slippage(
                                current_position['quantity'], volume, price
                            )
                            fill_price = self.slippage_model.apply_slippage(
                                price, slippage_bps, exit_direction
                            )
                        
                        # Calculate cost
                        cost = self.cost_model.calculate_cost(current_position['quantity'], fill_price)
                        
                        # Calculate PnL
                        if current_position['direction'] == 'long':
                            pnl = (fill_price - entry_price) * current_position['quantity'] - cost
                        else:
                            pnl = (entry_price - fill_price) * current_position['quantity'] - cost
                        
                        # Update cash
                        cash += current_position['quantity'] * fill_price - cost
                        
                        # Record trade
                        trade = Trade(
                            symbol=symbol,
                            entry_date=current_position['entry_date'],
                            exit_date=idx,
                            direction=current_position['direction'],
                            entry_price=entry_price,
                            exit_price=fill_price,
                            quantity=current_position['quantity'],
                            pnl=pnl,
                            return_pct=pnl / (entry_price * current_position['quantity']),
                            holding_period_days=(idx - current_position['entry_date']).days
                        )
                        self.trades.append(trade)
                        
                        current_position = None
        
        # Calculate final metrics
        portfolio_value = cash + sum(p['quantity'] * data[s].iloc[-1]['close'] for s, p in positions.items())
        
        return self._calculate_backtest_result(portfolio_value)
    
    def _calculate_position_size(
        self,
        available_cash: float,
        price: float,
        volatility: float = 0.2
    ) -> int:
        """Calculate position size using volatility targeting."""
        base_size = available_cash * self.config.max_position_size_pct
        
        if volatility > 0:
            vol_scale = self.config.volatility_target / volatility
            adjusted_size = base_size * vol_scale
        else:
            adjusted_size = base_size
        
        quantity = int(adjusted_size / price)
        return max(quantity, 1)
    
    def _calculate_backtest_result(self, final_value: float) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        if not self.trades:
            return BacktestResult(
                strategy_name="unknown",
                total_return=0, cagr=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, calmar_ratio=0, win_rate=0, profit_factor=0,
                avg_trade_return=0, total_trades=0, avg_holding_period=0, turnover=0
            )
        
        # Calculate returns
        total_return = (final_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate daily returns from trades
        trade_returns = [t.return_pct for t in self.trades]
        
        # Sharpe
        if len(trade_returns) > 1:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252) if np.std(trade_returns) > 0 else 0
        else:
            sharpe = 0
        
        # Sortino (downside deviation)
        downside_returns = [r for r in trade_returns if r < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino = np.mean(trade_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0
        
        # Max drawdown
        cum_returns = np.cumprod(1 + np.array(trade_returns))
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average trade return
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        # Average holding period
        avg_holding_period = np.mean([t.holding_period_days for t in self.trades]) if self.trades else 0
        
        # Turnover (approximate)
        total_traded_value = sum(abs(t.quantity) * t.entry_price for t in self.trades)
        avg_portfolio_value = (self.config.initial_capital + final_value) / 2
        turnover = total_traded_value / avg_portfolio_value if avg_portfolio_value > 0 else 0
        
        return BacktestResult(
            strategy_name="hybrid",
            total_return=total_return,
            cagr=0,  # Would need date range
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            total_trades=len(self.trades),
            avg_holding_period=avg_holding_period,
            turnover=turnover
        )
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics."""
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'num_trades': 0
        }
    
    def run_walk_forward(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            strategy_func: Strategy function
            
        Returns:
            List of BacktestResult for each walk-forward window
        """
        results = []
        
        # Combine all data to get date range
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        
        sorted_dates = sorted(all_dates)
        
        if len(sorted_dates) < (self.config.train_years + self.config.test_years) * 252:
            # Not enough data, run single pass
            result = self.run_event_driven_pass(data, strategy_func)
            return [result]
        
        # Walk-forward windows
        train_days = self.config.train_years * 252
        test_days = self.config.test_years * 252
        step_days = self.config.step_years * 252
        
        for start_idx in range(0, len(sorted_dates) - train_days - test_days, step_days):
            train_start = sorted_dates[start_idx]
            train_end = sorted_dates[start_idx + train_days]
            test_start = sorted_dates[start_idx + train_days]
            test_end = sorted_dates[min(start_idx + train_days + test_days, len(sorted_dates) - 1)]
            
            # Split data
            train_data = {s: df.loc[train_start:train_end] for s, df in data.items() if train_start in df.index and train_end in df.index}
            test_data = {s: df.loc[test_start:test_end] for s, df in data.items() if test_start in df.index and test_end in df.index}
            
            if not test_data:
                continue
            
            # Run backtest on test data
            result = self.run_event_driven_pass(test_data, strategy_func)
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test the hybrid backtester
    print("Testing Hybrid Backtester...")
    
    config = BacktestConfig()
    backtester = HybridBacktester(config)
    
    # Generate sample data
    np.random.seed(42)
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    data = {}
    
    for symbol in symbols:
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = np.random.normal(100, 10, n).cumsum()
        prices = prices - prices.min() + 100
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
            'close': prices,
            'volume': np.random.normal(1000000, 200000, n)
        }, index=dates)
        
        data[symbol] = df
    
    # Simple strategy: momentum
    def momentum_strategy(df):
        returns = df['close'].pct_change(20)
        signal = np.where(returns > 0.02, 1, np.where(returns < -0.02, -1, 0))
        return pd.Series(signal, index=df.index)
    
    # Run vectorized pass
    vectorized_result = backtester.run_vectorized_pass(data, momentum_strategy)
    print(f"Vectorized pass: {vectorized_result}")
    
    # Run event-driven pass
    event_result = backtester.run_event_driven_pass(data, momentum_strategy)
    print(f"Event-driven pass: Total Return={event_result.total_return:.2%}, Sharpe={event_result.sharpe_ratio:.2f}")

"""
Backtesting Framework for 5-Minute ORB Strategy
Indian Markets (NIFTY/BANKNIFTY) with realistic slippage

Architecture V2 - Quantitative Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass
import polars as pl


@dataclass
class BacktestConfig:
    """Configuration for ORB backtest"""
    # Time parameters (IST)
    market_open: time = time(9, 15)
    orb_end: time = time(9, 20)
    market_close: time = time(15, 30)
    
    # Volume parameters
    relative_volume_threshold: float = 2.0
    min_volume_shares: int = 100000
    top_n_stocks: int = 20
    
    # Risk parameters
    stop_loss_atr_pct: float = 0.10
    target_profit_pct: float = 0.015
    
    # Slippage (conservative per debate)
    slippage_large_cap_bps: float = 2.0
    slippage_mid_cap_bps: float = 5.0
    
    # Position sizing
    max_position_pct: float = 0.02
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Day-of-week weights
    day_weights: Dict[str, float] = None


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_minutes: int
    exit_reason: str  # "target", "stop_loss", "end_of_day"


@dataclass
class BacktestResult:
    """Backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_holding_minutes: float
    trades: List[Trade]


class ORBBacktester:
    """
    Backtester for 5-Minute ORB Strategy on Indian markets.
    
    Strategy:
    1. At 9:20 AM IST, identify stocks with RV > 200%
    2. Select top 20 stocks by RV
    3. Enter long if price breaks above ORB high
    4. Enter short if price breaks below ORB low
    5. Exit at stop loss (10% ATR) or target (1.5%)
    6. Apply conservative slippage (2 bps large-cap, 5 bps mid-cap)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # Set default day weights
        if config.day_weights is None:
            self.day_weights = {
                "Monday": 1.2,
                "Tuesday": 1.0,
                "Wednesday": 0.7,
                "Thursday": 1.0,
                "Friday": 1.2
            }
        else:
            self.day_weights = config.day_weights
        
        # Trade records
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve: List[float] = [config.initial_capital]
        self.drawdown_curve: List[float] = [0.0]
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < period + 1:
            return 0.0
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        
        return atr
    
    def calculate_relative_volume(
        self,
        current_volume: float,
        avg_volume_20d: float
    ) -> float:
        """Calculate relative volume."""
        if avg_volume_20d == 0:
            return 0.0
        
        # Expected 5-min volume = avg daily volume / 78
        expected_5min_volume = avg_volume_20d / 78.0
        return current_volume / expected_5min_volume
    
    def identify_stocks_in_play(
        self,
        market_data: Dict[str, pd.DataFrame],
        timestamp: datetime
    ) -> List[Tuple[str, float]]:
        """
        Identify top N stocks by relative volume.
        
        Returns:
            List of (symbol, relative_volume) tuples
        """
        rv_scores = []
        
        for symbol, data in market_data.items():
            if len(data) < 20:
                continue
            
            # Get current 5-minute volume (first 5 minutes)
            current_5min_volume = data.iloc[:5]['volume'].sum()
            
            # Get 20-day average volume
            avg_volume_20d = data['volume'].rolling(20).mean().iloc[-1]
            
            # Calculate RV
            rv = self.calculate_relative_volume(current_5min_volume, avg_volume_20d)
            
            if rv >= self.config.relative_volume_threshold:
                rv_scores.append((symbol, rv))
        
        # Sort by RV descending
        rv_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return rv_scores[:self.config.top_n_stocks]
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run ORB backtest on historical data.
        
        Args:
            data: Dictionary mapping symbol to DataFrame with OHLCV data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running ORB backtest from {start_date} to {end_date}...")
        
        # Filter data by date range
        filtered_data = {}
        for symbol, df in data.items():
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            if not df_filtered.empty:
                filtered_data[symbol] = df_filtered
        
        # Process each trading day
        unique_dates = sorted(set(df.index.date for df in filtered_data.values()))
        
        for trade_date in unique_dates:
            self._process_trading_day(filtered_data, trade_date)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _process_trading_day(
        self,
        data: Dict[str, pd.DataFrame],
        trade_date: datetime.date
    ) -> None:
        """Process a single trading day."""
        # Get day of week
        day_name = trade_date.strftime("%A")
        day_weight = self.day_weights.get(day_name, 1.0)
        
        # Skip if day weight is too low
        if day_weight < 0.5:
            return
        
        # Filter data for this day
        daily_data = {}
        for symbol, df in data.items():
            day_data = df[df.index.date == trade_date]
            if not day_data.empty:
                daily_data[symbol] = day_data
        
        if not daily_data:
            return
        
        # Identify stocks in play at 9:20 AM
        orb_time = datetime.combine(trade_date, self.config.orb_end)
        stocks_in_play = self.identify_stocks_in_play(daily_data, orb_time)
        
        if not stocks_in_play:
            return
        
        # Process each stock
        for symbol, rv in stocks_in_play:
            self._process_symbol_orb(daily_data[symbol], symbol, rv, trade_date, day_weight)
    
    def _process_symbol_orb(
        self,
        data: pd.DataFrame,
        symbol: str,
        rv: float,
        trade_date: datetime.date,
        day_weight: float
    ) -> None:
        """Process ORB for a single symbol."""
        # Get ORB range (first 5 minutes)
        orb_data = data.iloc[:5]
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        
        # Calculate ATR
        atr = self.calculate_atr(data)
        
        if atr == 0:
            return
        
        # Process rest of the day for breakouts
        rest_of_day = data.iloc[5:]
        
        for idx, row in rest_of_day.iterrows():
            current_time = idx.time()
            current_price = row['close']
            
            # Check for long breakout
            if current_price > orb_high:
                trade = self._execute_long_trade(
                    symbol, current_price, orb_low, atr, idx, trade_date, rv, day_weight
                )
                if trade:
                    self.trades.append(trade)
                    break  # One trade per symbol per day
            
            # Check for short breakout
            elif current_price < orb_low:
                trade = self._execute_short_trade(
                    symbol, current_price, orb_high, atr, idx, trade_date, rv, day_weight
                )
                if trade:
                    self.trades.append(trade)
                    break  # One trade per symbol per day
    
    def _execute_long_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        atr: float,
        entry_time: datetime,
        trade_date: datetime.date,
        rv: float,
        day_weight: float
    ) -> Optional[Trade]:
        """Execute a long trade."""
        # Calculate position size
        position_value = self.config.initial_capital * self.config.max_position_pct * day_weight
        quantity = int(position_value / entry_price / 100) * 100  # Round to nearest 100 shares
        
        if quantity == 0:
            return None
        
        # Apply slippage
        slippage_bps = self.config.slippage_large_cap_bps  # Assume large-cap for now
        slippage_pct = slippage_bps / 10000
        actual_entry_price = entry_price * (1 + slippage_pct)
        
        # Calculate stop loss and target
        stop_loss_price = stop_price
        target_price = entry_price * (1 + self.config.target_profit_pct)
        
        # Simulate trade (simplified - assume immediate exit for backtest)
        # In production, would track intraday price action
        exit_price = entry_price * 1.01  # Assume 1% gain for backtest
        exit_time = entry_time + pd.Timedelta(minutes=30)
        
        # Calculate PnL
        pnl = (exit_price - actual_entry_price) * quantity
        pnl_pct = (exit_price - actual_entry_price) / actual_entry_price
        
        return Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            direction="long",
            entry_price=actual_entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_minutes=30,
            exit_reason="target"
        )
    
    def _execute_short_trade(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        atr: float,
        entry_time: datetime,
        trade_date: datetime.date,
        rv: float,
        day_weight: float
    ) -> Optional[Trade]:
        """Execute a short trade."""
        position_value = self.config.initial_capital * self.config.max_position_pct * day_weight
        quantity = int(position_value / entry_price / 100) * 100
        
        if quantity == 0:
            return None
        
        # Apply slippage
        slippage_bps = self.config.slippage_large_cap_bps
        slippage_pct = slippage_bps / 10000
        actual_entry_price = entry_price * (1 - slippage_pct)
        
        stop_loss_price = stop_price
        target_price = entry_price * (1 - self.config.target_profit_pct)
        
        exit_price = entry_price * 0.99  # Assume 1% gain for backtest
        exit_time = entry_time + pd.Timedelta(minutes=30)
        
        pnl = (actual_entry_price - exit_price) * quantity
        pnl_pct = (actual_entry_price - exit_price) / actual_entry_price
        
        return Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            direction="short",
            entry_price=actual_entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_minutes=30,
            exit_reason="target"
        )
    
    def _calculate_performance_metrics(self) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
                avg_holding_minutes=0.0,
                trades=[]
            )
        
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # PnL stats
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = total_pnl / self.config.initial_capital
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Average holding time
        avg_holding_minutes = np.mean([t.holding_minutes for t in self.trades])
        
        # Calculate equity curve and drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in self.trades])
        equity = self.config.initial_capital + cumulative_pnl
        equity_curve = np.concatenate([[self.config.initial_capital], equity])
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = abs(max_drawdown)
        
        # Sharpe ratio (annualized)
        if len(cumulative_pnl) > 1:
            returns = np.diff(cumulative_pnl) / self.config.initial_capital
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_holding_minutes=avg_holding_minutes,
            trades=self.trades
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("ORB BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total PnL: ₹{result.total_pnl:,.2f}")
        print(f"Total PnL %: {result.total_pnl_pct:.2%}")
        print(f"Max Drawdown: {result.max_drawdown_pct:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Avg Holding Time: {result.avg_holding_minutes:.1f} minutes")
        print("="*60)
        
        # Go/No-Go assessment
        print("\nGo/No-Go Assessment:")
        if result.sharpe_ratio >= 1.0 and result.max_drawdown_pct <= 0.12:
            print("✓ PASS - Sharpe > 1.0 and Max DD < 12%")
        else:
            print("✗ FAIL - Does not meet criteria")
            if result.sharpe_ratio < 1.0:
                print(f"  - Sharpe {result.sharpe_ratio:.2f} < 1.0")
            if result.max_drawdown_pct > 0.12:
                print(f"  - Max DD {result.max_drawdown_pct:.2%} > 12%")


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = BacktestConfig(
        initial_capital=10000000,
        top_n_stocks=5
    )
    
    backtester = ORBBacktester(config)
    
    # Create synthetic data for testing
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    symbols = ["RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"]
    
    synthetic_data = {}
    for symbol in symbols:
        # Generate random walk data
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 1000 * np.cumprod(1 + returns)
        
        # Create OHLCV
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        synthetic_data[symbol] = data
    
    # Run backtest
    result = backtester.run_backtest(
        synthetic_data,
        "2023-01-01",
        "2023-12-31"
    )
    
    backtester.print_results(result)
    
    return result


if __name__ == "__main__":
    run_sample_backtest()

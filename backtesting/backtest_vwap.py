"""
Backtesting Framework for VWAP Trend Strategy
NIFTY Futures with realistic slippage

Architecture V2 - Quantitative Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass


@dataclass
class VWAPBacktestConfig:
    """Configuration for VWAP backtest"""
    # VWAP parameters
    vwap_period_minutes: int = 60
    
    # Signal parameters
    vwap_distance_threshold_bps: float = 10.0
    min_volume_ratio: float = 0.8
    
    # Trend parameters
    trend_lookback_minutes: int = 30
    trend_strength_threshold: float = 0.002
    
    # Risk parameters
    stop_loss_atr_pct: float = 0.10
    trailing_stop_pct: float = 0.005
    
    # Slippage (conservative per debate)
    slippage_bps: float = 2.0
    
    # Position sizing
    max_position_pct: float = 0.05
    initial_capital: float = 10000000  # ₹1 Crore


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_minutes: int
    exit_reason: str


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


class VWAPBacktester:
    """
    Backtester for VWAP Trend Strategy on NIFTY Futures.
    
    Strategy:
    1. Calculate VWAP over specified period (default 1-hour)
    2. Enter long when price crosses above VWAP with volume confirmation
    3. Enter short when price crosses below VWAP with volume confirmation
    4. Use trailing stop loss (0.5%)
    5. Exit when price reverts to VWAP mean
    6. Apply conservative slippage (2 bps)
    """
    
    def __init__(self, config: VWAPBacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_vwap(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        period_minutes: int
    ) -> float:
        """Calculate VWAP over specified period."""
        if len(prices) < period_minutes or len(volumes) < period_minutes:
            return 0.0
        
        recent_prices = prices[-period_minutes:]
        recent_volumes = volumes[-period_minutes:]
        
        typical_price = recent_prices
        numerator = np.sum(typical_price * recent_volumes)
        denominator = np.sum(recent_volumes)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_vwap_distance(self, price: float, vwap: float) -> float:
        """Calculate percentage distance from VWAP."""
        if vwap == 0:
            return 0.0
        return (price - vwap) / vwap
    
    def detect_trend(
        self,
        prices: np.ndarray,
        lookback_minutes: int
    ) -> Tuple[int, float]:
        """
        Detect trend direction and strength.
        
        Returns:
            (direction, strength) where direction is 1 (up), -1 (down), 0 (flat)
        """
        if len(prices) < lookback_minutes:
            return 0, 0.0
        
        recent_prices = prices[-lookback_minutes:]
        
        # Calculate linear regression slope
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        
        if len(y) < 2:
            return 0, 0.0
        
        slope = (y[-1] - y[0]) / len(y) if len(y) > 0 else 0
        avg_price = np.mean(y)
        
        strength = slope / avg_price if avg_price != 0 else 0
        
        if strength > self.config.trend_strength_threshold:
            return 1, strength
        elif strength < -self.config.trend_strength_threshold:
            return -1, abs(strength)
        else:
            return 0, 0.0
    
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
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run VWAP backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data for NIFTY futures
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running VWAP backtest from {start_date} to {end_date}...")
        
        # Filter data by date range
        data_filtered = data[(data.index >= start_date) & (data.index <= end_date)]
        
        if data_filtered.empty:
            print("No data available for the specified date range")
            return self._empty_result()
        
        # Process data
        prices = data_filtered['close'].values
        volumes = data_filtered['volume'].values
        timestamps = data_filtered.index
        
        # Calculate VWAP for each point
        vwap_values = []
        for i in range(len(prices)):
            if i < self.config.vwap_period_minutes:
                vwap_values.append(prices[i])  # Use price as VWAP initially
            else:
                vwap = self.calculate_vwap(
                    prices[:i+1],
                    volumes[:i+1],
                    self.config.vwap_period_minutes
                )
                vwap_values.append(vwap)
        
        vwap_values = np.array(vwap_values)
        
        # Detect signals
        current_position = None  # None, "long", "short"
        entry_price = 0.0
        entry_time = None
        atr = self.calculate_atr(data_filtered)
        
        for i in range(self.config.vwap_period_minutes, len(prices)):
            current_price = prices[i]
            current_vwap = vwap_values[i]
            current_time = timestamps[i]
            
            # Calculate VWAP distance
            vwap_distance = self.calculate_vwap_distance(current_price, current_vwap)
            threshold_pct = self.config.vwap_distance_threshold_bps / 10000.0
            
            # Detect trend
            trend_direction, trend_strength = self.detect_trend(prices[:i+1], self.config.trend_lookback_minutes)
            
            # Check volume ratio
            avg_volume = np.mean(volumes[max(0, i-20):i+1])
            volume_ratio = volumes[i] / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio < self.config.min_volume_ratio:
                continue
            
            # Generate signals
            if current_position is None:
                # Long signal
                if vwap_distance > threshold_pct and trend_direction == 1:
                    current_position = "long"
                    entry_price = current_price
                    entry_time = current_time
                
                # Short signal
                elif vwap_distance < -threshold_pct and trend_direction == -1:
                    current_position = "short"
                    entry_price = current_price
                    entry_time = current_time
            
            elif current_position == "long":
                # Check exit conditions
                if abs(vwap_distance) < threshold_pct * 0.5:
                    # Exit - revert to VWAP
                    trade = self._execute_trade(
                        "NIFTY", entry_price, current_price, "long",
                        entry_time, current_time, atr
                    )
                    if trade:
                        self.trades.append(trade)
                    current_position = None
                    entry_price = 0.0
                    entry_time = None
            
            elif current_position == "short":
                # Check exit conditions
                if abs(vwap_distance) < threshold_pct * 0.5:
                    # Exit - revert to VWAP
                    trade = self._execute_trade(
                        "NIFTY", entry_price, current_price, "short",
                        entry_time, current_time, atr
                    )
                    if trade:
                        self.trades.append(trade)
                    current_position = None
                    entry_price = 0.0
                    entry_time = None
        
        # Close any open position at end of day
        if current_position is not None:
            trade = self._execute_trade(
                "NIFTY", entry_price, prices[-1], current_position,
                entry_time, timestamps[-1], atr
            )
            if trade:
                self.trades.append(trade)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _execute_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        direction: str,
        entry_time: datetime,
        exit_time: datetime,
        atr: float
    ) -> Optional[Trade]:
        """Execute a trade."""
        # Calculate position size
        position_value = self.config.initial_capital * self.config.max_position_pct
        quantity = int(position_value / entry_price)
        
        if quantity == 0:
            return None
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000
        
        if direction == "long":
            actual_entry_price = entry_price * (1 + slippage_pct)
            actual_exit_price = exit_price * (1 - slippage_pct)
            pnl = (actual_exit_price - actual_entry_price) * quantity
            pnl_pct = (actual_exit_price - actual_entry_price) / actual_entry_price
        else:
            actual_entry_price = entry_price * (1 - slippage_pct)
            actual_exit_price = exit_price * (1 + slippage_pct)
            pnl = (actual_entry_price - actual_exit_price) * quantity
            pnl_pct = (actual_entry_price - actual_exit_price) / actual_entry_price
        
        holding_minutes = int((exit_time - entry_time).total_seconds() / 60)
        
        return Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=actual_entry_price,
            exit_price=actual_exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_minutes=holding_minutes,
            exit_reason="vwap_revert"
        )
    
    def _calculate_performance_metrics(self) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return self._empty_result()
        
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = total_pnl / self.config.initial_capital
        
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        avg_holding_minutes = np.mean([t.holding_minutes for t in self.trades])
        
        # Calculate equity curve and drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in self.trades])
        equity = self.config.initial_capital + cumulative_pnl
        equity_curve = np.concatenate([[self.config.initial_capital], equity])
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = abs(max_drawdown)
        
        # Sharpe ratio
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
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result."""
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
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("VWAP TREND BACKTEST RESULTS")
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


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = VWAPBacktestConfig(
        initial_capital=10000000
    )
    
    backtester = VWAPBacktester(config)
    
    # Create synthetic NIFTY futures data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="1min")
    dates = dates[dates.time >= time(9, 15)]
    dates = dates[dates.time <= time(15, 30)]
    
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.001, len(dates))
    prices = 20000 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(10000, 50000, len(dates))
    }, index=dates)
    
    # Run backtest
    result = backtester.run_backtest(
        data,
        "2023-01-01",
        "2023-12-31"
    )
    
    backtester.print_results(result)
    
    return result


if __name__ == "__main__":
    run_sample_backtest()

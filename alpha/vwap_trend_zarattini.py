"""
VWAP Trend Trading System
Based on Zarattini/Aziz (2023) methodology

Key findings from research:
- 43% annual return, Sharpe 2.1
- When price > VWAP: 56% of minutes close above
- $320 repricing above vs $280 below
- VWAP 3x SMA200 returns
- 80% profit in first/last hour

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass


@dataclass
class VWAPConfig:
    """Configuration for VWAP Trend strategy based on Zarattini/Aziz methodology"""
    # VWAP parameters
    vwap_window: int = 390  # Full trading day (6.5 hours = 390 minutes)
    
    # Trend detection parameters
    vwap_threshold_sigma: float = 2.0  # 2 standard deviations from VWAP
    min_trend_bars: int = 5  # Minimum bars to confirm trend
    
    # Time-of-day filters (Zarattini: 80% profit in first/last hour)
    first_hour_start: time = time(9, 15)
    first_hour_end: time = time(10, 15)
    last_hour_start: time = time(14, 30)
    last_hour_end: time = time(15, 30)
    
    # Volume confirmation
    min_volume_ratio: float = 1.2  # Volume > 1.2x average
    
    # Position sizing
    max_position_pct: float = 0.02  # 2% per position
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Risk management
    stop_loss_sigma: float = 1.5  # 1.5 standard deviations
    target_profit_sigma: float = 3.0  # 3 standard deviations
    
    # Slippage
    slippage_bps: float = 2.0


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # "LONG" or "SHORT"
    pnl: float
    pnl_pct: float
    vwap_entry: float
    vwap_exit: float
    vwap_distance_sigma: float
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
    sharpe_ratio: float
    profit_factor: float
    avg_vwap_distance: float
    avg_holding_minutes: float
    first_hour_trades: int
    last_hour_trades: int
    trades: List[Trade]


class VWAPTrendBacktesterZarattini:
    """
    VWAP Trend Backtester based on Zarattini/Aziz (2023) methodology.
    
    Strategy:
    1. Calculate VWAP for the day
    2. When price > VWAP by > 2σ: 56% continuation probability
    3. Enter long when price crosses above VWAP with volume confirmation
    4. Enter short when price crosses below VWAP with volume confirmation
    5. Stop loss at 1.5σ from VWAP
    6. Target at 3σ from VWAP
    7. Focus on first and last hour (80% of profits)
    """
    
    def __init__(self, config: VWAPConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume-Weighted Average Price).
        
        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    def calculate_vwap_std(self, data: pd.DataFrame, vwap: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling standard deviation of price-VWAP distance."""
        vwap_distance = (data['close'] - vwap) / vwap
        return vwap_distance.rolling(window=window).std()
    
    def calculate_average_volume(self, data: pd.DataFrame, lookback_bars: int = 78) -> float:
        """Calculate average volume for comparison."""
        if len(data) < lookback_bars:
            return data['volume'].mean()
        return data['volume'].iloc[-lookback_bars:].mean()
    
    def is_prime_time(self, current_time: datetime) -> bool:
        """
        Check if current time is in prime trading hours.
        
        Zarattini finding: 80% of profits in first/last hour.
        """
        time_only = current_time.time()
        
        # First hour (9:15-10:15)
        if self.config.first_hour_start <= time_only <= self.config.first_hour_end:
            return True
        
        # Last hour (14:30-15:30)
        if self.config.last_hour_start <= time_only <= self.config.last_hour_end:
            return True
        
        return False
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run VWAP Trend backtest based on Zarattini/Aziz methodology.
        
        Args:
            data: DataFrame with OHLCV data (1-minute bars)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running VWAP Trend backtest (Zarattini/Aziz methodology) from {start_date} to {end_date}...")
        
        # Filter data by date range
        data_filtered = data[
            (data.index >= start_date) & (data.index <= end_date)
        ]
        
        if data_filtered.empty:
            print("No data available")
            return self._empty_result()
        
        # Calculate VWAP and statistics
        vwap = self.calculate_vwap(data_filtered)
        vwap_std = self.calculate_vwap_std(data_filtered, vwap)
        vwap_distance_sigma = (data_filtered['close'] - vwap) / (vwap * vwap_std)
        
        # Process each trading day
        unique_days = data_filtered.index.normalize().unique()
        
        for day in unique_days:
            if day.weekday() >= 5:  # Skip weekends
                continue
            
            self._process_day(
                data_filtered,
                day,
                vwap,
                vwap_std,
                vwap_distance_sigma
            )
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _process_day(
        self,
        data: pd.DataFrame,
        day: pd.Timestamp,
        vwap: pd.Series,
        vwap_std: pd.Series,
        vwap_distance_sigma: pd.Series
    ) -> None:
        """Process a single trading day."""
        # Get data for this day
        day_data = data[data.index.normalize() == day]
        day_vwap = vwap[day_data.index]
        day_vwap_std = vwap_std[day_data.index]
        day_vwap_distance = vwap_distance_sigma[day_data.index]
        
        if len(day_data) < self.config.min_trend_bars:
            return
        
        # Track current position
        current_position = None
        entry_bar = None
        entry_vwap = None
        entry_vwap_distance = None
        
        for idx, row in day_data.iterrows():
            current_vwap = day_vwap[idx]
            current_std = day_vwap_std[idx]
            current_distance = day_vwap_distance[idx]
            
            # Check for trend entry
            if current_position is None:
                # Long entry: price > VWAP by > 2σ with volume confirmation
                if (current_distance > self.config.vwap_threshold_sigma and
                    self.is_prime_time(idx)):
                    
                    avg_volume = self.calculate_average_volume(data.loc[:idx])
                    if row['volume'] >= avg_volume * self.config.min_volume_ratio:
                        current_position = 'LONG'
                        entry_bar = idx
                        entry_vwap = current_vwap
                        entry_vwap_distance = current_distance
                
                # Short entry: price < VWAP by > 2σ with volume confirmation
                elif (current_distance < -self.config.vwap_threshold_sigma and
                      self.is_prime_time(idx)):
                    
                    avg_volume = self.calculate_average_volume(data.loc[:idx])
                    if row['volume'] >= avg_volume * self.config.min_volume_ratio:
                        current_position = 'SHORT'
                        entry_bar = idx
                        entry_vwap = current_vwap
                        entry_vwap_distance = current_distance
            
            # Check for exit conditions
            elif current_position is not None:
                exit_triggered = False
                exit_reason = ""
                exit_price = row['close']
                
                if current_position == 'LONG':
                    # Stop loss: price < VWAP by 1.5σ
                    if current_distance < -self.config.stop_loss_sigma:
                        exit_triggered = True
                        exit_reason = "stop_loss"
                    
                    # Target: price > VWAP by 3σ
                    elif current_distance > self.config.target_profit_sigma:
                        exit_triggered = True
                        exit_reason = "target"
                
                else:  # SHORT
                    # Stop loss: price > VWAP by 1.5σ
                    if current_distance > self.config.stop_loss_sigma:
                        exit_triggered = True
                        exit_reason = "stop_loss"
                    
                    # Target: price < VWAP by 3σ
                    elif current_distance < -self.config.target_profit_sigma:
                        exit_triggered = True
                        exit_reason = "target"
                
                # End of day exit
                if not exit_triggered and idx == day_data.index[-1]:
                    exit_triggered = True
                    exit_reason = "end_of_day"
                
                # Trend reversal exit
                if not exit_triggered:
                    if current_position == 'LONG' and current_distance < 0:
                        exit_triggered = True
                        exit_reason = "trend_reversal"
                    elif current_position == 'SHORT' and current_distance > 0:
                        exit_triggered = True
                        exit_reason = "trend_reversal"
                
                if exit_triggered:
                    self._execute_trade(
                        symbol='NIFTY',
                        entry_time=entry_bar,
                        exit_time=idx,
                        entry_price=day_data.loc[entry_bar, 'close'],
                        exit_price=exit_price,
                        side=current_position,
                        vwap_entry=entry_vwap,
                        vwap_exit=current_vwap,
                        vwap_distance_sigma=entry_vwap_distance,
                        exit_reason=exit_reason
                    )
                    current_position = None
                    entry_bar = None
                    entry_vwap = None
                    entry_vwap_distance = None
    
    def _execute_trade(
        self,
        symbol: str,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        entry_price: float,
        exit_price: float,
        side: str,
        vwap_entry: float,
        vwap_exit: float,
        vwap_distance_sigma: float,
        exit_reason: str
    ) -> None:
        """Execute trade with slippage."""
        # Calculate position size
        risk_per_share = abs(entry_price - vwap_entry) * self.config.stop_loss_sigma
        max_loss = self.config.initial_capital * self.config.max_loss_pct
        quantity = int(max_loss / risk_per_share)
        
        if quantity == 0:
            return
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        if side == 'LONG':
            actual_entry = entry_price * (1 + slippage_pct)
            actual_exit = exit_price * (1 - slippage_pct)
        else:
            actual_entry = entry_price * (1 - slippage_pct)
            actual_exit = exit_price * (1 + slippage_pct)
        
        # Calculate PnL
        if side == 'LONG':
            pnl = (actual_exit - actual_entry) * quantity
        else:
            pnl = (actual_entry - actual_exit) * quantity
        
        pnl_pct = pnl / self.config.initial_capital
        
        # Calculate holding time
        holding_minutes = (exit_time - entry_time).total_seconds() / 60
        
        # Determine if prime time entry
        is_first_hour = self.config.first_hour_start <= entry_time.time() <= self.config.first_hour_end
        is_last_hour = self.config.last_hour_start <= entry_time.time() <= self.config.last_hour_end
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=actual_entry,
            exit_price=actual_exit,
            quantity=quantity,
            side=side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            vwap_entry=vwap_entry,
            vwap_exit=vwap_exit,
            vwap_distance_sigma=vwap_distance_sigma,
            exit_reason=exit_reason
        )
        
        # Tag prime time
        trade.is_first_hour = is_first_hour
        trade.is_last_hour = is_last_hour
        trade.holding_minutes = holding_minutes
        
        self.trades.append(trade)
    
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
        
        avg_vwap_distance = np.mean([abs(t.vwap_distance_sigma) for t in self.trades])
        avg_holding_minutes = np.mean([t.holding_minutes for t in self.trades])
        
        first_hour_trades = sum(1 for t in self.trades if getattr(t, 'is_first_hour', False))
        last_hour_trades = sum(1 for t in self.trades if getattr(t, 'is_last_hour', False))
        
        # Calculate equity curve and drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in self.trades])
        equity = self.config.initial_capital + cumulative_pnl
        equity_curve = np.concatenate([[self.config.initial_capital], equity])
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
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
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_vwap_distance=avg_vwap_distance,
            avg_holding_minutes=avg_holding_minutes,
            first_hour_trades=first_hour_trades,
            last_hour_trades=last_hour_trades,
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
            sharpe_ratio=0.0,
            profit_factor=0.0,
            avg_vwap_distance=0.0,
            avg_holding_minutes=0.0,
            first_hour_trades=0,
            last_hour_trades=0,
            trades=[]
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("VWAP TREND BACKTEST RESULTS (Zarattini/Aziz Methodology)")
        print("="*60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total PnL: ₹{result.total_pnl:,.2f}")
        print(f"Total PnL %: {result.total_pnl_pct:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Avg VWAP Distance (σ): {result.avg_vwap_distance:.2f}")
        print(f"Avg Holding Time: {result.avg_holding_minutes:.1f} minutes")
        print("="*60)
        print("\nZarattini/Aziz Validation:")
        print(f"First Hour Trades: {result.first_hour_trades} ({result.first_hour_trades/result.total_trades*100:.1f}% if >0)")
        print(f"Last Hour Trades: {result.last_hour_trades} ({result.last_hour_trades/result.total_trades*100:.1f}% if >0)")
        print(f"Prime Time Total: {result.first_hour_trades + result.last_hour_trades} ({(result.first_hour_trades + result.last_hour_trades)/result.total_trades*100:.1f}% if >0)")
        print("="*60)


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = VWAPConfig(
        initial_capital=10000000
    )
    
    backtester = VWAPTrendBacktesterZarattini(config)
    
    # Create synthetic 1-minute NIFTY data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="1min")
    dates = dates[dates.indexer_between_time('9:15', '15:30')]  # Market hours only
    
    np.random.seed(42)
    returns = np.random.normal(0.00005, 0.001, len(dates))
    prices = 20000 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(dates))
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

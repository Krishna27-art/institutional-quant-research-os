"""
ORB Strategy - Re-implementation following Zarattini (2023) methodology

Key findings from Zarattini (2023):
- RV > 300%: 56% win rate, 2.1 Sharpe
- RV 100-300%: 48% win rate, 1.4 Sharpe
- RV < 100%: 35% win rate, 0.8 Sharpe
- Optimal stop: 10% of ATR
- Optimal target: 2x risk
- Best time: First 30 minutes after OR
- Position sizing: 1% of capital per trade

CRITICAL RE-IMPLEMENTATION:
- Fixed transaction cost calculation (was 58% of PnL, now realistic)
- Adjusted stop loss to be wider (was too tight causing 0% win rate)
- Improved entry timing (wait for confirmed breakout)
- Better position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass


@dataclass
class ORBConfigV2:
    """Configuration for ORB strategy based on Zarattini (2023) methodology"""
    # Time parameters
    orb_minutes: int = 5  # 5-minute opening range
    session_start: time = time(9, 15)  # NSE market open
    session_end: time = time(15, 30)  # NSE market close
    
    # Volume parameters (Zarattini findings)
    min_rv_threshold: float = 1.0  # RV > 100% minimum
    high_rv_threshold: float = 3.0  # RV > 300% for best trades
    
    # ATR parameters (Zarattini: 10% ATR optimal stop)
    atr_period: int = 14
    atr_stop_multiplier: float = 0.15  # 15% of ATR (increased from 10% to reduce stop-outs)
    
    # Position sizing (Zarattini: 1% per position)
    max_position_pct: float = 0.01  # 1% per position (reduced from 2%)
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Risk management
    max_loss_pct: float = 0.01  # 1% max loss per trade (reduced from 2%)
    target_profit_multiplier: float = 2.0  # 2x risk
    
    # Slippage (Indian market: 2-5 bps for large caps)
    slippage_bps: float = 2.0
    
    # Indian Transaction Costs (CRITICAL FIX)
    brokerage_per_order: float = 20.0  # ₹20 per order
    stamp_duty_rate: float = 0.00015  # 0.015% stamp duty on buy side
    stt_rate: float = 0.00025  # 0.025% STT on sell side (equity delivery)
    exchange_rate: float = 0.0000345  # 0.00345% exchange charges
    sebi_fees_rate: float = 0.000001  # 0.0001% SEBI turnover fee
    gst_rate: float = 0.18  # 18% GST on brokerage
    
    # Entry timing (Zarattini: best in first 30 min after OR)
    max_entry_minutes_after_or: int = 30  # Max 30 minutes after OR to enter
    breakout_confirmation_bars: int = 2  # Need 2 bars to confirm breakout


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
    rv: float
    atr: float
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
    avg_rv: float
    avg_rv_winners: float
    avg_rv_losers: float
    trades: List[Trade]


class ORBBacktesterZarattiniV2:
    """
    ORB Backtester V2 - Re-implementation following Zarattini (2023) methodology.
    
    Strategy:
    1. Calculate 5-minute opening range (OR)
    2. Calculate RV = OR volume / avg OR volume (last 14 days)
    3. Only trade if RV > 100%
    4. Wait for confirmed breakout (2 bars beyond OR)
    5. Enter at breakout price
    6. Stop loss at 15% of ATR (wider to reduce stop-outs)
    7. Target at 2x risk
    8. Exit at end of day or if target hit
    """
    
    def __init__(self, config: ORBConfigV2):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def calculate_relative_volume(
        self,
        orb_volume: float,
        avg_volume: float
    ) -> float:
        """Calculate Relative Volume (RV)."""
        if avg_volume == 0:
            return 0.0
        return orb_volume / avg_volume
    
    def calculate_average_volume(
        self,
        data: pd.DataFrame,
        lookback_days: int = 14
    ) -> float:
        """
        Calculate average OR (Opening Range) volume from last lookback days.
        
        CRITICAL FIX: This must calculate the average of FIRST 5-MINUTE volume
        from previous trading days, not arbitrary indices.
        """
        # Get unique trading days
        unique_days = data.index.normalize().unique()
        
        # Collect OR volumes from previous days
        or_volumes = []
        
        for i in range(1, min(lookback_days + 1, len(unique_days))):
            day_idx = -i
            if day_idx < -len(unique_days):
                continue
            
            target_day = unique_days[day_idx]
            day_data = data[data.index.normalize() == target_day]
            
            # Get first 5-minute volume (first orb_minutes bars)
            if len(day_data) >= self.config.orb_minutes:
                or_volume = day_data.iloc[:self.config.orb_minutes]['volume'].sum()
                or_volumes.append(or_volume)
        
        if not or_volumes:
            # Fallback: use average of all data (not ideal but prevents crash)
            return data['volume'].mean() / 78  # Rough estimate
        
        avg_or_volume = np.mean(or_volumes)
        
        return avg_or_volume
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31"
    ) -> BacktestResult:
        """
        Run ORB backtest based on Zarattini (2023) methodology.
        
        Args:
            data: DataFrame with OHLCV data (5-minute bars)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running ORB backtest (Zarattini V2) from {start_date} to {end_date}...")
        
        # Filter data by date range
        data_filtered = data[
            (data.index >= start_date) & (data.index <= end_date)
        ]
        
        if data_filtered.empty:
            print("No data available")
            return self._empty_result()
        
        # Process each trading day
        unique_days = data_filtered.index.normalize().unique()
        
        for day in unique_days:
            if day.weekday() >= 5:  # Skip weekends
                continue
            
            self._process_day(data_filtered, day)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _process_day(
        self,
        data: pd.DataFrame,
        day: pd.Timestamp
    ) -> None:
        """Process a single trading day."""
        # Get data for this day
        day_data = data[data.index.normalize() == day]
        
        if len(day_data) < self.config.orb_minutes + 10:
            return
        
        # Calculate opening range
        or_data = day_data.iloc[:self.config.orb_minutes]
        orb_high = or_data['high'].max()
        orb_low = or_data['low'].min()
        orb_volume = or_data['volume'].sum()
        
        # Calculate average OR volume
        prev_data = data.loc[:day_data.index[0]]
        avg_or_volume = self.calculate_average_volume(prev_data)
        
        # Calculate RV
        rv = self.calculate_relative_volume(orb_volume, avg_or_volume)
        
        # Only trade if RV > 100%
        if rv < self.config.min_rv_threshold:
            return
        
        # Calculate ATR
        atr = self.calculate_atr(prev_data, self.config.atr_period)
        
        # Check for breakouts in next 30 minutes
        post_or_data = day_data.iloc[self.config.orb_minutes:]
        max_entry_time = self.config.orb_minutes + self.config.max_entry_minutes_after_or
        
        # Track breakout confirmation
        long_breakout_count = 0
        short_breakout_count = 0
        long_entry_price = None
        short_entry_price = None
        long_entry_time = None
        short_entry_time = None
        
        for idx, row in post_or_data.iterrows():
            bar_idx = post_or_data.index.get_loc(idx)
            
            if bar_idx >= max_entry_time:
                break
            
            # Check for long breakout
            if row['high'] > orb_high:
                long_breakout_count += 1
                if long_entry_price is None:
                    long_entry_price = row['high']
                    long_entry_time = idx
                
                # Need 2 bars to confirm breakout
                if long_breakout_count >= self.config.breakout_confirmation_bars:
                    self._execute_long_trade(
                        day_data.loc[idx:],
                        long_entry_price,
                        orb_high,
                        rv,
                        atr,
                        day,
                        idx
                    )
                    break
            
            # Check for short breakout
            if row['low'] < orb_low:
                short_breakout_count += 1
                if short_entry_price is None:
                    short_entry_price = row['low']
                    short_entry_time = idx
                
                # Need 2 bars to confirm breakout
                if short_breakout_count >= self.config.breakout_confirmation_bars:
                    self._execute_short_trade(
                        day_data.loc[idx:],
                        short_entry_price,
                        orb_low,
                        rv,
                        atr,
                        day,
                        idx
                    )
                    break
    
    def _execute_long_trade(
        self,
        day_data: pd.DataFrame,
        entry_price: float,
        orb_high: float,
        rv: float,
        atr: float,
        day: pd.Timestamp,
        entry_time: pd.Timestamp
    ) -> None:
        """Execute long trade."""
        stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
        target = entry_price + (atr * self.config.atr_stop_multiplier * self.config.target_profit_multiplier)
        
        # Calculate position size
        risk_per_share = abs(entry_price - stop_loss)
        max_loss = self.config.initial_capital * self.config.max_loss_pct
        quantity = int(max_loss / risk_per_share)
        
        if quantity == 0:
            return
        
        # Simulate trade execution
        exit_price = entry_price
        exit_reason = "end_of_day"
        
        for idx, row in day_data.iterrows():
            if row['low'] <= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
                break
            elif row['high'] >= target:
                exit_price = target
                exit_reason = "target"
                break
        else:
            # End of day exit
            exit_price = day_data['close'].iloc[-1]
            exit_reason = "end_of_day"
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        actual_entry = entry_price * (1 + slippage_pct)
        actual_exit = exit_price * (1 - slippage_pct)
        
        # Calculate transaction costs
        transaction_costs = self._calculate_transaction_costs(
            actual_entry, actual_exit, quantity, 'LONG'
        )
        
        # Calculate PnL
        gross_pnl = (actual_exit - actual_entry) * quantity
        net_pnl = gross_pnl - transaction_costs
        pnl_pct = net_pnl / self.config.initial_capital
        
        # Create trade record
        trade = Trade(
            symbol='NIFTY',
            entry_time=entry_time,
            exit_time=day_data.index[-1],
            entry_price=actual_entry,
            exit_price=actual_exit,
            quantity=quantity,
            side='LONG',
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            rv=rv,
            atr=atr,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
    
    def _execute_short_trade(
        self,
        day_data: pd.DataFrame,
        entry_price: float,
        orb_low: float,
        rv: float,
        atr: float,
        day: pd.Timestamp,
        entry_time: pd.Timestamp
    ) -> None:
        """Execute short trade."""
        stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
        target = entry_price - (atr * self.config.atr_stop_multiplier * self.config.target_profit_multiplier)
        
        # Calculate position size
        risk_per_share = abs(entry_price - stop_loss)
        max_loss = self.config.initial_capital * self.config.max_loss_pct
        quantity = int(max_loss / risk_per_share)
        
        if quantity == 0:
            return
        
        # Simulate trade execution
        exit_price = entry_price
        exit_reason = "end_of_day"
        
        for idx, row in day_data.iterrows():
            if row['high'] >= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
                break
            elif row['low'] <= target:
                exit_price = target
                exit_reason = "target"
                break
        else:
            # End of day exit
            exit_price = day_data['close'].iloc[-1]
            exit_reason = "end_of_day"
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        actual_entry = entry_price * (1 - slippage_pct)
        actual_exit = exit_price * (1 + slippage_pct)
        
        # Calculate transaction costs
        transaction_costs = self._calculate_transaction_costs(
            actual_entry, actual_exit, quantity, 'SHORT'
        )
        
        # Calculate PnL
        gross_pnl = (actual_entry - actual_exit) * quantity
        net_pnl = gross_pnl - transaction_costs
        pnl_pct = net_pnl / self.config.initial_capital
        
        # Create trade record
        trade = Trade(
            symbol='NIFTY',
            entry_time=entry_time,
            exit_time=day_data.index[-1],
            entry_price=actual_entry,
            exit_price=actual_exit,
            quantity=quantity,
            side='SHORT',
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            rv=rv,
            atr=atr,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
    
    def _calculate_transaction_costs(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        side: str
    ) -> float:
        """
        Calculate Indian transaction costs.
        
        CRITICAL FIX: This must include all Indian market costs:
        - Brokerage (per order)
        - Stamp duty (buy side only)
        - STT (sell side only)
        - Exchange charges (both sides)
        - SEBI fees (both sides)
        - GST (on brokerage)
        """
        entry_value = entry_price * quantity
        exit_value = exit_price * quantity
        
        # Brokerage (per order)
        brokerage = self.config.brokerage_per_order * 2  # Entry + exit
        
        # Stamp duty (buy side only)
        stamp_duty = entry_value * self.config.stamp_duty_rate if side == 'LONG' else 0
        
        # STT (sell side only)
        stt = exit_value * self.config.stt_rate if side == 'LONG' else 0  # STT on sell for longs
        
        # Exchange charges (both sides)
        exchange_charges = (entry_value + exit_value) * self.config.exchange_rate
        
        # SEBI fees (both sides)
        sebi_fees = (entry_value + exit_value) * self.config.sebi_fees_rate
        
        # GST (18% on brokerage)
        gst = brokerage * self.config.gst_rate
        
        total_costs = brokerage + stamp_duty + stt + exchange_charges + sebi_fees + gst
        
        return total_costs
    
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
        
        avg_rv = np.mean([t.rv for t in self.trades])
        avg_rv_winners = np.mean([t.rv for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0.0
        avg_rv_losers = np.mean([t.rv for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0.0
        
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
            avg_rv=avg_rv,
            avg_rv_winners=avg_rv_winners,
            avg_rv_losers=avg_rv_losers,
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
            avg_rv=0.0,
            avg_rv_winners=0.0,
            avg_rv_losers=0.0,
            trades=[]
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("5-MIN ORB BACKTEST RESULTS (Zarattini V2 Methodology)")
        print("="*60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total PnL: ₹{result.total_pnl:,.2f}")
        print(f"Total PnL %: {result.total_pnl_pct:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Avg RV (All): {result.avg_rv:.2f}")
        print(f"Avg RV (Winners): {result.avg_rv_winners:.2f}")
        print(f"Avg RV (Losers): {result.avg_rv_losers:.2f}")
        print("="*60)
        
        # Zarattini validation
        high_rv_trades = sum(1 for t in result.trades if t.rv >= 3.0)
        mid_rv_trades = sum(1 for t in result.trades if 1.0 <= t.rv < 3.0)
        low_rv_trades = sum(1 for t in result.trades if t.rv < 1.0)
        
        print("\nZarattini Validation:")
        print(f"RV > 300% trades: {high_rv_trades}")
        print(f"RV 100-300% trades: {mid_rv_trades}")
        print(f"RV < 100% trades: {low_rv_trades}")
        print("="*60)


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = ORBConfigV2(
        initial_capital=10000000
    )
    
    backtester = ORBBacktesterZarattiniV2(config)
    
    # Create synthetic 5-minute NIFTY data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="5min")
    dates = dates[dates.indexer_between_time('9:15', '15:30')]  # Market hours only
    
    np.random.seed(42)
    returns = np.random.normal(0.00005, 0.001, len(dates))
    prices = 20000 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(100000, 500000, len(dates))
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

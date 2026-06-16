"""
5-Minute Opening Range Breakout Strategy
Based on Zarattini et al. (2024) methodology

Key findings from research:
- 1,637% return (2016-2023) with Sharpe 2.81
- RV < 100%: -0.02R per trade
- RV 100-300%: +0.08R per trade  
- RV > 300%: +0.38R per trade
- 10% ATR optimal stop loss

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Install with: pip install TA-Lib")


@dataclass
class ORBConfig:
    """Configuration for ORB strategy based on Zarattini methodology"""
    # Time parameters
    orb_minutes: int = 5  # 5-minute opening range
    session_start: time = time(9, 15)  # NSE market open
    session_end: time = time(15, 30)  # NSE market close
    
    # Volume parameters (Zarattini findings)
    min_rv_threshold: float = 1.0  # RV > 100% minimum
    high_rv_threshold: float = 3.0  # RV > 300% for best trades
    
    # ATR parameters (Zarattini: 10% ATR optimal stop, but increased for NIFTY futures)
    atr_period: int = 14
    atr_stop_multiplier: float = 0.1  # 10% of ATR (restored to original paper's parameter to remove in-sample bias)
    
    # Position sizing
    max_position_pct: float = 0.02  # 2% per position
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Risk management
    max_loss_pct: float = 0.02  # 2% max loss per trade
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
    rv: float  # Relative Volume
    atr: float  # ATR at entry
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


class ORBBacktesterZarattini:
    """
    5-Minute ORB Backtester based on Zarattini et al. (2024) methodology.
    
    Strategy:
    1. Calculate 5-minute opening range (9:15-9:20)
    2. Calculate Relative Volume (RV) = OR volume / avg volume
    3. Only trade if RV > 100%
    4. Enter on breakout above/below OR
    5. Stop loss at 10% ATR (Zarattini optimal)
    6. Target at 2x risk
    """
    
    def __init__(self, config: ORBConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR using Wilder's smoothing."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        if TALIB_AVAILABLE:
            atr = talib.ATR(high, low, close, timeperiod=period)
            return pd.Series(atr, index=data.index)
        else:
            # Fallback: manual ATR calculation
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            tr[0] = 0  # First value is NaN
            
            # Wilder's smoothing
            atr = np.zeros_like(tr)
            atr[period - 1] = np.mean(tr[:period])
            
            for i in range(period, len(tr)):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            
            return pd.Series(atr, index=data.index)
    
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
            print(f"WARNING: Could not calculate OR volume average. Using fallback.")
            return data['volume'].mean() / 78  # Rough estimate
        
        avg_or_volume = np.mean(or_volumes)
        
        return avg_or_volume
    
    def identify_stocks_in_play(
        self,
        data: pd.DataFrame,
        current_time: datetime
    ) -> Dict[str, float]:
        """
        Identify stocks in play based on RV > 100%.
        
        Returns:
            Dictionary mapping symbol to RV ratio
        """
        stocks_in_play = {}
        
        # For single symbol backtest, calculate RV
        orb_volume = data['volume'].iloc[-1]  # Last 5-min volume
        avg_volume = self.calculate_average_volume(data)
        rv = self.calculate_relative_volume(orb_volume, avg_volume)
        
        if rv >= self.config.min_rv_threshold:
            stocks_in_play['NIFTY'] = rv
        
        return stocks_in_play
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run ORB backtest based on Zarattini methodology.
        
        Args:
            data: DataFrame with OHLCV data (5-minute bars)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running 5-min ORB backtest (Zarattini methodology) from {start_date} to {end_date}...")
        
        # Filter data by date range
        data_filtered = data[
            (data.index >= start_date) & (data.index <= end_date)
        ]
        
        if data_filtered.empty:
            print("No data available")
            return self._empty_result()
        
        # Calculate ATR
        atr_series = self.calculate_atr(data_filtered, self.config.atr_period)
        
        # Process each trading day
        unique_days = data_filtered.index.normalize().unique()
        
        for day in unique_days:
            if day.weekday() >= 5:  # Skip weekends
                continue
            
            self._process_day(data_filtered, day, atr_series)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _process_day(
        self,
        data: pd.DataFrame,
        day: pd.Timestamp,
        atr_series: pd.Series
    ) -> None:
        """Process a single trading day."""
        # Get data for this day
        day_data = data[data.index.normalize() == day]
        
        if len(day_data) < self.config.orb_minutes + 1:
            return
        
        # Get opening range (first 5 minutes)
        orb_data = day_data.iloc[:self.config.orb_minutes]
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        orb_close = orb_data['close'].iloc[-1]
        orb_volume = orb_data['volume'].sum()
        
        # Calculate RV
        avg_volume = self.calculate_average_volume(data.loc[:day])
        rv = self.calculate_relative_volume(orb_volume, avg_volume)
        
        # Check if stock is in play (RV > 100%)
        if rv < self.config.min_rv_threshold:
            return
        
        # Use the last completed bar before today's session. Using same-day high/low
        # would make the stop depend on information not known at the open.
        prior_atr = atr_series[atr_series.index < day_data.index[0]].dropna()
        if prior_atr.empty:
            return
        atr = float(prior_atr.iloc[-1])
        
        # Determine direction (Zarattini: trade in direction of OR close)
        if orb_close > (orb_high + orb_low) / 2:
            # Bullish OR - look for breakout above OR high
            self._check_long_breakout(day_data, orb_high, rv, atr, day)
        else:
            # Bearish OR - look for breakdown below OR low
            self._check_short_breakout(day_data, orb_low, rv, atr, day)
    
    def _check_long_breakout(
        self,
        day_data: pd.DataFrame,
        orb_high: float,
        rv: float,
        atr: float,
        day: pd.Timestamp
    ) -> None:
        """Check for long breakout above OR high."""
        # Look for breakout in next 10 minutes
        post_orb_data = day_data.iloc[self.config.orb_minutes:self.config.orb_minutes + 10]
        
        breakout_mask = post_orb_data['high'] > orb_high
        if breakout_mask.any():
            idx = breakout_mask.idxmax()
            entry_price = orb_high
            stop_loss = entry_price - (atr * self.config.atr_stop_multiplier)
            target = entry_price + (atr * self.config.atr_stop_multiplier * self.config.target_profit_multiplier)

            if stop_loss >= entry_price:
                raise ValueError(f"Stop loss {stop_loss:.2f} >= entry {entry_price:.2f}")
            
            # Simulate execution
            self._execute_trade(
                symbol='NIFTY',
                entry_time=idx,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                side='LONG',
                rv=rv,
                atr=atr,
                day_data=day_data.loc[idx:]
            )
    
    def _check_short_breakout(
        self,
        day_data: pd.DataFrame,
        orb_low: float,
        rv: float,
        atr: float,
        day: pd.Timestamp
    ) -> None:
        """Check for short breakdown below OR low."""
        # Look for breakdown in next 10 minutes
        post_orb_data = day_data.iloc[self.config.orb_minutes:self.config.orb_minutes + 10]
        
        breakdown_mask = post_orb_data['low'] < orb_low
        if breakdown_mask.any():
            idx = breakdown_mask.idxmax()
            entry_price = orb_low
            stop_loss = entry_price + (atr * self.config.atr_stop_multiplier)
            target = entry_price - (atr * self.config.atr_stop_multiplier * self.config.target_profit_multiplier)

            if stop_loss <= entry_price:
                raise ValueError(f"Stop loss {stop_loss:.2f} <= entry {entry_price:.2f}")
            
            # Simulate execution
            self._execute_trade(
                symbol='NIFTY',
                entry_time=idx,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target=target,
                side='SHORT',
                rv=rv,
                atr=atr,
                day_data=day_data.loc[idx:]
            )
    
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
        
        buy_value = entry_value if side == 'LONG' else exit_value
        sell_value = exit_value if side == 'LONG' else entry_value

        # Stamp duty is charged on the buy leg; STT is charged on the sell leg.
        stamp_duty = buy_value * self.config.stamp_duty_rate
        stt = sell_value * self.config.stt_rate
        
        # Exchange charges (both sides)
        exchange_charges = (entry_value + exit_value) * self.config.exchange_rate
        
        # SEBI fees (both sides)
        sebi_fees = (entry_value + exit_value) * self.config.sebi_fees_rate
        
        # GST (18% on brokerage)
        gst = brokerage * self.config.gst_rate
        
        total_costs = brokerage + stamp_duty + stt + exchange_charges + sebi_fees + gst
        
        return total_costs
    
    def _execute_trade(
        self,
        symbol: str,
        entry_time: pd.Timestamp,
        entry_price: float,
        stop_loss: float,
        target: float,
        side: str,
        rv: float,
        atr: float,
        day_data: pd.DataFrame
    ) -> None:
        """Execute trade with stop loss and target."""
        # Calculate position size
        risk_per_share = abs(entry_price - stop_loss)
        max_loss = self.config.initial_capital * self.config.max_loss_pct
        quantity = int(max_loss / risk_per_share)
        
        if quantity == 0:
            return
        
        # Simulate trade execution
        exit_price = entry_price
        exit_reason = "end_of_day"
        
        future_data = day_data[day_data.index > entry_time]
        if not future_data.empty:
            if side == 'LONG':
                sl_mask = future_data['low'] <= stop_loss
                tp_mask = future_data['high'] >= target
            else:
                sl_mask = future_data['high'] >= stop_loss
                tp_mask = future_data['low'] <= target
            
            sl_idx = sl_mask.idxmax() if sl_mask.any() else pd.Timestamp.max
            tp_idx = tp_mask.idxmax() if tp_mask.any() else pd.Timestamp.max
            
            if sl_idx == pd.Timestamp.max and tp_idx == pd.Timestamp.max:
                exit_price = day_data['close'].iloc[-1]
                exit_reason = "end_of_day"
            elif sl_idx <= tp_idx:
                exit_price = stop_loss
                exit_reason = "stop_loss"
            else:
                exit_price = target
                exit_reason = "target"
        else:
            # End of day exit
            exit_price = day_data['close'].iloc[-1]
            exit_reason = "end_of_day"
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        if side == 'LONG':
            actual_entry = entry_price * (1 + slippage_pct)
            actual_exit = exit_price * (1 - slippage_pct)
        else:
            actual_entry = entry_price * (1 - slippage_pct)
            actual_exit = exit_price * (1 + slippage_pct)
        
        # Calculate transaction costs (CRITICAL FIX)
        transaction_costs = self._calculate_transaction_costs(
            actual_entry, actual_exit, quantity, side
        )
        
        # Calculate PnL
        if side == 'LONG':
            gross_pnl = (actual_exit - actual_entry) * quantity
        else:
            gross_pnl = (actual_entry - actual_exit) * quantity
        
        net_pnl = gross_pnl - transaction_costs
        pnl_pct = net_pnl / self.config.initial_capital
        
        # Debug output for costs
        print(f"DEBUG COSTS ({side}):")
        print(f"  Gross PnL: ₹{gross_pnl:,.2f}")
        print(f"  Transaction Costs: ₹{transaction_costs:,.2f}")
        print(f"  Net PnL: ₹{net_pnl:,.2f}")
        print(f"  Cost % of gross: {transaction_costs/abs(gross_pnl)*100 if gross_pnl != 0 else 0:.2f}%")
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_time=entry_time,
            exit_time=day_data.index[-1],
            entry_price=actual_entry,
            exit_price=actual_exit,
            quantity=quantity,
            side=side,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            rv=rv,
            atr=atr,
            exit_reason=exit_reason
        )
        
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
        
        avg_rv = np.mean([t.rv for t in self.trades])
        avg_rv_winners = np.mean([t.rv for t in self.trades if t.pnl > 0])
        avg_rv_losers = np.mean([t.rv for t in self.trades if t.pnl < 0])
        
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
        print("5-MIN ORB BACKTEST RESULTS (Zarattini Methodology)")
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
        print("\nZarattini Validation:")
        print(f"RV > 300% trades: {sum(1 for t in result.trades if t.rv > 3.0)}")
        print(f"RV 100-300% trades: {sum(1 for t in result.trades if 1.0 <= t.rv <= 3.0)}")
        print(f"RV < 100% trades: {sum(1 for t in result.trades if t.rv < 1.0)}")
        print("="*60)


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = ORBConfig(
        initial_capital=10000000
    )
    
    backtester = ORBBacktesterZarattini(config)
    
    # Create synthetic 5-minute NIFTY data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="5min")
    dates = dates[dates.indexer_between_time('9:15', '15:30')]  # Market hours only
    
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.001, len(dates))
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


def scan_symbols(data: dict, current_time) -> List[dict]:
    """Scan a universe for opening-range breakout signals.

    Expects each value in ``data`` to be an OHLCV DataFrame. The latest session
    is scanned and prior sessions are used to estimate normal opening volume.
    """
    signals = []
    for symbol, df in data.items():
        if df is None or df.empty:
            continue

        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            continue

        frame = df.sort_index().copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            continue

        config = ORBConfig()
        latest_day = frame.index.normalize().max()
        day_data = frame[frame.index.normalize() == latest_day]
        if len(day_data) < config.orb_minutes + 1:
            continue

        orb_data = day_data.iloc[:config.orb_minutes]
        post_orb = day_data.iloc[config.orb_minutes:]
        if post_orb.empty:
            continue

        orb_volume = float(orb_data['volume'].sum())
        prior = frame[frame.index.normalize() < latest_day]
        prior_or_volumes = []
        for _, prior_day in prior.groupby(prior.index.normalize()):
            if len(prior_day) >= config.orb_minutes:
                prior_or_volumes.append(float(prior_day.iloc[:config.orb_minutes]['volume'].sum()))

        avg_volume = float(np.mean(prior_or_volumes[-14:])) if prior_or_volumes else float(frame['volume'].tail(78).mean() * config.orb_minutes)
        if avg_volume <= 0:
            continue

        rv = orb_volume / avg_volume
        if rv < config.min_rv_threshold:
            continue

        orb_high = float(orb_data['high'].max())
        orb_low = float(orb_data['low'].min())
        latest = post_orb.iloc[-1]
        latest_price = float(latest['close'])
        atr = ORBBacktesterZarattini(config).calculate_atr(frame, config.atr_period).dropna()
        volatility = float(atr.iloc[-1] / latest_price) if not atr.empty and latest_price > 0 else float(frame['close'].pct_change().tail(20).std())
        volatility = max(volatility, 1e-4)

        range_size = max(orb_high - orb_low, latest_price * volatility)
        confidence = float(np.clip(rv / config.high_rv_threshold, 0.0, 1.0))

        if latest_price > orb_high:
            target = latest_price + config.target_profit_multiplier * range_size
            signals.append({
                'symbol': symbol,
                'rv': rv,
                'direction': 1,
                'entry': latest_price,
                'stop': orb_low,
                'target': target,
                'expected_return': (target - latest_price) / latest_price,
                'confidence': confidence,
                'volatility': volatility,
            })
        elif latest_price < orb_low:
            target = latest_price - config.target_profit_multiplier * range_size
            signals.append({
                'symbol': symbol,
                'rv': rv,
                'direction': -1,
                'entry': latest_price,
                'stop': orb_high,
                'target': target,
                'expected_return': (latest_price - target) / latest_price,
                'confidence': confidence,
                'volatility': volatility,
            })
    # Sort by RV descending, take top 5
    signals.sort(key=lambda x: x['rv'], reverse=True)
    return signals[:5]

if __name__ == "__main__":
    run_sample_backtest()

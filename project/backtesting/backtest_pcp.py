"""
Backtesting Framework for Put-Call Carry Strategy
Weekly Options Expiry with IV filtering

Architecture V2 - Quantitative Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
from dataclasses import dataclass


@dataclass
class PCPBacktestConfig:
    """Configuration for Put-Call Carry backtest"""
    # Strategy parameters
    entry_day: str = "Wednesday"  # Entry on Wednesday
    exit_day: str = "Thursday"  # Exit on Thursday (expiry)
    
    # IV parameters
    iv_percentile_threshold: float = 0.70  # 70th percentile
    iv_lookback_days: int = 30
    min_iv: float = 0.15  # 15% minimum IV
    
    # Strike selection
    otm_distance_pct: float = 0.05  # 5% OTM
    min_days_to_expiry: int = 1
    max_days_to_expiry: int = 3
    lot_size: int = 50
    margin_pct_notional: float = 0.15
    
    # Position sizing
    max_position_pct: float = 0.02
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Slippage (options have wider spreads)
    slippage_bps: float = 5.0
    brokerage_per_order: float = 40.0
    stt_rate: float = 0.0005
    exchange_rate: float = 0.00005
    sebi_fees_rate: float = 0.000001
    gst_rate: float = 0.18
    
    # Risk parameters
    max_loss_pct: float = 0.10  # 10% max loss per position


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    option_type: str  # "call" or "put"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str
    pnl: float
    pnl_pct: float
    iv_entry: float
    iv_exit: float
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
    avg_holding_days: float
    avg_iv_entry: float
    trades: List[Trade]


class PCPBacktester:
    """
    Backtester for Put-Call Carry Strategy on weekly options.
    
    Strategy:
    1. On Wednesday, identify high IV environment (IV > 70th percentile)
    2. Sell OTM strangle (both call and put)
    3. Close on Thursday before expiry
    4. Profit from theta decay
    5. Apply conservative slippage (5 bps for options)
    """
    
    def __init__(self, config: PCPBacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_iv_percentile(self, current_iv: float, iv_history: List[float]) -> float:
        """Calculate IV percentile based on historical data."""
        if not iv_history:
            return 0.5
        
        iv_history_sorted = sorted(iv_history)
        rank = sum(1 for iv in iv_history_sorted if iv <= current_iv)
        percentile = rank / len(iv_history)
        
        return percentile
    
    def select_otm_strikes(
        self,
        underlying_price: float,
        iv: float,
        days_to_expiry: int
    ) -> Tuple[float, float]:
        """
        Select OTM strike prices for strangle.
        
        Returns:
            (call_strike, put_strike) tuple
        """
        # Calculate OTM distance based on IV and time
        otm_amount = underlying_price * self.config.otm_distance_pct
        
        call_strike = underlying_price + otm_amount
        put_strike = underlying_price - otm_amount
        
        return call_strike, put_strike
    
    def estimate_option_price(
        self,
        spot: float,
        strike: float,
        iv: float,
        days_to_expiry: int,
        option_type: str
    ) -> float:
        """
        Estimate option price using Black-Scholes approximation.
        """
        from scipy.stats import norm
        
        # Convert to years
        T = days_to_expiry / 365.0
        
        if T <= 0:
            return 0.0
        
        # Risk-free rate (assume 5%)
        r = 0.05
        
        # Calculate d1 and d2
        d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(price, 0.01)  # Minimum price

    def calculate_option_transaction_costs(
        self,
        entry_premium: float,
        exit_premium: float,
        quantity: int,
        side: str,
    ) -> float:
        """Calculate option costs for a single leg."""
        entry_value = entry_premium * quantity
        exit_value = exit_premium * quantity
        sell_value = entry_value if side == "SHORT" else exit_value
        turnover = entry_value + exit_value

        brokerage = self.config.brokerage_per_order * 2
        stt = sell_value * self.config.stt_rate
        exchange = turnover * self.config.exchange_rate
        sebi = turnover * self.config.sebi_fees_rate
        gst = brokerage * self.config.gst_rate
        return brokerage + stt + exchange + sebi + gst
    
    def run_backtest(
        self,
        options_data: Dict[str, pd.DataFrame],
        underlying_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run Put-Call Carry backtest on historical data.
        """
        print(f"Running Put-Call Carry backtest from {start_date} to {end_date}...")
        
        # Filter data by date range
        underlying_filtered = underlying_data[
            (underlying_data.index >= start_date) & (underlying_data.index <= end_date)
        ]
        
        if underlying_filtered.empty:
            print("No underlying data available")
            return self._empty_result()
        
        # Process each week
        unique_weeks = self._get_weekly_dates(underlying_filtered.index)
        
        for week_start, week_end in unique_weeks:
            self._process_week(underlying_filtered, week_start, week_end)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _get_weekly_dates(self, dates: pd.DatetimeIndex) -> List[Tuple[datetime, datetime]]:
        """Get weekly date ranges (Wednesday to Thursday)."""
        weeks = []
        
        for date in dates:
            if date.weekday() == 2:  # Wednesday
                week_start = date
                week_end = date + timedelta(days=1)  # Thursday
                weeks.append((week_start, week_end))
        
        return weeks
    
    def _process_week(
        self,
        underlying_data: pd.DataFrame,
        week_start: datetime,
        week_end: datetime
    ) -> None:
        """Process a single week (Wednesday to Thursday)."""
        # Get data for this week
        week_data = underlying_data[
            (underlying_data.index >= week_start) & (underlying_data.index <= week_end)
        ]
        
        if len(week_data) < 2:
            return
        
        # Get Wednesday data (entry)
        wednesday_data = week_data[week_data.index.dayofweek == 2]
        if wednesday_data.empty:
            return
        
        wednesday_close = wednesday_data['close'].iloc[-1]
        
        # Calculate IV history for percentile
        iv_history = self._get_iv_history(underlying_data, week_start)
        
        # Assume current IV
        current_iv = 0.20  # Placeholder
        
        # Calculate IV percentile
        iv_percentile = self.calculate_iv_percentile(current_iv, iv_history)
        
        # Check if IV is high enough
        if iv_percentile < self.config.iv_percentile_threshold or current_iv < self.config.min_iv:
            return
        
        # Select strikes
        call_strike, put_strike = self.select_otm_strikes(wednesday_close, current_iv, 1)
        
        # Estimate option prices at entry
        call_price_entry = self.estimate_option_price(
            wednesday_close, call_strike, current_iv, 1, "call"
        )
        put_price_entry = self.estimate_option_price(
            wednesday_close, put_strike, current_iv, 1, "put"
        )
        
        # Get Thursday data (exit)
        thursday_data = week_data[week_data.index.dayofweek == 3]
        if thursday_data.empty:
            return
        
        thursday_close = thursday_data['close'].iloc[-1]
        
        # Assume IV at exit (theta decay)
        iv_exit = current_iv * 0.8  # IV typically drops near expiry
        
        # Estimate option prices at exit
        call_price_exit = self.estimate_option_price(
            thursday_close, call_strike, iv_exit, 0, "call"
        )
        put_price_exit = self.estimate_option_price(
            thursday_close, put_strike, iv_exit, 0, "put"
        )
        
        # Execute strangle trades
        self._execute_strangle_trade(
            call_strike, call_price_entry, call_price_exit,
            put_strike, put_price_entry, put_price_exit,
            current_iv, iv_exit, week_start, week_end
        )
    
    def _get_iv_history(self, data: pd.DataFrame, current_date: datetime) -> List[float]:
        """Get IV history for percentile calculation."""
        # Get data for lookback period
        lookback_start = current_date - timedelta(days=self.config.iv_lookback_days)
        historical_data = data[data.index >= lookback_start]
        historical_data = historical_data[historical_data.index < current_date]
        
        # Calculate historical IV
        if len(historical_data) < 5:
            return []
        
        returns = historical_data['close'].pct_change().dropna()
        ivs = []
        
        for i in range(20, len(returns), 5):
            window_returns = returns[i-20:i]
            realized_vol = np.std(window_returns) * np.sqrt(252)
            ivs.append(realized_vol)
        
        return ivs
    
    def _execute_strangle_trade(
        self,
        call_strike: float,
        call_entry: float,
        call_exit: float,
        put_strike: float,
        put_entry: float,
        put_exit: float,
        iv_entry: float,
        iv_exit: float,
        entry_time: datetime,
        exit_time: datetime
    ) -> None:
        """Execute strangle trade (sell both call and put)."""
        position_value = self.config.initial_capital * self.config.max_position_pct
        total_premium = call_entry + put_entry
        
        if total_premium == 0:
            return
        
        underlying_ref = (call_strike + put_strike) / 2
        margin_per_lot = underlying_ref * self.config.lot_size * self.config.margin_pct_notional
        num_lots = int(position_value / margin_per_lot)
        num_strangles = num_lots * self.config.lot_size
        
        if num_strangles == 0:
            return
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        
        # Call trade (short)
        call_actual_entry = call_entry * (1 - slippage_pct)
        call_actual_exit = call_exit * (1 + slippage_pct)
        call_costs = self.calculate_option_transaction_costs(
            call_actual_entry, call_actual_exit, num_strangles, side="SHORT"
        )
        call_pnl = (call_actual_entry - call_actual_exit) * num_strangles - call_costs
        
        # Put trade (short)
        put_actual_entry = put_entry * (1 - slippage_pct)
        put_actual_exit = put_exit * (1 + slippage_pct)
        put_costs = self.calculate_option_transaction_costs(
            put_actual_entry, put_actual_exit, num_strangles, side="SHORT"
        )
        put_pnl = (put_actual_entry - put_actual_exit) * num_strangles - put_costs
        
        # Total PnL
        total_pnl = call_pnl + put_pnl
        
        # Create trade records
        call_trade = Trade(
            symbol=f"NIFTY_{int(call_strike)}_CE",
            option_type="call",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=call_actual_entry,
            exit_price=call_actual_exit,
            quantity=num_strangles,
            side="SHORT",
            pnl=call_pnl,
            pnl_pct=call_pnl / self.config.initial_capital,
            iv_entry=iv_entry,
            iv_exit=iv_exit,
            exit_reason="expiry"
        )
        
        put_trade = Trade(
            symbol=f"NIFTY_{int(put_strike)}_PE",
            option_type="put",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=put_actual_entry,
            exit_price=put_actual_exit,
            quantity=num_strangles,
            side="SHORT",
            pnl=put_pnl,
            pnl_pct=put_pnl / self.config.initial_capital,
            iv_entry=iv_entry,
            iv_exit=iv_exit,
            exit_reason="expiry"
        )
        
        self.trades.extend([call_trade, put_trade])
    
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
        
        avg_holding_days = 1.0  # Always 1 day for this strategy
        
        avg_iv_entry = np.mean([t.iv_entry for t in self.trades])
        
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
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0.0  # Weekly
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
            avg_holding_days=avg_holding_days,
            avg_iv_entry=avg_iv_entry,
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
            avg_holding_days=0.0,
            avg_iv_entry=0.0,
            trades=[]
        )

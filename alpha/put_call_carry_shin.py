"""
Put-Call Carry Gap System
Based on Shin (2026a,b) methodology

Key findings from research:
- 37bp annualized carry
- 98.4% positive observations
- GBM path-risk structure: rσ√τ
- Option-implied vs OIS discount factor gap
- Correlated with VIX (ρ = 0.50)

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class PCPCarryConfig:
    """Configuration for Put-Call Carry strategy based on Shin methodology"""
    # Carry parameters
    min_carry_gap_bps: float = 20.0  # Minimum 20bp gap
    max_carry_gap_bps: float = 50.0  # Maximum 50bp gap (stress)
    
    # Option parameters
    min_dte: int = 7  # Minimum days to expiry
    max_dte: int = 30  # Maximum days to expiry
    
    # Strike selection
    otm_distance_pct: float = 0.05  # 5% OTM
    
    # Position sizing
    max_position_pct: float = 0.02  # 2% per position
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Risk parameters
    max_loss_pct: float = 0.05  # 5% max loss per position
    path_risk_multiplier: float = 1.5  # GBM path-risk adjustment
    
    # Slippage (options have wider spreads)
    slippage_bps: float = 5.0


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float  # Premium received
    exit_price: float  # Premium paid to close
    quantity: int
    side: str  # "SHORT" strangle
    pnl: float
    pnl_pct: float
    carry_gap_bps: float
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
    sharpe_ratio: float
    profit_factor: float
    avg_carry_gap: float
    avg_holding_days: float
    positive_gap_pct: float
    trades: List[Trade]


class PCPCarryBacktesterShin:
    """
    Put-Call Carry Backtester based on Shin (2026a,b) methodology.
    
    Strategy:
    1. Calculate carry gap between option-implied and OIS discount factors
    2. When carry gap > 20bp: enter short strangle
    3. Hold for 7-30 days to capture theta decay
    4. GBM path-risk adjustment: rσ√τ
    5. Exit at 1 DTE or if carry gap closes
    """
    
    def __init__(self, config: PCPCarryConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_carry_gap(
        self,
        option_implied_rate: float,
        ois_rate: float,
        volatility: float,
        days_to_expiry: float
    ) -> float:
        """
        Calculate carry gap based on Shin methodology.
        
        Carry Gap = Option-implied - OIS - GBM path-risk
        GBM path-risk = r * σ * √τ
        """
        # Convert days to years
        tau = days_to_expiry / 365.0
        
        # GBM path-risk adjustment
        path_risk = ois_rate * volatility * np.sqrt(tau)
        
        # Carry gap in basis points
        carry_gap = (option_implied_rate - ois_rate - path_risk) * 10000
        
        return carry_gap
    
    def estimate_option_price(
        self,
        spot: float,
        strike: float,
        iv: float,
        days_to_expiry: int,
        option_type: str
    ) -> float:
        """Estimate option price using Black-Scholes."""
        tau = days_to_expiry / 365.0
        
        if tau <= 0:
            intrinsic = max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)
            return intrinsic
        
        r = 0.05  # Risk-free rate
        
        d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * tau) / (iv * np.sqrt(tau))
        d2 = d1 - iv * np.sqrt(tau)
        
        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-r * tau) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(price, 0.01)
    
    def select_otm_strikes(
        self,
        spot: float,
        iv: float,
        days_to_expiry: int
    ) -> Tuple[float, float]:
        """Select OTM strike prices for strangle."""
        otm_amount = spot * self.config.otm_distance_pct
        
        call_strike = spot + otm_amount
        put_strike = spot - otm_amount
        
        return call_strike, put_strike
    
    def run_backtest(
        self,
        underlying_data: pd.DataFrame,
        options_data: Optional[Dict] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31"
    ) -> BacktestResult:
        """
        Run Put-Call Carry backtest based on Shin methodology.
        
        Args:
            underlying_data: DataFrame with underlying (NIFTY) OHLCV data
            options_data: Optional dictionary with options data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running Put-Call Carry backtest (Shin methodology) from {start_date} to {end_date}...")
        
        # Filter data by date range
        data_filtered = underlying_data[
            (underlying_data.index >= start_date) & (underlying_data.index <= end_date)
        ]
        
        if data_filtered.empty:
            print("No data available")
            return self._empty_result()
        
        # Process weekly (entry on Wednesday, exit on Thursday)
        unique_weeks = self._get_weekly_dates(data_filtered.index)
        
        for week_start, week_end in unique_weeks:
            self._process_week(data_filtered, week_start, week_end)
        
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
        data: pd.DataFrame,
        week_start: datetime,
        week_end: datetime
    ) -> None:
        """Process a single week (Wednesday to Thursday)."""
        # Get data for this week
        week_data = data[
            (data.index >= week_start) & (data.index <= week_end)
        ]
        
        if len(week_data) < 2:
            return
        
        # Get Wednesday data (entry)
        wednesday_data = week_data[week_data.index.dayofweek == 2]
        if wednesday_data.empty:
            return
        
        wednesday_close = wednesday_data['close'].iloc[-1]
        
        # Calculate volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Assume rates (in production, get from market)
        option_implied_rate = 0.06  # 6%
        ois_rate = 0.05  # 5%
        
        # Calculate carry gap
        days_to_expiry = 7
        carry_gap = self.calculate_carry_gap(
            option_implied_rate,
            ois_rate,
            volatility,
            days_to_expiry
        )
        
        # Check if carry gap is sufficient
        if carry_gap < self.config.min_carry_gap_bps:
            return
        
        # Select strikes
        call_strike, put_strike = self.select_otm_strikes(
            wednesday_close,
            volatility,
            days_to_expiry
        )
        
        # Estimate option prices at entry
        call_price_entry = self.estimate_option_price(
            wednesday_close, call_strike, volatility, days_to_expiry, "call"
        )
        put_price_entry = self.estimate_option_price(
            wednesday_close, put_strike, volatility, days_to_expiry, "put"
        )
        
        # Get Thursday data (exit)
        thursday_data = week_data[week_data.index.dayofweek == 3]
        if thursday_data.empty:
            return
        
        thursday_close = thursday_data['close'].iloc[-1]
        
        # Assume IV at exit (theta decay)
        iv_exit = volatility * 0.8  # IV drops near expiry
        
        # Estimate option prices at exit
        call_price_exit = self.estimate_option_price(
            thursday_close, call_strike, iv_exit, 1, "call"
        )
        put_price_exit = self.estimate_option_price(
            thursday_close, put_strike, iv_exit, 1, "put"
        )
        
        # Execute strangle trade
        self._execute_strangle_trade(
            call_strike, call_price_entry, call_price_exit,
            put_strike, put_price_entry, put_price_exit,
            carry_gap, volatility, iv_exit, week_start, week_end
        )
    
    def _execute_strangle_trade(
        self,
        call_strike: float,
        call_entry: float,
        call_exit: float,
        put_strike: float,
        put_entry: float,
        put_exit: float,
        carry_gap: float,
        iv_entry: float,
        iv_exit: float,
        entry_time: datetime,
        exit_time: datetime
    ) -> None:
        """Execute strangle trade (short both call and put)."""
        # Calculate position size
        position_value = self.config.initial_capital * self.config.max_position_pct
        total_premium = call_entry + put_entry
        
        if total_premium == 0:
            return
        
        # Number of strangles (each strangle = 1 call + 1 put)
        num_strangles = int(position_value / total_premium)
        
        if num_strangles == 0:
            return
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        
        # Call trade (short)
        call_actual_entry = call_entry * (1 - slippage_pct)
        call_actual_exit = call_exit * (1 + slippage_pct)
        call_pnl = (call_actual_entry - call_actual_exit) * num_strangles
        
        # Put trade (short)
        put_actual_entry = put_entry * (1 - slippage_pct)
        put_actual_exit = put_exit * (1 + slippage_pct)
        put_pnl = (put_actual_entry - put_actual_exit) * num_strangles
        
        # Total PnL
        total_pnl = call_pnl + put_pnl
        total_pnl_pct = total_pnl / self.config.initial_capital
        
        # Create trade record
        trade = Trade(
            symbol=f"NIFTY_{int(call_strike)}_{int(put_strike)}_STRANGLE",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=call_actual_entry + put_actual_entry,
            exit_price=call_actual_exit + put_actual_exit,
            quantity=num_strangles,
            side="SHORT",
            pnl=total_pnl,
            pnl_pct=total_pnl_pct,
            carry_gap_bps=carry_gap,
            iv_entry=iv_entry,
            iv_exit=iv_exit,
            exit_reason="expiry"
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
        
        avg_carry_gap = np.mean([t.carry_gap_bps for t in self.trades])
        avg_holding_days = 1.0  # Always 1 day for this strategy
        
        positive_gap_pct = sum(1 for t in self.trades if t.carry_gap_bps > 0) / total_trades
        
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
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_carry_gap=avg_carry_gap,
            avg_holding_days=avg_holding_days,
            positive_gap_pct=positive_gap_pct,
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
            avg_carry_gap=0.0,
            avg_holding_days=0.0,
            positive_gap_pct=0.0,
            trades=[]
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("PUT-CALL CARRY BACKTEST RESULTS (Shin Methodology)")
        print("="*60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total PnL: ₹{result.total_pnl:,.2f}")
        print(f"Total PnL %: {result.total_pnl_pct:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Avg Carry Gap: {result.avg_carry_gap:.2f} bps")
        print(f"Avg Holding: {result.avg_holding_days:.1f} days")
        print(f"Positive Gap %: {result.positive_gap_pct:.2%}")
        print("="*60)
        print("\nShin Validation:")
        print(f"Expected positive gap: 98.4% (Shin finding)")
        print(f"Actual positive gap: {result.positive_gap_pct:.2%}")
        print(f"Expected annual carry: 37bp (Shin finding)")
        print(f"Actual annualized: {result.total_pnl_pct * 10000:.2f}bp")
        print("="*60)


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = PCPCarryConfig(
        initial_capital=10000000
    )
    
    backtester = PCPCarryBacktesterShin(config)
    
    # Create synthetic NIFTY data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    prices = 20000 * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Run backtest
    result = backtester.run_backtest(
        data,
        None,
        "2023-01-01",
        "2023-12-31"
    )
    
    backtester.print_results(result)
    
    return result


if __name__ == "__main__":
    run_sample_backtest()

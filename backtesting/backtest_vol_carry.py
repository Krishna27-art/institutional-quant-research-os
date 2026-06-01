"""
Backtesting Framework for Volatility Carry Strategy
Short Straddle with Delta Hedging

Architecture V2 - Quantitative Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class VolCarryBacktestConfig:
    """Configuration for Volatility Carry backtest"""
    # Strategy parameters
    min_dte: int = 5  # Minimum days to expiry
    max_dte: int = 10  # Maximum days to expiry
    
    # Volatility parameters
    iv_rv_spread_threshold: float = 0.10  # 10% spread
    iv_lookback_days: int = 20
    min_iv: float = 0.12  # 12% minimum IV
    
    # Position sizing
    max_position_pct: float = 0.02
    initial_capital: float = 10000000  # ₹1 Crore
    
    # Slippage
    slippage_bps: float = 5.0
    
    # Delta hedging
    hedge_threshold: float = 0.20  # Hedge when |delta| > 0.20
    hedge_frequency: str = "daily"  # daily or intraday
    
    # Risk parameters
    max_loss_pct: float = 0.15  # 15% max loss per position


@dataclass
class Trade:
    """Trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float  # Straddle premium received
    exit_price: float  # Straddle premium paid to close
    quantity: int
    pnl: float
    pnl_pct: float
    iv_entry: float
    iv_exit: float
    hedge_pnl: float
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
    avg_hedge_pnl: float
    trades: List[Trade]


class VolCarryBacktester:
    """
    Backtester for Volatility Carry Strategy (Short Straddle).
    
    Strategy:
    1. Identify when IV > RV by > 10%
    2. Sell ATM straddle with 5-10 DTE
    3. Delta hedge when |delta| > 0.20
    4. Close at 1 DTE or if IV spikes
    5. Profit from vol risk premium and theta decay
    """
    
    def __init__(self, config: VolCarryBacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
    
    def calculate_realized_volatility(self, prices: np.ndarray, days: int = 5) -> float:
        """Calculate realized volatility."""
        if len(prices) < days + 1:
            return 0.0
        
        returns = np.diff(np.log(prices[-days-1:]))
        rv = np.std(returns) * np.sqrt(252)
        
        return rv
    
    def calculate_delta(
        self,
        spot: float,
        strike: float,
        iv: float,
        days_to_expiry: int,
        option_type: str
    ) -> float:
        """Calculate option delta using Black-Scholes."""
        T = days_to_expiry / 365.0
        
        if T <= 0:
            return 1.0 if option_type == "call" else 0.0
        
        r = 0.05  # Risk-free rate
        
        d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        
        if option_type == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        return delta
    
    def estimate_option_price(
        self,
        spot: float,
        strike: float,
        iv: float,
        days_to_expiry: int,
        option_type: str
    ) -> float:
        """Estimate option price using Black-Scholes."""
        T = days_to_expiry / 365.0
        
        if T <= 0:
            intrinsic = max(spot - strike, 0) if option_type == "call" else max(strike - spot, 0)
            return intrinsic
        
        r = 0.05
        
        d1 = (np.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
        d2 = d1 - iv * np.sqrt(T)
        
        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return max(price, 0.01)
    
    def run_backtest(
        self,
        underlying_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run Volatility Carry backtest on historical data.
        
        Args:
            underlying_data: DataFrame with underlying (NIFTY) OHLCV data
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"Running Volatility Carry backtest from {start_date} to {end_date}...")
        
        # Filter data by date range
        data_filtered = underlying_data[
            (underlying_data.index >= start_date) & (underlying_data.index <= end_date)
        ]
        
        if data_filtered.empty:
            print("No data available")
            return self._empty_result()
        
        # Process each day
        dates = data_filtered.index
        
        for i in range(self.config.iv_lookback_days + 1, len(dates) - 1):
            current_date = dates[i]
            
            # Check if we should enter a position
            if self._should_enter_position(data_filtered, i):
                # Enter straddle position
                self._enter_strangle(data_filtered, i, current_date)
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics()
        
        return result
    
    def _should_enter_position(self, data: pd.DataFrame, current_idx: int) -> bool:
        """Check if conditions are met to enter a position."""
        # Get current price
        current_price = data['close'].iloc[current_idx]
        
        # Calculate realized volatility
        prices = data['close'].iloc[:current_idx+1].values
        rv = self.calculate_realized_volatility(prices, 5)
        
        # Assume current IV (in production, would get from options data)
        current_iv = 0.18  # Placeholder
        
        # Check IV-RV spread
        iv_rv_spread = current_iv - rv
        
        if iv_rv_spread < self.config.iv_rv_spread_threshold:
            return False
        
        if current_iv < self.config.min_iv:
            return False
        
        return True
    
    def _enter_strangle(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        entry_date: datetime
    ) -> None:
        """Enter straddle position."""
        current_price = data['close'].iloc[entry_idx]
        
        # Calculate IV
        prices = data['close'].iloc[:entry_idx+1].values
        rv = self.calculate_realized_volatility(prices, 5)
        iv_entry = 0.18  # Placeholder
        
        # Select ATM strike
        atm_strike = round(current_price / 100) * 100
        
        # Estimate option prices
        call_price = self.estimate_option_price(current_price, atm_strike, iv_entry, 7, "call")
        put_price = self.estimate_option_price(current_price, atm_strike, iv_entry, 7, "put")
        
        straddle_premium = call_price + put_price
        
        # Calculate position size
        position_value = self.config.initial_capital * self.config.max_position_pct
        num_straddles = int(position_value / straddle_premium)
        
        if num_straddles == 0:
            return
        
        # Simulate holding period (5 days)
        exit_idx = min(entry_idx + 5, len(data) - 1)
        exit_date = data.index[exit_idx]
        exit_price = data['close'].iloc[exit_idx]
        
        # Calculate IV at exit (theta decay)
        iv_exit = iv_entry * 0.7  # IV drops as expiry approaches
        
        # Estimate option prices at exit
        call_exit = self.estimate_option_price(exit_price, atm_strike, iv_exit, 2, "call")
        put_exit = self.estimate_option_price(exit_price, atm_strike, iv_exit, 2, "put")
        
        straddle_exit_premium = call_exit + put_exit
        
        # Calculate delta hedging PnL
        hedge_pnl = self._calculate_hedge_pnl(
            data, entry_idx, exit_idx, atm_strike, iv_entry, num_straddles
        )
        
        # Apply slippage
        slippage_pct = self.config.slippage_bps / 10000.0
        
        # Entry: receive premium (short straddle)
        actual_entry_premium = straddle_premium * (1 - slippage_pct)
        
        # Exit: pay premium to close
        actual_exit_premium = straddle_exit_premium * (1 + slippage_pct)
        
        # Calculate PnL
        straddle_pnl = (actual_entry_premium - actual_exit_premium) * num_straddles
        total_pnl = straddle_pnl + hedge_pnl
        total_pnl_pct = total_pnl / self.config.initial_capital
        
        # Create trade record
        trade = Trade(
            symbol=f"NIFTY_{int(atm_strike)}_STRADDLE",
            entry_time=entry_date,
            exit_time=exit_date,
            entry_price=actual_entry_premium,
            exit_price=actual_exit_premium,
            quantity=num_straddles,
            pnl=total_pnl,
            pnl_pct=total_pnl_pct,
            iv_entry=iv_entry,
            iv_exit=iv_exit,
            hedge_pnl=hedge_pnl,
            exit_reason="expiry"
        )
        
        self.trades.append(trade)
    
    def _calculate_hedge_pnl(
        self,
        data: pd.DataFrame,
        entry_idx: int,
        exit_idx: int,
        strike: float,
        iv: float,
        quantity: int
    ) -> float:
        """Calculate delta hedging PnL."""
        hedge_pnl = 0.0
        
        # Calculate initial delta
        spot_entry = data['close'].iloc[entry_idx]
        days_to_expiry = 7
        
        call_delta = self.calculate_delta(spot_entry, strike, iv, days_to_expiry, "call")
        put_delta = self.calculate_delta(spot_entry, strike, iv, days_to_expiry, "put")
        
        # Straddle delta (short both)
        straddle_delta = -(call_delta + put_delta)
        
        # If delta is significant, hedge
        if abs(straddle_delta) > self.config.hedge_threshold:
            # Hedge by buying/selling underlying
            hedge_quantity = int(abs(straddle_delta) * quantity)
            
            # Simulate hedge PnL (simplified)
            spot_exit = data['close'].iloc[exit_idx]
            hedge_pnl = (spot_exit - spot_entry) * hedge_quantity * np.sign(straddle_delta)
        
        return hedge_pnl
    
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
        
        avg_holding_days = 5.0  # Average holding period
        
        avg_iv_entry = np.mean([t.iv_entry for t in self.trades])
        avg_hedge_pnl = np.mean([t.hedge_pnl for t in self.trades])
        
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
            avg_holding_days=avg_holding_days,
            avg_iv_entry=avg_iv_entry,
            avg_hedge_pnl=avg_hedge_pnl,
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
            avg_hedge_pnl=0.0,
            trades=[]
        )
    
    def print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "="*60)
        print("VOLATILITY CARRY BACKTEST RESULTS")
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
        print(f"Avg Holding: {result.avg_holding_days:.1f} days")
        print(f"Avg IV Entry: {result.avg_iv_entry:.2%}")
        print(f"Avg Hedge PnL: ₹{result.avg_hedge_pnl:,.2f}")
        print("="*60)


def run_sample_backtest():
    """Run a sample backtest with synthetic data."""
    config = VolCarryBacktestConfig(
        initial_capital=10000000
    )
    
    backtester = VolCarryBacktester(config)
    
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
        "2023-01-01",
        "2023-12-31"
    )
    
    backtester.print_results(result)
    
    return result


if __name__ == "__main__":
    run_sample_backtest()

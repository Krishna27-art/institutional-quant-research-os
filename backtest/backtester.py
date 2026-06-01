"""
Vectorized Backtesting Engine for strategy validation.
Supports multi-strategy portfolio backtests with realistic 
transaction costs and slippage for Indian markets.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000_000  # 1 Cr INR
    brokerage_pct: float = 0.0003       # 0.03% (Zerodha equity)
    stt_pct: float = 0.00025           # Securities Transaction Tax
    transaction_charges_pct: float = 0.0000345  # NSE charges
    gst_pct: float = 0.18              # 18% on brokerage + transaction charges
    sebi_charges_pct: float = 0.000001 # SEBI turnover fees
    stamp_duty_pct: float = 0.00003    # Stamp duty (buyer)
    slippage_bps: float = 2.0          # 2 bps execution slippage
    impact_coefficient: float = 0.1    # Market impact coefficient
    benchmark_symbol: str = "NIFTY"


@dataclass
class BacktestResult:
    strategy_name: str
    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: List[Dict]
    
    # Metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    total_trades: int
    calmar_ratio: float
    
    # Costs
    total_brokerage: float
    total_slippage: float
    total_impact_cost: float


def compute_transaction_cost(
    trade_value: float,
    config: BacktestConfig,
    is_buy: bool = True
) -> float:
    """Compute all-in transaction cost for Indian markets."""
    brokerage = trade_value * config.brokerage_pct
    stt = trade_value * config.stt_pct
    transaction_charges = trade_value * config.transaction_charges_pct
    gst = (brokerage + transaction_charges) * config.gst_pct
    sebi = trade_value * config.sebi_charges_pct
    stamp_duty = trade_value * config.stamp_duty_pct if is_buy else 0
    
    return brokerage + stt + transaction_charges + gst + sebi + stamp_duty


def compute_market_impact(
    quantity: float,
    avg_daily_volume: float,
    config: BacktestConfig
) -> float:
    """
    Square-root market impact model: Impact = sigma * sqrt(Q/V)
    Simplified version using participation rate.
    """
    if avg_daily_volume == 0:
        return 0.0
    
    participation_rate = quantity / avg_daily_volume
    impact_bps = config.impact_coefficient * np.sqrt(participation_rate) * 100
    
    return impact_bps


class VectorizedBacktester:
    """
    Fast vectorized backtester for strategy evaluation.
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
    
    def run_signal_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy"
    ) -> BacktestResult:
        """
        Run backtest from signal DataFrame.
        
        Args:
            signals: DataFrame with columns [symbol, timestamp, direction, strength]
                    direction: +1 (long), -1 (short), 0 (flat)
            prices: DataFrame with columns [symbol, timestamp, open, high, low, close]
            volumes: Optional volume DataFrame for impact calculation
            strategy_name: Name for reporting
            
        Returns:
            BacktestResult with full metrics
        """
        if signals.empty:
            raise ValueError("No signals provided")
        
        # Pivot prices to wide format
        price_pivot = prices.pivot(columns="symbol", values="close")
        
        # Pivot signals to wide format
        signal_pivot = signals.pivot(columns="symbol", values="direction", index="timestamp")
        signal_pivot = signal_pivot.reindex(price_pivot.index).ffill().fillna(0)
        
        # Calculate returns
        returns = price_pivot.pct_change()
        
        # Strategy returns (shift signals by 1 to avoid look-ahead)
        shifted_signals = signal_pivot.shift(1)
        strategy_returns = (shifted_signals * returns).sum(axis=1) / (shifted_signals.abs().sum(axis=1).replace(0, 1))
        
        # Apply transaction costs on signal changes
        signal_changes = shifted_signals.diff().abs()
        avg_price_level = price_pivot.mean(axis=1)
        
        # Estimate costs
        turnover = signal_changes.sum(axis=1) * avg_price_level
        costs = turnover.apply(
            lambda x: compute_transaction_cost(x, self.config) / self.config.initial_capital
        )
        strategy_returns = strategy_returns - costs
        
        # Build equity curve
        equity_curve = (1 + strategy_returns).cumprod() * self.config.initial_capital
        
        # Compute metrics
        result = self._compute_metrics(
            strategy_returns, equity_curve, strategy_name
        )
        
        return result
    
    def run_orb_backtest(
        self,
        data_dict: Dict[str, pd.DataFrame],
        orb_config: dict
    ) -> BacktestResult:
        """
        Specific backtester for ORB strategy.
        Simulates intraday opening range and breakout entry.
        """
        from alpha.orb_strategy import ORBStrategy
        
        strategy = ORBStrategy({"alpha": {"orb": orb_config}})
        
        all_trades = []
        equity = self.config.initial_capital
        
        for symbol, df in data_dict.items():
            if df.empty:
                continue
            
            # Resample to 1-minute if needed
            if df.index.freq and df.index.freq > pd.Timedelta("1min"):
                continue
            
            # Iterate through trading days
            for date, day_data in df.groupby(df.index.date):
                if len(day_data) < 10:
                    continue
                
                # Scan opening range
                intraday_dict = {symbol: day_data}
                candidates = strategy.scan_opening_range(intraday_dict)
                
                if symbol not in candidates:
                    continue
                
                # Simulate intraday signals
                for i in range(len(day_data)):
                    bar = day_data.iloc[i]
                    signal, position = strategy.generate_signal(symbol, bar)
                    
                    if signal.name in ["LONG_BREAKOUT", "SHORT_BREAKOUT"]:
                        entry = position.entry_price
                        sl = position.stop_loss
                        tgt = position.target_price
                        direction = position.direction
                        
                        # Simulate rest of day
                        remaining = day_data.iloc[i+1:]
                        
                        for _, future_bar in remaining.iterrows():
                            if direction == "long":
                                if future_bar["Low"] <= sl:
                                    pnl = (sl - entry) / entry
                                    all_trades.append({
                                        "symbol": symbol, "date": date,
                                        "direction": direction,
                                        "entry": entry, "exit": sl,
                                        "pnl_pct": pnl, "status": "SL"
                                    })
                                    break
                                elif future_bar["High"] >= tgt:
                                    pnl = (tgt - entry) / entry
                                    all_trades.append({
                                        "symbol": symbol, "date": date,
                                        "direction": direction,
                                        "entry": entry, "exit": tgt,
                                        "pnl_pct": pnl, "status": "TGT"
                                    })
                                    break
                            else:  # short
                                if future_bar["High"] >= sl:
                                    pnl = (entry - sl) / entry
                                    all_trades.append({
                                        "symbol": symbol, "date": date,
                                        "direction": direction,
                                        "entry": entry, "exit": sl,
                                        "pnl_pct": pnl, "status": "SL"
                                    })
                                    break
                                elif future_bar["Low"] <= tgt:
                                    pnl = (entry - tgt) / entry
                                    all_trades.append({
                                        "symbol": symbol, "date": date,
                                        "direction": direction,
                                        "entry": entry, "exit": tgt,
                                        "pnl_pct": pnl, "status": "TGT"
                                    })
                                    break
                        else:
                            # EOD force close at last price
                            exit_price = remaining.iloc[-1]["Close"] if len(remaining) > 0 else entry
                            if direction == "long":
                                pnl = (exit_price - entry) / entry
                            else:
                                pnl = (entry - exit_price) / entry
                            all_trades.append({
                                "symbol": symbol, "date": date,
                                "direction": direction,
                                "entry": entry, "exit": exit_price,
                                "pnl_pct": pnl, "status": "EOD"
                            })
                        
                        # Only one trade per symbol per day
                        break
        
        if not all_trades:
            return self._empty_result("ORB")
        
        trades_df = pd.DataFrame(all_trades)
        
        # Apply transaction costs
        for i, trade in trades_df.iterrows():
            trade_value = self.config.initial_capital * orb_config.get("max_position_size_pct", 0.05)
            cost = compute_transaction_cost(trade_value, self.config) / self.config.initial_capital
            trades_df.loc[i, "pnl_pct"] -= cost
        
        # Build equity curve
        daily_returns = trades_df.groupby("date")["pnl_pct"].sum()
        equity_curve = (1 + daily_returns).cumprod() * self.config.initial_capital
        
        result = self._compute_metrics(daily_returns, equity_curve, "ORB")
        result.trades = all_trades
        
        return result
    
    def _compute_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        strategy_name: str
    ) -> BacktestResult:
        """Compute comprehensive backtest metrics."""
        returns = returns.dropna()
        equity_curve = equity_curve.dropna()
        
        if len(returns) == 0:
            return self._empty_result(strategy_name)
        
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # CAGR
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Sharpe Ratio (risk-free = 6% for India)
        rf_daily = 0.06 / 252
        excess_returns = returns - rf_daily
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Sortino Ratio
        downside = returns[returns < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0
        
        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        is_dd = drawdown < 0
        dd_groups = (is_dd != is_dd.shift()).cumsum()
        dd_durations = is_dd.groupby(dd_groups).sum()
        max_dd_duration = dd_durations.max() if len(dd_durations) > 0 else 0
        
        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade metrics
        winning = returns[returns > 0]
        losing = returns[returns < 0]
        win_rate = len(winning) / len(returns) if len(returns) > 0 else 0
        profit_factor = winning.sum() / abs(losing.sum()) if len(losing) > 0 and losing.sum() != 0 else float('inf')
        
        return BacktestResult(
            strategy_name=strategy_name,
            returns=returns,
            equity_curve=equity_curve,
            positions=pd.DataFrame(),
            trades=[],
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration=int(max_dd_duration),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=returns.mean(),
            avg_winning_trade=winning.mean() if len(winning) > 0 else 0,
            avg_losing_trade=losing.mean() if len(losing) > 0 else 0,
            total_trades=len(returns),
            calmar_ratio=calmar,
            total_brokerage=0,
            total_slippage=0,
            total_impact_cost=0
        )
    
    def _empty_result(self, name: str) -> BacktestResult:
        return BacktestResult(
            strategy_name=name, returns=pd.Series(), equity_curve=pd.Series(),
            positions=pd.DataFrame(), trades=[], total_return=0, cagr=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, win_rate=0, profit_factor=0,
            avg_trade_return=0, avg_winning_trade=0, avg_losing_trade=0,
            total_trades=0, calmar_ratio=0, total_brokerage=0,
            total_slippage=0, total_impact_cost=0
        )

"""
Vectorized Backtester - Fast screening for strategy evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class BacktestMetrics:
    """Backtest performance metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    win_loss_ratio: float
    annual_volatility: float
    calmar_ratio: float


class VectorizedBacktester:
    """Fast vectorized backtester for strategy screening"""
    
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
    
    def run(self, data: pd.DataFrame, signal_generator: Callable,
            position_sizing: Optional[Callable] = None) -> BacktestMetrics:
        """
        Run vectorized backtest
        
        Args:
            data: OHLCV data with datetime index
            signal_generator: Function that generates signals from data
            position_sizing: Optional function for position sizing
            
        Returns:
            BacktestMetrics
        """
        # Resample to daily if needed
        if data.index.freq != 'D':
            daily_data = data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        else:
            daily_data = data.copy()
        
        # Generate signals
        signals = signal_generator(daily_data)
        
        # Position sizing
        if position_sizing:
            positions = position_sizing(signals, daily_data)
        else:
            positions = signals
        
        # Compute returns — no shift needed; positions.shift(1) below
        # ensures signals are lagged by one bar to avoid lookahead.
        returns = daily_data['close'].pct_change()
        
        # Compute PnL (vectorized)
        pnl = (positions.shift(1) * returns).sum(axis=1) if isinstance(positions, pd.DataFrame) else positions.shift(1) * returns
        
        # Compute metrics
        metrics = self._calculate_metrics(pnl)
        
        return metrics
    
    def _calculate_metrics(self, pnl: pd.Series) -> BacktestMetrics:
        """Calculate backtest metrics"""
        # Total return
        total_return = (1 + pnl).prod() - 1
        
        # Annualized Sharpe
        if pnl.std() > 0:
            sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + pnl).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Hit rate
        hit_rate = (pnl > 0).sum() / len(pnl) if len(pnl) > 0 else 0.0
        
        # Win/loss ratio
        wins = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
        losses = abs(pnl[pnl < 0].mean()) if (pnl < 0).any() else 1
        win_loss_ratio = wins / losses if losses > 0 else 0.0
        
        # Annual volatility
        annual_volatility = pnl.std() * np.sqrt(252)
        
        # Calmar ratio
        calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            win_loss_ratio=win_loss_ratio,
            annual_volatility=annual_volatility,
            calmar_ratio=calmar
        )
    
    def run_multi_asset(self, data_dict: Dict[str, pd.DataFrame], 
                       signal_generator: Callable,
                       rebalance_freq: str = 'M') -> BacktestMetrics:
        """
        Run multi-asset vectorized backtest
        
        Args:
            data_dict: Dict mapping symbol to OHLCV data
            signal_generator: Function that generates signals
            rebalance_freq: Rebalancing frequency
            
        Returns:
            BacktestMetrics
        """
        # Align data
        aligned_data = self._align_data(data_dict)
        
        # Generate signals for each asset
        signals_dict = {}
        for symbol, data in aligned_data.items():
            signals_dict[symbol] = signal_generator(data)
        
        # Combine signals
        signals_df = pd.DataFrame(signals_dict)
        
        # Rebalance at specified frequency
        if rebalance_freq == 'M':
            signals_df = signals_df.resample('M').last()
        elif rebalance_freq == 'W':
            signals_df = signals_df.resample('W').last()
        
        # Normalize to sum to 1 (long-only) or sum abs to 1 (long-short)
        signals_df = signals_df.div(signals_df.abs().sum(axis=1), axis=0)
        
        # Compute returns
        returns_dict = {}
        for symbol, data in aligned_data.items():
            returns_dict[symbol] = data['close'].pct_change()
        
        returns_df = pd.DataFrame(returns_dict)
        
        # Compute portfolio returns
        portfolio_returns = (signals_df.shift(1) * returns_df).sum(axis=1)
        
        return self._calculate_metrics(portfolio_returns)
    
    def _align_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align data across assets by date"""
        # Get common dates
        common_dates = data_dict[list(data_dict.keys())[0]].index
        for data in data_dict.values():
            common_dates = common_dates.intersection(data.index)
        
        # Filter to common dates
        aligned = {}
        for symbol, data in data_dict.items():
            aligned[symbol] = data.loc[common_dates]
        
        return aligned

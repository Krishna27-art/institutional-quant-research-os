"""
Walk-Forward Backtester - Walk-forward validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from ..vectorized.vectorized_backtester import VectorizedBacktester, BacktestMetrics


class WalkForwardBacktester:
    """Walk-forward backtesting with rolling windows"""
    
    def __init__(self, train_window: int = 1260, test_window: int = 252, step: int = 21):
        """
        Args:
            train_window: Training window in days
            test_window: Test window in days
            step: Step size in days
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.vectorized = VectorizedBacktester()
    
    def run(self, data: pd.DataFrame, signal_generator: Callable,
            strategy_trainer: Optional[Callable] = None) -> Dict:
        """
        Run walk-forward backtest
        
        Args:
            data: OHLCV data
            signal_generator: Function that generates signals
            strategy_trainer: Optional function to train strategy on train window
            
        Returns:
            Dict with walk-forward results
        """
        results = []
        
        for start_idx in range(0, len(data) - self.train_window - self.test_window, self.step):
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window
            
            if test_end_idx > len(data):
                break
            
            # Split data
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]
            
            # Train strategy if trainer provided
            if strategy_trainer:
                trained_signal_gen = strategy_trainer(train_data)
                signal_gen = trained_signal_gen
            else:
                signal_gen = signal_generator
            
            # Run backtest on test period
            metrics = self.vectorized.run(test_data, signal_gen)
            
            results.append({
                'train_start': data.index[start_idx],
                'train_end': data.index[train_end_idx],
                'test_start': data.index[train_end_idx],
                'test_end': data.index[test_end_idx],
                'metrics': metrics
            })
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results"""
        if not results:
            return {}
        
        # Extract metrics
        sharpe_list = [r['metrics'].sharpe_ratio for r in results]
        return_list = [r['metrics'].total_return for r in results]
        drawdown_list = [r['metrics'].max_drawdown for r in results]
        
        return {
            'num_folds': len(results),
            'mean_sharpe': np.mean(sharpe_list),
            'std_sharpe': np.std(sharpe_list),
            'mean_return': np.mean(return_list),
            'std_return': np.std(return_list),
            'mean_drawdown': np.mean(drawdown_list),
            'worst_drawdown': np.min(drawdown_list),
            'fold_results': results
        }

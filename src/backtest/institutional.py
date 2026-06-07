"""
Institutional Backtester - Two-stage: vectorized screening + event-driven validation
"""

import pandas as pd
from typing import Dict, Callable, Optional
from .vectorized.vectorized_backtester import VectorizedBacktester, BacktestMetrics
from .event_driven.event_backtester import EventDrivenBacktester


class InstitutionalBacktester:
    """
    Two-stage institutional backtester:
    1. Vectorized screening (fast) - filter out poor strategies
    2. Event-driven validation (accurate) - final evaluation with realistic costs
    """
    
    def __init__(self, initial_capital: float = 1_000_000,
                 screening_threshold: float = 0.5):
        self.initial_capital = initial_capital
        self.screening_threshold = screening_threshold  # Min Sharpe for screening
        self.vectorized = VectorizedBacktester(initial_capital)
        self.event_driven = EventDrivenBacktester(initial_capital)
    
    def run(self, data: pd.DataFrame, signal_generator: Callable,
            position_sizing: Optional[Callable] = None) -> Dict:
        """
        Run two-stage backtest
        
        Args:
            data: OHLCV data
            signal_generator: Function that generates signals
            position_sizing: Optional position sizing function
            
        Returns:
            Dict with both screening and validation results
        """
        # Stage 1: Vectorized screening
        print("Stage 1: Vectorized screening...")
        screening_metrics = self.vectorized.run(data, signal_generator, position_sizing)
        
        # Check if passes screening
        if screening_metrics.sharpe_ratio < self.screening_threshold:
            return {
                'passed_screening': False,
                'screening_metrics': screening_metrics,
                'validation_metrics': None
            }
        
        print(f"Passed screening (Sharpe: {screening_metrics.sharpe_ratio:.2f})")
        print("Stage 2: Event-driven validation...")
        
        # Stage 2: Event-driven validation
        validation_results = self.event_driven.run(data, signal_generator)
        
        return {
            'passed_screening': True,
            'screening_metrics': screening_metrics,
            'validation_metrics': validation_results
        }
    
    def run_walk_forward(self, data: pd.DataFrame, signal_generator: Callable,
                        train_window: int = 1260, test_window: int = 252) -> Dict:
        """
        Run walk-forward with two-stage validation
        
        Args:
            data: OHLCV data
            signal_generator: Function that generates signals
            train_window: Training window
            test_window: Test window
            
        Returns:
            Dict with walk-forward results
        """
        from .walk_forward.walk_forward import WalkForwardBacktester
        
        walk_forward = WalkForwardBacktester(train_window, test_window)
        
        # Use institutional backtester for each fold
        def fold_backtester(test_data):
            return self.run(test_data, signal_generator)
        
        return walk_forward.run(data, signal_generator, fold_backtester)

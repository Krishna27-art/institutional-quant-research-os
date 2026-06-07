"""
Walk-Forward Trainer - Training with walk-forward validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .ensemble import MLEnsemble


class WalkForwardTrainer:
    """Walk-forward training with purged cross-validation"""
    
    def __init__(
        self,
        train_window: int = 1260,
        test_window: int = 252,
        step: int = 21,
        purge_window: int = 5,
        embargo_window: int = 5,
    ):
        """
        Args:
            train_window: Training window in days (5 years)
            test_window: Test window in days (1 year)
            step: Step size in days (1 month)
            purge_window: Rows removed immediately before the test window
            embargo_window: Rows skipped between train and test windows
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.purge_window = purge_window
        self.embargo_window = embargo_window
    
    def train(self, data: pd.DataFrame, feature_columns: List[str], 
              target_column: str) -> List[Dict]:
        """
        Perform walk-forward training
        
        Args:
            data: Full dataset
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            List of training results
        """
        results = []
        
        total_window = self.train_window + self.purge_window + self.embargo_window + self.test_window
        for start_idx in range(0, len(data) - total_window + 1, self.step):
            train_end_idx = start_idx + self.train_window
            test_start_idx = train_end_idx + self.purge_window + self.embargo_window
            test_end_idx = test_start_idx + self.test_window
            
            if test_end_idx > len(data):
                break
            
            # Split data
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]
            
            # Prepare features and target
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Train model
            ensemble = MLEnsemble()
            ensemble.fit(X_train, y_train)
            
            # Evaluate
            pred = ensemble.predict(X_test)
            
            # Compute metrics
            sharpe = self._compute_sharpe(y_test, pred)
            hit_rate = self._compute_hit_rate(y_test, pred)
            
            results.append({
                'train_start': data.index[start_idx],
                'train_end': data.index[train_end_idx - 1],
                'test_start': data.index[test_start_idx],
                'test_end': data.index[test_end_idx - 1],
                'sharpe': sharpe,
                'hit_rate': hit_rate,
                'model': ensemble
            })
        
        return results
    
    def train_with_regime(self, data: pd.DataFrame, feature_columns: List[str],
                         target_column: str, regime_column: str) -> Dict[str, List[Dict]]:
        """
        Train separate models for each regime
        
        Args:
            data: Full dataset
            feature_columns: List of feature column names
            target_column: Target column name
            regime_column: Regime column name
            
        Returns:
            Dict mapping regime to training results
        """
        regime_results = {}
        
        for regime in data[regime_column].unique():
            regime_data = data[data[regime_column] == regime]
            
            if len(regime_data) < self.train_window + self.test_window:
                continue
            
            results = self.train(regime_data, feature_columns, target_column)
            regime_results[regime] = results
        
        return regime_results
    
    def _compute_sharpe(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute Sharpe ratio of predictions"""
        # Use prediction as signal, actual return as target
        returns = y_true * np.sign(y_pred)
        
        if returns.std() == 0:
            return 0.0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe
    
    def _compute_hit_rate(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Compute hit rate (percentage of correct directional predictions)"""
        correct = (np.sign(y_true) == np.sign(y_pred)).sum()
        total = len(y_true)
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def get_ensemble_model(self, results: List[Dict]) -> MLEnsemble:
        """Get the best model from training results"""
        if not results:
            raise ValueError("No training results available")
        
        # Select model with highest Sharpe
        best_result = max(results, key=lambda x: x['sharpe'])
        return best_result['model']
    
    def run(self, data: pd.DataFrame, feature_columns: List[str], 
             target_column: str) -> List[Dict]:
        """
        Alias for train() method for compatibility.
        
        Args:
            data: Full dataset
            feature_columns: List of feature column names
            target_column: Target column name
            
        Returns:
            List of training results
        """
        return self.train(data, feature_columns, target_column)

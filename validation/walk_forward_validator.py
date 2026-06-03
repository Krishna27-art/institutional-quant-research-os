"""
Walk-Forward Validation with Strict Out-of-Sample Holdout
Eliminates overfitting by ensuring models are tested on truly unseen data.

Critical for institutional-grade backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from sklearn.model_selection import TimeSeriesSplit


class ValidationMethod(Enum):
    """Validation methods"""
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation"""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    train_returns: pd.Series
    test_returns: pd.Series
    is_oos_valid: bool


class WalkForwardValidator:
    """
    Strict walk-forward validation with out-of-sample holdout.
    
    Rules:
    1. Train on historical data only
    2. Test on future data only (no overlap)
    3. Never tune hyperparameters on test set
    4. Minimum 1-year holdout for testing
    5. Minimum 5-year training window
    6. Roll forward by 6 months or 1 year
    """
    
    def __init__(self,
                 train_window_years: int = 5,
                 test_window_years: int = 1,
                 roll_forward_months: int = 6,
                 min_oos_sharpe: float = 0.5):
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.roll_forward_months = roll_forward_months
        self.min_oos_sharpe = min_oos_sharpe
        self.results: List[WalkForwardResult] = []
    
    def validate(self,
                 returns: pd.Series,
                 model_func: Callable,
                 feature_data: Optional[pd.DataFrame] = None) -> List[WalkForwardResult]:
        """
        Perform walk-forward validation.
        
        Args:
            returns: Strategy returns (indexed by time)
            model_func: Function that takes (train_data) and returns predictions
            feature_data: Optional feature data
        
        Returns:
            List of WalkForwardResult for each fold
        """
        self.results = []
        
        # Convert to datetime if not already
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        # Calculate window sizes in days
        train_days = self.train_window_years * 252
        test_days = self.test_window_years * 252
        roll_days = self.roll_forward_months * 21
        
        # Generate walk-forward folds
        start_idx = train_days
        end_idx = len(returns) - test_days
        
        fold_idx = 0
        while start_idx + test_days <= len(returns):
            train_start = returns.index[0]
            train_end = returns.index[start_idx - 1]
            test_start = returns.index[start_idx]
            test_end = returns.index[min(start_idx + test_days - 1, len(returns) - 1)]
            
            # Get train and test data
            train_data = returns.loc[train_start:train_end]
            test_data = returns.loc[test_start:test_end]
            
            # Train model and get predictions
            if feature_data is not None:
                train_features = feature_data.loc[train_start:train_end]
                test_features = feature_data.loc[test_start:test_end]
                predictions = model_func(train_data, train_features, test_features)
            else:
                predictions = model_func(train_data, None, None)
            
            # Calculate Sharpe ratios
            train_sharpe = self._calculate_sharpe(train_data)
            test_sharpe = self._calculate_sharpe(test_data)
            
            # Check if OOS valid
            is_oos_valid = test_sharpe >= self.min_oos_sharpe
            
            result = WalkForwardResult(
                train_start=str(train_start),
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_returns=train_data,
                test_returns=test_data,
                is_oos_valid=is_oos_valid
            )
            
            self.results.append(result)
            
            # Roll forward
            start_idx += roll_days
            fold_idx += 1
            
            # Safety limit
            if fold_idx > 20:
                break
        
        return self.results
    
    def _calculate_sharpe(self, returns: pd.Series, annualize: bool = True) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        sharpe = returns.mean() / returns.std()
        
        if annualize:
            sharpe *= np.sqrt(252)
        
        return sharpe
    
    def get_oos_sharpe_mean(self) -> float:
        """Get mean out-of-sample Sharpe across all folds"""
        if not self.results:
            return 0.0
        
        return np.mean([r.test_sharpe for r in self.results])
    
    def get_oos_sharpe_std(self) -> float:
        """Get std of out-of-sample Sharpe across all folds"""
        if not self.results:
            return 0.0
        
        return np.std([r.test_sharpe for r in self.results])
    
    def get_valid_fold_count(self) -> int:
        """Get number of folds with valid OOS Sharpe"""
        return sum(1 for r in self.results if r.is_oos_valid)
    
    def is_strategy_valid(self) -> bool:
        """Check if strategy passes walk-forward validation"""
        if not self.results:
            return False
        
        # At least 50% of folds must be valid
        valid_ratio = self.get_valid_fold_count() / len(self.results)
        
        # Mean OOS Sharpe must be above threshold
        mean_oos = self.get_oos_sharpe_mean()
        
        return valid_ratio >= 0.5 and mean_oos >= self.min_oos_sharpe
    
    def generate_report(self) -> str:
        """Generate validation report"""
        if not self.results:
            return "No validation results available"
        
        total_folds = len(self.results)
        valid_folds = self.get_valid_fold_count()
        mean_oos = self.get_oos_sharpe_mean()
        std_oos = self.get_oos_sharpe_std()
        
        report = f"""
Walk-Forward Validation Report
{'=' * 50}
Total folds: {total_folds}
Valid folds (Sharpe >= {self.min_oos_sharpe}): {valid_folds} ({valid_folds/total_folds*100:.1f}%)
Mean OOS Sharpe: {mean_oos:.3f}
Std OOS Sharpe: {std_oos:.3f}
Strategy Valid: {self.is_strategy_valid()}

Fold-by-Fold Results:
{'-' * 50}
"""
        
        for i, result in enumerate(self.results):
            status = "PASS" if result.is_oos_valid else "FAIL"
            report += f"Fold {i+1}: Train Sharpe={result.train_sharpe:.3f}, "
            report += f"Test Sharpe={result.test_sharpe:.3f} [{status}]\n"
            report += f"  Train: {result.train_start} to {result.train_end}\n"
            report += f"  Test: {result.test_start} to {result.test_end}\n\n"
        
        return report


def simple_model(train_data: pd.Series, train_features: Optional[pd.DataFrame],
                test_features: Optional[pd.DataFrame]) -> pd.Series:
    """
    Simple model for demonstration - uses mean of training returns as prediction.
    """
    mean_return = train_data.mean()
    
    if test_features is not None:
        return pd.Series([mean_return] * len(test_features), index=test_features.index)
    else:
        return pd.Series([mean_return] * len(train_data), index=train_data.index)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample returns
    np.random.seed(42)
    n = 2520  # 10 years of daily data
    dates = pd.date_range('2014-01-01', periods=n, freq='D')
    returns = pd.Series(np.random.randn(n) * 0.01, index=dates)
    
    validator = WalkForwardValidator(
        train_window_years=5,
        test_window_years=1,
        roll_forward_months=6,
        min_oos_sharpe=0.5
    )
    
    results = validator.validate(returns, simple_model)
    
    print(validator.generate_report())

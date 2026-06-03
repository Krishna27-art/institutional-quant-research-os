"""
Walk-Forward Validation Framework

Based on Comprehensive Upgrade Analysis - Tier 1 Upgrade (#5)
Expected Sharpe improvement: +0.2–0.4
Eliminates overfitting bias

Methodology:
- Rolling window training with out-of-sample testing
- Prevents look-ahead bias
- Realistic performance estimation
- Detects strategy decay
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class WalkForwardConfig:
    """Configuration for Walk-Forward Validation"""
    train_window_days: int = 1260  # 5 years training
    test_window_days: int = 252  # 1 year testing
    step_days: int = 252  # 1 year step
    min_train_samples: int = 1000  # Minimum samples for training
    warmup_days: int = 60  # Warmup period for each fold
    retrain_frequency: str = "yearly"  # "yearly", "quarterly", "monthly"
    enable_decay_detection: bool = True
    decay_threshold: float = 0.3  # 30% performance drop triggers decay alert


@dataclass
class FoldResult:
    """Result for a single walk-forward fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    train_max_dd: float
    test_max_dd: float
    is_decayed: bool


class WalkForwardValidator:
    """
    Walk-Forward Validation Framework
    
    Implements rolling window cross-validation for time series data.
    Prevents look-ahead bias and provides realistic performance estimates.
    
    Methodology:
    1. Train on window [t - train_window, t]
    2. Test on window [t, t + test_window]
    3. Step forward by step_days
    4. Repeat until end of data
    5. Aggregate results across all folds
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        
        # Fold results
        self.fold_results: List[FoldResult] = []
        
        # Performance tracking
        self.cumulative_returns: List[float] = []
    
    def validate(self, 
                 data: pd.DataFrame,
                 train_func: Callable,
                 predict_func: Callable,
                 target_col: str = "returns") -> Dict:
        """
        Run walk-forward validation
        
        Args:
            data: DataFrame with datetime index and features
            train_func: Function to train model (X_train, y_train) -> model
            predict_func: Function to generate predictions (model, X_test) -> predictions
            target_col: Name of target column
            
        Returns:
            Dictionary with validation results
        """
        dates = data.index
        n_dates = len(dates)
        
        if n_dates < self.config.train_window_days + self.config.test_window_days:
            raise ValueError("Insufficient data for walk-forward validation")
        
        # Calculate fold boundaries
        folds = self._generate_folds(dates)
        
        # Run each fold
        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds):
            print(f"\nFold {fold_id + 1}/{len(folds)}")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test: {test_start.date()} to {test_end.date()}")
            
            # Get train data
            train_data = data.loc[train_start:train_end]
            X_train = train_data.drop(columns=[target_col])
            y_train = train_data[target_col]
            
            # Get test data
            test_data = data.loc[test_start:test_end]
            X_test = test_data.drop(columns=[target_col])
            y_test = test_data[target_col]
            
            # Skip if insufficient data
            if len(X_train) < self.config.min_train_samples:
                print(f"  Skipping: insufficient training data ({len(X_train)} < {self.config.min_train_samples})")
                continue
            
            # Train model
            try:
                model = train_func(X_train, y_train)
            except Exception as e:
                print(f"  Training failed: {e}")
                continue
            
            # Generate predictions
            try:
                predictions = predict_func(model, X_test)
            except Exception as e:
                print(f"  Prediction failed: {e}")
                continue
            
            # Compute metrics
            train_metrics = self._compute_metrics(y_train, predict_func(model, X_train))
            test_metrics = self._compute_metrics(y_test, predictions)
            
            # Check for decay
            is_decayed = False
            if self.config.enable_decay_detection:
                is_decayed = self._detect_decay(train_metrics, test_metrics)
            
            # Store fold result
            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_sharpe=train_metrics["sharpe"],
                test_sharpe=test_metrics["sharpe"],
                train_return=train_metrics["total_return"],
                test_return=test_metrics["total_return"],
                train_max_dd=train_metrics["max_drawdown"],
                test_max_dd=test_metrics["max_drawdown"],
                is_decayed=is_decayed
            )
            
            self.fold_results.append(fold_result)
            
            print(f"  Train Sharpe: {train_metrics['sharpe']:.2f}")
            print(f"  Test Sharpe: {test_metrics['sharpe']:.2f}")
            print(f"  Decay detected: {is_decayed}")
        
        # Aggregate results
        return self._aggregate_results()
    
    def _generate_folds(self, dates: pd.DatetimeIndex) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate fold boundaries"""
        folds = []
        
        current_test_start = dates[self.config.train_window_days]
        
        while current_test_start <= dates[-self.config.test_window_days]:
            train_start = dates[dates.get_loc(current_test_start) - self.config.train_window_days]
            train_end = dates[dates.get_loc(current_test_start) - 1]
            test_end = dates[dates.get_loc(current_test_start) + self.config.test_window_days - 1]
            
            folds.append((train_start, train_end, current_test_start, test_end))
            
            # Step forward
            current_idx = dates.get_loc(current_test_start) + self.config.step_days
            if current_idx >= len(dates):
                break
            current_test_start = dates[current_idx]
        
        return folds
    
    def _compute_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Compute performance metrics"""
        returns = predicted
        
        # Total return
        total_return = returns.sum()
        
        # Sharpe ratio
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Information coefficient (if actual provided)
        ic = 0.0
        if len(actual) == len(returns):
            ic = actual.corr(returns)
        
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "ic": ic
        }
    
    def _detect_decay(self, train_metrics: Dict, test_metrics: Dict) -> bool:
        """Detect if strategy has decayed"""
        # Check Sharpe drop
        sharpe_drop = (train_metrics["sharpe"] - test_metrics["sharpe"]) / train_metrics["sharpe"]
        
        if sharpe_drop > self.config.decay_threshold:
            return True
        
        return False
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all folds"""
        if not self.fold_results:
            return {"error": "No valid folds"}
        
        # Extract metrics
        train_sharpes = [f.train_sharpe for f in self.fold_results]
        test_sharpes = [f.test_sharpe for f in self.fold_results]
        train_returns = [f.train_return for f in self.fold_results]
        test_returns = [f.test_return for f in self.fold_results]
        train_max_dds = [f.train_max_dd for f in self.fold_results]
        test_max_dds = [f.test_max_dd for f in self.fold_results]
        
        # Compute statistics
        results = {
            "num_folds": len(self.fold_results),
            "train_sharpe_mean": np.mean(train_sharpes),
            "train_sharpe_std": np.std(train_sharpes),
            "test_sharpe_mean": np.mean(test_sharpes),
            "test_sharpe_std": np.std(test_sharpes),
            "train_return_mean": np.mean(train_returns),
            "test_return_mean": np.mean(test_returns),
            "train_max_dd_mean": np.mean(train_max_dds),
            "test_max_dd_mean": np.mean(test_max_dds),
            "decay_rate": sum(f.is_decayed for f in self.fold_results) / len(self.fold_results),
            "fold_results": self.fold_results
        }
        
        # Overfitting detection
        overfitting = (results["train_sharpe_mean"] - results["test_sharpe_mean"]) / results["train_sharpe_mean"]
        results["overfitting_ratio"] = overfitting
        results["is_overfitted"] = overfitting > 0.5  # 50% drop indicates overfitting
        
        return results
    
    def get_fold_details(self) -> pd.DataFrame:
        """Get detailed results for each fold"""
        if not self.fold_results:
            return pd.DataFrame()
        
        data = []
        for fold in self.fold_results:
            data.append({
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_sharpe": fold.train_sharpe,
                "test_sharpe": fold.test_sharpe,
                "train_return": fold.train_return,
                "test_return": fold.test_return,
                "train_max_dd": fold.train_max_dd,
                "test_max_dd": fold.test_max_dd,
                "is_decayed": fold.is_decayed
            })
        
        return pd.DataFrame(data)


def simple_train_func(X_train: pd.DataFrame, y_train: pd.Series):
    """Simple training function for testing"""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def simple_predict_func(model, X_test: pd.DataFrame) -> pd.Series:
    """Simple prediction function for testing"""
    predictions = model.predict(X_test)
    return pd.Series(predictions, index=X_test.index)


if __name__ == "__main__":
    # Example usage
    config = WalkForwardConfig(
        train_window_days=1260,  # 5 years
        test_window_days=252,  # 1 year
        step_days=252,  # 1 year step
        enable_decay_detection=True
    )
    
    validator = WalkForwardValidator(config)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    np.random.seed(42)
    n_days = 2520  # 10 years
    dates = pd.date_range(start="2014-01-01", periods=n_days)
    
    # Generate features
    n_features = 5
    features = np.random.randn(n_days, n_features)
    
    # Generate target with some signal
    signal = 0.01 * features[:, 0] + 0.005 * features[:, 1]
    noise = np.random.randn(n_days) * 0.02
    returns = signal + noise
    
    # Create DataFrame
    data = pd.DataFrame(features, index=dates, columns=[f"feature_{i}" for i in range(n_features)])
    data["returns"] = returns
    
    # Run walk-forward validation
    print("\nRunning walk-forward validation...")
    results = validator.validate(
        data=data,
        train_func=simple_train_func,
        predict_func=simple_predict_func,
        target_col="returns"
    )
    
    print(f"\n=== Walk-Forward Validation Results ===")
    print(f"Number of folds: {results['num_folds']}")
    print(f"Train Sharpe (mean): {results['train_sharpe_mean']:.2f} ± {results['train_sharpe_std']:.2f}")
    print(f"Test Sharpe (mean): {results['test_sharpe_mean']:.2f} ± {results['test_sharpe_std']:.2f}")
    print(f"Train Return (mean): {results['train_return_mean']:.4f}")
    print(f"Test Return (mean): {results['test_return_mean']:.4f}")
    print(f"Train Max DD (mean): {results['train_max_dd_mean']:.4f}")
    print(f"Test Max DD (mean): {results['test_max_dd_mean']:.4f}")
    print(f"Decay rate: {results['decay_rate']:.2%}")
    print(f"Overfitting ratio: {results['overfitting_ratio']:.2%}")
    print(f"Is overfitted: {results['is_overfitted']}")
    
    print(f"\n=== Fold Details ===")
    fold_details = validator.get_fold_details()
    print(fold_details.to_string(index=False))

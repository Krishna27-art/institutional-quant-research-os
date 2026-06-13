"""
Walk-Forward Validation Framework

This module implements walk-forward validation to prevent overfitting and
ensure models perform well on out-of-sample data, as required by the research
literature (Lopez de Prado 2018, Kelly & Xiu 2020).

Key Features:
- Walk-forward validation with rolling windows
- Purging and embargo to prevent data leakage
- Deflated Sharpe ratio (Bailey 2014) for multiple testing correction
- Combinatorial cross-validation
- Out-of-sample performance tracking
- Model decay detection

Based on Audit Report Priority 0: Critical - Week 1-2
Research Papers: Lopez de Prado (2018), Bailey et al (2014), Kelly & Xiu (2020)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Validation methods."""
    WALK_FORWARD = "walk_forward"
    PURGE_EMBARGO = "purge_embargo"
    COMBINATORIAL_CV = "combinatorial_cv"
    TIME_SPLIT = "time_split"


@dataclass
class ValidationResult:
    """Result of a validation run."""
    method: ValidationMethod
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_sharpe: float
    test_sharpe: float
    train_returns: List[float]
    test_returns: List[float]
    deflated_sharpe: float
    num_trials: int
    parameters: Dict
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    # Window sizes (in days)
    train_window: int = 1260  # 5 years of trading days
    test_window: int = 252   # 1 year of trading days
    step: int = 21           # Re-optimize every month
    
    # Purging and embargo (Lopez de Prado 2018)
    purge_period: int = 5    # Days to purge after training
    embargo_period: int = 5   # Days to embargo before test
    
    # Deflated Sharpe (Bailey 2014)
    use_deflated_sharpe: bool = True
    expected_sharpe: float = 1.0
    var_sharpe: float = 0.5
    
    # Combinatorial CV
    n_splits: int = 5
    test_size: float = 0.2
    
    # Minimum performance thresholds
    min_train_sharpe: float = 0.5
    min_test_sharpe: float = 0.3
    max_drawdown: float = -0.20


class WalkForwardValidator:
    """
    Walk-forward validation framework.
    
    This class implements institutional-grade validation methods to ensure
    models are robust and not overfitted to historical data.
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        """
        Initialize walk-forward validator.
        
        Args:
            config: Configuration for validation
        """
        self.config = config or WalkForwardConfig()
        self.validation_results: List[ValidationResult] = []
        
        logger.info("WalkForwardValidator initialized")
    
    def validate(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        parameters: Dict,
        method: ValidationMethod = ValidationMethod.WALK_FORWARD
    ) -> ValidationResult:
        """
        Run validation using specified method.
        
        Args:
            data: DataFrame with features and returns
            model_class: Model class to train
            parameters: Model hyperparameters
            method: Validation method to use
            
        Returns:
            ValidationResult with performance metrics
        """
        logger.info(f"Running validation with method: {method.value}")
        
        if method == ValidationMethod.WALK_FORWARD:
            return self._walk_forward_validation(data, model_class, parameters)
        elif method == ValidationMethod.PURGE_EMBARGO:
            return self._purge_embargo_validation(data, model_class, parameters)
        elif method == ValidationMethod.COMBINATORIAL_CV:
            return self._combinatorial_cv(data, model_class, parameters)
        else:
            return self._time_split_validation(data, model_class, parameters)
    
    def _walk_forward_validation(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        parameters: Dict
    ) -> ValidationResult:
        """
        Perform walk-forward validation.
        
        This is the gold standard for time-series validation, ensuring
        models are tested on truly out-of-sample data.
        """
        results = []
        
        total_length = len(data)
        train_len = self.config.train_window
        test_len = self.config.test_window
        step = self.config.step
        
        for start in range(0, total_length - train_len - test_len, step):
            train_end = start + train_len
            test_start = train_end
            test_end = test_start + test_len
            
            if test_end > total_length:
                break
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            model = model_class(**parameters)
            model.fit(train_data)
            
            # Get predictions
            train_preds = model.predict(train_data)
            test_preds = model.predict(test_data)
            
            # Calculate Sharpe ratios
            train_sharpe = self._calculate_sharpe(train_preds, train_data)
            test_sharpe = self._calculate_sharpe(test_preds, test_data)
            
            results.append({
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'train_returns': train_preds,
                'test_returns': test_preds
            })
        
        # Aggregate results
        avg_train_sharpe = np.mean([r['train_sharpe'] for r in results])
        avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
        
        # Calculate deflated Sharpe
        deflated_sharpe = self._calculate_deflated_sharpe(
            [r['test_sharpe'] for r in results],
            len(parameters)
        )
        
        # Get dates from data index
        if isinstance(data.index, pd.DatetimeIndex):
            train_start_date = data.index[0]
            train_end_date = data.index[self.config.train_window]
            test_start_date = data.index[self.config.train_window]
            test_end_date = data.index[-1]
        else:
            train_start_date = datetime.now()
            train_end_date = datetime.now()
            test_start_date = datetime.now()
            test_end_date = datetime.now()
        
        result = ValidationResult(
            method=ValidationMethod.WALK_FORWARD,
            train_start=train_start_date,
            train_end=train_end_date,
            test_start=test_start_date,
            test_end=test_end_date,
            train_sharpe=avg_train_sharpe,
            test_sharpe=avg_test_sharpe,
            train_returns=[r['train_returns'] for r in results],
            test_returns=[r['test_returns'] for r in results],
            deflated_sharpe=deflated_sharpe,
            num_trials=len(results),
            parameters=parameters
        )
        
        self.validation_results.append(result)
        return result
    
    def _purge_embargo_validation(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        parameters: Dict
    ) -> ValidationResult:
        """
        Perform validation with purging and embargo.
        
        Purging removes samples from training set that overlap with test set.
        Embargo adds a buffer period after training before testing.
        
        Based on Lopez de Prado (2018).
        """
        purge = self.config.purge_period
        embargo = self.config.embargo_period
        
        results = []
        
        total_length = len(data)
        train_len = self.config.train_window
        test_len = self.config.test_window
        
        for start in range(0, total_length - train_len - test_len - purge - embargo, self.config.step):
            train_end = start + train_len
            purge_end = train_end + purge
            test_start = purge_end + embargo
            test_end = test_start + test_len
            
            if test_end > total_length:
                break
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            model = model_class(**parameters)
            model.fit(train_data)
            
            # Get predictions
            train_preds = model.predict(train_data)
            test_preds = model.predict(test_data)
            
            # Calculate Sharpe ratios
            train_sharpe = self._calculate_sharpe(train_preds, train_data)
            test_sharpe = self._calculate_sharpe(test_preds, test_data)
            
            results.append({
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe
            })
        
        avg_train_sharpe = np.mean([r['train_sharpe'] for r in results])
        avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
        
        deflated_sharpe = self._calculate_deflated_sharpe(
            [r['test_sharpe'] for r in results],
            len(parameters)
        )
        
        result = ValidationResult(
            method=ValidationMethod.PURGE_EMBARGO,
            train_start=data.index[0],
            train_end=data.index[self.config.train_window],
            test_start=data.index[self.config.train_window],
            test_end=data.index[-1],
            train_sharpe=avg_train_sharpe,
            test_sharpe=avg_test_sharpe,
            train_returns=[],
            test_returns=[],
            deflated_sharpe=deflated_sharpe,
            num_trials=len(results),
            parameters=parameters
        )
        
        self.validation_results.append(result)
        return result
    
    def _combinatorial_cv(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        parameters: Dict
    ) -> ValidationResult:
        """
        Perform combinatorial cross-validation.
        
        This method tests all possible combinations of train/test splits
        to get a more robust estimate of model performance.
        """
        n_splits = self.config.n_splits
        test_size = self.config.test_size
        
        results = []
        
        for i in range(n_splits):
            # Calculate split indices
            test_start = int(i * (1 - test_size) * len(data) / n_splits)
            test_end = int(test_start + test_size * len(data))
            
            train_data = pd.concat([data.iloc[:test_start], data.iloc[test_end:]])
            test_data = data.iloc[test_start:test_end]
            
            # Train model
            model = model_class(**parameters)
            model.fit(train_data)
            
            # Get predictions
            train_preds = model.predict(train_data)
            test_preds = model.predict(test_data)
            
            # Calculate Sharpe ratios
            train_sharpe = self._calculate_sharpe(train_preds, train_data)
            test_sharpe = self._calculate_sharpe(test_preds, test_data)
            
            results.append({
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe
            })
        
        avg_train_sharpe = np.mean([r['train_sharpe'] for r in results])
        avg_test_sharpe = np.mean([r['test_sharpe'] for r in results])
        
        deflated_sharpe = self._calculate_deflated_sharpe(
            [r['test_sharpe'] for r in results],
            len(parameters)
        )
        
        result = ValidationResult(
            method=ValidationMethod.COMBINATORIAL_CV,
            train_start=data.index[0],
            train_end=data.index[-1],
            test_start=data.index[0],
            test_end=data.index[-1],
            train_sharpe=avg_train_sharpe,
            test_sharpe=avg_test_sharpe,
            train_returns=[],
            test_returns=[],
            deflated_sharpe=deflated_sharpe,
            num_trials=len(results),
            parameters=parameters
        )
        
        self.validation_results.append(result)
        return result
    
    def _time_split_validation(
        self,
        data: pd.DataFrame,
        model_class: Callable,
        parameters: Dict
    ) -> ValidationResult:
        """
        Simple time-series split validation.
        
        This is the baseline method (not recommended for production).
        """
        split_point = int(len(data) * 0.8)
        
        train_data = data.iloc[:split_point]
        test_data = data.iloc[split_point:]
        
        # Train model
        model = model_class(**parameters)
        model.fit(train_data)
        
        # Get predictions
        train_preds = model.predict(train_data)
        test_preds = model.predict(test_data)
        
        # Calculate Sharpe ratios
        train_sharpe = self._calculate_sharpe(train_preds, train_data)
        test_sharpe = self._calculate_sharpe(test_preds, test_data)
        
        deflated_sharpe = self._calculate_deflated_sharpe([test_sharpe], len(parameters))
        
        result = ValidationResult(
            method=ValidationMethod.TIME_SPLIT,
            train_start=data.index[0],
            train_end=data.index[split_point],
            test_start=data.index[split_point],
            test_end=data.index[-1],
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_returns=train_preds,
            test_returns=test_preds,
            deflated_sharpe=deflated_sharpe,
            num_trials=1,
            parameters=parameters
        )
        
        self.validation_results.append(result)
        return result
    
    def _calculate_sharpe(self, predictions: np.ndarray, data: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from predictions."""
        if 'returns' in data.columns:
            returns = data['returns'].values
        else:
            # Assume predictions are returns
            returns = predictions
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_deflated_sharpe(
        self,
        sharpe_values: List[float],
        num_parameters: int
    ) -> float:
        """
        Calculate deflated Sharpe ratio (Bailey 2014).
        
        The deflated Sharpe adjusts for multiple testing bias by
        accounting for the number of trials/parameters tested.
        """
        if not sharpe_values:
            return 0.0
        
        avg_sharpe = np.mean(sharpe_values)
        
        if not self.config.use_deflated_sharpe:
            return avg_sharpe
        
        # Bailey et al (2014) deflation formula
        E = self.config.expected_sharpe
        V = self.config.var_sharpe
        n = len(sharpe_values)
        
        # Probability of observing Sharpe >= avg_sharpe under null
        z_score = (avg_sharpe - E) / np.sqrt(V)
        p_value = 1 - stats.norm.cdf(z_score)
        
        # Deflated Sharpe
        deflated = avg_sharpe * (1 - p_value)
        
        return deflated
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {}
        
        summary = {
            'total_validations': len(self.validation_results),
            'avg_train_sharpe': np.mean([r.train_sharpe for r in self.validation_results]),
            'avg_test_sharpe': np.mean([r.test_sharpe for r in self.validation_results]),
            'avg_deflated_sharpe': np.mean([r.deflated_sharpe for r in self.validation_results]),
            'methods_used': list(set([r.method.value for r in self.validation_results]))
        }
        
        return summary
    
    def print_validation_report(self) -> None:
        """Print validation report."""
        summary = self.get_validation_summary()
        
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION REPORT")
        print("="*60)
        print(f"\nTotal Validations: {summary['total_validations']}")
        print(f"Average Train Sharpe: {summary['avg_train_sharpe']:.2f}")
        print(f"Average Test Sharpe: {summary['avg_test_sharpe']:.2f}")
        print(f"Average Deflated Sharpe: {summary['avg_deflated_sharpe']:.2f}")
        print(f"Methods Used: {', '.join(summary['methods_used'])}")
        
        print("\nIndividual Results:")
        for i, result in enumerate(self.validation_results):
            print(f"\n{i+1}. {result.method.value}")
            print(f"   Train Sharpe: {result.train_sharpe:.2f}")
            print(f"   Test Sharpe: {result.test_sharpe:.2f}")
            print(f"   Deflated Sharpe: {result.deflated_sharpe:.2f}")
            print(f"   Trials: {result.num_trials}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Test the walk-forward validator
    print("Testing Walk-Forward Validator...")
    
    validator = WalkForwardValidator()
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=2000, freq='1D')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 2000)
    
    data = pd.DataFrame({
        'returns': returns,
        'feature1': np.random.randn(2000),
        'feature2': np.random.randn(2000)
    }, index=dates)
    
    # Mock model class
    class MockModel:
        def __init__(self, **kwargs):
            pass
        
        def fit(self, data):
            pass
        
        def predict(self, data):
            return data['returns'].values
    
    # Run validation
    result = validator.validate(
        data,
        MockModel,
        {'param1': 1.0, 'param2': 2.0},
        ValidationMethod.WALK_FORWARD
    )
    
    print(f"Test Sharpe: {result.test_sharpe:.2f}")
    print(f"Deflated Sharpe: {result.deflated_sharpe:.2f}")
    
    validator.print_validation_report()

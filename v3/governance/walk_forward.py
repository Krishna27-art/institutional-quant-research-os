"""
Walk-Forward Research OS
Monthly automated validation for all active strategies with train/validate/test/forward splits.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    """Data split for walk-forward validation"""
    split_name: str
    start_date: date
    end_date: date
    data: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict:
        return {
            "split_name": self.split_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "days": (self.end_date - self.start_date).days,
        }


@dataclass
class WalkForwardResult:
    """Results of walk-forward validation for a strategy"""
    strategy_id: str
    validation_date: date
    
    # Data splits
    train_split: Optional[DataSplit] = None
    validate_split: Optional[DataSplit] = None
    test_split: Optional[DataSplit] = None
    forward_split: Optional[DataSplit] = None
    
    # Performance metrics
    train_sharpe: float = 0.0
    validate_sharpe: float = 0.0
    test_sharpe: float = 0.0
    forward_sharpe: float = 0.0
    
    # Hyperparameters
    best_hyperparameters: Dict = field(default_factory=dict)
    
    # Pass/fail
    passed: bool = False
    degradation_pct: float = 0.0
    flags: List[str] = field(default_factory=list)
    
    # Metadata
    validated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "validation_date": self.validation_date.isoformat(),
            "train_split": self.train_split.to_dict() if self.train_split else None,
            "validate_split": self.validate_split.to_dict() if self.validate_split else None,
            "test_split": self.test_split.to_dict() if self.test_split else None,
            "forward_split": self.forward_split.to_dict() if self.forward_split else None,
            "train_sharpe": self.train_sharpe,
            "validate_sharpe": self.validate_sharpe,
            "test_sharpe": self.test_sharpe,
            "forward_sharpe": self.forward_sharpe,
            "best_hyperparameters": self.best_hyperparameters,
            "passed": self.passed,
            "degradation_pct": self.degradation_pct,
            "flags": self.flags,
            "validated_at": self.validated_at.isoformat(),
        }


class WalkForwardOS:
    """
    Automated walk-forward validation framework.
    Runs monthly for all active strategies with train/validate/test/forward splits.
    """
    
    def __init__(
        self,
        train_years: int = 3,
        validate_years: int = 1,
        test_years: int = 1,
        forward_years: int = 1,
        min_test_sharpe: float = 1.0,
        max_degradation_pct: float = 30.0
    ):
        self.train_years = train_years
        self.validate_years = validate_years
        self.test_years = test_years
        self.forward_years = forward_years
        self.min_test_sharpe = min_test_sharpe
        self.max_degradation_pct = max_degradation_pct
        
        self.validation_results: Dict[str, List[WalkForwardResult]] = {}
        self.data_loader: Optional[Callable] = None
    
    def set_data_loader(self, data_loader: Callable) -> None:
        """
        Set data loader function for fetching historical market data.
        
        Args:
            data_loader: Function that takes (start_date, end_date) and returns DataFrame
        """
        self.data_loader = data_loader
    
    def create_splits(self, validation_date: date) -> Tuple[DataSplit, DataSplit, DataSplit, DataSplit]:
        """
        Create train/validate/test/forward data splits.
        
        Args:
            validation_date: Date when validation is run
        
        Returns:
            Tuple of (train, validate, test, forward) splits
        """
        # Calculate split dates
        forward_end = validation_date - timedelta(days=1)
        forward_start = forward_end - timedelta(days=self.forward_years * 365)
        
        test_end = forward_start - timedelta(days=1)
        test_start = test_end - timedelta(days=self.test_years * 365)
        
        validate_end = test_start - timedelta(days=1)
        validate_start = validate_end - timedelta(days=self.validate_years * 365)
        
        train_end = validate_start - timedelta(days=1)
        train_start = train_end - timedelta(days=self.train_years * 365)
        
        # Create split objects
        train_split = DataSplit("train", train_start, train_end)
        validate_split = DataSplit("validate", validate_start, validate_end)
        test_split = DataSplit("test", test_start, test_end)
        forward_split = DataSplit("forward", forward_start, forward_end)
        
        return train_split, validate_split, test_split, forward_split
    
    def load_split_data(self, split: DataSplit) -> pd.DataFrame:
        """Load data for a split"""
        if self.data_loader is None:
            raise ValueError("Data loader not set. Call set_data_loader() first.")
        
        data = self.data_loader(split.start_date, split.end_date)
        split.data = data
        return data
    
    def optimize_hyperparameters(
        self,
        train_data: pd.DataFrame,
        validate_data: pd.DataFrame,
        strategy_function: Callable,
        hyperparameter_grid: Dict[str, List]
    ) -> Dict:
        """
        Optimize hyperparameters on train+validate data.
        
        Args:
            train_data: Training data
            validate_data: Validation data
            strategy_function: Strategy function to evaluate
            hyperparameter_grid: Grid of hyperparameters to search
        
        Returns:
            Best hyperparameters
        """
        # Simplified: grid search
        # In production, use Bayesian optimization
        best_params = {}
        best_score = -np.inf
        
        # Generate all combinations
        from itertools import product
        keys = hyperparameter_grid.keys()
        values = hyperparameter_grid.values()
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            # Evaluate on validation data
            try:
                score = strategy_function(train_data, validate_data, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue
        
        return best_params
    
    def evaluate_strategy(
        self,
        strategy_id: str,
        strategy_function: Callable,
        hyperparameter_grid: Optional[Dict[str, List]] = None,
        validation_date: Optional[date] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward validation for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            strategy_function: Strategy function that takes (data, params) and returns Sharpe
            hyperparameter_grid: Optional hyperparameter grid for optimization
            validation_date: Date to run validation (default: today)
        
        Returns:
            WalkForwardResult with validation results
        """
        if validation_date is None:
            validation_date = date.today()
        
        result = WalkForwardResult(
            strategy_id=strategy_id,
            validation_date=validation_date
        )
        
        try:
            # Create splits
            train_split, validate_split, test_split, forward_split = self.create_splits(validation_date)
            result.train_split = train_split
            result.validate_split = validate_split
            result.test_split = test_split
            result.forward_split = forward_split
            
            # Load data
            train_data = self.load_split_data(train_split)
            validate_data = self.load_split_data(validate_split)
            test_data = self.load_split_data(test_split)
            forward_data = self.load_split_data(forward_split)
            
            # Optimize hyperparameters if grid provided
            if hyperparameter_grid:
                best_params = self.optimize_hyperparameters(
                    train_data, validate_data, strategy_function, hyperparameter_grid
                )
                result.best_hyperparameters = best_params
            else:
                best_params = {}
            
            # Evaluate on train
            result.train_sharpe = strategy_function(train_data, train_data, best_params)
            
            # Evaluate on validate
            result.validate_sharpe = strategy_function(train_data, validate_data, best_params)
            
            # Evaluate on test
            result.test_sharpe = strategy_function(train_data, test_data, best_params)
            
            # Check if test Sharpe meets threshold
            if result.test_sharpe < self.min_test_sharpe:
                result.flags.append(f"Test Sharpe {result.test_sharpe:.2f} below threshold {self.min_test_sharpe}")
                result.passed = False
                return result
            
            # Evaluate on forward (out-of-sample)
            result.forward_sharpe = strategy_function(train_data, forward_data, best_params)
            
            # Calculate degradation
            if result.test_sharpe > 0:
                result.degradation_pct = ((result.test_sharpe - result.forward_sharpe) / result.test_sharpe) * 100
            else:
                result.degradation_pct = 0.0
            
            # Check degradation
            if result.degradation_pct > self.max_degradation_pct:
                result.flags.append(
                    f"Forward Sharpe degradation {result.degradation_pct:.1f}% exceeds threshold "
                    f"{self.max_degradation_pct}%"
                )
                result.passed = False
            else:
                result.passed = True
            
        except Exception as e:
            result.flags.append(f"Validation failed: {str(e)}")
            result.passed = False
        
        # Store result
        if strategy_id not in self.validation_results:
            self.validation_results[strategy_id] = []
        self.validation_results[strategy_id].append(result)
        
        return result
    
    def run_monthly_validation(
        self,
        strategies: Dict[str, Callable],
        hyperparameter_grids: Optional[Dict[str, Dict[str, List]]] = None
    ) -> Dict[str, WalkForwardResult]:
        """
        Run monthly validation for all strategies.
        
        Args:
            strategies: Dictionary of strategy_id -> strategy_function
            hyperparameter_grids: Optional hyperparameter grids per strategy
        
        Returns:
            Dictionary of strategy_id -> WalkForwardResult
        """
        results = {}
        
        for strategy_id, strategy_function in strategies.items():
            hyperparams = hyperparameter_grids.get(strategy_id) if hyperparameter_grids else None
            result = self.evaluate_strategy(strategy_id, strategy_function, hyperparams)
            results[strategy_id] = result
        
        return results
    
    def get_validation_history(
        self,
        strategy_id: str,
        months: int = 12
    ) -> List[Dict]:
        """Get validation history for a strategy"""
        if strategy_id not in self.validation_results:
            return []
        
        cutoff_date = date.today() - timedelta(days=months * 30)
        recent_results = [
            r for r in self.validation_results[strategy_id]
            if r.validation_date >= cutoff_date
        ]
        
        return [r.to_dict() for r in recent_results]
    
    def get_all_strategies(self) -> List[str]:
        """Get list of all validated strategies"""
        return list(self.validation_results.keys())
    
    def get_degraded_strategies(self) -> List[str]:
        """Get strategies with significant degradation"""
        degraded = []
        
        for strategy_id, results in self.validation_results.items():
            if results:
                latest = results[-1]
                if not latest.passed and "degradation" in " ".join(latest.flags).lower():
                    degraded.append(strategy_id)
        
        return degraded
    
    def generate_report(self, strategy_id: str) -> Dict:
        """
        Generate comprehensive validation report for a strategy.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Report with summary and recommendations
        """
        history = self.validation_results.get(strategy_id, [])
        
        if not history:
            return {
                "strategy_id": strategy_id,
                "status": "No validations run",
                "recommendations": ["Run walk-forward validation first"]
            }
        
        latest = history[-1]
        
        # Calculate statistics
        test_sharpes = [r.test_sharpe for r in history]
        forward_sharpes = [r.forward_sharpe for r in history]
        degradation_pcts = [r.degradation_pct for r in history]
        
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0
        avg_forward_sharpe = np.mean(forward_sharpes) if forward_sharpes else 0.0
        avg_degradation = np.mean(degradation_pcts) if degradation_pcts else 0.0
        
        pass_rate = sum(1 for r in history if r.passed) / len(history) if history else 0.0
        
        # Generate recommendations
        recommendations = []
        
        if not latest.passed:
            if "degradation" in " ".join(latest.flags).lower():
                recommendations.append(
                    "Strategy shows significant degradation in forward validation. "
                    "Possible overfitting. Review hyperparameters and feature selection."
                )
            elif "threshold" in " ".join(latest.flags).lower():
                recommendations.append(
                    "Strategy does not meet minimum Sharpe threshold. "
                    "Consider retiring or retraining with different parameters."
                )
        
        if pass_rate < 0.5:
            recommendations.append(
                f"Strategy pass rate is {pass_rate:.1%}. Consider reviewing strategy design."
            )
        
        if avg_degradation > 20.0:
            recommendations.append(
                f"Average degradation is {avg_degradation:.1f}%. Strategy may be overfitting."
            )
        
        if latest.passed and pass_rate > 0.8:
            recommendations.append(
                "Strategy validation looks healthy. Ready for continued use."
            )
        
        return {
            "strategy_id": strategy_id,
            "latest_validation": latest.to_dict(),
            "validation_count": len(history),
            "avg_test_sharpe": avg_test_sharpe,
            "avg_forward_sharpe": avg_forward_sharpe,
            "avg_degradation_pct": avg_degradation,
            "pass_rate": pass_rate,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
        }
    
    def clear_results(self, strategy_id: Optional[str] = None) -> None:
        """Clear validation results, optionally filtered by strategy"""
        if strategy_id is None:
            self.validation_results.clear()
        else:
            self.validation_results.pop(strategy_id, None)


def simple_strategy_wrapper(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    params: Dict
) -> float:
    """
    Simple strategy wrapper for testing walk-forward validation.
    In production, replace with actual strategy function.
    
    Args:
        train_data: Training data
        test_data: Test data
        params: Strategy parameters
    
    Returns:
        Sharpe ratio
    """
    # Simplified: return random Sharpe between 0.5 and 1.5
    # In production, run actual backtest
    np.random.seed(hash(str(params)) % 2**32)
    return np.random.uniform(0.5, 1.5)

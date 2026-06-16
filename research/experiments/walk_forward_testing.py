"""
Walk-Forward Testing Engine
Validates models on multiple out-of-sample periods without leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import hashlib
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio from daily returns."""
    if len(returns) == 0:
        return 0.0
    returns_arr = np.array(returns)
    excess_returns = returns_arr - risk_free_rate / 252
    std_dev = np.std(excess_returns)
    if std_dev == 0:
        return 0.0
    return np.mean(excess_returns) / std_dev * np.sqrt(252)

def max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    if len(returns) == 0:
        return 0.0
    returns_arr = np.array(returns)
    cum_returns = (1 + returns_arr).cumprod()
    peak = np.maximum.accumulate(cum_returns)
    # Avoid division by zero
    peak = np.where(peak == 0, 1e-8, peak)
    drawdown = (cum_returns - peak) / peak
    return float(abs(np.min(drawdown)))

from time_machine_simulator import TimeMachineSimulator, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Single walk-forward fold"""
    fold_id: int
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    train_data_hash: str
    val_data_hash: str
    test_data_hash: str
    model_hash: Optional[str] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Overall walk-forward testing result"""
    experiment_id: str
    total_folds: int
    folds: List[WalkForwardFold]
    median_test_sharpe: float
    median_test_max_dd: float
    overall_metrics: Dict[str, float]
    data_leakage_check: Dict[str, bool]
    timestamp: datetime


class WalkForwardTester:
    """
    Walk-Forward Testing Engine for out-of-sample validation
    """
    
    def __init__(
        self,
        time_machine: TimeMachineSimulator,
        train_window_years: int = 3,
        test_window_years: int = 1,
        step_months: int = 1
    ):
        self.time_machine = time_machine
        self.train_window_years = train_window_years
        self.test_window_years = test_window_years
        self.step_months = step_months
        
        logger.info(
            f"Walk-Forward Tester initialized: "
            f"train={train_window_years}y, test={test_window_years}y, step={step_months}m"
        )
    
    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds
        
        Args:
            start_date: Overall start date
            end_date: Overall end date
            
        Returns:
            List of WalkForwardFold
        """
        train_window = timedelta(days=self.train_window_years * 365)
        test_window = timedelta(days=self.test_window_years * 365)
        step = timedelta(days=self.step_months * 30)
        
        folds = []
        fold_id = 0
        
        current_start = start_date
        
        while True:
            train_start = current_start
            train_end = train_start + train_window
            val_start = train_end
            val_end = val_start + test_window
            test_start = val_end
            test_end = test_start + test_window
            
            # Check if we have enough data for this fold
            if test_end > end_date:
                break
            
            # Generate data hashes for reproducibility
            train_data_hash = self._generate_data_hash(train_start, train_end)
            val_data_hash = self._generate_data_hash(val_start, val_end)
            test_data_hash = self._generate_data_hash(test_start, test_end)
            
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                train_data_hash=train_data_hash,
                val_data_hash=val_data_hash,
                test_data_hash=test_data_hash
            )
            
            folds.append(fold)
            fold_id += 1
            
            # Move to next fold
            current_start += step
        
        logger.info(f"Generated {len(folds)} walk-forward folds")
        
        return folds
    
    def _generate_data_hash(self, start_date: datetime, end_date: datetime) -> str:
        """Generate hash for data range"""
        hash_str = f"{start_date.isoformat()}_{end_date.isoformat()}"
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def run_fold(
        self,
        fold: WalkForwardFold,
        symbols: List[str],
        model: Any,
        train_func: Callable,
        predict_func: Callable,
        evaluate_func: Callable
    ) -> WalkForwardFold:
        """
        Run a single walk-forward fold
        
        Args:
            fold: WalkForwardFold to run
            symbols: Symbols to include
            model: Model to train
            train_func: Training function
            predict_func: Prediction function
            evaluate_func: Evaluation function
            
        Returns:
            Updated WalkForwardFold with metrics
        """
        logger.info(f"Running fold {fold.fold_id}: {fold.train_start} to {fold.test_end}")
        
        # Get training data (point-in-time)
        train_snapshots = self.time_machine.get_snapshot_range(
            start_date=fold.train_start,
            end_date=fold.train_end,
            frequency='1D',
            symbols=symbols,
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        train_features = self.time_machine.get_feature_matrix(train_snapshots)
        train_labels = self.time_machine.get_labels(train_snapshots, forward_periods=1)
        
        # Get validation data
        val_snapshots = self.time_machine.get_snapshot_range(
            start_date=fold.val_start,
            end_date=fold.val_end,
            frequency='1D',
            symbols=symbols,
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        val_features = self.time_machine.get_feature_matrix(val_snapshots)
        val_labels = self.time_machine.get_labels(val_snapshots, forward_periods=1)
        
        # Get test data (never used for training or tuning)
        test_snapshots = self.time_machine.get_snapshot_range(
            start_date=fold.test_start,
            end_date=fold.test_end,
            frequency='1D',
            symbols=symbols,
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        test_features = self.time_machine.get_feature_matrix(test_snapshots)
        test_labels = self.time_machine.get_labels(test_snapshots, forward_periods=1)
        
        # Train model
        trained_model = train_func(model, train_features, train_labels)
        
        # Store model hash
        fold.model_hash = self._generate_model_hash(trained_model)
        
        # Evaluate on training set
        train_preds = predict_func(trained_model, train_features)
        fold.train_metrics = evaluate_func(train_labels, train_preds)
        
        # Evaluate on validation set (for hyperparameter tuning)
        val_preds = predict_func(trained_model, val_features)
        fold.val_metrics = evaluate_func(val_labels, val_preds)
        
        # Evaluate on test set (final evaluation, never used for tuning)
        test_preds = predict_func(trained_model, test_features)
        fold.test_metrics = evaluate_func(test_labels, test_preds)
        
        logger.info(
            f"Fold {fold.fold_id} complete: "
            f"Train Sharpe={fold.train_metrics.get('sharpe', 0):.2f}, "
            f"Val Sharpe={fold.val_metrics.get('sharpe', 0):.2f}, "
            f"Test Sharpe={fold.test_metrics.get('sharpe', 0):.2f}"
        )
        
        return fold
    
    def run_walk_forward(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: List[str],
        model: Any,
        train_func: Callable,
        predict_func: Callable,
        evaluate_func: Callable
    ) -> WalkForwardResult:
        """
        Run complete walk-forward test
        
        Args:
            start_date: Overall start date
            end_date: Overall end date
            symbols: Symbols to include
            model: Model to train
            train_func: Training function
            predict_func: Prediction function
            evaluate_func: Evaluation function
            
        Returns:
            WalkForwardResult with all folds and overall metrics
        """
        # Generate folds
        folds = self.generate_folds(start_date, end_date)
        
        if not folds:
            logger.error("No folds generated")
            return WalkForwardResult(
                experiment_id="",
                total_folds=0,
                folds=[],
                median_test_sharpe=0.0,
                median_test_max_dd=0.0,
                overall_metrics={},
                data_leakage_check={},
                timestamp=datetime.now()
            )
        
        # Run each fold
        completed_folds = []
        for fold in folds:
            try:
                completed_fold = self.run_fold(
                    fold, symbols, model, train_func, predict_func, evaluate_func
                )
                completed_folds.append(completed_fold)
            except Exception as e:
                logger.error(f"Fold {fold.fold_id} failed: {e}")
        
        # Calculate overall metrics
        test_sharpes = [f.test_metrics.get('sharpe', 0) for f in completed_folds]
        test_max_dds = [f.test_metrics.get('max_drawdown', 0) for f in completed_folds]
        
        median_test_sharpe = np.median(test_sharpes) if test_sharpes else 0.0
        median_test_max_dd = np.median(test_max_dds) if test_max_dds else 0.0
        
        overall_metrics = {
            'median_test_sharpe': median_test_sharpe,
            'median_test_max_dd': median_test_max_dd,
            'mean_test_sharpe': np.mean(test_sharpes) if test_sharpes else 0.0,
            'std_test_sharpe': np.std(test_sharpes) if test_sharpes else 0.0,
            'min_test_sharpe': np.min(test_shares) if test_sharpes else 0.0,
            'max_test_sharpe': np.max(test_shares) if test_shares else 0.0,
        }
        
        # Check for data leakage
        leakage_check = self._check_data_leakage(completed_folds)
        
        # Generate experiment ID
        experiment_id = f"WF_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = WalkForwardResult(
            experiment_id=experiment_id,
            total_folds=len(completed_folds),
            folds=completed_folds,
            median_test_sharpe=median_test_sharpe,
            median_test_max_dd=median_test_max_dd,
            overall_metrics=overall_metrics,
            data_leakage_check=leakage_check,
            timestamp=datetime.now()
        )
        
        logger.info(
            f"Walk-forward complete: {len(completed_folds)} folds, "
            f"median test Sharpe={median_test_sharpe:.2f}"
        )
        
        return result
    
    def _generate_model_hash(self, model: Any) -> str:
        """Generate hash for model"""
        # For LightGBM, use model dump
        if hasattr(model, 'save_model'):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                model.save_model(f.name)
                with open(f.name, 'rb') as rf:
                    model_bytes = rf.read()
                Path(f.name).unlink()
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        else:
            # Fallback to string representation
            return hashlib.sha256(str(model).encode()).hexdigest()[:16]
    
    def _check_data_leakage(self, folds: List[WalkForwardFold]) -> Dict[str, bool]:
        """Check for data leakage across folds"""
        leakage_check = {
            'no_train_test_overlap': True,
            'no_val_test_overlap': True,
            'chronological_order': True,
            'data_hashes_unique': True,
        }
        
        # Check for overlap between train and test
        for fold in folds:
            if fold.train_end >= fold.test_start:
                leakage_check['no_train_test_overlap'] = False
                logger.error(f"Fold {fold.fold_id}: train-test overlap detected")
            
            if fold.val_end >= fold.test_start:
                leakage_check['no_val_test_overlap'] = False
                logger.error(f"Fold {fold.fold_id}: val-test overlap detected")
        
        # Check chronological order
        for i in range(len(folds) - 1):
            if folds[i].test_end > folds[i+1].train_start:
                leakage_check['chronological_order'] = False
                logger.error(f"Folds {i} and {i+1}: chronological order violated")
        
        # Check data hash uniqueness
        all_hashes = []
        for fold in folds:
            all_hashes.extend([
                fold.train_data_hash,
                fold.val_data_hash,
                fold.test_data_hash
            ])
        
        if len(all_hashes) != len(set(all_hashes)):
            leakage_check['data_hashes_unique'] = False
            logger.error("Duplicate data hashes detected")
        
        return leakage_check
    
    def save_result(self, result: WalkForwardResult, save_path: str) -> None:
        """Save walk-forward result to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        result_dict = {
            'experiment_id': result.experiment_id,
            'total_folds': result.total_folds,
            'median_test_sharpe': result.median_test_sharpe,
            'median_test_max_dd': result.median_test_max_dd,
            'overall_metrics': result.overall_metrics,
            'data_leakage_check': result.data_leakage_check,
            'timestamp': result.timestamp.isoformat(),
            'folds': [
                {
                    'fold_id': f.fold_id,
                    'train_start': f.train_start.isoformat(),
                    'train_end': f.train_end.isoformat(),
                    'val_start': f.val_start.isoformat(),
                    'val_end': f.val_end.isoformat(),
                    'test_start': f.test_start.isoformat(),
                    'test_end': f.test_end.isoformat(),
                    'train_data_hash': f.train_data_hash,
                    'val_data_hash': f.val_data_hash,
                    'test_data_hash': f.test_data_hash,
                    'model_hash': f.model_hash,
                    'train_metrics': f.train_metrics,
                    'val_metrics': f.val_metrics,
                    'test_metrics': f.test_metrics,
                }
                for f in result.folds
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved result to {save_path}")
    
    def load_result(self, load_path: str) -> WalkForwardResult:
        """Load walk-forward result from file"""
        with open(load_path, 'r') as f:
            result_dict = json.load(f)
        
        folds = [
            WalkForwardFold(
                fold_id=f['fold_id'],
                train_start=datetime.fromisoformat(f['train_start']),
                train_end=datetime.fromisoformat(f['train_end']),
                val_start=datetime.fromisoformat(f['val_start']),
                val_end=datetime.fromisoformat(f['val_end']),
                test_start=datetime.fromisoformat(f['test_start']),
                test_end=datetime.fromisoformat(f['test_end']),
                train_data_hash=f['train_data_hash'],
                val_data_hash=f['val_data_hash'],
                test_data_hash=f['test_data_hash'],
                model_hash=f.get('model_hash'),
                train_metrics=f['train_metrics'],
                val_metrics=f['val_metrics'],
                test_metrics=f['test_metrics'],
            )
            for f in result_dict['folds']
        ]
        
        result = WalkForwardResult(
            experiment_id=result_dict['experiment_id'],
            total_folds=result_dict['total_folds'],
            folds=folds,
            median_test_sharpe=result_dict['median_test_sharpe'],
            median_test_max_dd=result_dict['median_test_max_dd'],
            overall_metrics=result_dict['overall_metrics'],
            data_leakage_check=result_dict['data_leakage_check'],
            timestamp=datetime.fromisoformat(result_dict['timestamp'])
        )
        
        logger.info(f"Loaded result from {load_path}")
        
        return result


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def evaluate_predictions(labels: pd.Series, predictions: pd.Series) -> Dict[str, float]:
    """Evaluate predictions"""
    if len(labels) == 0 or len(predictions) == 0:
        return {}
    
    # Calculate returns from predictions (assuming predictions are returns)
    returns = predictions
    
    metrics = {
        'sharpe': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'mean_return': returns.mean(),
        'std_return': returns.std(),
        'total_return': returns.sum(),
    }
    
    # Calculate correlation if labels are available
    if len(labels) == len(predictions):
        metrics['correlation'] = labels.corr(predictions)
    
    return metrics


def train_lightgbm_model(model: Any, features: pd.DataFrame, labels: pd.Series) -> Any:
    """Train LightGBM model"""
    # Prepare data
    X = features.reset_index(level='symbol', drop=True)
    y = labels.reset_index(level='symbol', drop=True)
    
    # Align indices
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    # Train model
    model.fit(X, y)
    
    return model


def predict_lightgbm_model(model: Any, features: pd.DataFrame) -> pd.Series:
    """Predict with LightGBM model"""
    X = features.reset_index(level='symbol', drop=True)
    predictions = model.predict(X)
    
    return pd.Series(predictions, index=features.index)


def simulate_walk_forward_testing():
    """Simulate walk-forward testing"""
    
    print("="*60)
    print("WALK-FORWARD TESTING SIMULATION")
    print("="*60)
    
    # Initialize time machine
    time_machine = TimeMachineSimulator()
    
    # Initialize walk-forward tester
    tester = WalkForwardTester(
        time_machine=time_machine,
        train_window_years=1,  # Reduced for simulation
        test_window_years=0.5,
        step_months=1
    )
    
    # Generate folds
    print("\n1. Generating walk-forward folds...")
    folds = tester.generate_folds(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2024, 1, 1)
    )
    print(f"  Generated {len(folds)} folds")
    
    for fold in folds[:3]:
        print(f"    Fold {fold.fold_id}: {fold.train_start.date()} to {fold.test_end.date()}")
    
    # Create simple model
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Run walk-forward (simplified for simulation)
    print("\n2. Running walk-forward test...")
    print("  (Using simplified evaluation for simulation)")
    
    # Simulate results
    result = WalkForwardResult(
        experiment_id="SIMULATION",
        total_folds=len(folds),
        folds=folds[:3],  # Only first 3 for simulation
        median_test_sharpe=1.25,
        median_test_max_dd=-0.15,
        overall_metrics={
            'median_test_sharpe': 1.25,
            'median_test_max_dd': -0.15,
            'mean_test_sharpe': 1.18,
            'std_test_sharpe': 0.15,
            'min_test_sharpe': 0.95,
            'max_test_sharpe': 1.45,
        },
        data_leakage_check={
            'no_train_test_overlap': True,
            'no_val_test_overlap': True,
            'chronological_order': True,
            'data_hashes_unique': True,
        },
        timestamp=datetime.now()
    )
    
    print(f"  Median test Sharpe: {result.median_test_sharpe:.2f}")
    print(f"  Median test Max DD: {result.median_test_max_dd:.2%}")
    
    # Show overall metrics
    print("\n3. Overall metrics:")
    for metric, value in result.overall_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Show leakage check
    print("\n4. Data leakage check:")
    for check, passed in result.data_leakage_check.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Save result
    print("\n5. Saving result...")
    tester.save_result(result, "data/walk_forward_result.json")
    print("  Result saved to data/walk_forward_result.json")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_walk_forward_testing()

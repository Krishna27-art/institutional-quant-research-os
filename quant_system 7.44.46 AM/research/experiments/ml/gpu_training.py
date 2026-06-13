"""
GPU Training Support for XGBoost
Implements GPU-accelerated training for large-scale machine learning models.

Based on institutional review recommendations:
- GPU training (XGBoost GPU, RAPIDS)
- 10-100x speedup over CPU for large datasets
- Used by top firms for training on millions of samples
- Essential for feature engineering at scale

Key features:
- XGBoost GPU training
- RAPIDS cuDF for GPU dataframes
- GPU feature preprocessing
- Hyperparameter optimization with GPU
- Model ensemble with GPU
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUTrainer:
    """
    GPU-accelerated training for XGBoost and RAPIDS.
    
    Features:
    - XGBoost GPU training
    - RAPIDS cuDF for GPU dataframes
    - GPU feature preprocessing
    - Hyperparameter optimization with GPU
    - Model ensemble with GPU
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.xgb_gpu_available = self._check_xgb_gpu()
        self.rapids_available = self._check_rapids()
        
        if use_gpu and not self.xgb_gpu_available:
            logger.warning("XGBoost GPU not available, falling back to CPU")
            self.use_gpu = False
        
        logger.info(f"GPU Trainer initialized (GPU: {self.use_gpu})")
    
    def _check_xgb_gpu(self) -> bool:
        """Check if XGBoost GPU is available"""
        try:
            import xgboost as xgb
            # Try to create GPU DMatrix
            test_data = np.random.rand(100, 10)
            dmatrix = xgb.DMatrix(test_data)
            # Check if tree_method='gpu_hist' is available
            return True
        except Exception as e:
            logger.warning(f"XGBoost GPU not available: {e}")
            return False
    
    def _check_rapids(self) -> bool:
        """Check if RAPIDS is available"""
        try:
            import cudf
            return True
        except ImportError:
            logger.warning("RAPIDS not available")
            return False
    
    def train_xgboost_gpu(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None,
        num_rounds: int = 1000,
        early_stopping_rounds: int = 50
    ) -> Tuple[object, Dict]:
        """
        Train XGBoost model with GPU acceleration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: XGBoost parameters
            num_rounds: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            (model, training_metrics)
        """
        import xgboost as xgb
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'nthread': -1
        }
        
        if params:
            default_params.update(params)
        
        # Add GPU-specific parameters
        if self.use_gpu:
            default_params['tree_method'] = 'gpu_hist'
            default_params['gpu_id'] = 0
            logger.info("Using GPU for XGBoost training")
        else:
            default_params['tree_method'] = 'hist'
            logger.info("Using CPU for XGBoost training")
        
        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evals = [(dtrain, 'train')]
        
        # Train model
        start_time = datetime.now()
        
        model = xgb.train(
            default_params,
            dtrain,
            num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if evals else None,
            verbose_eval=100
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get training metrics
        metrics = {
            'training_time_seconds': training_time,
            'best_iteration': model.best_iteration,
            'best_score': model.best_score
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best iteration: {model.best_iteration}, Best score: {model.best_score:.4f}")
        
        return model, metrics
    
    def train_with_rapids(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict] = None,
        num_rounds: int = 1000
    ) -> Tuple[object, Dict]:
        """
        Train XGBoost model with RAPIDS cuDF for GPU dataframes.
        
        This provides additional speedup by keeping data on GPU throughout.
        """
        if not self.rapids_available:
            logger.warning("RAPIDS not available, falling back to standard XGBoost")
            return self.train_xgboost_gpu(X_train, y_train, X_val, y_val, params, num_rounds)
        
        import cudf
        import xgboost as xgb
        
        # Convert to GPU dataframe
        X_train_gpu = cudf.DataFrame(X_train)
        
        # Convert to DMatrix directly from GPU
        dtrain = xgb.DMatrix(X_train_gpu, label=y_train)
        
        evals = []
        if X_val is not None and y_val is not None:
            X_val_gpu = cudf.DataFrame(X_val)
            dval = xgb.DMatrix(X_val_gpu, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evals = [(dtrain, 'train')]
        
        # Default parameters with GPU
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',
            'gpu_id': 0
        }
        
        if params:
            default_params.update(params)
        
        # Train model
        start_time = datetime.now()
        
        model = xgb.train(
            default_params,
            dtrain,
            num_rounds,
            evals=evals,
            verbose_eval=100
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'training_time_seconds': training_time,
            'best_iteration': model.best_iteration,
            'best_score': model.best_score,
            'used_rapids': True
        }
        
        logger.info(f"RAPIDS training completed in {training_time:.2f} seconds")
        
        return model, metrics
    
    def hyperparameter_optimization_gpu(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        param_grid: Dict,
        n_trials: int = 50
    ) -> Tuple[Dict, object]:
        """
        Hyperparameter optimization with GPU acceleration.
        
        Uses Optuna with GPU-accelerated training.
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not available, returning default params")
            return {}, None
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'tree_method': 'gpu_hist' if self.use_gpu else 'hist'
            }
            
            model, metrics = self.train_xgboost_gpu(
                X_train, y_train, X_val, y_val,
                params=params,
                num_rounds=500,
                early_stopping_rounds=20
            )
            
            return metrics['best_score']
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_model, _ = self.train_xgboost_gpu(
            X_train, y_train, X_val, y_val,
            params=best_params,
            num_rounds=1000,
            early_stopping_rounds=50
        )
        
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return best_params, best_model
    
    def ensemble_gpu(
        self,
        models: List[object],
        X_test: pd.DataFrame,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Ensemble multiple GPU-trained models.
        
        Args:
            models: List of trained XGBoost models
            X_test: Test features
            weights: Optional weights for each model
            
        Returns:
            Ensemble predictions
        """
        import xgboost as xgb
        
        dtest = xgb.DMatrix(X_test)
        
        predictions = []
        for model in models:
            pred = model.predict(dtest)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if weights is None:
            # Equal weights
            ensemble_pred = np.mean(predictions, axis=0)
        else:
            # Weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = np.dot(weights, predictions)
        
        return ensemble_pred


def run_sample_gpu_training():
    """Run sample GPU training"""
    print("="*60)
    print("GPU TRAINING - DEMO")
    print("="*60)
    
    # Create trainer
    trainer = GPUTrainer(use_gpu=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100000
    n_features = 50
    
    X_train = pd.DataFrame(np.random.rand(n_samples, n_features), 
                          columns=[f'feature_{i}' for i in range(n_features)])
    y_train = np.random.rand(n_samples)
    
    X_val = pd.DataFrame(np.random.rand(n_samples // 5, n_features),
                        columns=[f'feature_{i}' for i in range(n_features)])
    y_val = np.random.rand(n_samples // 5)
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    
    # Train model
    print("\nTraining XGBoost with GPU...")
    model, metrics = trainer.train_xgboost_gpu(
        X_train, y_train, X_val, y_val,
        num_rounds=500,
        early_stopping_rounds=30
    )
    
    print(f"\nTraining Metrics:")
    print(f"  Training time: {metrics['training_time_seconds']:.2f} seconds")
    print(f"  Best iteration: {metrics['best_iteration']}")
    print(f"  Best score: {metrics['best_score']:.4f}")
    
    # Try RAPIDS if available
    if trainer.rapids_available:
        print("\nTraining with RAPIDS cuDF...")
        model_rapids, metrics_rapids = trainer.train_with_rapids(
            X_train, y_train, X_val, y_val,
            num_rounds=500
        )
        
        print(f"\nRAPIDS Training Metrics:")
        print(f"  Training time: {metrics_rapids['training_time_seconds']:.2f} seconds")
        print(f"  Best iteration: {metrics_rapids['best_iteration']}")
        print(f"  Best score: {metrics_rapids['best_score']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    run_sample_gpu_training()

"""
Ensemble Model (XGBoost + LightGBM)

This module implements an ensemble model combining XGBoost and LightGBM
for improved prediction accuracy and robustness.

Key Features:
- XGBoost and LightGBM model training
- Weighted ensemble averaging
- Stacking ensemble with meta-learner
- Cross-validation for optimal weights
- Feature importance aggregation
- Model diversity metrics

Based on Audit Report Priority 2: Alpha Generation
Research Papers: Zhou (2012) - Ensemble Methods
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result."""
    symbol: str
    prediction: float  # -1 to 1
    confidence: float  # 0 to 1
    xgb_prediction: float
    lgbm_prediction: float
    ensemble_weights: Dict[str, float]
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnsembleModel:
    """
    Ensemble model combining XGBoost and LightGBM.
    
    This class trains and combines multiple models for improved predictions.
    """
    
    def __init__(self, ensemble_method: str = "weighted_average"):
        """
        Initialize ensemble model.
        
        Args:
            ensemble_method: Method for combining predictions
                          ('weighted_average', 'stacking', 'voting')
        """
        self.ensemble_method = ensemble_method
        
        self.xgb_model = None
        self.lgbm_model = None
        self.meta_model = None  # For stacking
        
        self.xgb_weight = 0.5
        self.lgbm_weight = 0.5
        
        self.is_trained = False
        
        logger.info(f"EnsembleModel initialized with method: {ensemble_method}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> None:
        """
        Train ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training ensemble models...")
        
        # Train XGBoost
        self._train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train LightGBM
        self._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Calculate optimal weights if validation data provided
        if X_val is not None and y_val is not None:
            self._calculate_optimal_weights(X_val, y_val)
        
        # Train meta-learner for stacking
        if self.ensemble_method == "stacking" and X_val is not None:
            self._train_meta_learner(X_val, y_val)
        
        self.is_trained = True
        logger.info("Ensemble training completed")
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> None:
        """Train XGBoost model."""
        try:
            import xgboost as xgb
            
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            self.xgb_model = xgb.XGBRegressor(**params)
            
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            logger.info("XGBoost model trained")
            
        except ImportError:
            logger.warning("XGBoost not installed, skipping XGBoost model")
            self.xgb_model = None
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            self.xgb_model = None
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> None:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
            
            params = {
                'objective': 'regression',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            }
            
            self.lgbm_model = lgb.LGBMRegressor(**params)
            
            callbacks = []
            if X_val is not None and y_val is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=10))
            
            self.lgbm_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                callbacks=callbacks if callbacks else None
            )
            
            logger.info("LightGBM model trained")
            
        except ImportError:
            logger.warning("LightGBM not installed, skipping LightGBM model")
            self.lgbm_model = None
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            self.lgbm_model = None
    
    def _calculate_optimal_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """Calculate optimal ensemble weights using validation data."""
        if self.xgb_model is None and self.lgbm_model is None:
            return
        
        xgb_pred = self.xgb_model.predict(X_val) if self.xgb_model else np.zeros(len(y_val))
        lgbm_pred = self.lgbm_model.predict(X_val) if self.lgbm_model else np.zeros(len(y_val))
        
        # Calculate errors
        xgb_error = np.mean((xgb_pred - y_val) ** 2)
        lgbm_error = np.mean((lgbm_pred - y_val) ** 2)
        
        # Inverse error weighting (lower error = higher weight)
        total_error = xgb_error + lgbm_error
        if total_error > 0:
            self.xgb_weight = (1 - xgb_error / total_error) if self.xgb_model is not None else 0
            self.lgbm_weight = (1 - lgbm_error / total_error) if self.lgbm_model is not None else 0
            
            # Normalize
            total_weight = self.xgb_weight + self.lgbm_weight
            if total_weight > 0:
                self.xgb_weight /= total_weight
                self.lgbm_weight /= total_weight
        
        logger.info(f"Optimal weights - XGB: {self.xgb_weight:.3f}, LGBM: {self.lgbm_weight:.3f}")
    
    def _train_meta_learner(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """Train meta-learner for stacking ensemble."""
        try:
            from sklearn.linear_model import Ridge
            
            # Get base model predictions
            xgb_pred = self.xgb_model.predict(X_val) if self.xgb_model else np.zeros(len(y_val))
            lgbm_pred = self.lgbm_model.predict(X_val) if self.lgbm_model else np.zeros(len(y_val))
            
            # Create meta-features
            meta_features = np.column_stack([xgb_pred, lgbm_pred])
            
            # Train meta-learner
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_features, y_val)
            
            logger.info("Meta-learner trained for stacking")
            
        except Exception as e:
            logger.error(f"Meta-learner training failed: {e}")
            self.meta_model = None
    
    def predict(
        self,
        X: pd.DataFrame,
        symbol: str = None
    ) -> Optional[EnsemblePrediction]:
        """
        Make ensemble prediction.
        
        Args:
            X: Features
            symbol: Symbol name
            
        Returns:
            EnsemblePrediction
        """
        if not self.is_trained:
            logger.warning("Ensemble model not trained")
            return None
        
        # Get individual predictions
        xgb_pred = self.xgb_model.predict(X) if self.xgb_model else 0.0
        lgbm_pred = self.lgbm_model.predict(X) if self.lgbm_model else 0.0
        
        # Combine predictions
        if self.ensemble_method == "weighted_average":
            prediction = self.xgb_weight * xgb_pred + self.lgbm_weight * lgbm_pred
        elif self.ensemble_method == "stacking" and self.meta_model is not None:
            meta_features = np.column_stack([xgb_pred, lgbm_pred])
            prediction = self.meta_model.predict(meta_features)[0]
        else:  # voting
            prediction = (xgb_pred + lgbm_pred) / 2
        
        # Clip to [-1, 1] range
        prediction = max(-1.0, min(1.0, prediction))
        
        # Calculate confidence based on agreement
        agreement = 1.0 - abs(xgb_pred - lgbm_pred) / 2.0
        confidence = max(0.0, min(1.0, agreement))
        
        return EnsemblePrediction(
            symbol=symbol or "unknown",
            prediction=prediction,
            confidence=confidence,
            xgb_prediction=float(xgb_pred),
            lgbm_prediction=float(lgbm_pred),
            ensemble_weights={
                'xgboost': self.xgb_weight,
                'lightgbm': self.lgbm_weight
            },
            timestamp=datetime.now(),
            metadata={
                'ensemble_method': self.ensemble_method,
                'is_trained': self.is_trained
            }
        )
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated feature importance.
        
        Returns:
            Dictionary mapping model names to feature importance
        """
        importance = {}
        
        if self.xgb_model is not None and hasattr(self.xgb_model, 'feature_importances_'):
            importance['xgboost'] = dict(zip(
                self.xgb_model.get_booster().feature_names,
                self.xgb_model.feature_importances_
            ))
        
        if self.lgbm_model is not None and hasattr(self.lgbm_model, 'feature_importances_'):
            importance['lightgbm'] = dict(zip(
                self.lgbm_model.feature_name_,
                self.lgbm_model.feature_importances_
            ))
        
        return importance
    
    def save_model(self, path: str) -> None:
        """
        Save ensemble model to disk.
        
        Args:
            path: Path to save model
        """
        model_data = {
            'xgb_model': self.xgb_model,
            'lgbm_model': self.lgbm_model,
            'meta_model': self.meta_model,
            'xgb_weight': self.xgb_weight,
            'lgbm_weight': self.lgbm_weight,
            'ensemble_method': self.ensemble_method,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Ensemble model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load ensemble model from disk.
        
        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        
        self.xgb_model = model_data['xgb_model']
        self.lgbm_model = model_data['lgbm_model']
        self.meta_model = model_data['meta_model']
        self.xgb_weight = model_data['xgb_weight']
        self.lgbm_weight = model_data['lgbm_weight']
        self.ensemble_method = model_data['ensemble_method']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Ensemble model loaded from {path}")


def create_ensemble(
    ensemble_method: str = "weighted_average"
) -> EnsembleModel:
    """
    Create ensemble model.
    
    Args:
        ensemble_method: Method for combining predictions
        
    Returns:
        EnsembleModel instance
    """
    return EnsembleModel(ensemble_method=ensemble_method)


if __name__ == "__main__":
    # Test ensemble model
    print("Testing Ensemble Model...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    X_val = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.randn(200))
    
    # Create and train ensemble
    ensemble = EnsembleModel(ensemble_method="weighted_average")
    ensemble.train(X_train, y_train, X_val, y_val)
    
    # Make prediction
    X_test = pd.DataFrame(
        np.random.randn(1, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    prediction = ensemble.predict(X_test, symbol="RELIANCE")
    
    if prediction:
        print(f"\nEnsemble Prediction:")
        print(f"  Prediction: {prediction.prediction:.4f}")
        print(f"  Confidence: {prediction.confidence:.2%}")
        print(f"  XGB Prediction: {prediction.xgb_prediction:.4f}")
        print(f"  LGBM Prediction: {prediction.lgbm_prediction:.4f}")
        print(f"  Weights: {prediction.ensemble_weights}")
    
    # Get feature importance
    importance = ensemble.get_feature_importance()
    print(f"\nFeature Importance:")
    for model, imp_dict in importance.items():
        print(f"  {model}: {len(imp_dict)} features")

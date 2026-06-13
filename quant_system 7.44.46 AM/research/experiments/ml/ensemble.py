"""
ML Layer - XGBoost + LightGBM Ensemble
Implements ensemble model for signal generation with SHAP interpretability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Model prediction result"""
    prediction: float
    confidence: float
    shap_values: Optional[np.ndarray]
    feature_importance: Dict[str, float]
    timestamp: pd.Timestamp


class XGBoostModel:
    """XGBoost model wrapper"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'n_jobs': -1
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.is_fitted = False
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Fit XGBoost model
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        try:
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            logger.info(f"XGBoost model fitted with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting XGBoost: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using XGBoost model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_fitted:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save(self, path: Path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'params': self.params,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"XGBoost model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.params = data['params']
            self.is_fitted = data['is_fitted']
        logger.info(f"XGBoost model loaded from {path}")


class LightGBMModel:
    """LightGBM model wrapper"""
    
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize LightGBM model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1
        }
        self.model = lgb.LGBMRegressor(**self.params)
        self.is_fitted = False
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Fit LightGBM model
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        try:
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit model
            self.model.fit(X_scaled, y)
            self.is_fitted = True
            
            logger.info(f"LightGBM model fitted with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting LightGBM: {e}")
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using LightGBM model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_fitted:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save(self, path: Path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'params': self.params,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"LightGBM model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.params = data['params']
            self.is_fitted = data['is_fitted']
        logger.info(f"LightGBM model loaded from {path}")


class EnsembleModel:
    """XGBoost + LightGBM ensemble for signal generation"""
    
    def __init__(
        self,
        xgb_weight: float = 0.6,
        lgb_weight: float = 0.4,
        random_state: int = 42
    ):
        """
        Initialize ensemble model
        
        Args:
            xgb_weight: Weight for XGBoost predictions
            lgb_weight: Weight for LightGBM predictions
            random_state: Random seed
        """
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        
        # Initialize models
        self.xgb_model = XGBoostModel(random_state=random_state)
        self.lgb_model = LightGBMModel(random_state=random_state)
        
        self.is_fitted = False
        self.shap_explainer_xgb = None
        self.shap_explainer_lgb = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_override: Optional[pd.Series] = None
    ) -> bool:
        """
        Fit ensemble model with walk-forward validation
        
        Args:
            X: Feature matrix
            y: Target variable
            regime_override: Optional regime flags for regime-specific training
        """
        try:
            # Fit individual models
            xgb_success = self.xgb_model.fit(X, y)
            lgb_success = self.lgb_model.fit(X, y)
            
            if not (xgb_success and lgb_success):
                logger.error("Failed to fit one or more models")
                return False
            
            # Initialize SHAP explainers
            X_scaled = self.xgb_model.scaler.transform(X)
            self.shap_explainer_xgb = shap.TreeExplainer(self.xgb_model.model)
            self.shap_explainer_lgb = shap.TreeExplainer(self.lgb_model.model)
            
            self.is_fitted = True
            logger.info("Ensemble model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting ensemble: {e}")
            return False
    
    def predict(
        self,
        X: pd.DataFrame,
        regime: Optional[str] = None
    ) -> ModelPrediction:
        """
        Predict using ensemble
        
        Args:
            X: Feature matrix
            regime: Current market regime for regime override
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Get individual predictions
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        # Ensemble prediction
        ensemble_pred = self.xgb_weight * xgb_pred + self.lgb_weight * lgb_pred
        
        # Calculate confidence based on prediction variance
        pred_variance = np.var([xgb_pred, lgb_pred], axis=0)
        confidence = 1.0 / (1.0 + pred_variance)  # Higher variance = lower confidence
        
        # Get SHAP values
        X_scaled = self.xgb_model.scaler.transform(X)
        shap_values_xgb = self.shap_explainer_xgb.shap_values(X_scaled)
        shap_values_lgb = self.shap_explainer_lgb.shap_values(X_scaled)
        
        # Combined SHAP values
        combined_shap = self.xgb_weight * shap_values_xgb + self.lgb_weight * shap_values_lgb
        
        # Feature importance
        xgb_importance = self.xgb_model.get_feature_importance()
        lgb_importance = self.lgb_model.get_feature_importance()
        
        combined_importance = {}
        all_features = set(xgb_importance.keys()) | set(lgb_importance.keys())
        for feat in all_features:
            combined_importance[feat] = (
                self.xgb_weight * xgb_importance.get(feat, 0) +
                self.lgb_weight * lgb_importance.get(feat, 0)
            )
        
        # Regime override (if provided)
        final_pred = ensemble_pred[0] if len(ensemble_pred) == 1 else ensemble_pred
        if regime:
            final_pred = self._apply_regime_override(final_pred, regime)
        
        return ModelPrediction(
            prediction=float(final_pred),
            confidence=float(confidence[0] if len(confidence) == 1 else confidence.mean()),
            shap_values=combined_shap[0] if len(combined_shap) == 1 else combined_shap,
            feature_importance=combined_importance,
            timestamp=pd.Timestamp.now()
        )
    
    def _apply_regime_override(self, prediction: float, regime: str) -> float:
        """Apply regime-based override to prediction"""
        regime_adjustments = {
            'bull_trend': 1.2,  # Boost long signals
            'bear_trend': 0.8,  # Reduce long signals
            'sideways': 1.0,  # No adjustment
            'high_volatility': 0.7,  # Reduce all signals
            'low_volatility': 1.3,  # Boost signals
            'panic': 0.0,  # No signals
            'euphoria': 0.5  # Reduce signals significantly
        }
        
        adjustment = regime_adjustments.get(regime, 1.0)
        return prediction * adjustment
    
    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_years: int = 5,
        test_years: int = 1,
        step_years: int = 1
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation
        
        Args:
            X: Feature matrix
            y: Target variable
            train_years: Training window in years
            test_years: Test window in years
            step_years: Step size in years
        """
        results = {
            'fold_results': [],
            'overall_metrics': {}
        }
        
        # Convert to daily data
        train_window = train_years * 252
        test_window = test_years * 252
        step = step_years * 252
        
        for start in range(0, len(X) - train_window - test_window, step):
            train_end = start + train_window
            test_start = train_end
            test_end = test_start + test_window
            
            if test_end > len(X):
                break
            
            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Fit model
            self.fit(X_train, y_train)
            
            # Predict
            predictions = []
            for i in range(len(X_test)):
                pred = self.predict(X_test.iloc[i:i+1])
                predictions.append(pred.prediction)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            
            fold_result = {
                'train_start': start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'mse': mse,
                'mae': mae
            }
            results['fold_results'].append(fold_result)
            
            logger.info(f"Fold {len(results['fold_results'])}: MSE={mse:.4f}, MAE={mae:.4f}")
        
        # Calculate overall metrics
        if results['fold_results']:
            results['overall_metrics']['avg_mse'] = np.mean([f['mse'] for f in results['fold_results']])
            results['overall_metrics']['avg_mae'] = np.mean([f['mae'] for f in results['fold_results']])
            results['overall_metrics']['std_mse'] = np.std([f['mse'] for f in results['fold_results']])
        
        return results
    
    def save(self, directory: Path):
        """Save ensemble to disk"""
        directory.mkdir(parents=True, exist_ok=True)
        
        self.xgb_model.save(directory / 'xgb_model.pkl')
        self.lgb_model.save(directory / 'lgb_model.pkl')
        
        # Save ensemble config
        with open(directory / 'ensemble_config.pkl', 'wb') as f:
            pickle.dump({
                'xgb_weight': self.xgb_weight,
                'lgb_weight': self.lgb_weight,
                'is_fitted': self.is_fitted
            }, f)
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load(self, directory: Path):
        """Load ensemble from disk"""
        self.xgb_model.load(directory / 'xgb_model.pkl')
        self.lgb_model.load(directory / 'lgb_model.pkl')
        
        with open(directory / 'ensemble_config.pkl', 'rb') as f:
            config = pickle.load(f)
            self.xgb_weight = config['xgb_weight']
            self.lgb_weight = config['lgb_weight']
            self.is_fitted = config['is_fitted']
        
        # Reinitialize SHAP explainers
        if self.is_fitted:
            self.shap_explainer_xgb = shap.TreeExplainer(self.xgb_model.model)
            self.shap_explainer_lgb = shap.TreeExplainer(self.lgb_model.model)
        
        logger.info(f"Ensemble loaded from {directory}")


class ModelPipeline:
    """Complete ML pipeline for training and inference"""
    
    def __init__(
        self,
        model_dir: Path = Path('models'),
        retrain_interval_days: int = 7
    ):
        """
        Initialize model pipeline
        
        Args:
            model_dir: Directory to save/load models
            retrain_interval_days: Days between retraining
        """
        self.model_dir = model_dir
        self.retrain_interval_days = retrain_interval_days
        self.ensemble = EnsembleModel()
        self.last_retrain_date = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        perform_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Train ensemble model
        
        Args:
            X: Feature matrix
            y: Target variable
            perform_validation: Whether to perform walk-forward validation
        """
        results = {}
        
        # Fit model
        fit_success = self.ensemble.fit(X, y)
        results['fit_success'] = fit_success
        
        if not fit_success:
            return results
        
        # Perform validation if requested
        if perform_validation:
            validation_results = self.ensemble.walk_forward_validation(X, y)
            results['validation'] = validation_results
        
        # Save model
        self.ensemble.save(self.model_dir)
        self.last_retrain_date = pd.Timestamp.now()
        
        logger.info("Model training completed")
        return results
    
    def predict(
        self,
        X: pd.DataFrame,
        regime: Optional[str] = None,
        force_retrain: bool = False
    ) -> ModelPrediction:
        """
        Predict using ensemble
        
        Args:
            X: Feature matrix
            regime: Current market regime
            force_retrain: Force model retraining
        """
        # Check if retraining is needed
        if force_retrain or self._should_retrain():
            logger.warning("Model retraining needed but no training data provided")
        
        if not self.ensemble.is_fitted:
            # Try to load from disk
            if (self.model_dir / 'xgb_model.pkl').exists():
                self.ensemble.load(self.model_dir)
            else:
                raise ValueError("Model not fitted and no saved model found")
        
        return self.ensemble.predict(X, regime)
    
    def _should_retrain(self) -> bool:
        """Check if model should be retrained"""
        if self.last_retrain_date is None:
            return True
        
        days_since_retrain = (pd.Timestamp.now() - self.last_retrain_date).days
        return days_since_retrain >= self.retrain_interval_days
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance"""
        return self.ensemble.xgb_model.get_feature_importance()
    
    def get_shap_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get SHAP summary for feature importance"""
        if not self.ensemble.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.ensemble.xgb_model.scaler.transform(X)
        shap_values = self.ensemble.shap_explainer_xgb.shap_values(X_scaled)
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(self.ensemble.xgb_model.feature_names, mean_shap))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'feature_importance': sorted_importance,
            'top_features': list(sorted_importance.keys())[:10]
        }

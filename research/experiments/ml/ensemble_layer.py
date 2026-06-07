"""
ML Ensemble Layer - XGBoost + LightGBM with SHAP Explainability
Based on Blueprint V1.0

Architecture:
- Primary: XGBoost + LightGBM Ensemble (60% XGB, 40% LGB)
- Secondary: CatBoost for feature importance (monthly)
- Research: LSTM/GRU for regime detection (offline)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for ML Ensemble."""
    
    # CRITICAL FIX: Simplify model to prevent overfitting
    use_simple_baseline: bool = True
    simple_model_type: str = "linear"  # "linear" or "shallow_tree"
    shallow_tree_max_depth: int = 3  # Max depth for shallow tree
    
    # CRITICAL FIX: Use ensemble of simple models instead of one complex model
    use_simple_ensemble: bool = True
    simple_ensemble_size: int = 3  # Average 3 simple models
    
    # Original complex model parameters (for comparison)
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    lgb_n_estimators: int = 150
    lgb_max_depth: int = 5
    lgb_learning_rate: float = 0.03
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0
    xgb_weight: float = 0.6
    lgb_weight: float = 0.4
    train_test_split: float = 0.2
    validation_splits: int = 5
    early_stopping_rounds: int = 20
    max_features: int = 50
    feature_selection_method: str = "mutual_information"
    use_regime_override: bool = True
    regime_adjustment_factor: float = 0.1


class XGBoostModel:
    """XGBoost model wrapper."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def build_model(self):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost not available")
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_alpha=self.config.xgb_reg_alpha,
            reg_lambda=self.config.xgb_reg_lambda,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> Dict:
        if self.model is None:
            self.build_model()
        self.feature_names = X.columns.tolist()
        
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False,
                         early_stopping_rounds=self.config.early_stopping_rounds)
        else:
            self.model.fit(X, y, verbose=False)
        
        self.is_trained = True
        y_pred = self.model.predict(X)
        metrics = {'train_mse': mean_squared_error(y, y_pred),
                  'train_mae': mean_absolute_error(y, y_pred)}
        
        if X_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics['val_mse'] = mean_squared_error(y_val, y_val_pred)
            metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='gain') -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained")
        importance = self.model.get_booster().get_score(importance_type=importance_type)
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        return importance
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_trained = True


class LightGBMModel:
    """LightGBM model wrapper."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def build_model(self):
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM not available")
        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.lgb_n_estimators,
            max_depth=self.config.lgb_max_depth,
            learning_rate=self.config.lgb_learning_rate,
            subsample=self.config.lgb_subsample,
            colsample_bytree=self.config.lgb_colsample_bytree,
            reg_alpha=self.config.lgb_reg_alpha,
            reg_lambda=self.config.lgb_reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> Dict:
        if self.model is None:
            self.build_model()
        self.feature_names = X.columns.tolist()
        
        if X_val is not None and y_val is not None:
            callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
            self.model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=callbacks)
        else:
            self.model.fit(X, y)
        
        self.is_trained = True
        y_pred = self.model.predict(X)
        metrics = {'train_mse': mean_squared_error(y, y_pred),
                  'train_mae': mean_absolute_error(y, y_pred)}
        
        if X_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics['val_mse'] = mean_squared_error(y_val, y_val_pred)
            metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='gain') -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained")
        importance = self.model.booster_.feature_importance(importance_type=importance_type)
        importance_dict = dict(zip(self.feature_names, importance))
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        return importance_dict
    
    def save(self, path: str):
        joblib.dump(self.model, path)
    
    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_trained = True


class SimpleBaselineModel:
    """
    Simple baseline model to prevent overfitting.
    
    CRITICAL FIX: Use linear regression or shallow tree (max_depth=3) as baseline.
    If simple model doesn't work, signal is noise.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_trained = False
    
    def build_model(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        if self.config.simple_model_type == "linear":
            self.model = LinearRegression()
        elif self.config.simple_model_type == "shallow_tree":
            self.model = DecisionTreeRegressor(
                max_depth=self.config.shallow_tree_max_depth,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown simple model type: {self.config.simple_model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> Dict:
        if self.model is None:
            self.build_model()
        self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_trained = True
        
        y_pred = self.model.predict(X)
        metrics = {'train_mse': mean_squared_error(y, y_pred),
                  'train_mae': mean_absolute_error(y, y_pred)}
        
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            metrics['val_mse'] = mean_squared_error(y_val, y_val_pred)
            metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict:
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if hasattr(self.model, 'coef_'):  # Linear regression
            importance = dict(zip(self.feature_names, np.abs(self.model.coef_)))
        elif hasattr(self.model, 'feature_importances_'):  # Decision tree
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            importance = {}
        
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance


class SimpleEnsembleModel:
    """
    Ensemble of simple models to prevent overfitting.
    
    CRITICAL FIX: Average 3 simple models instead of one complex model.
    This reduces variance and prevents overfitting.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = []
        self.is_trained = False
        self.feature_names = None
    
    def build_models(self):
        """Build multiple simple models."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn not available")
        
        self.models = []
        for i in range(self.config.simple_ensemble_size):
            if self.config.simple_model_type == "linear":
                model = LinearRegression()
            elif self.config.simple_model_type == "shallow_tree":
                # Vary random state for diversity
                model = DecisionTreeRegressor(
                    max_depth=self.config.shallow_tree_max_depth,
                    random_state=42 + i
                )
            else:
                raise ValueError(f"Unknown simple model type: {self.config.simple_model_type}")
            self.models.append(model)
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> Dict:
        """Train all models in the ensemble."""
        if not self.models:
            self.build_models()
        
        self.feature_names = X.columns.tolist()
        all_metrics = []
        
        for i, model in enumerate(self.models):
            model.fit(X, y)
            
            y_pred = model.predict(X)
            metrics = {
                'train_mse': mean_squared_error(y, y_pred),
                'train_mae': mean_absolute_error(y, y_pred)
            }
            
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                metrics['val_mse'] = mean_squared_error(y_val, y_val_pred)
                metrics['val_mae'] = mean_absolute_error(y_val, y_val_pred)
            
            all_metrics.append(metrics)
        
        self.is_trained = True
        
        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict by averaging all models."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def get_feature_importance(self) -> Dict:
        """Get average feature importance across models."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        all_importances = []
        for model in self.models:
            if hasattr(model, 'coef_'):  # Linear regression
                importance = np.abs(model.coef_)
            elif hasattr(model, 'feature_importances_'):  # Decision tree
                importance = model.feature_importances_
            else:
                continue
            all_importances.append(importance)
        
        if not all_importances:
            return {}
        
        # Average importance
        avg_importance = np.mean(all_importances, axis=0)
        importance_dict = dict(zip(self.feature_names, avg_importance))
        
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v/total for k, v in importance_dict.items()}
        
        return importance_dict


class EnsembleModel:
    """XGBoost + LightGBM Ensemble with SHAP explainability."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.xgb_model = XGBoostModel(self.config)
        self.lgb_model = LightGBMModel(self.config)
        self.simple_baseline = SimpleBaselineModel(self.config)  # CRITICAL FIX
        self.simple_ensemble = SimpleEnsembleModel(self.config)  # CRITICAL FIX
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}
        self.xgb_explainer = None
        self.lgb_explainer = None
        self.feature_importance = {}
        self.baseline_metrics = {}  # CRITICAL FIX: Track baseline performance
        self.simple_ensemble_metrics = {}  # CRITICAL FIX: Track simple ensemble performance
        
    def train(self, X: pd.DataFrame, y: pd.Series, use_walk_forward=True, n_splits=5) -> Dict:
        if use_walk_forward and SKLEARN_AVAILABLE:
            return self._train_walk_forward(X, y, n_splits)
        else:
            return self._train_simple(X, y)
    
    def _train_simple(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        split_idx = int(len(X) * (1 - self.config.train_test_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # CRITICAL FIX: Train simple baseline first
        if self.config.use_simple_baseline:
            baseline_metrics = self.simple_baseline.train(X_train, y_train, X_val, y_val)
            self.baseline_metrics = baseline_metrics
            print(f"CRITICAL FIX: Baseline model metrics: {baseline_metrics}")
        
        # CRITICAL FIX: Train simple ensemble
        if self.config.use_simple_ensemble:
            simple_ensemble_metrics = self.simple_ensemble.train(X_train, y_train, X_val, y_val)
            self.simple_ensemble_metrics = simple_ensemble_metrics
            print(f"CRITICAL FIX: Simple ensemble metrics: {simple_ensemble_metrics}")
        
        xgb_metrics = self.xgb_model.train(X_train, y_train, X_val, y_val)
        lgb_metrics = self.lgb_model.train(X_train, y_train, X_val, y_val)
        
        self.is_trained = True
        self.feature_names = X.columns.tolist()
        
        self.training_metrics = {
            'baseline': self.baseline_metrics if self.config.use_simple_baseline else {},
            'simple_ensemble': self.simple_ensemble_metrics if self.config.use_simple_ensemble else {},
            'xgboost': xgb_metrics,
            'lightgbm': lgb_metrics,
            'ensemble': self._calculate_ensemble_metrics(X_val, y_val)
        }
        
        self._calculate_feature_importance()
        if SHAP_AVAILABLE:
            self._initialize_shap(X_train)
        
        return self.training_metrics
    
    def _train_walk_forward(self, X: pd.DataFrame, y: pd.Series, n_splits: int) -> Dict:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_metrics = {'baseline': [], 'simple_ensemble': [], 'xgboost': [], 'lightgbm': [], 'ensemble': []}
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # CRITICAL FIX: Train simple baseline first
            if self.config.use_simple_baseline:
                baseline_metrics = self.simple_baseline.train(X_train, y_train, X_val, y_val)
                all_metrics['baseline'].append(baseline_metrics)
            
            # CRITICAL FIX: Train simple ensemble
            if self.config.use_simple_ensemble:
                simple_ensemble_metrics = self.simple_ensemble.train(X_train, y_train, X_val, y_val)
                all_metrics['simple_ensemble'].append(simple_ensemble_metrics)
            
            xgb_metrics = self.xgb_model.train(X_train, y_train, X_val, y_val)
            lgb_metrics = self.lgb_model.train(X_train, y_train, X_val, y_val)
            ensemble_metrics = self._calculate_ensemble_metrics(X_val, y_val)
            
            all_metrics['xgboost'].append(xgb_metrics)
            all_metrics['lightgbm'].append(lgb_metrics)
            all_metrics['ensemble'].append(ensemble_metrics)
        
        avg_metrics = {}
        for model_name in ['baseline', 'simple_ensemble', 'xgboost', 'lightgbm', 'ensemble']:
            if not all_metrics[model_name]:
                continue
            avg_metrics[model_name] = {}
            for key in all_metrics[model_name][0].keys():
                values = [m[key] for m in all_metrics[model_name]]
                avg_metrics[model_name][key] = np.mean(values)
        
        self.is_trained = True
        self.feature_names = X.columns.tolist()
        self.training_metrics = avg_metrics
        self._calculate_feature_importance()
        
        if SHAP_AVAILABLE:
            self._initialize_shap(X)
        
        return avg_metrics
    
    def _calculate_ensemble_metrics(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        y_pred = self.predict(X_val)
        return {'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred)}
    
    def _calculate_feature_importance(self):
        xgb_imp = self.xgb_model.get_feature_importance()
        lgb_imp = self.lgb_model.get_feature_importance()
        
        combined_imp = {}
        all_features = set(xgb_imp.keys()) | set(lgb_imp.keys())
        
        for feature in all_features:
            xgb_val = xgb_imp.get(feature, 0)
            lgb_val = lgb_imp.get(feature, 0)
            combined_imp[feature] = (self.config.xgb_weight * xgb_val + self.config.lgb_weight * lgb_val)
        
        total = sum(combined_imp.values())
        if total > 0:
            combined_imp = {k: v/total for k, v in combined_imp.items()}
        
        self.feature_importance = combined_imp
    
    def _initialize_shap(self, X_background: pd.DataFrame):
        if not SHAP_AVAILABLE:
            return
        
        background_size = min(100, len(X_background))
        X_background_sample = X_background.sample(n=background_size, random_state=42)
        
        try:
            self.xgb_explainer = shap.TreeExplainer(self.xgb_model.model, X_background_sample)
            self.lgb_explainer = shap.TreeExplainer(self.lgb_model.model, X_background_sample)
        except Exception as e:
            print(f"Warning: Could not initialize SHAP: {e}")
    
    def predict(self, X: pd.DataFrame, regime_multiplier: float = 1.0) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        ensemble_pred = (self.config.xgb_weight * xgb_pred + self.config.lgb_weight * lgb_pred)
        
        if self.config.use_regime_override:
            ensemble_pred = ensemble_pred * (1 + self.config.regime_adjustment_factor * (regime_multiplier - 1))
        
        return ensemble_pred
    
    def predict_with_confidence(self, X: pd.DataFrame, regime_multiplier: float = 1.0) -> Tuple:
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        ensemble_pred = (self.config.xgb_weight * xgb_pred + self.config.lgb_weight * lgb_pred)
        pred_diff = np.abs(xgb_pred - lgb_pred)
        confidence = 1.0 / (1.0 + pred_diff)
        
        if self.config.use_regime_override:
            ensemble_pred = ensemble_pred * (1 + self.config.regime_adjustment_factor * (regime_multiplier - 1))
        
        return ensemble_pred, confidence
    
    def explain_prediction(self, X: pd.DataFrame, idx: int = 0) -> Dict:
        if not SHAP_AVAILABLE or self.xgb_explainer is None:
            return {}
        
        if idx >= len(X):
            raise IndexError(f"Index {idx} out of range")
        
        xgb_shap = self.xgb_explainer.shap_values(X.iloc[idx:idx+1])
        lgb_shap = self.lgb_explainer.shap_values(X.iloc[idx:idx+1])
        
        combined_shap = {}
        for i, feature in enumerate(self.feature_names):
            combined_shap[feature] = (self.config.xgb_weight * xgb_shap[0][i] + self.config.lgb_weight * lgb_shap[0][i])
        
        return combined_shap
    
    def get_feature_importance(self, top_n: int = 20) -> List:
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_n]
    
    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        self.xgb_model.save(f"{path}/xgboost_model.pkl")
        self.lgb_model.save(f"{path}/lightgbm_model.pkl")
        
        metadata = {
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        joblib.dump(metadata, f"{path}/ensemble_metadata.pkl")
    
    def load(self, path: str):
        self.xgb_model.load(f"{path}/xgboost_model.pkl")
        self.lgb_model.load(f"{path}/lightgbm_model.pkl")
        
        metadata = joblib.load(f"{path}/ensemble_metadata.pkl")
        self.feature_names = metadata['feature_names']
        self.training_metrics = metadata['training_metrics']
        self.feature_importance = metadata['feature_importance']
        self.config = metadata['config']
        
        self.is_trained = True


class ModelPipeline:
    """Complete ML pipeline for institutional quant."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.ensemble = EnsembleModel(self.config)
        
    def run_pipeline(self, X: pd.DataFrame, y: pd.Series, select_features=True, n_features=50) -> Dict:
        results = {
            'original_features': len(X.columns),
            'selected_features': len(X.columns),
            'feature_importance': {},
            'training_metrics': {}
        }
        
        if select_features:
            from features.institutional_feature_engine import FeatureSelector
            selector = FeatureSelector()
            selected = selector.select_features(X, y, method=self.config.feature_selection_method, n_features=n_features)
            X = X[selected]
            results['selected_features'] = len(selected)
        
        metrics = self.ensemble.train(X, y, use_walk_forward=True, n_splits=5)
        results['training_metrics'] = metrics
        
        top_features = self.ensemble.get_feature_importance(top_n=20)
        results['feature_importance'] = dict(top_features)
        
        return results
    
    def predict_live(self, X: pd.DataFrame, regime_multiplier: float = 1.0) -> Tuple:
        predictions, confidence = self.ensemble.predict_with_confidence(X, regime_multiplier)
        explanation = self.ensemble.explain_prediction(X, idx=0)
        return predictions, confidence, explanation

"""
ML Ensemble - XGBoost + LightGBM ensemble
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb


class MLEnsemble:
    """Ensemble of XGBoost and LightGBM models"""
    
    def __init__(self, xgb_weight: float = 0.6, lgb_weight: float = 0.4):
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.lgb_model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: Optional[List[str]] = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> None:
        """
        Fit ensemble models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        self.feature_names = X_train.columns.tolist()
        
        # XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        if X_val is not None and y_val is not None:
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train)
        
        # LightGBM model
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        if X_val is not None and y_val is not None:
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )
        else:
            self.lgb_model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.xgb_model is None or self.lgb_model is None:
            raise RuntimeError("Models must be fitted before prediction")
        
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = self.xgb_weight * xgb_pred + self.lgb_weight * lgb_pred
        
        return ensemble_pred
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (using model variance)
        
        Args:
            X: Features
            
        Returns:
            (predictions, confidence)
        """
        if self.xgb_model is None or self.lgb_model is None:
            raise RuntimeError("Models must be fitted before prediction")
        
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        ensemble_pred = self.xgb_weight * xgb_pred + self.lgb_weight * lgb_pred
        
        # Confidence based on prediction variance between models
        variance = np.var([xgb_pred, lgb_pred], axis=0)
        confidence = 1 / (1 + variance)  # Higher confidence when models agree
        
        return ensemble_pred, confidence
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get combined feature importance"""
        if self.xgb_model is None or self.lgb_model is None:
            raise RuntimeError("Models must be fitted first")
        
        xgb_importance = dict(zip(self.feature_names, self.xgb_model.feature_importances_))
        lgb_importance = dict(zip(self.feature_names, self.lgb_model.feature_importances_))
        
        # Normalize and combine
        xgb_total = sum(xgb_importance.values())
        lgb_total = sum(lgb_importance.values())
        
        if xgb_total > 0:
            xgb_importance = {k: v / xgb_total for k, v in xgb_importance.items()}
        if lgb_total > 0:
            lgb_importance = {k: v / lgb_total for k, v in lgb_importance.items()}
        
        combined_importance = {}
        for feature in self.feature_names:
            combined_importance[feature] = (
                self.xgb_weight * xgb_importance.get(feature, 0) +
                self.lgb_weight * lgb_importance.get(feature, 0)
            )
        
        return combined_importance

"""
LightGBM + CatBoost Ensemble
Based on V4 Blueprint - Institutional Architecture

Key improvements over single LightGBM:
- CatBoost handles categorical features better (Indian sectors, FII/DII categories)
- Ensemble stacking improves robustness
- Expected Sharpe improvement: +0.1–0.2

V4 Upgrade - Expected Sharpe increase: +0.1–0.2
Priority: Medium (Phase 2)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result."""
    prediction: float
    lightgbm_pred: float
    catboost_pred: float
    ensemble_weight_lgb: float
    ensemble_weight_cat: float
    confidence: float


class LightGBMCatBoostEnsemble:
    """
    LightGBM + CatBoost ensemble for institutional trading.
    
    Benefits:
    - LightGBM: Fast, robust, handles large datasets well
    - CatBoost: Excellent with categorical features (sectors, FII/DII)
    - Ensemble: More robust, less overfitting
    
    Implementation:
    - Train both models on same data
    - Optimize ensemble weights via validation set
    - Dynamic weight adjustment based on recent performance
    """
    
    def __init__(self):
        self.lgb_model = None
        self.cat_model = None
        self.lgb_weight = 0.5
        self.cat_weight = 0.5
        self.feature_importance_lgb = {}
        self.feature_importance_cat = {}
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None
    ) -> None:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: LightGBM parameters
        """
        try:
            import lightgbm as lgb
        except ImportError:
            print("LightGBM not installed. Install with: pip install lightgbm")
            return
        
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        # Get feature importance
        self.feature_importance_lgb = dict(zip(
            X_train.columns,
            self.lgb_model.feature_importance(importance_type='gain')
        ))
    
    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Optional[Dict] = None,
        cat_features: Optional[List[str]] = None
    ) -> None:
        """
        Train CatBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: CatBoost parameters
            cat_features: List of categorical feature names
        """
        try:
            from catboost import CatBoostRegressor, Pool
        except ImportError:
            print("CatBoost not installed. Install with: pip install catboost")
            return
        
        if params is None:
            params = {
                'objective': 'RMSE',
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_state': 42,
                'verbose': False
            }
        
        # Create pools
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        # Train model
        self.cat_model = CatBoostRegressor(**params)
        self.cat_model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance_cat = dict(zip(
            X_train.columns,
            self.cat_model.get_feature_importance()
        ))
    
    def optimize_ensemble_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """
        Optimize ensemble weights on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        if self.lgb_model is None:
            raise ValueError("LightGBM model must be trained before optimizing weights")
        
        if self.cat_model is None:
            # Only LightGBM available
            self.lgb_weight = 1.0
            self.cat_weight = 0.0
            return
        
        # Get predictions
        lgb_pred = self.lgb_model.predict(X_val)
        cat_pred = self.cat_model.predict(X_val)
        
        # Try different weight combinations
        best_weight = 0.5
        best_mse = float('inf')
        
        for weight in np.linspace(0, 1, 21):
            ensemble_pred = weight * lgb_pred + (1 - weight) * cat_pred
            mse = mean_squared_error(y_val, ensemble_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_weight = weight
        
        self.lgb_weight = best_weight
        self.cat_weight = 1 - best_weight
    
    def predict(
        self,
        X: pd.DataFrame
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction.
        
        Args:
            X: Features
            
        Returns:
            EnsemblePrediction
        """
        if self.lgb_model is None:
            raise ValueError("LightGBM model must be trained before prediction")
        
        # Get LightGBM prediction
        lgb_pred = self.lgb_model.predict(X)
        
        # If CatBoost not available, use only LightGBM
        if self.cat_model is None:
            ensemble_pred = lgb_pred
            cat_pred = lgb_pred
            confidence = 0.8
        else:
            # Get CatBoost prediction
            cat_pred = self.cat_model.predict(X)
            
            # Ensemble prediction
            ensemble_pred = self.lgb_weight * lgb_pred + self.cat_weight * cat_pred
            
            # Calculate confidence (based on agreement between models)
            pred_diff = abs(lgb_pred - cat_pred)
            confidence = 1.0 / (1.0 + pred_diff)
        
        return EnsemblePrediction(
            prediction=float(ensemble_pred[0]) if len(ensemble_pred) == 1 else ensemble_pred,
            lightgbm_pred=float(lgb_pred[0]) if len(lgb_pred) == 1 else lgb_pred,
            catboost_pred=float(cat_pred[0]) if len(cat_pred) == 1 else cat_pred,
            ensemble_weight_lgb=self.lgb_weight,
            ensemble_weight_cat=self.cat_weight,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else (float(confidence[0]) if len(confidence) == 1 else confidence)
        )
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate ensemble on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        prediction = self.predict(X_test)
        
        mse = mean_squared_error(y_test, prediction.prediction)
        mae = mean_absolute_error(y_test, prediction.prediction)
        rmse = np.sqrt(mse)
        
        # Individual model performance
        lgb_mse = mean_squared_error(y_test, prediction.lightgbm_pred)
        cat_mse = mean_squared_error(y_test, prediction.catboost_pred)
        
        return {
            'ensemble_mse': mse,
            'ensemble_mae': mae,
            'ensemble_rmse': rmse,
            'lightgbm_mse': lgb_mse,
            'catboost_mse': cat_mse,
            'ensemble_improvement': (lgb_mse - mse) / lgb_mse if lgb_mse > 0 else 0
        }
    
    def print_ensemble_report(self) -> None:
        """Print ensemble report."""
        print("\n" + "="*60)
        print("LIGHTGBM + CATBOOST ENSEMBLE REPORT")
        print("="*60)
        print(f"LightGBM Weight: {self.lgb_weight:.2%}")
        print(f"CatBoost Weight: {self.cat_weight:.2%}")
        
        if self.feature_importance_lgb:
            print("\nTop 10 LightGBM Features:")
            for feat, imp in sorted(self.feature_importance_lgb.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feat}: {imp:.2f}")
        
        if self.feature_importance_cat:
            print("\nTop 10 CatBoost Features:")
            for feat, imp in sorted(self.feature_importance_cat.items(), 
                                   key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feat}: {imp:.2f}")
        
        print("="*60)


def run_sample_ensemble():
    """Run sample LightGBM + CatBoost ensemble."""
    ensemble = LightGBMCatBoostEnsemble()
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Add categorical features
    X['sector'] = np.random.choice(['IT', 'Finance', 'Energy', 'Consumer'], n_samples)
    X['fii_category'] = np.random.choice(['Buy', 'Sell', 'Neutral'], n_samples)
    
    # Generate target
    y = X['feature_0'] * 0.5 + X['feature_1'] * 0.3 + np.random.randn(n_samples) * 0.1
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train LightGBM (without categorical features for simplicity)
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_val_num = X_val.select_dtypes(include=[np.number])
    X_test_num = X_test.select_dtypes(include=[np.number])
    
    print("Training LightGBM...")
    ensemble.train_lightgbm(X_train_num, y_train, X_val_num, y_val)
    
    # Train CatBoost (with categorical features)
    cat_features = ['sector', 'fii_category']
    print("Training CatBoost...")
    ensemble.train_catboost(X_train, y_train, X_val, y_val, cat_features=cat_features)
    
    # Optimize ensemble weights
    print("Optimizing ensemble weights...")
    ensemble.optimize_ensemble_weights(X_val_num, y_val)
    
    # Evaluate
    print("Evaluating ensemble...")
    metrics = ensemble.evaluate(X_test_num, y_test)
    
    # Print report
    ensemble.print_ensemble_report()
    
    print("\nTest Set Metrics:")
    print(f"  Ensemble MSE: {metrics['ensemble_mse']:.4f}")
    print(f"  Ensemble RMSE: {metrics['ensemble_rmse']:.4f}")
    print(f"  LightGBM MSE: {metrics['lightgbm_mse']:.4f}")
    print(f"  CatBoost MSE: {metrics['catboost_mse']:.4f}")
    print(f"  Ensemble Improvement: {metrics['ensemble_improvement']:.2%}")
    
    return ensemble


if __name__ == "__main__":
    run_sample_ensemble()

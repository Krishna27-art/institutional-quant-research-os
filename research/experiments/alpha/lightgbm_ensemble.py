"""
LightGBM Ensemble for Alpha Combination
Based on research recommendations for Indian markets

Key findings from research:
- LightGBM: 5-20ms inference, handles categoricals, native missing values
- XGBoost: Slightly better accuracy, use as ensemble
- Ensemble: Rolling Sharpe optimization
- Rebalance: Daily
- Risk: Correlation penalty

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class AlphaSignal:
    """Individual alpha signal"""
    name: str
    value: float
    confidence: float
    timestamp: pd.Timestamp


@dataclass
class EnsembleResult:
    """Result of ensemble combination"""
    combined_signal: float
    individual_signals: Dict[str, float]
    weights: Dict[str, float]
    confidence: float
    regime_adjusted: bool


class LightGBMEnsemble:
    """
    LightGBM Ensemble for Alpha Combination.
    
    Architecture:
    - Primary: LightGBM (fastest, lowest latency)
    - Secondary: XGBoost (slightly better accuracy)
    - Ensemble: Stacked with rolling Sharpe optimization
    - Rebalance: Daily
    - Risk: Correlation penalty
    
    Why LightGBM over alternatives:
    - 5-20ms inference vs LSTM (100-500ms)
    - Handles categorical features (sector, time-of-day)
    - Native missing value handling
    - Built-in regularization
    - Tree-based = no scaling required
    - Feature importance native
    """
    
    def __init__(
        self,
        alpha_names: List[str],
        use_xgboost: bool = True,
        lookback_days: int = 252
    ):
        self.alpha_names = alpha_names
        self.use_xgboost = use_xgboost
        self.lookback_days = lookback_days
        
        # Models
        self.lgb_model = None
        self.xgb_model = None
        
        # Weights (rolling Sharpe optimization)
        self.weights = {name: 1.0 / len(alpha_names) for name in alpha_names}
        
        # Correlation matrix for risk penalty
        self.correlation_matrix = None
        
        # Performance tracking
        self.performance_history = {name: [] for name in alpha_names}
        
        self.is_fitted = False
    
    def prepare_features(
        self,
        alpha_signals: Dict[str, List[float]],
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for ensemble.
        
        Features:
        - Individual alpha signals
        - Alpha signal lags (1, 5, 10 periods)
        - Market features (returns, volatility, volume)
        - Time features (hour, day of week)
        - Regime features (if available)
        """
        features = []
        
        # Get signal history
        signal_df = pd.DataFrame(alpha_signals)
        
        for i in range(len(signal_df)):
            row_features = {}
            
            # Current alpha signals
            for alpha in self.alpha_names:
                row_features[f'{alpha}_current'] = signal_df[alpha].iloc[i]
            
            # Lagged signals
            for lag in [1, 5, 10]:
                for alpha in self.alpha_names:
                    if i >= lag:
                        row_features[f'{alpha}_lag_{lag}'] = signal_df[alpha].iloc[i - lag]
                    else:
                        row_features[f'{alpha}_lag_{lag}'] = 0.0
            
            # Market features
            if i < len(market_data):
                row_features['market_return'] = market_data['close'].pct_change().iloc[i]
                row_features['market_volatility'] = market_data['close'].pct_change().rolling(20).std().iloc[i]
                row_features['market_volume_ratio'] = market_data['volume'].iloc[i] / market_data['volume'].rolling(20).mean().iloc[i]
            
            # Time features
            timestamp = signal_df.index[i] if hasattr(signal_df, 'index') else pd.Timestamp.now()
            row_features['hour'] = timestamp.hour
            row_features['day_of_week'] = timestamp.dayofweek
            row_features['is_month_end'] = 1 if timestamp.is_month_end else 0
            
            features.append(row_features)
        
        return pd.DataFrame(features)
    
    def fit(
        self,
        alpha_signals: Dict[str, List[float]],
        returns: List[float],
        market_data: pd.DataFrame
    ) -> None:
        """
        Fit ensemble models on historical data.
        
        Args:
            alpha_signals: Dictionary mapping alpha names to signal values
            returns: Target returns to predict
            market_data: Market data for feature engineering
        """
        print(f"Fitting LightGBM ensemble on {len(returns)} samples...")
        
        # Prepare features
        features_df = self.prepare_features(alpha_signals, market_data)
        
        # Prepare target
        target = np.array(returns)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Fit LightGBM
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        lgb_scores = []
        for train_idx, val_idx in tscv.split(features_df):
            X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
            y_train, y_val = target[train_idx], target[val_idx]
            
            self.lgb_model.fit(X_train, y_train)
            pred = self.lgb_model.predict(X_val)
            score = mean_squared_error(y_val, pred)
            lgb_scores.append(score)
        
        print(f"LightGBM CV MSE: {np.mean(lgb_scores):.6f}")
        
        # Fit on full data
        self.lgb_model.fit(features_df, target)
        
        # Fit XGBoost if enabled
        if self.use_xgboost:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            xgb_scores = []
            for train_idx, val_idx in tscv.split(features_df):
                X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
                y_train, y_val = target[train_idx], target[val_idx]
                
                self.xgb_model.fit(X_train, y_train)
                pred = self.xgb_model.predict(X_val)
                score = mean_squared_error(y_val, pred)
                xgb_scores.append(score)
            
            print(f"XGBoost CV MSE: {np.mean(xgb_scores):.6f}")
            
            # Fit on full data
            self.xgb_model.fit(features_df, target)
        
        # Calculate correlation matrix for risk penalty
        signal_df = pd.DataFrame(alpha_signals)
        self.correlation_matrix = signal_df.corr()
        
        self.is_fitted = True
        print("Ensemble fitted successfully")
    
    def optimize_weights_rolling_sharpe(
        self,
        alpha_returns: Dict[str, List[float]],
        window: int = 60
    ) -> Dict[str, float]:
        """
        Optimize weights using rolling Sharpe ratio.
        
        Args:
            alpha_returns: Dictionary mapping alpha names to returns
            window: Rolling window for Sharpe calculation
            
        Returns:
            Optimized weights dictionary
        """
        weights = {}
        
        for alpha in self.alpha_names:
            returns = alpha_returns[alpha][-window:]
            
            if len(returns) < 10:
                weights[alpha] = 1.0 / len(self.alpha_names)
                continue
            
            # Calculate Sharpe ratio
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Store performance
            self.performance_history[alpha].append(sharpe)
            
            # Keep only recent history
            if len(self.performance_history[alpha]) > window:
                self.performance_history[alpha] = self.performance_history[alpha][-window:]
        
        # Calculate weights based on Sharpe
        sharpe_values = []
        for alpha in self.alpha_names:
            if self.performance_history[alpha]:
                sharpe = np.mean(self.performance_history[alpha])
                sharpe_values.append(max(sharpe, 0.1))  # Minimum 0.1
            else:
                sharpe_values.append(0.1)
        
        # Normalize to sum to 1
        total = sum(sharpe_values)
        for i, alpha in enumerate(self.alpha_names):
            weights[alpha] = sharpe_values[i] / total
        
        # Apply correlation penalty
        if self.correlation_matrix is not None:
            weights = self._apply_correlation_penalty(weights)
        
        self.weights = weights
        return weights
    
    def _apply_correlation_penalty(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply correlation penalty to weights."""
        penalized_weights = weights.copy()
        
        for i, alpha1 in enumerate(self.alpha_names):
            penalty = 0.0
            for j, alpha2 in enumerate(self.alpha_names):
                if i != j:
                    corr = self.correlation_matrix.loc[alpha1, alpha2]
                    if corr > 0.5:  # High correlation threshold
                        penalty += weights[alpha2] * (corr - 0.5)
            
            penalized_weights[alpha1] = max(weights[alpha1] - penalty, 0.01)
        
        # Re-normalize
        total = sum(penalized_weights.values())
        if total > 0:
            penalized_weights = {k: v/total for k, v in penalized_weights.items()}
        
        return penalized_weights
    
    def predict(
        self,
        current_signals: Dict[str, float],
        market_data: pd.DataFrame,
        regime_weights: Optional[Dict[str, float]] = None
    ) -> EnsembleResult:
        """
        Generate combined signal using ensemble.
        
        Args:
            current_signals: Current alpha signals
            market_data: Current market data
            regime_weights: Optional regime-based weights
            
        Returns:
            EnsembleResult with combined signal
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")
        
        # Prepare current features
        current_features = {}
        for alpha in self.alpha_names:
            current_features[f'{alpha}_current'] = current_signals.get(alpha, 0.0)
        
        # Add market features
        current_features['market_return'] = market_data['close'].pct_change().iloc[-1]
        current_features['market_volatility'] = market_data['close'].pct_change().rolling(20).std().iloc[-1]
        current_features['market_volume_ratio'] = market_data['volume'].iloc[-1] / market_data['volume'].rolling(20).mean().iloc[-1]
        
        # Add time features
        timestamp = market_data.index[-1]
        current_features['hour'] = timestamp.hour
        current_features['day_of_week'] = timestamp.dayofweek
        current_features['is_month_end'] = 1 if timestamp.is_month_end else 0
        
        # Convert to DataFrame
        features_df = pd.DataFrame([current_features])
        
        # Get predictions
        lgb_pred = self.lgb_model.predict(features_df)[0]
        
        if self.use_xgboost and self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(features_df)[0]
            # Simple average ensemble
            combined = (lgb_pred + xgb_pred) / 2
        else:
            combined = lgb_pred
        
        # Apply weights
        weighted_signals = {}
        for alpha in self.alpha_names:
            weighted_signals[alpha] = current_signals.get(alpha, 0.0) * self.weights[alpha]
        
        # Apply regime weights if provided
        if regime_weights:
            for alpha in self.alpha_names:
                if alpha in regime_weights:
                    weighted_signals[alpha] *= regime_weights[alpha]
        
        # Final combined signal
        final_signal = sum(weighted_signals.values()) + combined * 0.5
        
        # Calculate confidence (based on weight concentration)
        weight_entropy = -sum(w * np.log(w) for w in self.weights.values() if w > 0)
        max_entropy = np.log(len(self.alpha_names))
        confidence = 1.0 - (weight_entropy / max_entropy)
        
        return EnsembleResult(
            combined_signal=final_signal,
            individual_signals=current_signals,
            weights=self.weights,
            confidence=confidence,
            regime_adjusted=regime_weights is not None
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from LightGBM model."""
        if not self.is_fitted or self.lgb_model is None:
            return {}
        
        importance = self.lgb_model.feature_importances_
        feature_names = self.lgb_model.feature_name_
        
        return dict(zip(feature_names, importance))
    
    def print_ensemble_info(self, result: EnsembleResult) -> None:
        """Print ensemble results."""
        print("\n" + "="*60)
        print("LIGHTGBM ENSEMBLE RESULTS")
        print("="*60)
        print(f"Combined Signal: {result.combined_signal:.6f}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Regime Adjusted: {result.regime_adjusted}")
        
        print("\nIndividual Signals:")
        for alpha, signal in result.individual_signals.items():
            print(f"  {alpha:<20}: {signal:>8.4f}")
        
        print("\nEnsemble Weights:")
        for alpha, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 20)
            print(f"  {alpha:<20}: {weight:>6.2%} {bar}")
        
        print("="*60)


def run_sample_ensemble():
    """Run sample ensemble with synthetic data."""
    # Create synthetic alpha signals
    np.random.seed(42)
    n_samples = 500
    
    alpha_names = ["ORB", "VWAP", "PCP", "VOL_CARRY"]
    
    alpha_signals = {}
    for alpha in alpha_names:
        alpha_signals[alpha] = np.random.normal(0, 1, n_samples)
    
    # Create synthetic returns
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Create synthetic market data
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")
    prices = 20000 * np.cumprod(1 + returns)
    
    market_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_samples)
    }, index=dates)
    
    # Initialize and fit ensemble
    ensemble = LightGBMEnsemble(alpha_names=alpha_names, use_xgboost=True)
    ensemble.fit(alpha_signals, returns, market_data)
    
    # Optimize weights
    alpha_returns = {alpha: np.random.normal(0.001, 0.02, 100) for alpha in alpha_names}
    weights = ensemble.optimize_weights_rolling_sharpe(alpha_returns)
    
    # Make prediction
    current_signals = {alpha: np.random.normal(0, 1) for alpha in alpha_names}
    result = ensemble.predict(current_signals, market_data)
    
    ensemble.print_ensemble_info(result)
    
    # Print feature importance
    print("\nFeature Importance (Top 10):")
    importance = ensemble.get_feature_importance()
    for feature, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feature:<30}: {imp:>8.4f}")
    
    return result


if __name__ == "__main__":
    run_sample_ensemble()

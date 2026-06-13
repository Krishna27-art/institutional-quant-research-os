"""
Meta-labeling for Signal Quality Assessment
Based on the critique: Determine which signals actually predict outcomes

Meta-labeling approach:
1. Primary model generates trading signals
2. Secondary model predicts if primary signal will be profitable
3. Only take trades where secondary model predicts success
4. Improves signal quality and reduces false positives

Reference: Lopez de Prado (2018) - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LabelType(Enum):
    """Types of meta-labels."""
    BINARY = "binary"  # Profitable or not
    TRIPLE_BARRIER = "triple_barrier"  # Hit upper, lower, or time barrier
    FIXED_HORIZON = "fixed_horizon"  # Return after fixed period


@dataclass
class MetaLabelResult:
    """Result of meta-labeling."""
    signal_timestamp: datetime
    primary_signal: float  # -1, 0, 1
    meta_label: int  # 0 or 1 (profitable or not)
    predicted_return: float
    actual_return: float
    meta_probability: float  # Probability of being profitable


class MetaLabelingEngine:
    """
    Meta-labeling engine for signal quality assessment.
    
    Process:
    1. Generate primary signals from strategy
    2. Label signals based on actual outcomes
    3. Train secondary model to predict signal success
    4. Use secondary model to filter signals in production
    """
    
    def __init__(self):
        self.meta_model = None
        self.feature_columns: List[str] = []
        self.is_trained = False
        
        # Meta-labeling parameters
        self.holding_period = 5  # Days to hold position
        self.profit_threshold = 0.01  # 1% return threshold for "profitable"
    
    def generate_meta_labels(
        self,
        signals: pd.Series,
        returns: pd.Series,
        label_type: LabelType = LabelType.BINARY
    ) -> pd.Series:
        """
        Generate meta-labels for signals.
        
        Args:
            signals: Primary trading signals (-1, 0, 1)
            returns: Future returns for each signal
            label_type: Type of labeling
            
        Returns:
            Series of meta-labels (0 or 1)
        """
        # Align signals and returns
        aligned = pd.DataFrame({
            'signal': signals,
            'return': returns
        }).dropna()
        
        # Generate labels based on return threshold
        if label_type == LabelType.BINARY:
            labels = (aligned['return'] > self.profit_threshold).astype(int)
        elif label_type == LabelType.FIXED_HORIZON:
            labels = (aligned['return'] > 0).astype(int)
        else:
            labels = (aligned['return'] > 0).astype(int)
        
        return labels
    
    def extract_meta_features(
        self,
        data: pd.DataFrame,
        signal_timestamp: datetime,
        lookback: int = 20
    ) -> Dict:
        """
        Extract features for meta-labeling model.
        
        Features include:
        - Recent volatility
        - Recent trend
        - Volume profile
        - Market regime indicators
        """
        # Get historical data up to signal timestamp
        historical = data.loc[:signal_timestamp].tail(lookback)
        
        if len(historical) < lookback:
            return {}
        
        features = {}
        
        # Volatility features
        returns = historical['close'].pct_change().dropna()
        features['volatility_5d'] = returns.tail(5).std() if len(returns) >= 5 else 0
        features['volatility_20d'] = returns.tail(20).std() if len(returns) >= 20 else 0
        
        # Trend features
        features['momentum_5d'] = historical['close'].iloc[-1] / historical['close'].iloc[-5] - 1 if len(historical) >= 5 else 0
        features['momentum_20d'] = historical['close'].iloc[-1] / historical['close'].iloc[-20] - 1 if len(historical) >= 20 else 0
        
        # Volume features
        features['volume_avg_5d'] = historical['volume'].tail(5).mean() if len(historical) >= 5 else 0
        features['volume_avg_20d'] = historical['volume'].tail(20).mean() if len(historical) >= 20 else 0
        features['volume_ratio'] = features['volume_avg_5d'] / features['volume_avg_20d'] if features['volume_avg_20d'] > 0 else 1
        
        # Price level features
        features['price_to_ma5'] = historical['close'].iloc[-1] / historical['close'].tail(5).mean() if len(historical) >= 5 else 1
        features['price_to_ma20'] = historical['close'].iloc[-1] / historical['close'].tail(20).mean() if len(historical) >= 20 else 1
        
        # Range features
        features['range_5d'] = (historical['high'].tail(5).max() - historical['low'].tail(5).min()) / historical['close'].iloc[-1] if len(historical) >= 5 else 0
        
        return features
    
    def train_meta_model(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        returns: pd.Series
    ) -> None:
        """
        Train meta-labeling model.
        
        Args:
            data: Historical OHLCV data
            signals: Primary trading signals
            returns: Future returns for each signal
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn not available. Cannot train meta model.")
            return
        
        # Generate meta-labels
        meta_labels = self.generate_meta_labels(signals, returns)
        
        # Extract features for each signal
        feature_rows = []
        label_rows = []
        
        for timestamp in signals[signals != 0].index:
            features = self.extract_meta_features(data, timestamp)
            if features:
                feature_rows.append(features)
                label_rows.append(meta_labels.get(timestamp, 0))
        
        if len(feature_rows) < 50:
            print("Insufficient data for training meta model")
            return
        
        # Create feature matrix
        X = pd.DataFrame(feature_rows)
        y = np.array(label_rows)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train Random Forest
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.meta_model.predict(X_test)
        y_prob = self.meta_model.predict_proba(X_test)[:, 1]
        
        print("Meta-labeling Model Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.meta_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance.head(10).to_string(index=False))
        
        self.is_trained = True
    
    def predict_signal_quality(
        self,
        data: pd.DataFrame,
        signal_timestamp: datetime,
        primary_signal: float
    ) -> Optional[MetaLabelResult]:
        """
        Predict if a signal will be profitable.
        
        Args:
            data: Historical OHLCV data
            signal_timestamp: Timestamp of signal
            primary_signal: Primary trading signal
            
        Returns:
            MetaLabelResult with prediction
        """
        if not self.is_trained or self.meta_model is None:
            return None
        
        # Extract features
        features = self.extract_meta_features(data, signal_timestamp)
        
        if not features or len(features) != len(self.feature_columns):
            return None
        
        # Create feature vector
        X = pd.DataFrame([features])[self.feature_columns]
        
        # Predict
        meta_probability = self.meta_model.predict_proba(X)[0, 1]
        meta_label = int(meta_probability > 0.5)
        
        return MetaLabelResult(
            signal_timestamp=signal_timestamp,
            primary_signal=primary_signal,
            meta_label=meta_label,
            predicted_return=0,  # Would need actual return prediction
            actual_return=0,  # Unknown at prediction time
            meta_probability=meta_probability
        )
    
    def filter_signals(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        min_probability: float = 0.6
    ) -> pd.Series:
        """
        Filter signals using meta-labeling model.
        
        Only keeps signals where meta-model predicts high probability of success.
        
        Args:
            data: Historical OHLCV data
            signals: Primary trading signals
            min_probability: Minimum probability threshold
            
        Returns:
            Filtered signals
        """
        if not self.is_trained:
            return signals
        
        filtered_signals = signals.copy()
        
        for timestamp in signals[signals != 0].index:
            result = self.predict_signal_quality(data, timestamp, signals[timestamp])
            
            if result and result.meta_probability < min_probability:
                filtered_signals[timestamp] = 0
        
        return filtered_signals
    
    def evaluate_filtering_performance(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        returns: pd.Series,
        min_probability: float = 0.6
    ) -> Dict:
        """
        Evaluate the performance of signal filtering.
        
        Compares performance of:
        - Original signals
        - Meta-filtered signals
        """
        # Original signal performance
        original_returns = (signals * returns).dropna()
        original_sharpe = original_returns.mean() / original_returns.std() * np.sqrt(252) if original_returns.std() > 0 else 0
        
        # Filtered signal performance
        filtered_signals = self.filter_signals(data, signals, min_probability)
        filtered_returns = (filtered_signals * returns).dropna()
        filtered_sharpe = filtered_returns.mean() / filtered_returns.std() * np.sqrt(252) if filtered_returns.std() > 0 else 0
        
        # Reduction in signal count
        original_count = len(signals[signals != 0])
        filtered_count = len(filtered_signals[filtered_signals != 0])
        reduction = (original_count - filtered_count) / original_count if original_count > 0 else 0
        
        return {
            'original_sharpe': original_sharpe,
            'filtered_sharpe': filtered_sharpe,
            'sharpe_improvement': filtered_sharpe - original_sharpe,
            'original_signals': original_count,
            'filtered_signals': filtered_count,
            'signal_reduction': reduction,
            'min_probability': min_probability
        }


if __name__ == "__main__":
    # Test the Meta-labeling Engine
    print("Testing Meta-labeling Engine...")
    
    engine = MetaLabelingEngine()
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = np.random.normal(100, 10, n).cumsum()
    prices = prices - prices.min() + 100
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
        'close': prices,
        'volume': np.random.normal(1000000, 200000, n)
    }, index=dates)
    
    # Generate sample signals
    signals = pd.Series(np.random.choice([-1, 0, 1], size=n, p=[0.1, 0.8, 0.1]), index=dates)
    
    # CRITICAL FIX: Use forward returns (shift(1)) instead of lookahead (shift(-1))
    # Generate future returns
    returns = data['close'].pct_change().shift(1)
    
    # Train meta model
    print("\nTraining meta-labeling model...")
    engine.train_meta_model(data, signals, returns)
    
    if engine.is_trained:
        # Evaluate filtering performance
        print("\nEvaluating filtering performance...")
        results = engine.evaluate_filtering_performance(data, signals, returns, min_probability=0.6)
        
        print(f"\nOriginal Sharpe: {results['original_sharpe']:.2f}")
        print(f"Filtered Sharpe: {results['filtered_sharpe']:.2f}")
        print(f"Sharpe Improvement: {results['sharpe_improvement']:.2f}")
        print(f"Signal Reduction: {results['signal_reduction']:.2%}")
        print(f"Original Signals: {results['original_signals']}")
        print(f"Filtered Signals: {results['filtered_signals']}")
        
        # Test prediction on a single signal
        print("\nTesting single signal prediction...")
        result = engine.predict_signal_quality(data, dates[300], 1)
        if result:
            print(f"Signal: {result.primary_signal}")
            print(f"Meta Label: {result.meta_label}")
            print(f"Meta Probability: {result.meta_probability:.2f}")
    else:
        print("\nMeta model not trained (sklearn not available or insufficient data)")

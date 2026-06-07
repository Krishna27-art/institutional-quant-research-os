"""
Meta-Learning Layer: Signal -> Meta Model -> Trade
Based on the critique: Build Meta-Learning Layer for when signals work

Current:
    Signal -> Trade

Advanced:
    Signal
    ↓
    Meta Model
    ↓
    Trade

Meta model learns:
    When ORB works
    When VWAP works
    When Carry works

This can improve Sharpe more than adding 50 indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


class SignalType(Enum):
    """Types of signals."""
    ORB = "orb"
    VWAP = "vwap"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CARRY = "carry"
    RESIDUAL_MOMENTUM = "residual_momentum"


@dataclass
class MetaFeature:
    """Meta-feature for signal prediction."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    market_regime: str
    volatility_regime: str
    time_of_day: str
    day_of_week: str
    spread: float
    volume_ratio: float
    recent_performance: float
    crowding_score: float


@dataclass
class MetaLabel:
    """Meta-label for signal performance."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    signal_value: float
    future_return: float
    is_profitable: bool
    profitability_score: float


@dataclass
class MetaPrediction:
    """Meta-model prediction."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    signal_value: float
    predicted_profitability: float
    confidence: float
    filtered_signal: float  # Signal after meta-model filtering


class MetaLearningEngine:
    """
    Meta-Learning Engine for signal filtering and enhancement.
    
    Features:
    - Meta-feature extraction
    - Meta-label generation
    - Meta-model training
    - Signal filtering
    - Performance prediction
    """
    
    def __init__(self):
        self.meta_features: Dict[str, List[MetaFeature]] = {}
        self.meta_labels: Dict[str, List[MetaLabel]] = {}
        self.meta_models: Dict[SignalType, object] = {}
        self.model_type = "random_forest"  # random_forest, gradient_boosting, logistic_regression
        
        # Initialize models for each signal type
        for signal_type in SignalType:
            if self.model_type == "random_forest":
                self.meta_models[signal_type] = RandomForestClassifier(n_estimators=100, max_depth=10)
            elif self.model_type == "gradient_boosting":
                self.meta_models[signal_type] = GradientBoostingClassifier(n_estimators=100, max_depth=5)
            else:
                self.meta_models[signal_type] = LogisticRegression()
    
    def extract_meta_features(
        self,
        timestamp: datetime,
        symbol: str,
        signal_type: SignalType,
        market_regime: str,
        volatility_regime: str,
        spread: float,
        volume_ratio: float,
        recent_performance: float,
        crowding_score: float
    ) -> MetaFeature:
        """
        Extract meta-features for signal prediction.
        
        Meta-features describe the market environment when the signal was generated.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            signal_type: Type of signal
            market_regime: Market regime (bull_trend, bear_trend, sideways)
            volatility_regime: Volatility regime (high_vol, low_vol)
            spread: Bid-ask spread
            volume_ratio: Volume ratio to average
            recent_performance: Recent signal performance
            crowding_score: Crowding score (0 to 1)
            
        Returns:
            MetaFeature
        """
        # Time-based features
        time_of_day = timestamp.strftime("%H:%M")
        day_of_week = timestamp.strftime("%A")
        
        meta_feature = MetaFeature(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            spread=spread,
            volume_ratio=volume_ratio,
            recent_performance=recent_performance,
            crowding_score=crowding_score
        )
        
        # Store in history
        key = f"{symbol}_{signal_type.value}"
        if key not in self.meta_features:
            self.meta_features[key] = []
        self.meta_features[key].append(meta_feature)
        
        return meta_feature
    
    def generate_meta_label(
        self,
        timestamp: datetime,
        symbol: str,
        signal_type: SignalType,
        signal_value: float,
        future_return: float
    ) -> MetaLabel:
        """
        Generate meta-label for signal performance.
        
        Meta-label indicates whether the signal was profitable.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            signal_type: Type of signal
            signal_value: Signal value
            future_return: Future return after signal
            
        Returns:
            MetaLabel
        """
        # Determine if profitable
        is_profitable = (signal_value * future_return) > 0
        
        # Calculate profitability score
        profitability_score = signal_value * future_return
        
        meta_label = MetaLabel(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            signal_value=signal_value,
            future_return=future_return,
            is_profitable=is_profitable,
            profitability_score=profitability_score
        )
        
        # Store in history
        key = f"{symbol}_{signal_type.value}"
        if key not in self.meta_labels:
            self.meta_labels[key] = []
        self.meta_labels[key].append(meta_label)
        
        return meta_label
    
    def prepare_training_data(
        self,
        signal_type: SignalType,
        min_samples: int = 100
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for meta-model.
        
        Args:
            signal_type: Type of signal
            min_samples: Minimum samples required
            
        Returns:
            X (features), y (labels)
        """
        # Get all meta-features and labels for this signal type
        features_list = []
        labels_list = []
        
        for key in self.meta_features:
            if signal_type.value in key:
                features_list.extend(self.meta_features[key])
        
        for key in self.meta_labels:
            if signal_type.value in key:
                labels_list.extend(self.meta_labels[key])
        
        if len(features_list) < min_samples or len(labels_list) < min_samples:
            return pd.DataFrame(), pd.Series()
        
        # Align features and labels by timestamp and symbol
        feature_df = pd.DataFrame([
            {
                'timestamp': f.timestamp,
                'symbol': f.symbol,
                'market_regime': f.market_regime,
                'volatility_regime': f.volatility_regime,
                'time_of_day': f.time_of_day,
                'day_of_week': f.day_of_week,
                'spread': f.spread,
                'volume_ratio': f.volume_ratio,
                'recent_performance': f.recent_performance,
                'crowding_score': f.crowding_score
            }
            for f in features_list
        ])
        
        label_df = pd.DataFrame([
            {
                'timestamp': l.timestamp,
                'symbol': l.symbol,
                'is_profitable': l.is_profitable
            }
            for l in labels_list
        ])
        
        # Merge on timestamp and symbol
        merged = pd.merge(feature_df, label_df, on=['timestamp', 'symbol'])
        
        if len(merged) < min_samples:
            return pd.DataFrame(), pd.Series()
        
        # Encode categorical variables
        merged = pd.get_dummies(merged, columns=['market_regime', 'volatility_regime', 'day_of_week'])
        
        # Drop non-feature columns
        X = merged.drop(['timestamp', 'symbol', 'is_profitable'], axis=1)
        y = merged['is_profitable']
        
        return X, y
    
    def train_meta_model(
        self,
        signal_type: SignalType,
        min_samples: int = 100
    ) -> Dict:
        """
        Train meta-model for a signal type.
        
        Args:
            signal_type: Type of signal
            min_samples: Minimum samples required
            
        Returns:
            Training metrics
        """
        X, y = self.prepare_training_data(signal_type, min_samples)
        
        if X.empty or y.empty:
            return {'status': 'insufficient_data', 'samples': 0}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = self.meta_models[signal_type]
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'status': 'success',
            'samples': len(X),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def predict_signal_profitability(
        self,
        timestamp: datetime,
        symbol: str,
        signal_type: SignalType,
        signal_value: float,
        market_regime: str,
        volatility_regime: str,
        spread: float,
        volume_ratio: float,
        recent_performance: float,
        crowding_score: float
    ) -> MetaPrediction:
        """
        Predict signal profitability using meta-model.
        
        Args:
            timestamp: Timestamp
            symbol: Trading symbol
            signal_type: Type of signal
            signal_value: Signal value
            market_regime: Market regime
            volatility_regime: Volatility regime
            spread: Bid-ask spread
            volume_ratio: Volume ratio
            volume_ratio: Volume ratio
            recent_performance: Recent performance
            crowding_score: Crowding score
            
        Returns:
            MetaPrediction
        """
        # Extract meta-features
        meta_feature = self.extract_meta_features(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            spread=spread,
            volume_ratio=volume_ratio,
            recent_performance=recent_performance,
            crowding_score=crowding_score
        )
        
        # Prepare feature vector
        feature_dict = {
            'spread': meta_feature.spread,
            'volume_ratio': meta_feature.volume_ratio,
            'recent_performance': meta_feature.recent_performance,
            'crowding_score': meta_feature.crowding_score
        }
        
        # Add encoded categorical variables
        for regime in ['bull_trend', 'bear_trend', 'sideways']:
            feature_dict[f'market_regime_{regime}'] = 1 if meta_feature.market_regime == regime else 0
        
        for vol_regime in ['high_vol', 'low_vol']:
            feature_dict[f'volatility_regime_{vol_regime}'] = 1 if meta_feature.volatility_regime == vol_regime else 0
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            feature_dict[f'day_of_week_{day}'] = 1 if meta_feature.day_of_week == day else 0
        
        # Get model
        model = self.meta_models[signal_type]
        
        # Predict
        try:
            X = pd.DataFrame([feature_dict])
            predicted_profitability = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else 0.5
            confidence = abs(predicted_profitability - 0.5) * 2  # 0 to 1
        except:
            predicted_profitability = 0.5
            confidence = 0.0
        
        # Filter signal based on prediction
        # If predicted profitability is low, reduce signal strength
        if predicted_profitability < 0.4:
            filtered_signal = signal_value * 0.3  # Reduce signal
        elif predicted_profitability > 0.6:
            filtered_signal = signal_value * 1.2  # Amplify signal
        else:
            filtered_signal = signal_value  # Keep signal
        
        # Clip to -1 to 1
        filtered_signal = max(-1, min(1, filtered_signal))
        
        prediction = MetaPrediction(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            signal_value=signal_value,
            predicted_profitability=predicted_profitability,
            confidence=confidence,
            filtered_signal=filtered_signal
        )
        
        return prediction
    
    def get_meta_model_summary(self) -> pd.DataFrame:
        """Get summary of meta-model performance."""
        data = []
        
        for signal_type in SignalType:
            metrics = self.train_meta_model(signal_type, min_samples=10)
            data.append({
                'Signal Type': signal_type.value,
                'Status': metrics['status'],
                'Samples': metrics.get('samples', 0),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0)
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the Meta-Learning Engine
    print("Testing Meta-Learning Layer: Signal -> Meta Model -> Trade...")
    
    engine = MetaLearningEngine()
    
    # Generate sample meta-features and labels
    print("\nGenerating sample data...")
    np.random.seed(42)
    n = 200
    
    for i in range(n):
        timestamp = datetime.now() - timedelta(days=i)
        symbol = "RELIANCE"
        signal_type = np.random.choice(list(SignalType))
        
        # Generate meta-features
        meta_feature = engine.extract_meta_features(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            market_regime=np.random.choice(['bull_trend', 'bear_trend', 'sideways']),
            volatility_regime=np.random.choice(['high_vol', 'low_vol']),
            spread=np.random.uniform(0.001, 0.01),
            volume_ratio=np.random.uniform(0.5, 3.0),
            recent_performance=np.random.uniform(-0.1, 0.1),
            crowding_score=np.random.uniform(0, 1)
        )
        
        # Generate meta-label
        signal_value = np.random.uniform(-1, 1)
        future_return = np.random.normal(0.001, 0.02)
        
        meta_label = engine.generate_meta_label(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=signal_type,
            signal_value=signal_value,
            future_return=future_return
        )
    
    print(f"Generated {n} meta-features and labels")
    
    # Train meta-models
    print("\nTraining Meta-Models...")
    for signal_type in SignalType:
        metrics = engine.train_meta_model(signal_type, min_samples=50)
        print(f"{signal_type.value}: {metrics['status']}")
        if metrics['status'] == 'success':
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall: {metrics['recall']:.2%}")
    
    # Predict signal profitability
    print("\nPredicting Signal Profitability...")
    prediction = engine.predict_signal_profitability(
        timestamp=datetime.now(),
        symbol="RELIANCE",
        signal_type=SignalType.ORB,
        signal_value=0.8,
        market_regime="bull_trend",
        volatility_regime="low_vol",
        spread=0.002,
        volume_ratio=1.5,
        recent_performance=0.05,
        crowding_score=0.3
    )
    
    print(f"Signal Type: {prediction.signal_type.value}")
    print(f"Original Signal: {prediction.signal_value:.2f}")
    print(f"Predicted Profitability: {prediction.predicted_profitability:.2%}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print(f"Filtered Signal: {prediction.filtered_signal:.2f}")
    
    # Get meta-model summary
    print("\nMeta-Model Summary:")
    summary = engine.get_meta_model_summary()
    print(summary.to_string(index=False))

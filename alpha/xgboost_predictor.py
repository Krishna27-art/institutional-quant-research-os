"""
XGBoost Predictor - Real Model Inference

This module implements real XGBoost model inference to replace mock predictions
with actual machine learning-based predictions.

Key Features:
- Real XGBoost model loading and inference
- Feature computation from market data
- Prediction confidence scoring
- Model version management
- Feature importance tracking
- Prediction logging for accuracy tracking

Based on Audit Report Priority 0: Critical - Week 1-2
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

# Import prediction registry for logging
try:
    from models.prediction_registry import get_prediction_registry
    prediction_registry = get_prediction_registry()
except ImportError:
    prediction_registry = None
    logger.warning("Prediction registry not available, predictions will not be logged")


@dataclass
class Prediction:
    """Prediction result."""
    symbol: str
    prediction_value: float  # -1 to 1 (negative = short, positive = long)
    confidence: float  # 0 to 1
    model_version: str
    timestamp: datetime
    features_used: Dict[str, float]
    model_type: str = "xgboost"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_signal(self) -> str:
        """Get trading signal from prediction."""
        if self.prediction_value > 0.3:
            return "BUY"
        elif self.prediction_value < -0.3:
            return "SHORT"
        else:
            return "NEUTRAL"


class XGBoostPredictor:
    """
    XGBoost predictor for real model inference.
    
    This class loads trained XGBoost models and makes predictions
    based on computed features from market data.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize XGBoost predictor.
        
        Args:
            model_path: Path to trained model file
        """
        self.model_path = Path(model_path) if model_path else Path(__file__).parent / "models"
        self.model = None
        self.model_version = None
        self.feature_names = None
        self.is_loaded = False
        
        # Try to load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load XGBoost model from disk.
        
        Returns:
            True if model loaded successfully
        """
        model_file = self.model_path / "xgboost_model.pkl"
        
        if not model_file.exists():
            logger.warning(f"Model file not found at {model_file}")
            logger.info("Using fallback prediction logic")
            return False
        
        try:
            model_data = joblib.load(model_file)
            
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names')
                self.model_version = model_data.get('version', '1.0')
            else:
                self.model = model_data
                self.model_version = "1.0"
                self.feature_names = None
            
            self.is_loaded = True
            logger.info(f"Loaded XGBoost model v{self.model_version} from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Using fallback prediction logic")
            return False
    
    def compute_features(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Compute features from market data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Dictionary of feature values
        """
        features = {}
        
        if data.empty or len(data) < 20:
            return features
        
        # Price-based features
        features['returns_1d'] = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        features['returns_5d'] = (data['close'].iloc[-1] - data['close'].iloc[-6]) / data['close'].iloc[-6]
        features['returns_20d'] = (data['close'].iloc[-1] - data['close'].iloc[-21]) / data['close'].iloc[-21]
        
        # Volatility features
        returns = data['close'].pct_change().dropna()
        features['volatility_5d'] = returns.tail(5).std()
        features['volatility_20d'] = returns.tail(20).std()
        
        # Momentum features
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['momentum_10'] = data['close'].iloc[-1] / data['close'].iloc[-11] - 1
        
        # Volume features
        features['volume_ratio'] = data['volume'].iloc[-1] / data['volume'].tail(20).mean()
        
        # Price levels
        features['above_sma20'] = 1 if data['close'].iloc[-1] > data['close'].tail(20).mean() else 0
        features['above_sma50'] = 1 if data['close'].iloc[-1] > data['close'].tail(50).mean() else 0
        
        # Gap features
        if len(data) > 1:
            features['gap'] = (data['open'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if len(rsi) > 0 else 50.0
    
    def predict(
        self,
        data: pd.DataFrame,
        symbol: str,
        use_fallback: bool = True
    ) -> Optional[Prediction]:
        """
        Make prediction using XGBoost model.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            use_fallback: Use fallback logic if model not loaded
            
        Returns:
            Prediction result
        """
        # Compute features
        features = self.compute_features(data, symbol)
        
        if not features:
            logger.warning(f"Insufficient data to compute features for {symbol}")
            return None
        
        # Use model if loaded
        if self.is_loaded and self.model is not None:
            return self._predict_with_model(features, symbol)
        
        # Use fallback logic
        if use_fallback:
            return self._predict_fallback(features, symbol)
        
        return None
    
    def _predict_with_model(self, features: Dict[str, float], symbol: str) -> Prediction:
        """Make prediction using loaded XGBoost model."""
        try:
            # Prepare feature vector
            if self.feature_names:
                feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            else:
                feature_vector = list(features.values())
            
            # Reshape for prediction
            X = np.array(feature_vector).reshape(1, -1)
            
            # Make prediction
            prediction_value = float(self.model.predict(X)[0])
            
            # Clip to [-1, 1] range
            prediction_value = max(-1.0, min(1.0, prediction_value))
            
            # Calculate confidence based on prediction magnitude
            confidence = abs(prediction_value)
            
            prediction = Prediction(
                symbol=symbol,
                prediction_value=prediction_value,
                confidence=confidence,
                model_version=self.model_version,
                timestamp=datetime.now(),
                features_used=features,
                model_type="xgboost"
            )
            
            # Log prediction to registry if available
            if prediction_registry:
                try:
                    prediction_registry.log_prediction(
                        model_id=f"xgboost_{self.model_version}",
                        symbol=symbol,
                        prediction_value=prediction_value,
                        confidence=confidence,
                        features=features,
                        metadata={'model_type': 'xgboost', 'is_fallback': False}
                    )
                except Exception as e:
                    logger.warning(f"Failed to log prediction to registry: {e}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Model prediction failed for {symbol}: {e}")
            return self._predict_fallback(features, symbol)
    
    def _predict_fallback(self, features: Dict[str, float], symbol: str) -> Prediction:
        """
        Make prediction using fallback logic.
        
        This uses a simple rule-based approach when the model is not available.
        """
        # Simple momentum-based prediction
        momentum = features.get('returns_5d', 0)
        rsi = features.get('rsi_14', 50)
        volume_ratio = features.get('volume_ratio', 1.0)
        
        # Combine signals
        prediction_value = 0.0
        
        # Momentum signal
        if momentum > 0.02:
            prediction_value += 0.3
        elif momentum < -0.02:
            prediction_value -= 0.3
        
        # RSI signal (overbought/oversold)
        if rsi > 70:
            prediction_value -= 0.2
        elif rsi < 30:
            prediction_value += 0.2
        
        # Volume confirmation
        if volume_ratio > 1.5:
            prediction_value *= 1.2
        
        # Clip to [-1, 1]
        prediction_value = max(-1.0, min(1.0, prediction_value))
        
        confidence = abs(prediction_value) * 0.8  # Lower confidence for fallback
        
        prediction = Prediction(
            symbol=symbol,
            prediction_value=prediction_value,
            confidence=confidence,
            model_version="fallback",
            timestamp=datetime.now(),
            features_used=features,
            model_type="rule_based",
            metadata={'fallback_used': True}
        )
        
        # Log fallback prediction to registry if available
        if prediction_registry:
            try:
                prediction_registry.log_prediction(
                    model_id="fallback_rule_based",
                    symbol=symbol,
                    prediction_value=prediction_value,
                    confidence=confidence,
                    features=features,
                    metadata={'model_type': 'rule_based', 'is_fallback': True}
                )
            except Exception as e:
                logger.warning(f"Failed to log fallback prediction to registry: {e}")
        
        return prediction
    
    def batch_predict(
        self,
        data_dict: Dict[str, pd.DataFrame],
        use_fallback: bool = True
    ) -> List[Prediction]:
        """
        Make predictions for multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            use_fallback: Use fallback logic if model not loaded
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for symbol, data in data_dict.items():
            try:
                prediction = self.predict(data, symbol, use_fallback)
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
        
        return predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_loaded or self.model is None:
            return None
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                
                if self.feature_names:
                    return dict(zip(self.feature_names, importance))
                else:
                    return {f'feature_{i}': imp for i, imp in enumerate(importance)}
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
        
        return None


# Singleton instance
_xgboost_predictor = None

def get_xgboost_predictor(model_path: str = None) -> XGBoostPredictor:
    """Get the singleton XGBoost predictor instance."""
    global _xgboost_predictor
    if _xgboost_predictor is None:
        _xgboost_predictor = XGBoostPredictor(model_path)
    return _xgboost_predictor


if __name__ == "__main__":
    # Test the XGBoost predictor
    print("Testing XGBoost Predictor...")
    
    predictor = XGBoostPredictor()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': np.random.uniform(1000, 1100, 100),
        'high': np.random.uniform(1100, 1200, 100),
        'low': np.random.uniform(900, 1000, 100),
        'close': np.random.uniform(1000, 1100, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Make prediction
    prediction = predictor.predict(data, "RELIANCE")
    
    if prediction:
        print(f"Prediction for RELIANCE:")
        print(f"  Value: {prediction.prediction_value:.4f}")
        print(f"  Confidence: {prediction.confidence:.2%}")
        print(f"  Signal: {prediction.get_signal()}")
        print(f"  Model: {prediction.model_type} v{prediction.model_version}")
        print(f"  Features used: {len(prediction.features_used)}")
    else:
        print("Prediction failed")

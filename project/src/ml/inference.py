"""
Inference Server - Real-time inference server
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .ensemble import MLEnsemble


class InferenceServer:
    """Real-time inference server for ML models"""
    
    def __init__(self):
        self.models: Dict[str, MLEnsemble] = {}  # regime -> model
        self.feature_store = None  # To be connected
    
    def load_model(self, regime: str, model: MLEnsemble) -> None:
        """Load a model for a specific regime"""
        self.models[regime] = model
    
    def predict(self, symbols: List[str], features: pd.DataFrame, 
                regime: str) -> Dict[str, Dict]:
        """
        Make predictions for multiple symbols
        
        Args:
            symbols: List of symbols
            features: DataFrame of features (indexed by symbol or time)
            regime: Current regime
            
        Returns:
            Dict mapping symbol to {signal, confidence}
        """
        if regime not in self.models:
            raise ValueError(f"No model loaded for regime: {regime}")
        
        model = self.models[regime]
        
        results = {}
        for symbol in symbols:
            if symbol in features.index:
                symbol_features = features.loc[[symbol]]
                
                # Make prediction
                pred, confidence = model.predict_with_confidence(symbol_features)
                
                results[symbol] = {
                    'signal': float(pred[0]),
                    'confidence': float(confidence[0])
                }
        
        return results
    
    def predict_batch(self, features: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Batch prediction for all symbols in features
        
        Args:
            features: DataFrame of features
            regime: Current regime
            
        Returns:
            DataFrame with signal and confidence columns
        """
        if regime not in self.models:
            raise ValueError(f"No model loaded for regime: {regime}")
        
        model = self.models[regime]
        
        signals, confidences = model.predict_with_confidence(features)
        
        results = pd.DataFrame({
            'signal': signals,
            'confidence': confidences
        }, index=features.index)
        
        return results
    
    def get_model_info(self, regime: str) -> Dict:
        """Get information about a model"""
        if regime not in self.models:
            return {}
        
        model = self.models[regime]
        
        return {
            'regime': regime,
            'feature_names': model.feature_names,
            'xgb_weight': model.xgb_weight,
            'lgb_weight': model.lgb_weight,
            'feature_importance': model.get_feature_importance()
        }

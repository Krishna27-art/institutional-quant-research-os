"""
Volatility Features - Computation logic for volatility-based features
"""

import pandas as pd
import numpy as np
from typing import Optional


class VolatilityFeatures:
    """Volatility feature computations"""
    
    def compute(self, feature_name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a volatility feature"""
        
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        if feature_name == "atr_14":
            return self._atr(data, window=14)
        elif feature_name == "realized_vol_10d":
            return self._realized_vol(data, window=10)
        elif feature_name == "realized_vol_21d":
            return self._realized_vol(data, window=21)
        elif feature_name == "realized_vol_63d":
            return self._realized_vol(data, window=63)
        elif feature_name == "parkinson_vol":
            return self._parkinson_vol(data, window=21)
        elif feature_name == "vol_of_vol":
            return self._vol_of_vol(data, window=21)
        else:
            raise ValueError(f"Unknown volatility feature: {feature_name}")
    
    def _atr(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute Average True Range"""
        if 'high' not in data.columns or 'low' not in data.columns:
            raise ValueError("Data must contain 'high' and 'low' columns for ATR")
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _realized_vol(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute realized volatility (annualized)"""
        returns = np.log(data['close'] / data['close'].shift(1))
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        return vol
    
    def _parkinson_vol(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute Parkinson volatility estimator"""
        if 'high' not in data.columns or 'low' not in data.columns:
            raise ValueError("Data must contain 'high' and 'low' columns for Parkinson vol")
        
        hl_ratio = np.log(data['high'] / data['low'])
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(window=window).mean())
        return parkinson * np.sqrt(252)
    
    def _vol_of_vol(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute volatility of volatility"""
        realized_vol = self._realized_vol(data, window=21)
        vov = realized_vol.rolling(window=window).std()
        return vov

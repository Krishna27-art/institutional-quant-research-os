"""
Volume Features - Computation logic for volume-based features
"""

import pandas as pd
import numpy as np
from typing import Optional


class VolumeFeatures:
    """Volume feature computations"""
    
    def compute(self, feature_name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a volume feature"""
        
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        if feature_name == "volume_ratio_20d":
            return self._volume_ratio(data, window=20)
        elif feature_name == "volume_zscore":
            return self._volume_zscore(data, window=20)
        elif feature_name == "obv":
            return self._obv(data)
        elif feature_name == "mfi_14":
            return self._mfi(data, window=14)
        else:
            raise ValueError(f"Unknown volume feature: {feature_name}")
    
    def _volume_ratio(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute volume ratio to moving average"""
        vol_ma = data['volume'].rolling(window=window).mean()
        return data['volume'] / vol_ma
    
    def _volume_zscore(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute volume Z-score"""
        vol_mean = data['volume'].rolling(window=window).mean()
        vol_std = data['volume'].rolling(window=window).std()
        return (data['volume'] - vol_mean) / vol_std
    
    def _obv(self, data: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume"""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for OBV")
        
        price_change = data['close'].diff()
        obv = (np.sign(price_change) * data['volume']).fillna(0).cumsum()
        return obv
    
    def _mfi(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute Money Flow Index (simplified version)"""
        if 'high' not in data.columns or 'low' not in data.columns:
            raise ValueError("Data must contain 'high' and 'low' columns for MFI")
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

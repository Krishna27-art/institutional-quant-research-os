"""
Price Features - Computation logic for price-based features
"""

import pandas as pd
import numpy as np
from typing import Optional


class PriceFeatures:
    """Price feature computations"""
    
    def compute(self, feature_name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a price feature"""
        
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        if feature_name == "returns_1d":
            return self._returns(data, periods=1)
        elif feature_name == "returns_5d":
            return self._returns(data, periods=5)
        elif feature_name == "returns_10d":
            return self._returns(data, periods=10)
        elif feature_name == "returns_21d":
            return self._returns(data, periods=21)
        elif feature_name == "returns_63d":
            return self._returns(data, periods=63)
        elif feature_name == "returns_252d":
            return self._returns(data, periods=252)
        elif feature_name == "distance_to_sma20":
            return self._distance_to_sma(data, window=20)
        elif feature_name == "distance_to_sma50":
            return self._distance_to_sma(data, window=50)
        elif feature_name == "distance_to_sma200":
            return self._distance_to_sma(data, window=200)
        elif feature_name == "pct_from_52w_high":
            return self._pct_from_52w_high(data)
        elif feature_name == "pct_from_52w_low":
            return self._pct_from_52w_low(data)
        elif feature_name == "gap_pct":
            return self._gap_pct(data)
        else:
            raise ValueError(f"Unknown price feature: {feature_name}")
    
    def _returns(self, data: pd.DataFrame, periods: int) -> pd.Series:
        """Compute log returns"""
        return np.log(data['close'] / data['close'].shift(periods))
    
    def _distance_to_sma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Compute distance to SMA as percentage"""
        sma = data['close'].rolling(window=window).mean()
        return (data['close'] - sma) / sma
    
    def _pct_from_52w_high(self, data: pd.DataFrame) -> pd.Series:
        """Compute percentage from 52-week high"""
        high_52w = data['close'].rolling(window=252).max()
        return (data['close'] - high_52w) / high_52w
    
    def _pct_from_52w_low(self, data: pd.DataFrame) -> pd.Series:
        """Compute percentage from 52-week low"""
        low_52w = data['close'].rolling(window=252).min()
        return (data['close'] - low_52w) / low_52w
    
    def _gap_pct(self, data: pd.DataFrame) -> pd.Series:
        """Compute gap percentage"""
        if 'open' not in data.columns:
            raise ValueError("Data must contain 'open' column for gap_pct")
        prev_close = data['close'].shift(1)
        return (data['open'] - prev_close) / prev_close

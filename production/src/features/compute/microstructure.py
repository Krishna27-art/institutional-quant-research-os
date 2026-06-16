"""
Breadth Features - Computation logic for market breadth features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class BreadthFeatures:
    """Market breadth feature computations"""
    
    def compute(self, feature_name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a breadth feature"""
        
        if feature_name == "advance_decline_ratio":
            return self._advance_decline_ratio(data)
        elif feature_name == "pct_above_ma50":
            return self._pct_above_ma(data, window=50)
        elif feature_name == "pct_above_ma200":
            return self._pct_above_ma(data, window=200)
        elif feature_name == "new_highs_minus_lows":
            return self._new_highs_minus_lows(data)
        else:
            raise ValueError(f"Unknown breadth feature: {feature_name}")
    
    def _advance_decline_ratio(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute Advance-Decline ratio
        Data should have columns for each stock with returns
        """
        if 'close' not in data.columns:
            # Assume multi-column data with stock returns
            returns = data.pct_change()
            advances = (returns > 0).sum(axis=1)
            declines = (returns < 0).sum(axis=1)
            return advances / declines.replace(0, np.nan)
        else:
            raise ValueError("For breadth features, data should be multi-column with stock data")
    
    def _pct_above_ma(self, data: pd.DataFrame, window: int) -> pd.Series:
        """
        Compute percentage of stocks above moving average
        Data should have columns for each stock with prices
        """
        if 'close' not in data.columns:
            ma = data.rolling(window=window).mean()
            above_ma = (data > ma).sum(axis=1)
            total = data.notna().sum(axis=1)
            return above_ma / total
        else:
            raise ValueError("For breadth features, data should be multi-column with stock data")
    
    def _new_highs_minus_lows(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute new highs minus new lows (52-week)
        Data should have columns for each stock with prices
        """
        if 'close' not in data.columns:
            high_52w = data.rolling(window=252).max()
            low_52w = data.rolling(window=252).min()
            
            new_highs = (data == high_52w).sum(axis=1)
            new_lows = (data == low_52w).sum(axis=1)
            
            return new_highs - new_lows
        else:
            raise ValueError("For breadth features, data should be multi-column with stock data")

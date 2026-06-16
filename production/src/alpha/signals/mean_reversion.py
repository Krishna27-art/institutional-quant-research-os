"""
Mean Reversion Alpha Strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseAlpha


class ORBAlpha(BaseAlpha):
    """Opening Range Breakout (5-minute)"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        default_params = {'lookback': 5}
        default_params.update(params)
        super().__init__("ORB_5min", default_params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute ORB signal"""
        lookback = self.parameters['lookback']
        
        # Compute opening range high and low
        data = data.copy()
        data['or_high'] = data['high'].rolling(lookback).max()
        data['or_low'] = data['low'].rolling(lookback).min()
        
        # Signal: long if break above OR, short if break below
        signal = np.where(
            data['close'] > data['or_high'].shift(1),
            1,
            np.where(
                data['close'] < data['or_low'].shift(1),
                -1,
                0
            )
        )
        
        return pd.Series(signal, index=data.index)
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on breakout strength"""
        lookback = self.parameters['lookback']
        
        data = data.copy()
        data['or_high'] = data['high'].rolling(lookback).max()
        data['or_low'] = data['low'].rolling(lookback).min()
        
        # Confidence based on distance from OR
        range_size = data['or_high'] - data['or_low']
        distance = np.where(
            data['close'] > data['or_high'].shift(1),
            (data['close'] - data['or_high'].shift(1)) / range_size,
            np.where(
                data['close'] < data['or_low'].shift(1),
                (data['or_low'].shift(1) - data['close']) / range_size,
                0
            )
        )
        
        confidence = np.clip(distance, 0, 1)
        return pd.Series(confidence, index=data.index)


class VWAPReversionAlpha(BaseAlpha):
    """VWAP Reversion"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        super().__init__("VWAP_Reversion", params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute VWAP reversion signal"""
        # Compute VWAP (simplified)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        
        # Signal: short if above VWAP, long if below
        signal = -np.sign(data['close'] - vwap)
        
        return self.normalize_signal(signal)
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on distance from VWAP"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        
        distance = np.abs(data['close'] - vwap) / vwap
        confidence = np.clip(distance, 0, 1)
        
        return confidence


class IBSAlpha(BaseAlpha):
    """Internal Bar Strength"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        super().__init__("IBS", params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute IBS signal"""
        # IBS = (close - low) / (high - low)
        ibs = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Signal: buy when IBS low (close near low), sell when IBS high
        signal = (0.5 - ibs) * 2  # Normalize to [-1, 1]
        
        return signal
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on IBS extremeness"""
        ibs = (data['close'] - data['low']) / (data['high'] - data['low'])
        confidence = np.abs(ibs - 0.5) * 2
        
        return confidence

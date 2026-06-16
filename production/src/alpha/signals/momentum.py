"""
Momentum Alpha Strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseAlpha


class TSMOMAlpha(BaseAlpha):
    """Time Series Momentum (12-1)"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        default_params = {'lookback': 12, 'skip': 1}
        default_params.update(params)
        super().__init__("TSMOM_12_1", default_params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute TSMOM signal"""
        lookback = self.parameters['lookback']
        skip = self.parameters['skip']
        
        # Compute returns over lookback period
        returns = data['close'].pct_change(lookback + skip)
        
        # Signal is sign of returns
        signal = np.sign(returns)
        
        return self.normalize_signal(signal)
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on return magnitude"""
        lookback = self.parameters['lookback']
        skip = self.parameters['skip']
        
        returns = data['close'].pct_change(lookback + skip)
        confidence = np.abs(returns) / returns.rolling(63).std()
        confidence = confidence.clip(0, 1)
        
        return confidence


class DualMomentumAlpha(BaseAlpha):
    """Dual Momentum (absolute + relative)"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        default_params = {'sma_period': 200}
        default_params.update(params)
        super().__init__("Dual_Momentum", default_params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute dual momentum signal"""
        sma_period = self.parameters['sma_period']
        
        # Absolute momentum: price above SMA
        sma = data['close'].rolling(sma_period).mean()
        absolute_momentum = (data['close'] > sma).astype(float)
        
        # Relative momentum: recent returns
        relative_momentum = data['close'].pct_change(21)
        
        # Combine
        signal = absolute_momentum * np.sign(relative_momentum)
        
        return self.normalize_signal(signal)
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on distance from SMA"""
        sma_period = self.parameters['sma_period']
        
        sma = data['close'].rolling(sma_period).mean()
        distance = (data['close'] - sma) / sma
        confidence = np.abs(distance) / distance.rolling(63).std()
        confidence = confidence.clip(0, 1)
        
        return confidence


class SectorMomentumAlpha(BaseAlpha):
    """Sector Momentum Strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        super().__init__("Sector_Momentum", params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """Compute sector momentum signal"""
        # Cross-sectional momentum
        returns = data['close'].pct_change(21)
        signal = returns.rank(pct=True) * 2 - 1  # Normalize to [-1, 1]
        
        return signal
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on cross-sectional rank"""
        returns = data['close'].pct_change(21)
        rank = returns.rank(pct=True)
        confidence = np.abs(rank - 0.5) * 2  # Higher confidence for extreme ranks
        
        return confidence

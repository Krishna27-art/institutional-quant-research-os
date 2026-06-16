"""
Base Alpha Class
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class BaseAlpha(ABC):
    """Base class for all alpha strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
    
    @abstractmethod
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute trading signal
        
        Args:
            data: OHLCV data
            
        Returns:
            Series of signals (normalized to [-1, 1])
        """
        pass
    
    @abstractmethod
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute confidence in signal
        
        Args:
            data: OHLCV data
            
        Returns:
            Series of confidence scores [0, 1]
        """
        pass
    
    def normalize_signal(self, signal: pd.Series) -> pd.Series:
        """Normalize signal to [-1, 1] range"""
        if signal.std() == 0:
            return signal * 0
        return (signal - signal.mean()) / signal.std()
    
    def clip_signal(self, signal: pd.Series, min_val: float = -1, max_val: float = 1) -> pd.Series:
        """Clip signal to range"""
        return signal.clip(min_val, max_val)

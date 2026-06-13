"""
Options and Volatility Carry/Hedging Alpha Strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..base import BaseAlpha


class TailHedgingAlpha(BaseAlpha):
    """Tail Hedging with OTM puts"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        default_params = {'delta': 0.1, 'notional_pct': 0.01}
        default_params.update(params)
        super().__init__("Tail_Hedging", default_params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute tail hedging signal
        Returns position size in OTM puts (negative = long puts)
        """
        # This is a defensive strategy, signal is based on VIX or market stress
        # For now, return constant notional
        notional_pct = self.parameters['notional_pct']
        signal = pd.Series(-notional_pct, index=data.index)
        
        return signal
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on market stress indicators"""
        # Higher confidence during high volatility
        returns = data['close'].pct_change()
        vol = returns.rolling(21).std() * np.sqrt(252)
        
        # Normalize confidence based on vol percentile
        confidence = (vol / vol.rolling(252).max()).clip(0, 1)
        
        return confidence


class VolatilityTargetingAlpha(BaseAlpha):
    """Volatility Targeting"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        params = parameters or {}
        default_params = {'target_vol': 0.15}
        default_params.update(params)
        super().__init__("Volatility_Targeting", default_params)
    
    def compute_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute volatility targeting signal
        This is more of a position sizing overlay than a pure signal
        """
        # Compute realized volatility
        returns = data['close'].pct_change()
        realized_vol = returns.rolling(21).std() * np.sqrt(252)
        
        # Scale factor to achieve target volatility
        target_vol = self.parameters['target_vol']
        scale_factor = target_vol / realized_vol
        scale_factor = scale_factor.clip(0.5, 2.0)
        
        # Return scale factor as signal (to be applied to base position)
        return scale_factor
    
    def compute_confidence(self, data: pd.DataFrame) -> pd.Series:
        """Confidence based on volatility stability"""
        returns = data['close'].pct_change()
        vol = returns.rolling(21).std() * np.sqrt(252)
        vol_of_vol = vol.rolling(21).std()
        
        # Higher confidence when volatility is stable
        confidence = 1 - (vol_of_vol / vol).clip(0, 1)
        
        return confidence

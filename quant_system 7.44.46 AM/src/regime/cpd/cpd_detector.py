"""
Change Point Detector - Detect structural breaks in market regime
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from typing import List, Optional, Tuple


class CPDDetector:
    """Change Point Detection using ruptures library"""
    
    def __init__(self, model: str = "l2", min_size: int = 10):
        self.model = model
        self.min_size = min_size
        self.algorithm = rpt.Binseg(model=model, min_size=min_size)
    
    def detect(self, series: pd.Series, n_bkps: int = 5) -> List[int]:
        """
        Detect change points in a time series
        
        Args:
            series: Time series data
            n_bkps: Number of change points to detect
            
        Returns:
            List of change point indices
        """
        values = series.dropna().values
        
        if len(values) < self.min_size * 2:
            return []
        
        # Fit and predict change points
        self.algorithm.fit(values)
        bkps = self.algorithm.predict(n_bkps=n_bkps)
        
        # Remove the last point (always the end of series)
        return bkps[:-1] if bkps else []
    
    def detect_with_penalty(self, series: pd.Series, pen: float = 10.0) -> List[int]:
        """
        Detect change points with penalty (automatically determines number)
        
        Args:
            series: Time series data
            pen: Penalty parameter (higher = fewer change points)
            
        Returns:
            List of change point indices
        """
        values = series.dropna().values
        
        if len(values) < self.min_size * 2:
            return []
        
        self.algorithm.fit(values)
        bkps = self.algorithm.predict(pen=pen)
        
        return bkps[:-1] if bkps else []
    
    def get_change_point_magnitude(self, series: pd.Series, change_point: int, 
                                   window: int = 10) -> float:
        """
        Get magnitude of change at a change point
        
        Args:
            series: Time series data
            change_point: Index of change point
            window: Window size for comparison
            
        Returns:
            Magnitude of change
        """
        if change_point < window or change_point >= len(series) - window:
            return 0.0
        
        pre_mean = series.iloc[change_point-window:change_point].mean()
        post_mean = series.iloc[change_point:change_point+window].mean()
        
        return abs(post_mean - pre_mean)
    
    def detect_volatility_regime_change(self, returns: pd.Series) -> Tuple[List[int], str]:
        """
        Detect volatility regime changes
        
        Args:
            returns: Return series
            
        Returns:
            (change_points, regime_label)
        """
        # Compute rolling volatility
        vol = returns.rolling(21).std() * np.sqrt(252)
        
        # Detect change points
        change_points = self.detect(vol.dropna(), n_bkps=3)
        
        # Determine current regime
        if len(vol) > 21:
            current_vol = vol.iloc[-1]
            vol_percentile = (vol < current_vol).sum() / len(vol)
            
            if vol_percentile > 0.8:
                regime = "high_vol"
            elif vol_percentile < 0.2:
                regime = "low_vol"
            else:
                regime = "normal_vol"
        else:
            regime = "unknown"
        
        return change_points, regime

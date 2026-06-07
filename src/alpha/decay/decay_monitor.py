"""
Alpha Decay Monitor - Detect and monitor alpha performance decay
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from enum import Enum


class DecayStatus(Enum):
    HEALTHY = "healthy"
    DECAY_WARNING = "decay_warning"
    DECAY_CRITICAL = "decay_critical"
    STABLE = "stable"


class AlphaDecayMonitor:
    """Monitor alpha performance for decay"""
    
    def __init__(self, warning_threshold_pct: float = 25, 
                 critical_threshold_pct: float = 10,
                 window_size: int = 21):
        self.warning_threshold = warning_threshold_pct  # Percentile
        self.critical_threshold = critical_threshold_pct  # Percentile
        self.window_size = window_size
        self.historical_sharpe: Dict[str, List[float]] = {}
    
    def check_decay(self, alpha_id: str, recent_sharpe: float) -> DecayStatus:
        """
        Check if an alpha is decaying based on Sharpe ratio
        
        Args:
            alpha_id: Alpha identifier
            recent_sharpe: Recent Sharpe ratio (e.g., 21-day rolling)
            
        Returns:
            Decay status
        """
        if alpha_id not in self.historical_sharpe or len(self.historical_sharpe[alpha_id]) < 30:
            return DecayStatus.STABLE
        
        historical = np.array(self.historical_sharpe[alpha_id])
        percentile = stats.percentileofscore(historical, recent_sharpe)
        
        if percentile < self.critical_threshold:
            return DecayStatus.DECAY_CRITICAL
        elif percentile < self.warning_threshold:
            return DecayStatus.DECAY_WARNING
        else:
            return DecayStatus.HEALTHY
    
    def update_historical(self, alpha_id: str, sharpe: float) -> None:
        """Update historical Sharpe for an alpha"""
        if alpha_id not in self.historical_sharpe:
            self.historical_sharpe[alpha_id] = []
        
        self.historical_sharpe[alpha_id].append(sharpe)
        
        # Keep only last N observations
        if len(self.historical_sharpe[alpha_id]) > 252:  # 1 year of daily data
            self.historical_sharpe[alpha_id] = self.historical_sharpe[alpha_id][-252:]
    
    def check_feature_drift(self, feature_name: str, current_values: np.ndarray, 
                           reference_distribution: np.ndarray) -> Tuple[bool, float]:
        """
        Check for feature drift using Kolmogorov-Smirnov test
        
        Args:
            feature_name: Feature identifier
            current_values: Current feature values
            reference_distribution: Reference (training) distribution
            
        Returns:
            (is_drifting, p_value)
        """
        ks_stat, p_value = stats.ks_2samp(current_values, reference_distribution)
        is_drifting = p_value < 0.05
        return is_drifting, p_value
    
    def compute_decay_rate(self, returns: pd.Series) -> float:
        """
        Compute exponential decay rate of alpha returns
        
        Args:
            returns: Time series of returns
            
        Returns:
            Decay rate (negative if decaying)
        """
        if len(returns) < self.window_size:
            return 0.0
        
        # Compute rolling Sharpe
        rolling_sharpe = returns.rolling(self.window_size).mean() / returns.rolling(self.window_size).std()
        
        # Fit exponential decay
        x = np.arange(len(rolling_sharpe))
        y = rolling_sharpe.dropna().values
        
        if len(y) < 10:
            return 0.0
        
        # Simple linear fit on log scale
        try:
            coeffs = np.polyfit(x[-len(y):], np.log(np.abs(y) + 1e-6), 1)
            decay_rate = coeffs[0]
            return decay_rate
        except:
            return 0.0
    
    def get_decay_report(self, alpha_id: str) -> Dict:
        """Generate decay report for an alpha"""
        if alpha_id not in self.historical_sharpe or not self.historical_sharpe[alpha_id]:
            return {
                'alpha_id': alpha_id,
                'status': 'insufficient_data',
                'historical_sharpe_mean': None,
                'historical_sharpe_std': None,
                'recent_sharpe': None
            }
        
        historical = np.array(self.historical_sharpe[alpha_id])
        recent_sharpe = historical[-1] if len(historical) > 0 else None
        
        status = self.check_decay(alpha_id, recent_sharpe) if recent_sharpe else DecayStatus.STABLE
        
        return {
            'alpha_id': alpha_id,
            'status': status.value,
            'historical_sharpe_mean': np.mean(historical),
            'historical_sharpe_std': np.std(historical),
            'recent_sharpe': recent_sharpe,
            'percentile': stats.percentileofscore(historical, recent_sharpe) if recent_sharpe else None
        }

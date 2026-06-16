"""
Drift Detector - Feature drift and alpha decay detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from enum import Enum


class DriftType(Enum):
    FEATURE_DRIFT = "feature_drift"
    ALPHA_DECAY = "alpha_decay"
    TARGET_DRIFT = "target_drift"


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftDetector:
    """Detect feature drift and alpha decay"""
    
    def __init__(self, feature_drift_threshold: float = 0.05,
                 alpha_decay_warning_pct: float = 25,
                 alpha_decay_critical_pct: float = 10):
        self.feature_drift_threshold = feature_drift_threshold
        self.alpha_decay_warning_pct = alpha_decay_warning_pct
        self.alpha_decay_critical_pct = alpha_decay_critical_pct
        
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.alpha_sharpe_history: Dict[str, List[float]] = {}
    
    def set_reference_distribution(self, feature_name: str, 
                                   reference_data: np.ndarray) -> None:
        """Set reference distribution for a feature"""
        self.reference_distributions[feature_name] = reference_data
    
    def detect_feature_drift(self, feature_name: str, 
                            current_values: np.ndarray) -> Tuple[DriftSeverity, float]:
        """
        Detect feature drift using Kolmogorov-Smirnov test
        
        Args:
            feature_name: Feature identifier
            current_values: Current feature values
            
        Returns:
            (severity, p_value)
        """
        if feature_name not in self.reference_distributions:
            return DriftSeverity.NONE, 1.0
        
        reference = self.reference_distributions[feature_name]
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(current_values, reference)
        
        # Determine severity
        if p_value < 0.01:
            severity = DriftSeverity.CRITICAL
        elif p_value < 0.05:
            severity = DriftSeverity.HIGH
        elif p_value < 0.1:
            severity = DriftSeverity.MEDIUM
        elif p_value < self.feature_drift_threshold:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE
        
        return severity, p_value
    
    def detect_alpha_decay(self, alpha_id: str, 
                          recent_sharpe: float) -> Tuple[DriftSeverity, float]:
        """
        Detect alpha decay based on Sharpe ratio percentile
        
        Args:
            alpha_id: Alpha identifier
            recent_sharpe: Recent Sharpe ratio
            
        Returns:
            (severity, percentile)
        """
        if alpha_id not in self.alpha_sharpe_history:
            self.alpha_sharpe_history[alpha_id] = []
        
        history = np.array(self.alpha_sharpe_history[alpha_id])
        
        if len(history) < 30:
            return DriftSeverity.NONE, 50.0
        
        # Compute percentile
        percentile = stats.percentileofscore(history, recent_sharpe)
        
        # Determine severity
        if percentile < self.alpha_decay_critical_pct:
            severity = DriftSeverity.CRITICAL
        elif percentile < self.alpha_decay_warning_pct:
            severity = DriftSeverity.HIGH
        elif percentile < 40:
            severity = DriftSeverity.MEDIUM
        elif percentile < 50:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE
        
        return severity, percentile
    
    def update_alpha_sharpe(self, alpha_id: str, sharpe: float) -> None:
        """Update Sharpe history for an alpha"""
        if alpha_id not in self.alpha_sharpe_history:
            self.alpha_sharpe_history[alpha_id] = []
        
        self.alpha_sharpe_history[alpha_id].append(sharpe)
        
        # Keep last 252 observations (1 year of daily data)
        if len(self.alpha_sharpe_history[alpha_id]) > 252:
            self.alpha_sharpe_history[alpha_id] = self.alpha_sharpe_history[alpha_id][-252:]
    
    def compute_psi(self, reference: np.ndarray, current: np.ndarray, 
                   bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI)
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins
            
        Returns:
            PSI value
        """
        # Create bins based on reference distribution
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Compute distributions
        ref_dist, _ = np.histogram(reference, bins=bin_edges)
        curr_dist, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to percentages
        ref_dist = ref_dist / len(reference)
        curr_dist = curr_dist / len(current)
        
        # Add small epsilon to avoid division by zero
        epsilon = 0.0001
        ref_dist = ref_dist + epsilon
        curr_dist = curr_dist + epsilon
        
        # Compute PSI
        psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))
        
        return psi
    
    def detect_target_drift(self, reference_targets: np.ndarray,
                           current_targets: np.ndarray) -> Tuple[DriftSeverity, float]:
        """
        Detect target distribution drift
        
        Args:
            reference_targets: Reference target values
            current_targets: Current target values
            
        Returns:
            (severity, psi)
        """
        psi = self.compute_psi(reference_targets, current_targets)
        
        # PSI thresholds
        if psi > 0.25:
            severity = DriftSeverity.CRITICAL
        elif psi > 0.2:
            severity = DriftSeverity.HIGH
        elif psi > 0.1:
            severity = DriftSeverity.MEDIUM
        elif psi > 0.05:
            severity = DriftSeverity.LOW
        else:
            severity = DriftSeverity.NONE
        
        return severity, psi
    
    def get_drift_report(self) -> Dict:
        """Generate comprehensive drift report"""
        report = {
            'feature_drift': {},
            'alpha_decay': {},
            'summary': {
                'total_features_monitored': len(self.reference_distributions),
                'total_alphas_monitored': len(self.alpha_sharpe_history)
            }
        }
        
        return report


def exponentially_weighted_sharpe(returns: np.ndarray | pd.Series, half_life: int = 21) -> float:
    """Exponentially weighted annualized Sharpe ratio."""
    values = pd.Series(returns, dtype=float).dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    decay = np.exp(-np.log(2) / max(half_life, 1))
    weights = decay ** np.arange(len(values))[::-1]
    weights = weights / weights.sum()
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    if var <= 0 or not np.isfinite(var):
        return 0.0
    return float(mean / np.sqrt(var) * np.sqrt(252))


def detect_feature_drift_status(
    reference_batch: np.ndarray | pd.Series,
    live_batch: np.ndarray | pd.Series,
    threshold: float = 0.05,
) -> tuple[str, float]:
    """KS-test feature drift helper returning a stable string status."""
    reference = pd.Series(reference_batch, dtype=float).dropna()
    live = pd.Series(live_batch, dtype=float).dropna()
    if len(reference) == 0 or len(live) == 0:
        return "INSUFFICIENT_DATA", 1.0
    ks_stat, p_value = stats.ks_2samp(reference, live)
    return ("DRIFT_DETECTED" if p_value < threshold else "STABLE", float(ks_stat))

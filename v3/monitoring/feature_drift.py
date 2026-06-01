"""
Feature Drift Monitor
Daily monitoring of feature distribution changes with PSI, KL divergence, and automatic retraining triggers.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


class DriftLevel(Enum):
    """Drift severity levels"""
    STABLE = "STABLE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


@dataclass
class PSIResult:
    """Population Stability Index result for a single feature"""
    feature_name: str
    psi_value: float
    drift_level: DriftLevel
    reference_bins: np.ndarray
    test_bins: np.ndarray
    reference_distribution: np.ndarray
    test_distribution: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "psi_value": self.psi_value,
            "drift_level": self.drift_level.value,
            "reference_bins": self.reference_bins.tolist(),
            "test_bins": self.test_bins.tolist(),
            "reference_distribution": self.reference_distribution.tolist(),
            "test_distribution": self.test_distribution.tolist(),
        }


@dataclass
class RetrainingTrigger:
    """Trigger for model retraining"""
    feature_name: str
    reason: str
    psi_value: float
    triggered_at: datetime
    urgency: str  # "immediate", "within_24h", "increase_frequency"
    
    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "reason": self.reason,
            "psi_value": self.psi_value,
            "triggered_at": self.triggered_at.isoformat(),
            "urgency": self.urgency,
        }


@dataclass
class DriftMetrics:
    """Overall drift metrics for a model"""
    model_id: str
    date: date
    feature_psi_results: Dict[str, PSIResult] = field(default_factory=dict)
    feature_kl_divergence: Dict[str, float] = field(default_factory=dict)
    feature_importance_change: Dict[str, float] = field(default_factory=dict)
    average_psi: float = 0.0
    max_psi: float = 0.0
    high_drift_features: List[str] = field(default_factory=list)
    moderate_drift_features: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "date": self.date.isoformat(),
            "feature_psi_results": {k: v.to_dict() for k, v in self.feature_psi_results.items()},
            "feature_kl_divergence": self.feature_kl_divergence,
            "feature_importance_change": self.feature_importance_change,
            "average_psi": self.average_psi,
            "max_psi": self.max_psi,
            "high_drift_features": self.high_drift_features,
            "moderate_drift_features": self.moderate_drift_features,
        }


class FeatureDriftMonitor:
    """
    Monitors feature distribution changes between reference (training) and test (production) periods.
    Triggers retraining when drift exceeds thresholds.
    """
    
    def __init__(
        self,
        psi_stable_threshold: float = 0.1,
        psi_moderate_threshold: float = 0.2,
        top_n_features: int = 10
    ):
        self.psi_stable_threshold = psi_stable_threshold
        self.psi_moderate_threshold = psi_moderate_threshold
        self.top_n_features = top_n_features
        
        self.drift_history: Dict[str, List[DriftMetrics]] = {}
        self.retraining_triggers: List[RetrainingTrigger] = []
        
        # Store reference distributions
        self.reference_distributions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    
    def set_reference_distribution(
        self,
        model_id: str,
        feature_name: str,
        reference_data: np.ndarray,
        n_bins: int = 10
    ) -> None:
        """
        Set reference distribution for a feature.
        
        Args:
            model_id: Model identifier
            feature_name: Feature name
            reference_data: Reference data array
            n_bins: Number of bins for PSI calculation
        """
        # Create bins and calculate distribution
        bins = np.linspace(np.min(reference_data), np.max(reference_data), n_bins + 1)
        distribution, _ = np.histogram(reference_data, bins=bins, density=True)
        
        key = f"{model_id}:{feature_name}"
        self.reference_distributions[key] = (bins, distribution)
    
    def calculate_psi(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        n_bins: int = 10
    ) -> PSIResult:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = sum((%test - %ref) * ln(%test / %ref))
        
        Args:
            reference_data: Reference period data
            test_data: Test period data
            n_bins: Number of bins
            feature_name: Feature name for result
        
        Returns:
            PSIResult with bins, distributions, and drift level
        """
        # Create bins from reference data
        min_val = min(np.min(reference_data), np.min(test_data))
        max_val = max(np.max(reference_data), np.max(test_data))
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(reference_data, bins=bins, density=True)
        test_dist, _ = np.histogram(test_data, bins=bins, density=True)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_dist = ref_dist + epsilon
        test_dist = test_dist + epsilon
        
        # Normalize to percentages
        ref_pct = ref_dist / np.sum(ref_dist)
        test_pct = test_dist / np.sum(test_dist)
        
        # Calculate PSI
        psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        
        # Determine drift level
        if psi < self.psi_stable_threshold:
            drift_level = DriftLevel.STABLE
        elif psi < self.psi_moderate_threshold:
            drift_level = DriftLevel.MODERATE
        else:
            drift_level = DriftLevel.HIGH
        
        return PSIResult(
            feature_name="",  # Set by caller
            psi_value=psi,
            drift_level=drift_level,
            reference_bins=bins,
            test_bins=bins,
            reference_distribution=ref_pct,
            test_distribution=test_pct
        )
    
    def calculate_kl_divergence(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate KL Divergence for categorical or binned continuous features.
        
        Args:
            reference_data: Reference period data
            test_data: Test period data
            n_bins: Number of bins
        
        Returns:
            KL divergence value
        """
        # Create bins
        min_val = min(np.min(reference_data), np.min(test_data))
        max_val = max(np.max(reference_data), np.max(test_data))
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(reference_data, bins=bins, density=True)
        test_dist, _ = np.histogram(test_data, bins=bins, density=True)
        
        # Add small epsilon
        epsilon = 1e-10
        ref_dist = ref_dist + epsilon
        test_dist = test_dist + epsilon
        
        # Normalize
        ref_pct = ref_dist / np.sum(ref_dist)
        test_pct = test_dist / np.sum(test_dist)
        
        # Calculate KL divergence
        kl_div = np.sum(ref_pct * np.log(ref_pct / test_pct))
        
        return kl_div
    
    def evaluate_drift(
        self,
        model_id: str,
        feature_data: Dict[str, np.ndarray],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> DriftMetrics:
        """
        Evaluate drift for all features of a model.
        
        Args:
            model_id: Model identifier
            feature_data: Dictionary of feature_name -> test_data
            feature_importance: Optional feature importance ranking
        
        Returns:
            DriftMetrics with results for all features
        """
        drift_metrics = DriftMetrics(
            model_id=model_id,
            date=date.today()
        )
        
        psi_values = []
        
        for feature_name, test_data in feature_data.items():
            key = f"{model_id}:{feature_name}"
            
            # Get reference distribution
            if key not in self.reference_distributions:
                continue
            
            ref_bins, ref_dist = self.reference_distributions[key]
            
            # Calculate PSI
            psi_result = self.calculate_psi(
                reference_data=np.random.choice(
                    np.linspace(ref_bins[0], ref_bins[-1], 1000),
                    size=1000,
                    p=ref_dist / np.sum(ref_dist)
                ),
                test_data=test_data,
                n_bins=len(ref_bins) - 1
            )
            psi_result.feature_name = feature_name
            drift_metrics.feature_psi_results[feature_name] = psi_result
            psi_values.append(psi_result.psi_value)
            
            # Track drift levels
            if psi_result.drift_level == DriftLevel.HIGH:
                drift_metrics.high_drift_features.append(feature_name)
            elif psi_result.drift_level == DriftLevel.MODERATE:
                drift_metrics.moderate_drift_features.append(feature_name)
        
        # Calculate aggregate metrics
        if psi_values:
            drift_metrics.average_psi = np.mean(psi_values)
            drift_metrics.max_psi = np.max(psi_values)
        
        # Store in history
        if model_id not in self.drift_history:
            self.drift_history[model_id] = []
        self.drift_history[model_id].append(drift_metrics)
        
        # Keep only last 90 days
        if len(self.drift_history[model_id]) > 90:
            self.drift_history[model_id] = self.drift_history[model_id][-90:]
        
        return drift_metrics
    
    def check_retraining_triggers(
        self,
        drift_metrics: DriftMetrics,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> List[RetrainingTrigger]:
        """
        Check if drift metrics trigger retraining.
        
        Args:
            drift_metrics: Drift metrics from evaluation
            feature_importance: Optional feature importance ranking
        
        Returns:
            List of retraining triggers
        """
        triggers = []
        
        # Get top N features by importance if provided
        top_features = []
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_features = [f[0] for f in sorted_features[:self.top_n_features]]
        
        # Check for high drift in top features
        for feature_name in drift_metrics.high_drift_features:
            if feature_name in top_features or len(top_features) == 0:
                psi_value = drift_metrics.feature_psi_results[feature_name].psi_value
                trigger = RetrainingTrigger(
                    feature_name=feature_name,
                    reason=f"Top feature has high drift (PSI={psi_value:.3f})",
                    psi_value=psi_value,
                    triggered_at=datetime.now(),
                    urgency="immediate"
                )
                triggers.append(trigger)
                self.retraining_triggers.append(trigger)
        
        # Check for average drift
        if drift_metrics.average_psi > 0.15:
            trigger = RetrainingTrigger(
                feature_name="average",
                reason=f"Average PSI exceeds threshold ({drift_metrics.average_psi:.3f})",
                psi_value=drift_metrics.average_psi,
                triggered_at=datetime.now(),
                urgency="increase_frequency"
            )
            triggers.append(trigger)
            self.retraining_triggers.append(trigger)
        
        return triggers
    
    def get_drift_history(
        self,
        model_id: str,
        days: int = 30
    ) -> List[Dict]:
        """Get drift history for a model"""
        if model_id not in self.drift_history:
            return []
        
        cutoff_date = date.today() - timedelta(days=days)
        recent_metrics = [
            m for m in self.drift_history[model_id]
            if m.date >= cutoff_date
        ]
        
        return [m.to_dict() for m in recent_metrics]
    
    def get_retraining_triggers(
        self,
        model_id: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict]:
        """Get recent retraining triggers"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        triggers = [
            t for t in self.retraining_triggers
            if t.triggered_at >= cutoff_time
            and (model_id is None or t.feature_name.startswith(model_id))
        ]
        
        return [t.to_dict() for t in triggers]
    
    def clear_triggers(self, model_id: Optional[str] = None) -> None:
        """Clear retraining triggers, optionally filtered by model"""
        if model_id is None:
            self.retraining_triggers.clear()
        else:
            self.retraining_triggers = [
                t for t in self.retraining_triggers
                if not t.feature_name.startswith(model_id)
            ]


def calculate_psi_simple(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10
) -> float:
    """
    Simple PSI calculation function.
    
    Args:
        expected: Expected/reference values
        actual: Actual/test values
        buckets: Number of buckets
    
    Returns:
        PSI value
    """
    # Define breakpoints
    breakpoints = np.linspace(np.min(expected), np.max(expected), buckets)
    
    # Bucketize
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Add small value to avoid division by zero
    expected_percents = np.clip(expected_percents, 0.0001, None)
    actual_percents = np.clip(actual_percents, 0.0001, None)
    
    # Calculate PSI
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    
    return psi

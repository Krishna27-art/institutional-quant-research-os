"""
Feature Drift Detection with PSI and Model Rollback

This module implements feature drift detection using Population Stability Index (PSI)
and automatic model rollback when drift is detected, ensuring model reliability
in changing market conditions.

Key Features:
- Population Stability Index (PSI) calculation
- Feature drift detection and monitoring
- Automatic model rollback on drift
- Drift severity classification
- Feature-wise drift analysis
- Historical drift tracking
- Alert generation for drift events

Based on V4 Blueprint - Institutional Architecture
Priority: Medium (Phase 4.3)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity of feature drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureDriftResult:
    """Result of feature drift detection."""
    feature_name: str
    psi: float
    severity: DriftSeverity
    baseline_mean: float
    current_mean: float
    baseline_std: float
    current_std: float
    drift_detected: bool
    timestamp: datetime
    
    def is_significant(self) -> bool:
        """Check if drift is significant."""
        return self.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]


@dataclass
class ModelRollbackEvent:
    """Model rollback event."""
    model_name: str
    rollback_from_version: str
    rollback_to_version: str
    trigger_feature: str
    trigger_psi: float
    timestamp: datetime
    reason: str


class FeatureDriftDetector:
    """
    Feature drift detector using PSI.
    
    This class monitors feature distributions and detects drift
    using Population Stability Index (PSI).
    """
    
    def __init__(
        self,
        psi_threshold_low: float = 0.1,
        psi_threshold_medium: float = 0.25,
        psi_threshold_high: float = 0.5,
        psi_threshold_critical: float = 1.0,
        window_size: int = 1000
    ):
        """
        Initialize drift detector.
        
        Args:
            psi_threshold_low: PSI threshold for low drift
            psi_threshold_medium: PSI threshold for medium drift
            psi_threshold_high: PSI threshold for high drift
            psi_threshold_critical: PSI threshold for critical drift
            window_size: Size of rolling window for baseline
        """
        self.psi_threshold_low = psi_threshold_low
        self.psi_threshold_medium = psi_threshold_medium
        self.psi_threshold_high = psi_threshold_high
        self.psi_threshold_critical = psi_threshold_critical
        self.window_size = window_size
        
        self.baseline_distributions: Dict[str, np.ndarray] = {}
        self.feature_history: Dict[str, deque] = {}
        self.drift_history: List[FeatureDriftResult] = []
        self.rollback_events: List[ModelRollbackEvent] = []
        
        logger.info(f"FeatureDriftDetector initialized with thresholds: low={psi_threshold_low}, medium={psi_threshold_medium}, high={psi_threshold_high}, critical={psi_threshold_critical}")
    
    def set_baseline(
        self,
        features: pd.DataFrame
    ) -> None:
        """
        Set baseline feature distributions.
        
        Args:
            features: Baseline feature DataFrame
        """
        for col in features.columns:
            self.baseline_distributions[col] = features[col].values
            self.feature_history[col] = deque(maxlen=self.window_size)
        
        logger.info(f"Baseline set for {len(self.baseline_distributions)} features")
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Expected/baseline distribution
            actual: Current distribution
            bins: Number of bins for PSI calculation
            
        Returns:
            PSI value
        """
        # Create bins based on expected distribution
        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        
        # Handle edge cases
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate distributions
        expected_hist, _ = np.histogram(expected, bins=bin_edges)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
        
        # Normalize to percentages
        expected_dist = expected_hist / len(expected)
        actual_dist = actual_hist / len(actual)
        
        # Avoid division by zero
        expected_dist = np.where(expected_dist == 0, 0.0001, expected_dist)
        actual_dist = np.where(actual_dist == 0, 0.0001, actual_dist)
        
        # Calculate PSI
        psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
        
        return psi
    
    def classify_drift_severity(self, psi: float) -> DriftSeverity:
        """
        Classify drift severity based on PSI.
        
        Args:
            psi: PSI value
            
        Returns:
            DriftSeverity
        """
        if psi < self.psi_threshold_low:
            return DriftSeverity.NONE
        elif psi < self.psi_threshold_medium:
            return DriftSeverity.LOW
        elif psi < self.psi_threshold_high:
            return DriftSeverity.MEDIUM
        elif psi < self.psi_threshold_critical:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def detect_drift(
        self,
        features: pd.DataFrame
    ) -> List[FeatureDriftResult]:
        """
        Detect feature drift.
        
        Args:
            features: Current feature DataFrame
            
        Returns:
            List of FeatureDriftResult
        """
        results = []
        
        for col in features.columns:
            if col not in self.baseline_distributions:
                logger.warning(f"No baseline for feature {col}, skipping")
                continue
            
            baseline = self.baseline_distributions[col]
            current = features[col].values
            
            # Calculate PSI
            psi = self.calculate_psi(baseline, current)
            
            # Classify severity
            severity = self.classify_drift_severity(psi)
            
            # Calculate statistics
            baseline_mean = np.mean(baseline)
            current_mean = np.mean(current)
            baseline_std = np.std(baseline)
            current_std = np.std(current)
            
            # Determine if drift detected
            drift_detected = severity != DriftSeverity.NONE
            
            result = FeatureDriftResult(
                feature_name=col,
                psi=psi,
                severity=severity,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                baseline_std=baseline_std,
                current_std=current_std,
                drift_detected=drift_detected,
                timestamp=datetime.now()
            )
            
            results.append(result)
            
            # Add to history
            self.feature_history[col].extend(current)
            
            # Add to drift history
            if drift_detected:
                self.drift_history.append(result)
        
        return results
    
    def get_drift_summary(self) -> Dict[str, any]:
        """
        Get drift summary.
        
        Returns:
            Dict with drift summary
        """
        if not self.drift_history:
            return {
                'total_drift_events': 0,
                'features_with_drift': 0,
                'severity_distribution': {},
                'recent_drift': []
            }
        
        # Count by severity
        severity_counts = {}
        for result in self.drift_history:
            severity = result.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Get recent drift
        recent_drift = self.drift_history[-10:] if len(self.drift_history) > 10 else self.drift_history
        
        return {
            'total_drift_events': len(self.drift_history),
            'features_with_drift': len(set(r.feature_name for r in self.drift_history)),
            'severity_distribution': severity_counts,
            'recent_drift': recent_drift
        }
    
    def print_drift_report(self) -> None:
        """Print drift detection report."""
        print("\n" + "="*60)
        print("FEATURE DRIFT DETECTION REPORT")
        print("="*60)
        
        summary = self.get_drift_summary()
        
        print(f"\nTotal Drift Events: {summary['total_drift_events']}")
        print(f"Features with Drift: {summary['features_with_drift']}")
        
        if summary['severity_distribution']:
            print(f"\nSeverity Distribution:")
            for severity, count in summary['severity_distribution'].items():
                print(f"  {severity}: {count}")
        
        if summary['recent_drift']:
            print(f"\nRecent Drift Events:")
            print(f"{'Feature':<20} {'PSI':<10} {'Severity':<15} {'Mean Change':<15}")
            print("-" * 65)
            
            for result in summary['recent_drift']:
                mean_change = result.current_mean - result.baseline_mean
                print(f"{result.feature_name:<20} {result.psi:>9.4f} {result.severity.value:<15} {mean_change:>14.4f}")
        
        print("\n" + "="*60)


class ModelRollbackManager:
    """
    Model rollback manager for handling drift-induced rollbacks.
    
    This class manages model versions and performs automatic rollback
    when significant feature drift is detected.
    """
    
    def __init__(
        self,
        model_name: str,
        drift_threshold: float = 0.5,  # PSI threshold for rollback
        max_model_versions: int = 5
    ):
        """
        Initialize rollback manager.
        
        Args:
            model_name: Name of the model
            drift_threshold: PSI threshold for triggering rollback
            max_model_versions: Maximum number of model versions to keep
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold
        self.max_model_versions = max_model_versions
        
        self.model_versions: Dict[str, any] = {}
        self.active_version: Optional[str] = None
        self.rollback_events: List[ModelRollbackEvent] = []
        
        logger.info(f"ModelRollbackManager initialized for {model_name}")
    
    def register_model_version(
        self,
        version_id: str,
        model: any,
        features_used: List[str]
    ) -> None:
        """
        Register a model version.
        
        Args:
            version_id: Version identifier
            model: Model object
            features_used: List of features used by model
        """
        self.model_versions[version_id] = {
            'model': model,
            'features_used': features_used,
            'registered_at': datetime.now()
        }
        
        if self.active_version is None:
            self.active_version = version_id
        
        logger.info(f"Registered model version {version_id}")
    
    def check_and_rollback(
        self,
        drift_results: List[FeatureDriftResult]
    ) -> Optional[ModelRollbackEvent]:
        """
        Check for drift and perform rollback if needed.
        
        Args:
            drift_results: List of drift detection results
            
        Returns:
            ModelRollbackEvent if rollback performed, None otherwise
        """
        if not self.active_version:
            logger.warning("No active model version")
            return None
        
        # Check for significant drift
        significant_drift = [r for r in drift_results if r.is_significant()]
        
        if not significant_drift:
            return None
        
        # Find the most severe drift
        most_severe = max(significant_drift, key=lambda x: x.psi)
        
        if most_severe.psi < self.drift_threshold:
            return None
        
        # Perform rollback
        rollback_event = self._perform_rollback(most_severe)
        
        return rollback_event
    
    def _perform_rollback(self, drift_result: FeatureDriftResult) -> ModelRollbackEvent:
        """
        Perform model rollback.
        
        Args:
            drift_result: Drift result that triggered rollback
            
        Returns:
            ModelRollbackEvent
        """
        previous_version = self.active_version
        
        # Find previous stable version (simplified - just deactivate current)
        available_versions = [v for v in self.model_versions.keys() if v != previous_version]
        
        if available_versions:
            # Rollback to previous version
            new_version = available_versions[-1]
            self.active_version = new_version
            reason = f"Rollback due to drift in {drift_result.feature_name} (PSI={drift_result.psi:.4f})"
        else:
            # No previous version, keep current but log warning
            new_version = previous_version
            reason = f"No previous version available, drift detected in {drift_result.feature_name}"
        
        event = ModelRollbackEvent(
            model_name=self.model_name,
            rollback_from_version=previous_version,
            rollback_to_version=new_version,
            trigger_feature=drift_result.feature_name,
            trigger_psi=drift_result.psi,
            timestamp=datetime.now(),
            reason=reason
        )
        
        self.rollback_events.append(event)
        
        logger.warning(f"Model rollback performed: {reason}")
        
        return event
    
    def get_active_model(self) -> Optional[any]:
        """Get active model."""
        if self.active_version and self.active_version in self.model_versions:
            return self.model_versions[self.active_version]['model']
        return None
    
    def print_rollback_report(self) -> None:
        """Print rollback report."""
        print("\n" + "="*60)
        print("MODEL ROLLBACK REPORT")
        print("="*60)
        
        print(f"\nModel: {self.model_name}")
        print(f"Active Version: {self.active_version}")
        print(f"Total Versions: {len(self.model_versions)}")
        print(f"Rollback Events: {len(self.rollback_events)}")
        
        if self.rollback_events:
            print(f"\nRollback History:")
            print(f"{'Timestamp':<20} {'From':<15} {'To':<15} {'Trigger':<20}")
            print("-" * 75)
            
            for event in self.rollback_events[-5:]:
                print(f"{event.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {event.rollback_from_version:<15} "
                      f"{event.rollback_to_version:<15} {event.trigger_feature:<20}")
        
        print("\n" + "="*60)


def sample_feature_drift_detection():
    """Demonstrate feature drift detection."""
    print("=== Feature Drift Detection Demo ===\n")
    
    # Initialize drift detector
    detector = FeatureDriftDetector(
        psi_threshold_low=0.1,
        psi_threshold_medium=0.25,
        psi_threshold_high=0.5,
        psi_threshold_critical=1.0
    )
    
    # Set baseline
    np.random.seed(42)
    baseline_features = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000)
    })
    
    detector.set_baseline(baseline_features)
    
    # Generate current data with drift
    current_features = pd.DataFrame({
        'feature_1': np.random.randn(100) + 0.5,  # Drifted
        'feature_2': np.random.randn(100),  # No drift
        'feature_3': np.random.randn(100) * 2.0  # Drifted (variance change)
    })
    
    # Detect drift
    print("Detecting feature drift...")
    drift_results = detector.detect_drift(current_features)
    
    # Print report
    detector.print_drift_report()
    
    # Initialize rollback manager
    rollback_manager = ModelRollbackManager(
        model_name="alpha_model",
        drift_threshold=0.5
    )
    
    # Register model versions
    rollback_manager.register_model_version("v1.0", "model_v1", ['feature_1', 'feature_2', 'feature_3'])
    rollback_manager.register_model_version("v2.0", "model_v2", ['feature_1', 'feature_2', 'feature_3'])
    
    # Check and rollback
    print("\nChecking for rollback conditions...")
    rollback_event = rollback_manager.check_and_rollback(drift_results)
    
    if rollback_event:
        print(f"Rollback performed: {rollback_event.reason}")
    else:
        print("No rollback needed")
    
    # Print rollback report
    rollback_manager.print_rollback_report()
    
    print("\n=== Feature Drift Detection Demo Complete ===")
    print("Key capabilities:")
    print("- Population Stability Index (PSI) calculation")
    print("- Feature drift detection and monitoring")
    print("- Automatic model rollback on drift")
    print("- Drift severity classification")
    print("- Feature-wise drift analysis")
    print("- Historical drift tracking")


if __name__ == "__main__":
    sample_feature_drift_detection()

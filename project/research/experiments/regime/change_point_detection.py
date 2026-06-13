"""
Change Point Detection - Complementary to HMM
Based on Blueprint V1.0

Uses ruptures library for structural break detection.
Works alongside HMM for enhanced regime detection.

Methods:
- Binary Segmentation (Binseg)
- Pelt (Pruned Exact Linear Time)
- Window-based detection
- Bottom-up segmentation

Integration with HMM:
- HMM provides continuous regime classification
- CPD detects structural breaks for re-initialization
- Ensemble decision combines both signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


class CPDMethod(Enum):
    """Change point detection methods."""
    BINSEG = "binseg"
    PELT = "pelt"
    WINDOW = "window"
    BOTTOMUP = "bottomup"


@dataclass
class ChangePoint:
    """Detected change point."""
    index: int
    timestamp: datetime
    confidence: float
    feature_name: str
    change_magnitude: float


@dataclass
class CPDConfig:
    """Configuration for Change Point Detection."""
    method: CPDMethod = CPDMethod.BINSEG
    model: str = "l2"  # "l2", "rbf", "linear"
    min_size: int = 10
    jump: int = 1
    penalty: float = 10.0
    n_bkps: int = 5  # Maximum number of change points
    
    # Window-based parameters
    window_size: int = 20
    
    # Confidence threshold
    confidence_threshold: float = 0.7


class ChangePointDetector:
    """
    Change Point Detection for regime structural breaks.
    
    Complements HMM by detecting abrupt changes in market structure.
    Used for:
    - Re-initializing HMM when regime shifts
    - Detecting market state transitions
    - Identifying volatility regime changes
    """
    
    def __init__(self, config: CPDConfig = None):
        self.config = config or CPDConfig()
        
        if not RUPTURES_AVAILABLE:
            raise ImportError("ruptures library not available. Install with: pip install ruptures")
        
        # Feature history for detection
        self.feature_history: Dict[str, List[float]] = {}
        self.timestamp_history: List[datetime] = []
        
        # Detected change points
        self.change_points: List[ChangePoint] = []
        
        # Last detection timestamp
        self.last_detection: Optional[datetime] = None
    
    def add_observation(
        self,
        features: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """
        Add observation to feature history.
        
        Args:
            features: Dictionary of feature values
            timestamp: Observation timestamp
        """
        self.timestamp_history.append(timestamp)
        
        for feature_name, value in features.items():
            if feature_name not in self.feature_history:
                self.feature_history[feature_name] = []
            self.feature_history[feature_name].append(value)
            
            # Keep last 500 points
            if len(self.feature_history[feature_name]) > 500:
                self.feature_history[feature_name] = self.feature_history[feature_name][-500:]
        
        # Keep timestamp history aligned
        if len(self.timestamp_history) > 500:
            self.timestamp_history = self.timestamp_history[-500:]
    
    def detect_change_points(
        self,
        feature_name: str = "realized_volatility_5d",
        retrain_threshold_hours: int = 24
    ) -> List[ChangePoint]:
        """
        Detect change points in a specific feature.
        
        Args:
            feature_name: Name of feature to analyze
            retrain_threshold_hours: Minimum hours between detections
            
        Returns:
            List of detected change points
        """
        if feature_name not in self.feature_history:
            return []
        
        signal = np.array(self.feature_history[feature_name])
        
        if len(signal) < self.config.min_size * 2:
            return []
        
        # Check if enough time has passed since last detection
        if self.last_detection:
            hours_since_last = (datetime.now() - self.last_detection).total_seconds() / 3600
            if hours_since_last < retrain_threshold_hours:
                return []
        
        # Select algorithm based on config
        if self.config.method == CPDMethod.BINSEG:
            algo = rpt.Binseg(
                model=self.config.model,
                min_size=self.config.min_size,
                jump=self.config.jump
            )
        elif self.config.method == CPDMethod.PELT:
            algo = rpt.Pelt(
                model=self.config.model,
                min_size=self.config.min_size,
                jump=self.config.jump
            )
        elif self.config.method == CPDMethod.WINDOW:
            algo = rpt.Window(
                width=self.config.window_size,
                model=self.config.model,
                jump=self.config.jump
            )
        elif self.config.method == CPDMethod.BOTTOMUP:
            algo = rpt.BottomUp(
                model=self.config.model,
                min_size=self.config.min_size,
                jump=self.config.jump
            )
        else:
            algo = rpt.Binseg(model=self.config.model)
        
        # Fit algorithm
        try:
            if self.config.method == CPDMethod.PELT:
                algo.fit(signal).predict(pen=self.config.penalty)
            else:
                algo.fit(signal).predict(n_bkps=self.config.n_bkps)
            
            bkps = algo.change_points
        except Exception as e:
            # Fallback to simple detection
            bkps = self._simple_change_point_detection(signal)
        
        # Convert to ChangePoint objects
        new_change_points = []
        for bkp in bkps:
            if bkp < len(self.timestamp_history):
                timestamp = self.timestamp_history[bkp]
                confidence = self._calculate_confidence(signal, bkp)
                magnitude = self._calculate_magnitude(signal, bkp)
                
                if confidence > self.config.confidence_threshold:
                    cp = ChangePoint(
                        index=bkp,
                        timestamp=timestamp,
                        confidence=confidence,
                        feature_name=feature_name,
                        change_magnitude=magnitude
                    )
                    new_change_points.append(cp)
        
        # Update state
        if new_change_points:
            self.change_points.extend(new_change_points)
            self.last_detection = datetime.now()
        
        return new_change_points
    
    def _simple_change_point_detection(self, signal: np.ndarray) -> List[int]:
        """Simple change point detection using CUSUM-like approach."""
        change_points = []
        
        if len(signal) < 20:
            return change_points
        
        # Calculate rolling mean and std
        window = 20
        rolling_mean = pd.Series(signal).rolling(window).mean()
        rolling_std = pd.Series(signal).rolling(window).std()
        
        # Detect points where signal deviates significantly from rolling mean
        z_scores = (signal - rolling_mean) / rolling_std
        
        # Find points where |z-score| > 3
        for i in range(window, len(signal)):
            if abs(z_scores.iloc[i]) > 3:
                change_points.append(i)
        
        return change_points
    
    def _calculate_confidence(self, signal: np.ndarray, change_idx: int) -> float:
        """Calculate confidence score for a change point."""
        if change_idx < 10 or change_idx >= len(signal) - 10:
            return 0.5
        
        # Compare means before and after
        before_mean = np.mean(signal[max(0, change_idx-20):change_idx])
        after_mean = np.mean(signal[change_idx:min(len(signal), change_idx+20)])
        
        before_std = np.std(signal[max(0, change_idx-20):change_idx])
        after_std = np.std(signal[change_idx:min(len(signal), change_idx+20)])
        
        # Confidence based on mean difference relative to std
        mean_diff = abs(after_mean - before_mean)
        avg_std = (before_std + after_std) / 2
        
        if avg_std > 0:
            confidence = min(1.0, mean_diff / (2 * avg_std))
        else:
            confidence = 0.5
        
        return confidence
    
    def _calculate_magnitude(self, signal: np.ndarray, change_idx: int) -> float:
        """Calculate magnitude of change."""
        if change_idx < 10 or change_idx >= len(signal) - 10:
            return 0.0
        
        before_mean = np.mean(signal[max(0, change_idx-20):change_idx])
        after_mean = np.mean(signal[change_idx:min(len(signal), change_idx+20)])
        
        if before_mean != 0:
            magnitude = abs(after_mean - before_mean) / abs(before_mean)
        else:
            magnitude = abs(after_mean)
        
        return magnitude
    
    def should_reinitialize_hmm(self) -> Tuple[bool, str]:
        """
        Determine if HMM should be reinitialized based on recent change points.
        
        Returns:
            (should_reinit, reason) tuple
        """
        if not self.change_points:
            return False, ""
        
        # Check for recent change points
        recent_threshold = timedelta(hours=24)
        now = datetime.now()
        recent_cps = [cp for cp in self.change_points if now - cp.timestamp < recent_threshold]
        
        if not recent_cps:
            return False, ""
        
        # Check for high-confidence change points
        high_confidence_cps = [cp for cp in recent_cps if cp.confidence > 0.8]
        
        if high_confidence_cps:
            return True, f"High-confidence change point detected: {high_confidence_cps[0].feature_name}"
        
        # Check for large magnitude changes
        large_magnitude_cps = [cp for cp in recent_cps if cp.change_magnitude > 0.5]
        
        if large_magnitude_cps:
            return True, f"Large magnitude change detected: {large_magnitude_cps[0].feature_name}"
        
        return False, ""
    
    def get_recent_change_points(self, hours: int = 24) -> List[ChangePoint]:
        """Get change points from last N hours."""
        threshold = timedelta(hours=hours)
        now = datetime.now()
        
        return [cp for cp in self.change_points if now - cp.timestamp < threshold]
    
    def reset(self) -> None:
        """Reset detector state."""
        self.feature_history.clear()
        self.timestamp_history.clear()
        self.change_points.clear()
        self.last_detection = None


class EnsembleRegimeDetector:
    """
    Ensemble of HMM and Change Point Detection.
    
    Combines continuous regime classification (HMM) with
    structural break detection (CPD) for robust regime detection.
    """
    
    def __init__(self, hmm_config: dict = None, cpd_config: CPDConfig = None):
        from research.regime.hmm_engine import HMMRegimeEngine
        
        self.hmm_engine = HMMRegimeEngine(hmm_config or {})
        self.cpd_detector = ChangePointDetector(cpd_config)
        
        # Ensemble weights
        self.hmm_weight = 0.7
        self.cpd_weight = 0.3
    
    def detect_regime(
        self,
        features: Dict[str, float],
        timestamp: datetime
    ) -> Dict:
        """
        Detect regime using ensemble of HMM and CPD.
        
        Args:
            features: Current feature values
            timestamp: Current timestamp
            
        Returns:
            Dictionary with regime information
        """
        # Update both detectors
        self.hmm_engine.add_observation(features, timestamp)
        self.cpd_detector.add_observation(features, timestamp)
        
        # Get HMM regime
        hmm_regime = self.hmm_engine.predict_regime(features, timestamp)
        
        # Check for change points
        change_points = self.cpd_detector.detect_change_points()
        
        # Check if HMM should be reinitialized
        should_reinit, reinit_reason = self.cpd_detector.should_reinitialize_hmm()
        
        if should_reinit:
            self.hmm_engine.reset()
            # Retrain HMM with recent data
            self.hmm_engine.train_model()
        
        # Ensemble decision
        ensemble_regime = hmm_regime.regime
        
        # If recent high-confidence change point, override HMM
        recent_cps = self.cpd_detector.get_recent_change_points(hours=6)
        if recent_cps:
            high_conf_cp = max(recent_cps, key=lambda x: x.confidence)
            if high_conf_cp.confidence > 0.9:
                # Override based on change direction
                if high_conf_cp.change_magnitude > 0:
                    ensemble_regime = self.hmm_engine.Regime.HIGH_VOL
                else:
                    ensemble_regime = self.hmm_engine.Regime.LOW_VOL
        
        return {
            'regime': ensemble_regime.value,
            'hmm_regime': hmm_regime.regime.value,
            'hmm_probability': hmm_regime.probability,
            'change_points_detected': len(change_points),
            'recent_change_points': len(recent_cps),
            'hmm_reinitialized': should_reinit,
            'reinit_reason': reinit_reason,
            'ensemble_confidence': self.hmm_weight * hmm_regime.probability + self.cpd_weight * (1.0 if not recent_cps else recent_cps[0].confidence)
        }
    
    def get_regime_weights(self) -> Dict[str, float]:
        """Get regime-based alpha weights from HMM."""
        return self.hmm_engine.get_regime_weights()


if __name__ == "__main__":
    # Test the change point detector
    print("Testing Change Point Detection...")
    
    config = CPDConfig(method=CPDMethod.BINSEG)
    detector = ChangePointDetector(config)
    
    # Generate sample data with change point
    np.random.seed(42)
    n = 200
    
    # First regime: low volatility
    signal1 = np.random.normal(0, 0.01, 100)
    
    # Second regime: high volatility (change point)
    signal2 = np.random.normal(0, 0.05, 100)
    
    signal = np.concatenate([signal1, signal2])
    
    # Add observations
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    for i, (val, ts) in enumerate(zip(signal, timestamps)):
        detector.add_observation({'volatility': val}, ts)
    
    # Detect change points
    change_points = detector.detect_change_points('volatility')
    
    print(f"Detected {len(change_points)} change points")
    for cp in change_points:
        print(f"  Index: {cp.index}, Timestamp: {cp.timestamp}, Confidence: {cp.confidence:.2f}, Magnitude: {cp.change_magnitude:.2f}")
    
    # Check if HMM should be reinitialized
    should_reinit, reason = detector.should_reinitialize_hmm()
    print(f"Should reinitialize HMM: {should_reinit}, Reason: {reason}")

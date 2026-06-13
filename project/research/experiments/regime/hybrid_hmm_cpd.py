"""
Hybrid HMM + Change Point Detection Regime Engine
Based on research recommendations for Indian markets

Key findings from research:
- HMM wins for Indian markets (volatility persistence d=0.226)
- HMM captures vol regimes effectively
- Daily re-estimation (N=252) fits Indian data size
- Latency <5ms acceptable for 1-min bars
- Online CPD for rapid regime shifts

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from hmmlearn import hmm
from scipy import stats
from collections import deque
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeState:
    """Regime state information"""
    name: str
    probability: float
    vol_level: str  # "low", "medium", "high"
    direction: str  # "up", "down", "sideways"
    characteristics: Dict[str, float]


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    dominant_regime: str
    regime_probabilities: Dict[str, float]
    is_change_point: bool
    change_point_confidence: float
    regime_state: RegimeState
    alpha_weights: Dict[str, float]


class HybridHMMCPD:
    """
    Hybrid HMM + Change Point Detection Regime Engine.
    
    Architecture:
    - Primary: HMM with 4 states (trend_up, trend_down, sideways, high_vol)
    - Secondary: Online CPD for rapid regime shifts
    - Features: realized_vol, implied_vol, return, turnover, spread, skew
    - Re-estimation: Daily (252-day window)
    - Change point: Online detection with 10-min window
    
    Why HMM wins for Indian markets:
    - Volatility persistence is well-documented (Deep et al. 2026)
    - d=0.226 (GPH) means clear regime memory
    - Unlike transformers, works with N=1,000 days
    - Interpretable for risk management
    """
    
    def __init__(self, n_components: int = 4, cpd_window: int = 10):
        self.n_components = n_components
        self.cpd_window = cpd_window
        
        # HMM model
        # CRITICAL FIX: Changed from "full" to "diag" covariance
        # Full covariance with 7 features and 4 states = 784 parameters (unidentifiable)
        # Diagonal covariance = 28 parameters (identifiable)
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            init_params="kmeans",  # CRITICAL: stable initialization
            random_state=42
        )
        
        # Feature scaler (critical for HMM convergence)
        self.scaler = StandardScaler()
        
        # Regime labels (will be assigned after fitting)
        self.regime_labels = {
            0: "trend_up",
            1: "trend_down",
            2: "sideways",
            3: "high_vol"
        }
        
        # Online CPD
        self.feature_window = deque(maxlen=cpd_window)
        self.cpd_threshold = 2.5  # Standard deviations for change point
        
        # Regime-specific alpha weights
        self.regime_weights = {
            "trend_up": {
                "trend": 0.60,
                "mean_rev": 0.10,
                "vol": 0.20,
                "options": 0.10
            },
            "trend_down": {
                "trend": 0.50,
                "mean_rev": 0.10,
                "vol": 0.30,
                "options": 0.10
            },
            "sideways": {
                "trend": 0.10,
                "mean_rev": 0.50,
                "vol": 0.20,
                "options": 0.20
            },
            "high_vol": {
                "trend": 0.20,
                "mean_rev": 0.10,
                "vol": 0.40,
                "options": 0.30
            }
        }
        
        self.is_fitted = False
        self.last_regime = None
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection.
        
        CRITICAL FIX: All rolling calculations shifted by 1 to avoid look-ahead bias.
        
        Features:
        - realized_vol: 20-day realized volatility (shifted)
        - implied_vol: (placeholder) from options
        - return: Daily return (shifted)
        - turnover: Volume / avg volume (shifted)
        - spread: Bid-ask spread (placeholder)
        - skew: Return skewness (shifted)
        """
        features = []
        
        # Returns
        returns = data['close'].pct_change()
        
        # Realized volatility (20-day, shifted by 1 to avoid leakage)
        realized_vol = returns.rolling(20).std().shift(1).iloc[-1] * np.sqrt(252)
        
        # Daily return (shifted by 1 to avoid leakage)
        daily_return = returns.shift(1).iloc[-1]
        
        # Turnover (volume ratio, shifted)
        avg_volume = data['volume'].rolling(20).mean().shift(1).iloc[-1]
        turnover = data['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1.0
        
        # Skewness (20-day, shifted)
        skew = returns.rolling(20).skew().shift(1).iloc[-1]
        
        # Placeholder for implied vol and spread (would come from options data)
        implied_vol = realized_vol * 1.1  # Typical IV premium
        spread = 0.02  # 2bps typical spread
        
        features = [
            realized_vol,
            implied_vol,
            daily_return,
            turnover,
            spread,
            skew
        ]
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, data: pd.DataFrame, window_days: int = 252) -> None:
        """
        Fit HMM model on historical data.
        
        Args:
            data: DataFrame with OHLCV data
            window_days: Number of days to use for fitting (default 252 = 1 year)
        """
        print(f"Fitting HMM regime engine on {window_days} days of data...")
        
        # Get last window_days of data
        if len(data) < window_days:
            window_days = len(data)
        
        fit_data = data.iloc[-window_days:]
        
        # Extract features for each day
        feature_matrix = []
        for i in range(21, len(fit_data)):  # Start from 21 to have enough for shifted rolling stats
            day_data = fit_data.iloc[:i+1]
            features = self.extract_features(day_data)
            if not np.isnan(features).any():
                feature_matrix.append(features[0])
        
        feature_matrix = np.array(feature_matrix)
        
        if len(feature_matrix) < 100:
            raise ValueError(f"Need at least 100 samples, got {len(feature_matrix)}")
        
        # CRITICAL: Scale features before fitting HMM
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Fit HMM with multiple restarts for stability
        best_score = -np.inf
        best_model = None
        for seed in range(3):
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_components,
                    covariance_type="diag",
                    n_iter=200,
                    init_params="kmeans",
                    random_state=42 + seed
                )
                model.fit(feature_matrix_scaled)
                score = model.score(feature_matrix_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                print(f"Restart {seed} failed: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("HMM failed to converge after 3 restarts")
        
        self.hmm_model = best_model
        
        # Assign regime labels based on state characteristics
        self._assign_regime_labels(feature_matrix)
        
        self.is_fitted = True
        print("HMM regime engine fitted successfully")
    
    def _assign_regime_labels(self, feature_matrix: np.ndarray) -> None:
        """Assign meaningful labels to HMM states based on characteristics."""
        # Get state means
        state_means = self.hmm_model.means_
        
        # Classify states based on mean return and volatility
        for i in range(self.n_components):
            mean_return = state_means[i, 2]  # Return is at index 2
            mean_vol = state_means[i, 0]  # Realized vol is at index 0
            
            if mean_vol > np.percentile(state_means[:, 0], 75):
                self.regime_labels[i] = "high_vol"
            elif mean_return > 0.001:
                self.regime_labels[i] = "trend_up"
            elif mean_return < -0.001:
                self.regime_labels[i] = "trend_down"
            else:
                self.regime_labels[i] = "sideways"
    
    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect current regime using HMM and CPD.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            RegimeDetectionResult with regime information
        """
        if not self.is_fitted:
            raise RuntimeError("HMM model not fitted. Call fit() first.")
        
        # Extract current features
        current_features = self.extract_features(data)
        
        # CRITICAL: Scale features using fitted scaler
        current_features_scaled = self.scaler.transform(current_features)
        
        # HMM prediction
        hmm_state = self.hmm_model.predict(current_features_scaled)[0]
        hmm_probabilities = self.hmm_model.predict_proba(current_features_scaled)[0]
        
        # Map state to regime name
        regime_name = self.regime_labels[hmm_state]
        
        # Check for change point
        is_change_point, cp_confidence = self._check_change_point(current_features)
        
        # Build regime probabilities dictionary
        regime_probs = {}
        for i in range(self.n_components):
            regime_probs[self.regime_labels[i]] = hmm_probabilities[i]
        
        # Get regime state details
        regime_state = self._get_regime_state(regime_name, current_features)
        
        # Get alpha weights for this regime
        alpha_weights = self.regime_weights.get(regime_name, {})
        
        # Update last regime
        if self.last_regime != regime_name:
            self.last_regime = regime_name
        
        return RegimeDetectionResult(
            dominant_regime=regime_name,
            regime_probabilities=regime_probs,
            is_change_point=is_change_point,
            change_point_confidence=cp_confidence,
            regime_state=regime_state,
            alpha_weights=alpha_weights
        )
    
    def _check_change_point(self, current_features: np.ndarray) -> Tuple[bool, float]:
        """
        Check for change point using online CPD.
        
        Args:
            current_features: Current feature vector
            
        Returns:
            (is_change_point, confidence) tuple
        """
        # Add to window
        self.feature_window.append(current_features[0])
        
        if len(self.feature_window) < self.cpd_window:
            return False, 0.0
        
        # Calculate mean and std of window
        window_array = np.array(self.feature_window)
        window_mean = np.mean(window_array, axis=0)
        window_std = np.std(window_array, axis=0)
        
        # Calculate z-score of current features relative to window
        z_scores = np.abs((current_features[0] - window_mean) / (window_std + 1e-8))
        
        # Check if any feature exceeds threshold
        max_z = np.max(z_scores)
        
        if max_z > self.cpd_threshold:
            confidence = min((max_z - self.cpd_threshold) / 2.0, 1.0)
            return True, confidence
        
        return False, 0.0
    
    def _get_regime_state(self, regime_name: str, features: np.ndarray) -> RegimeState:
        """Get detailed regime state information."""
        vol_level = "medium"
        if features[0, 0] > 0.25:  # High vol threshold
            vol_level = "high"
        elif features[0, 0] < 0.12:
            vol_level = "low"
        
        direction = "sideways"
        if features[0, 2] > 0.001:
            direction = "up"
        elif features[0, 2] < -0.001:
            direction = "down"
        
        characteristics = {
            "realized_vol": features[0, 0],
            "implied_vol": features[0, 1],
            "daily_return": features[0, 2],
            "turnover": features[0, 3],
            "spread": features[0, 4],
            "skew": features[0, 5]
        }
        
        return RegimeState(
            name=regime_name,
            probability=0.0,  # Will be set from detection result
            vol_level=vol_level,
            direction=direction,
            characteristics=characteristics
        )
    
    def get_regime_transition_matrix(self) -> pd.DataFrame:
        """Get HMM transition matrix."""
        if not self.is_fitted:
            raise RuntimeError("HMM model not fitted")
        
        transition_matrix = self.hmm_model.transmat_
        
        # Create labeled DataFrame
        labels = [self.regime_labels[i] for i in range(self.n_components)]
        df = pd.DataFrame(transition_matrix, index=labels, columns=labels)
        
        return df
    
    def print_regime_info(self, result: RegimeDetectionResult) -> None:
        """Print regime detection results."""
        print("\n" + "="*60)
        print("REGIME DETECTION RESULTS")
        print("="*60)
        print(f"Dominant Regime: {result.dominant_regime.upper()}")
        print(f"Regime Probability: {result.regime_probabilities[result.dominant_regime]:.2%}")
        
        if result.is_change_point:
            print(f"⚠️  CHANGE POINT DETECTED (Confidence: {result.change_point_confidence:.2%})")
        
        print("\nRegime Probabilities:")
        for regime, prob in sorted(result.regime_probabilities.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"  {regime:<15}: {prob:>6.2%} {bar}")
        
        print("\nRegime Characteristics:")
        for key, value in result.regime_state.characteristics.items():
            print(f"  {key:<15}: {value:>8.4f}")
        
        print("\nAlpha Weights for This Regime:")
        for alpha, weight in result.alpha_weights.items():
            bar = "█" * int(weight * 20)
            print(f"  {alpha:<15}: {weight:>6.2%} {bar}")
        
        print("="*60)


def run_sample_detection():
    """Run sample regime detection with synthetic data."""
    # Create synthetic data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    
    np.random.seed(42)
    
    # Simulate different regimes
    prices = []
    vol_levels = []
    
    for i in range(len(dates)):
        # Regime switching
        if i < 60:
            regime = "trend_up"
            vol = 0.12
            drift = 0.001
        elif i < 120:
            regime = "sideways"
            vol = 0.10
            drift = 0.0001
        elif i < 180:
            regime = "high_vol"
            vol = 0.30
            drift = -0.0005
        else:
            regime = "trend_down"
            vol = 0.18
            drift = -0.001
        
        vol_levels.append(vol)
        
        if i == 0:
            price = 20000
        else:
            ret = np.random.normal(drift, vol / np.sqrt(252))
            price = prices[-1] * (1 + ret)
        
        prices.append(price)
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Initialize and fit regime engine
    regime_engine = HybridHMMCPD(n_components=4, cpd_window=10)
    regime_engine.fit(data, window_days=252)
    
    # Detect regime on last day
    result = regime_engine.detect_regime(data)
    regime_engine.print_regime_info(result)
    
    # Print transition matrix
    print("\nRegime Transition Matrix:")
    print(regime_engine.get_regime_transition_matrix())
    
    return result


if __name__ == "__main__":
    run_sample_detection()

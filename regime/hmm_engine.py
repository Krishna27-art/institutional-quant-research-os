"""
HMM Regime Detection Engine
Hidden Markov Model for market regime classification

Architecture V2 - 4 states: bull_trend, bear_trend, sideways, high_vol
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from hmmlearn import hmm


class Regime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"


@dataclass
class RegimeState:
    """Current regime state"""
    regime: Regime
    probability: float
    timestamp: datetime
    features: Dict[str, float]
    transition_matrix: Optional[np.ndarray] = None


@dataclass
class HMMConfig:
    """Configuration for HMM Regime Engine"""
    n_states: int = 4
    states: List[str] = None
    
    # Training parameters
    training_window_days: int = 252  # 1 year
    retraining_frequency: str = "daily"
    min_samples: int = 100
    
    # HMM parameters
    covariance_type: str = "full"  # 'full', 'tied', 'diag', 'spherical'
    n_iter: int = 100
    tol: float = 1e-2
    
    # Features for regime detection
    features: List[str] = None
    
    # Change point detection
    use_change_point: bool = True
    change_point_method: str = "CUSUM"
    change_point_window_minutes: int = 10
    
    # Fallback
    fallback_to_vol_quantile: bool = True
    vol_quantile_threshold: float = 0.80  # 80th percentile


class HMMRegimeEngine:
    """
    Hidden Markov Model Regime Detection Engine
    
    Detects 4 market regimes:
    1. Bull Trend: Positive returns, moderate volatility
    2. Bear Trend: Negative returns, moderate volatility
    3. Sideways: Low returns, low volatility
    4. High Vol: High volatility, uncertain direction
    
    Uses Gaussian HMM with 4 states trained on:
    - Realized volatility (5-day)
    - Implied volatility (nearest expiry)
    - NIFTY return (5-day)
    - Turnover ratio (5-day)
    """
    
    def __init__(self, config: dict):
        self.config = HMMConfig(**config)
        
        # Set default states if not provided
        if self.config.states is None:
            self.config.states = ["bull_trend", "bear_trend", "sideways", "high_vol"]
        
        # Set default features if not provided
        if self.config.features is None:
            self.config.features = [
                "realized_vol_5d",
                "implied_vol",
                "nifty_return_5d",
                "turnover_ratio_5d"
            ]
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=self.config.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=self.config.n_iter,
            tol=self.config.tol,
            random_state=42
        )
        
        # Training data
        self.training_data = []  # List of feature vectors
        self.training_dates = []
        
        # Current state
        self.current_regime: Optional[RegimeState] = None
        self.last_retrain_date: Optional[datetime] = None
        
        # Feature history for change point detection
        self.feature_history = {}  # feature_name -> list of values
        
        # Volatility history for fallback
        self.vol_history = []
        
    def get_required_features(self) -> List[str]:
        """Return required features"""
        return self.config.features
    
    def add_observation(
        self,
        features: Dict[str, float],
        timestamp: datetime
    ) -> None:
        """Add observation to training data"""
        # Extract features in order
        feature_vector = [features.get(f, 0) for f in self.config.features]
        
        self.training_data.append(feature_vector)
        self.training_dates.append(timestamp)
        
        # Keep only training window
        max_samples = self.config.training_window_days
        if len(self.training_data) > max_samples:
            self.training_data = self.training_data[-max_samples:]
            self.training_dates = self.training_dates[-max_samples:]
        
        # Update feature history for change point detection
        for feature_name, value in features.items():
            if feature_name not in self.feature_history:
                self.feature_history[feature_name] = []
            self.feature_history[feature_name].append(value)
            
            # Keep last 100 points
            if len(self.feature_history[feature_name]) > 100:
                self.feature_history[feature_name] = self.feature_history[feature_name][-100:]
        
        # Update volatility history for fallback
        if "realized_vol_5d" in features:
            self.vol_history.append(features["realized_vol_5d"])
            if len(self.vol_history) > 252:
                self.vol_history = self.vol_history[-252:]
    
    def should_retrain(self, current_date: datetime) -> bool:
        """Check if model should be retrained"""
        if self.last_retrain_date is None:
            return True
        
        if self.config.retraining_frequency == "daily":
            # Retrain daily after market close
            return current_date.date() > self.last_retrain_date.date()
        
        return False
    
    def train_model(self) -> None:
        """Train HMM model on current data"""
        if len(self.training_data) < self.config.min_samples:
            print(f"Not enough data to train: {len(self.training_data)} < {self.config.min_samples}")
            return
        
        # Convert to numpy array
        X = np.array(self.training_data)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Train model
        try:
            self.model.fit(X_normalized)
            self.last_retrain_date = datetime.now()
            print(f"HMM model trained on {len(X)} samples")
        except Exception as e:
            print(f"Error training HMM model: {e}")
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using z-score"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (X - mean) / std
    
    def predict_regime(
        self,
        features: Dict[str, float],
        timestamp: datetime
    ) -> RegimeState:
        """
        Predict current regime.
        
        Args:
            features: Current feature values
            timestamp: Current timestamp
            
        Returns:
            RegimeState with predicted regime and probability
        """
        # Extract features in order
        feature_vector = [features.get(f, 0) for f in self.config.features]
        X = np.array([feature_vector])
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Normalize using same parameters as training
        if len(self.training_data) > 0:
            training_X = np.array(self.training_data)
            training_X = np.nan_to_num(training_X, nan=0.0)
            mean = np.mean(training_X, axis=0)
            std = np.std(training_X, axis=0)
            std[std == 0] = 1
            X = (X - mean) / std
        
        # Predict regime
        try:
            if len(self.training_data) >= self.config.min_samples:
                # Use HMM prediction
                state = self.model.predict(X)[0]
                probs = self.model.predict_proba(X)[0]
                probability = probs[state]
            else:
                # Fallback to simple rules
                state, probability = self._fallback_prediction(features)
        except Exception as e:
            print(f"Error predicting regime: {e}")
            state, probability = self._fallback_prediction(features)
        
        # Map state index to regime name
        regime_name = self._map_state_to_regime(state, features)
        regime = Regime(regime_name)
        
        # Get transition matrix if available
        transition_matrix = self.model.transmat_ if hasattr(self.model, 'transmat_') else None
        
        self.current_regime = RegimeState(
            regime=regime,
            probability=probability,
            timestamp=timestamp,
            features=features,
            transition_matrix=transition_matrix
        )
        
        return self.current_regime
    
    def _map_state_to_regime(self, state: int, features: Dict[str, float]) -> str:
        """
        Map HMM state to regime name based on feature characteristics.
        
        States are unlabeled, so we infer meaning from feature values.
        """
        rv = features.get("realized_vol_5d", 0)
        iv = features.get("implied_vol", 0)
        nifty_return = features.get("nifty_return_5d", 0)
        
        # High vol regime
        if rv > 0.25 or iv > 0.30:
            return "high_vol"
        
        # Bull trend
        if nifty_return > 0.02 and rv < 0.20:
            return "bull_trend"
        
        # Bear trend
        if nifty_return < -0.02 and rv < 0.20:
            return "bear_trend"
        
        # Sideways (default)
        return "sideways"
    
    def _fallback_prediction(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Fallback prediction using simple rules"""
        rv = features.get("realized_vol_5d", 0)
        nifty_return = features.get("nifty_return_5d", 0)
        
        # Use volatility quantile if enabled
        if self.config.fallback_to_vol_quantile and len(self.vol_history) > 30:
            vol_threshold = np.percentile(self.vol_history, self.config.vol_quantile_threshold * 100)
            if rv > vol_threshold:
                return 3, 0.8  # High vol state
        
        # Simple rule-based classification
        if rv > 0.25:
            return 3, 0.7  # High vol
        elif nifty_return > 0.02:
            return 0, 0.6  # Bull
        elif nifty_return < -0.02:
            return 1, 0.6  # Bear
        else:
            return 2, 0.5  # Sideways
    
    def detect_change_point(self, features: Dict[str, float]) -> bool:
        """
        Detect regime change using CUSUM.
        
        Returns:
            True if regime change detected
        """
        if not self.config.use_change_point:
            return False
        
        # Simple CUSUM on realized volatility
        if "realized_vol_5d" not in self.feature_history:
            return False
        
        rv_history = self.feature_history["realized_vol_5d"]
        if len(rv_history) < self.config.change_point_window_minutes:
            return False
        
        # Calculate mean of recent window
        window_size = self.config.change_point_window_minutes
        recent_rv = rv_history[-window_size:]
        recent_mean = np.mean(recent_rv)
        
        # Calculate mean of previous window
        if len(rv_history) < 2 * window_size:
            return False
        
        previous_rv = rv_history[-2*window_size:-window_size]
        previous_mean = np.mean(previous_rv)
        
        # CUSUM threshold
        threshold = 0.5 * previous_mean
        
        if abs(recent_mean - previous_mean) > threshold:
            return True
        
        return False
    
    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get regime-based weights for alpha combination.
        
        Returns:
            Dictionary mapping alpha names to weights based on current regime
        """
        if self.current_regime is None:
            # Default equal weights
            return {
                "ORB": 0.25,
                "VWAP": 0.25,
                "PCP": 0.25,
                "VolCarry": 0.25
            }
        
        regime = self.current_regime.regime.value
        
        # Regime-based weights from Architecture V2
        weights = {
            "bull_trend": {
                "ORB": 0.40,
                "VWAP": 0.30,
                "PCP": 0.15,
                "VolCarry": 0.10
            },
            "bear_trend": {
                "ORB": 0.20,
                "VWAP": 0.40,
                "PCP": 0.20,
                "VolCarry": 0.15
            },
            "sideways": {
                "ORB": 0.10,
                "VWAP": 0.10,
                "PCP": 0.30,
                "VolCarry": 0.40
            },
            "high_vol": {
                "ORB": 0.15,
                "VWAP": 0.15,
                "PCP": 0.20,
                "VolCarry": 0.40
            }
        }
        
        return weights.get(regime, weights["sideways"])
    
    def get_current_regime(self) -> Optional[RegimeState]:
        """Get current regime state"""
        return self.current_regime
    
    def reset(self) -> None:
        """Reset engine state"""
        self.training_data.clear()
        self.training_dates.clear()
        self.current_regime = None
        self.last_retrain_date = None
        self.feature_history.clear()
        self.vol_history.clear()

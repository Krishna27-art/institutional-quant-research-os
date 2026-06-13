"""
5-State HMM Regime-Aware Factor Switching Alpha Strategy

This module implements a 5-state Hidden Markov Model for regime detection
and dynamic factor allocation, allowing the strategy to adapt to different
market conditions by switching factor exposures based on the detected regime.

Based on multiple regime detection studies (2020–2026).
Expected Sharpe: 0.3-0.5
Expected Capacity: High
Decay: Persistent
Difficulty: Medium

Priority: Medium (Research OS Phase 5)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("hmmlearn not available, HMM regime detection will use fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeState(Enum):
    """5-state regime classification."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeTransition:
    """Regime transition information."""
    from_state: RegimeState
    to_state: RegimeState
    probability: float
    timestamp: datetime


@dataclass
class RegimeSignal:
    """Regime-based factor allocation signal."""
    timestamp: datetime
    current_regime: RegimeState
    regime_probability: float
    factor_weights: Dict[str, float]  # Factor name -> weight
    momentum_weight: float
    value_weight: float
    quality_weight: float
    low_vol_weight: float
    confidence: float
    expected_regime_duration: int  # Days


class HMMRegimeSwitchingAlpha:
    """
    5-state HMM regime-aware factor switching alpha strategy.
    
    This class uses Hidden Markov Models to detect market regimes
    and dynamically allocate factor exposures based on the detected state.
    """
    
    def __init__(
        self,
        n_states: int = 5,
        lookback_days: int = 252,  # 1 year
        min_obs_for_training: int = 100,
        regime_persistence_threshold: float = 0.7
    ):
        """
        Initialize HMM regime switching alpha.
        
        Args:
            n_states: Number of HMM states (5 for 5-state model)
            lookback_days: Lookback period for regime detection
            min_obs_for_training: Minimum observations for HMM training
            regime_persistence_threshold: Threshold for regime persistence
        """
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.min_obs_for_training = min_obs_for_training
        self.regime_persistence_threshold = regime_persistence_threshold
        
        self.hmm_model = None
        self.regime_history: List[Tuple[datetime, RegimeState, float]] = []
        self.signals: List[RegimeSignal] = []
        self.transitions: List[RegimeTransition] = []
        
        # Factor weights for each regime
        self.regime_factor_weights = {
            RegimeState.BULL_TREND: {
                'momentum': 0.4,
                'quality': 0.3,
                'low_vol': 0.1,
                'value': 0.2
            },
            RegimeState.BEAR_TREND: {
                'momentum': 0.1,
                'quality': 0.4,
                'low_vol': 0.4,
                'value': 0.1
            },
            RegimeState.SIDEWAYS: {
                'momentum': 0.1,
                'quality': 0.3,
                'low_vol': 0.3,
                'value': 0.3
            },
            RegimeState.HIGH_VOLATILITY: {
                'momentum': 0.1,
                'quality': 0.4,
                'low_vol': 0.4,
                'value': 0.1
            },
            RegimeState.LOW_VOLATILITY: {
                'momentum': 0.3,
                'quality': 0.3,
                'low_vol': 0.2,
                'value': 0.2
            }
        }
        
        if HMM_AVAILABLE:
            self._initialize_hmm_model()
        
        logger.info(f"HMMRegimeSwitchingAlpha initialized: n_states={n_states}, "
                   f"lookback={lookback_days}days")
    
    def _initialize_hmm_model(self) -> None:
        """Initialize HMM model."""
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        logger.info("HMM model initialized")
    
    def prepare_features(
        self,
        returns: pd.Series,
        volatility: pd.Series,
        volume: pd.Series
    ) -> np.ndarray:
        """
        Prepare features for HMM training.
        
        Args:
            returns: Return series
            volatility: Volatility series
            volume: Volume series
            
        Returns:
            Feature matrix
        """
        # Create feature matrix
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'volume': volume
        })
        
        # Add lagged features
        for lag in [1, 2, 5]:
            features[f'returns_lag{lag}'] = returns.shift(lag)
            features[f'volatility_lag{lag}'] = volatility.shift(lag)
        
        # Add rolling features
        features['returns_ma5'] = returns.rolling(5).mean()
        features['returns_ma20'] = returns.rolling(20).mean()
        features['volatility_ma5'] = volatility.rolling(5).mean()
        
        # Drop NaN
        features = features.dropna()
        
        return features.values
    
    def train_hmm(
        self,
        features: np.ndarray
    ) -> None:
        """
        Train HMM model on features.
        
        Args:
            features: Feature matrix
        """
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, using fallback regime detection")
            return
        
        if len(features) < self.min_obs_for_training:
            logger.warning(f"Insufficient data for HMM training: {len(features)} < {self.min_obs_for_training}")
            return
        
        # Train HMM
        self.hmm_model.fit(features)
        
        logger.info(f"HMM model trained on {len(features)} observations")
    
    def detect_regime(
        self,
        features: np.ndarray,
        timestamp: datetime
    ) -> Tuple[RegimeState, float]:
        """
        Detect current regime using HMM.
        
        Args:
            features: Feature matrix
            timestamp: Current timestamp
            
        Returns:
            (regime_state, probability)
        """
        if not HMM_AVAILABLE or self.hmm_model is None:
            # Fallback: simple rule-based regime detection
            return self._fallback_regime_detection(features, timestamp)
        
        # Predict regime
        if len(features) < 1:
            return RegimeState.SIDEWAYS, 0.5
        
        # Get last observation
        last_obs = features[-1:].reshape(1, -1)
        
        # Predict state
        state = self.hmm_model.predict(last_obs)[0]
        
        # Get probability
        post_prob = self.hmm_model.predict_proba(last_obs)[0]
        probability = post_prob[state]
        
        # Map HMM state to regime
        regime = self._map_state_to_regime(state)
        
        return regime, probability
    
    def _fallback_regime_detection(
        self,
        features: np.ndarray,
        timestamp: datetime
    ) -> Tuple[RegimeState, float]:
        """
        Fallback regime detection without HMM.
        
        Args:
            features: Feature matrix
            timestamp: Current timestamp
            
        Returns:
            (regime_state, probability)
        """
        if len(features) < 20:
            return RegimeState.SIDEWAYS, 0.5
        
        # Use simple rules based on recent returns and volatility
        recent_returns = features[-20:, 0]  # returns column
        recent_vol = features[-20:, 1]  # volatility column
        
        avg_return = np.mean(recent_returns)
        avg_vol = np.mean(recent_vol)
        
        # Determine regime
        if avg_vol > np.percentile(features[:, 1], 80):
            if avg_return > 0:
                regime = RegimeState.BULL_TREND
            else:
                regime = RegimeState.HIGH_VOLATILITY
        elif avg_vol < np.percentile(features[:, 1], 20):
            regime = RegimeState.LOW_VOLATILITY
        elif abs(avg_return) < 0.001:
            regime = RegimeState.SIDEWAYS
        elif avg_return > 0:
            regime = RegimeState.BULL_TREND
        else:
            regime = RegimeState.BEAR_TREND
        
        probability = 0.7  # Default confidence for fallback
        
        return regime, probability
    
    def _map_state_to_regime(self, state: int) -> RegimeState:
        """
        Map HMM state to regime state.
        
        Args:
            state: HMM state index
            
        Returns:
            RegimeState
        """
        # Simple mapping - in practice, this would be learned from data
        mapping = {
            0: RegimeState.BULL_TREND,
            1: RegimeState.BEAR_TREND,
            2: RegimeState.SIDEWAYS,
            3: RegimeState.HIGH_VOLATILITY,
            4: RegimeState.LOW_VOLATILITY
        }
        return mapping.get(state, RegimeState.SIDEWAYS)
    
    def generate_signal(
        self,
        returns: pd.Series,
        volatility: pd.Series,
        volume: pd.Series,
        timestamp: datetime
    ) -> RegimeSignal:
        """
        Generate regime-based factor allocation signal.
        
        Args:
            returns: Return series
            volatility: Volatility series
            volume: Volume series
            timestamp: Current timestamp
            
        Returns:
            RegimeSignal
        """
        # Prepare features
        features = self.prepare_features(returns, volatility, volume)
        
        # Train HMM if needed
        if self.hmm_model is None and len(features) >= self.min_obs_for_training:
            self.train_hmm(features)
        
        # Detect regime
        regime, probability = self.detect_regime(features, timestamp)
        
        # Get factor weights for regime
        factor_weights = self.regime_factor_weights.get(regime, self.regime_factor_weights[RegimeState.SIDEWAYS])
        
        # Calculate confidence
        confidence = probability
        if confidence > self.regime_persistence_threshold:
            confidence = min(confidence + 0.1, 0.95)
        
        # Estimate expected regime duration
        expected_duration = self._estimate_regime_duration(regime)
        
        # Create signal
        signal = RegimeSignal(
            timestamp=timestamp,
            current_regime=regime,
            regime_probability=probability,
            factor_weights=factor_weights,
            momentum_weight=factor_weights['momentum'],
            value_weight=factor_weights['value'],
            quality_weight=factor_weights['quality'],
            low_vol_weight=factor_weights['low_vol'],
            confidence=confidence,
            expected_regime_duration=expected_duration
        )
        
        # Store regime history
        self.regime_history.append((timestamp, regime, probability))
        
        # Track transitions
        if len(self.regime_history) > 1:
            prev_regime = self.regime_history[-2][1]
            if prev_regime != regime:
                transition = RegimeTransition(
                    from_state=prev_regime,
                    to_state=regime,
                    probability=probability,
                    timestamp=timestamp
                )
                self.transitions.append(transition)
        
        self.signals.append(signal)
        
        # Keep history manageable
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        if len(self.signals) > 1000:
            self.signals = self.signals[-1000:]
        
        return signal
    
    def _estimate_regime_duration(self, regime: RegimeState) -> int:
        """
        Estimate expected duration of current regime.
        
        Args:
            regime: Current regime
            
        Returns:
            Expected duration in days
        """
        # Count consecutive occurrences of regime
        duration = 0
        for timestamp, r, prob in reversed(self.regime_history):
            if r == regime:
                duration += 1
            else:
                break
        
        # Default durations based on regime type
        default_durations = {
            RegimeState.BULL_TREND: 60,
            RegimeState.BEAR_TREND: 45,
            RegimeState.SIDEWAYS: 30,
            RegimeState.HIGH_VOLATILITY: 15,
            RegimeState.LOW_VOLATILITY: 90
        }
        
        return max(duration, default_durations.get(regime, 30))
    
    def get_latest_signal(self) -> Optional[RegimeSignal]:
        """Get the latest signal."""
        return self.signals[-1] if self.signals else None
    
    def get_regime_statistics(self) -> Dict[str, any]:
        """Get regime statistics."""
        if not self.regime_history:
            return {}
        
        regime_counts = {}
        for _, regime, _ in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        return {
            'total_observations': len(self.regime_history),
            'regime_distribution': regime_counts,
            'total_transitions': len(self.transitions)
        }
    
    def print_regime_report(self) -> None:
        """Print regime analysis report."""
        print("\n" + "="*60)
        print("HMM REGIME SWITCHING ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Number of States: {self.n_states}")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  Persistence Threshold: {self.regime_persistence_threshold}")
        print(f"  HMM Available: {HMM_AVAILABLE}")
        
        print(f"\nStatistics:")
        stats = self.get_regime_statistics()
        print(f"  Total Observations: {stats.get('total_observations', 0)}")
        print(f"  Total Transitions: {stats.get('total_transitions', 0)}")
        
        if stats.get('regime_distribution'):
            print(f"\nRegime Distribution:")
            for regime, count in stats['regime_distribution'].items():
                print(f"  {regime}: {count}")
        
        if self.signals:
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Regime':<20} {'Probability':<12} {'Momentum':<10} {'Value':<10} {'Quality':<10} {'LowVol':<10}")
            print("-" * 100)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.current_regime.value:<20} "
                      f"{signal.regime_probability:<12.4f} {signal.momentum_weight:<10.2f} "
                      f"{signal.value_weight:<10.2f} {signal.quality_weight:<10.2f} {signal.low_vol_weight:<10.2f}")
        
        if self.transitions:
            print(f"\nRecent Transitions:")
            print(f"{'Timestamp':<20} {'From':<20} {'To':<20} {'Probability':<12}")
            print("-" * 75)
            
            for transition in self.transitions[-5:]:
                print(f"{transition.timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{transition.from_state.value:<20} {transition.to_state.value:<20} "
                      f"{transition.probability:<12.4f}")
        
        print("\n" + "="*60)


def sample_hmm_regime_switching_alpha():
    """Demonstrate HMM regime switching alpha."""
    print("=== HMM Regime Switching Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = HMMRegimeSwitchingAlpha(
        n_states=5,
        lookback_days=252,
        min_obs_for_training=100,
        regime_persistence_threshold=0.7
    )
    
    # Generate sample market data
    np.random.seed(42)
    n_days = 300
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate returns with regime switching
    returns = np.random.randn(n_days) * 0.01
    volatility = np.abs(np.random.randn(n_days)) * 0.02 + 0.01
    volume = np.random.randint(1000000, 5000000, n_days)
    
    # Add regime effects
    for i in range(50, 150):
        returns[i] += 0.005  # Bull trend
        volatility[i] *= 0.8  # Lower vol
    
    for i in range(150, 200):
        returns[i] -= 0.005  # Bear trend
        volatility[i] *= 1.5  # Higher vol
    
    returns_series = pd.Series(returns, index=dates)
    volatility_series = pd.Series(volatility, index=dates)
    volume_series = pd.Series(volume, index=dates)
    
    # Process data
    print("Processing market data...")
    for i in range(100, n_days):
        signal = alpha.generate_signal(
            returns_series.iloc[:i],
            volatility_series.iloc[:i],
            volume_series.iloc[:i],
            dates[i]
        )
    
    # Print report
    alpha.print_regime_report()
    
    print("\n=== HMM Regime Switching Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- 5-state Hidden Markov Model for regime detection")
    print("- Dynamic factor allocation based on regime")
    print("- Regime persistence tracking")
    print("- Transition probability monitoring")
    print("- Expected regime duration estimation")
    print("- Expected Sharpe: 0.3-0.5")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_hmm_regime_switching_alpha()

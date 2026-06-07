"""
Ensemble Regime Detector - Combines HMM, Change Point, and rule-based detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..hmm.hmm_detector import HMMDetector, Regime
from ..cpd.cpd_detector import CPDDetector


class EnsembleRegimeDetector:
    """Ensemble regime detector combining multiple methods"""
    
    def __init__(self, n_states: int = 5, smoothing_window: int = 5):
        self.hmm = HMMDetector(n_components=n_states)
        self.cpd = CPDDetector()
        self.smoothing_window = smoothing_window
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit ensemble detector on features
        
        Args:
            features: DataFrame with regime features
        """
        # Fit HMM
        self.hmm.fit(features)
        self.is_fitted = True
    
    def detect(self, features: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Detect current regime
        
        Args:
            features: DataFrame with regime features (should include recent data)
            
        Returns:
            (regime_name, probability_dict)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before detection")
        
        # Get HMM prediction
        hmm_regimes, hmm_probs = self.hmm.predict(features)
        
        if len(hmm_regimes) == 0:
            return Regime.SIDEWAYS.value, {Regime.SIDEWAYS.value: 1.0}
        
        # Get change point detection
        if 'vol_21d' in features.columns:
            cp_points, vol_regime = self.cpd.detect_volatility_regime_change(
                features.get('returns_1d', pd.Series([0]))
            )
        else:
            cp_points = []
            vol_regime = "unknown"
        
        # Ensemble voting
        current_hmm_regime = hmm_regimes.iloc[-1]
        current_hmm_probs = hmm_probs.iloc[-1].to_dict()
        
        # Add volatility regime vote
        votes = current_hmm_probs.copy()
        
        if vol_regime == "high_vol":
            votes[Regime.HIGH_VOL.value] = votes.get(Regime.HIGH_VOL.value, 0) + 0.3
        elif vol_regime == "low_vol":
            votes[Regime.LOW_VOL.value] = votes.get(Regime.LOW_VOL.value, 0) + 0.3
        
        # Check for recent change point (add panic vote)
        if cp_points and len(features) > 0:
            last_cp = cp_points[-1] if cp_points else -1
            if last_cp >= len(features) - 10:  # Change point in last 10 periods
                votes[Regime.PANIC.value] = votes.get(Regime.PANIC.value, 0) + 0.5
        
        # Normalize votes
        total = sum(votes.values())
        if total > 0:
            votes = {k: v / total for k, v in votes.items()}
        
        # Select max
        final_regime = max(votes, key=votes.get)
        
        return final_regime, votes
    
    def detect_with_persistence(self, features: pd.DataFrame, 
                               history: Optional[List[str]] = None) -> Tuple[str, Dict[str, float]]:
        """
        Detect regime with persistence smoothing
        
        Args:
            features: DataFrame with regime features
            history: List of recent regime predictions
            
        Returns:
            (regime_name, probability_dict)
        """
        regime, probs = self.detect(features)
        
        if history is None or len(history) < self.smoothing_window:
            return regime, probs
        
        # Apply persistence smoothing (mode filter)
        recent_regimes = history[-self.smoothing_window:]
        
        # Count occurrences
        regime_counts = {}
        for r in recent_regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        # If current regime matches most common in history, keep it
        most_common = max(regime_counts, key=regime_counts.get)
        
        if regime_counts.get(most_common, 0) >= self.smoothing_window / 2:
            # Smooth to most common
            regime = most_common
        
        return regime, probs
    
    def get_regime_transition_probability(self, from_regime: str, to_regime: str) -> float:
        """Get probability of transitioning from one regime to another"""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted first")
        
        transmat = self.hmm.get_transition_matrix()
        
        # Map regime names to state indices
        state_mapping = {v: k for k, v in self.hmm.state_to_regime_mapping.items()}
        
        from_state = state_mapping.get(from_regime)
        to_state = state_mapping.get(to_regime)
        
        if from_state is None or to_state is None:
            return 0.0
        
        return transmat[from_state, to_state]
    
    def get_regime_persistence(self, regime: str) -> float:
        """Get expected persistence (in periods) for a regime"""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted first")
        
        persistence = self.hmm.get_state_persistence()
        return persistence.get(regime, 0.0)

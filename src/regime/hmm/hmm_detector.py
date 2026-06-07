"""
HMM Regime Detector - Hidden Markov Model for regime detection
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Regime(Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    PANIC = "panic"
    EUPHORIA = "euphoria"


class HMMDetector:
    """HMM-based regime detector"""
    
    def __init__(self, n_components: int = 5, covariance_type: str = "full"):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=100,
            random_state=42
        )
        self.state_to_regime_mapping = {
            0: Regime.BULL_TREND,
            1: Regime.BEAR_TREND,
            2: Regime.SIDEWAYS,
            3: Regime.HIGH_VOL,
            4: Regime.LOW_VOL
        }
        self.is_fitted = False
    
    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit HMM model on features
        
        Args:
            features: DataFrame with features (returns_1d, vol_21d, vix, breadth)
        """
        # Select relevant features
        feature_cols = ['returns_1d', 'vol_21d', 'vix', 'breadth']
        available_cols = [col for col in feature_cols if col in features.columns]
        
        if not available_cols:
            raise ValueError("No valid features found for HMM")
        
        X = features[available_cols].dropna().values
        
        if len(X) < 100:
            raise ValueError("Insufficient data for HMM fitting (need at least 100 points)")
        
        self.model.fit(X)
        self.is_fitted = True
        
        # Map states to regimes based on state characteristics
        self._map_states_to_regimes(features[available_cols].dropna())
    
    def predict(self, features: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Predict regime states
        
        Args:
            features: DataFrame with features
            
        Returns:
            (regime_series, probability_dataframe)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        feature_cols = ['returns_1d', 'vol_21d', 'vix', 'breadth']
        available_cols = [col for col in feature_cols if col in features.columns]
        
        X = features[available_cols].dropna().values
        
        # Predict states
        states = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        # Convert states to regimes
        regimes = pd.Series(
            [self.state_to_regime_mapping.get(s, Regime.SIDEWAYS).value for s in states],
            index=features[available_cols].dropna().index
        )
        
        # Create probability DataFrame
        prob_df = pd.DataFrame(
            probs,
            index=features[available_cols].dropna().index,
            columns=[self.state_to_regime_mapping.get(i, Regime.SIDEWAYS).value for i in range(self.n_components)]
        )
        
        return regimes, prob_df
    
    def _map_states_to_regimes(self, features: pd.DataFrame) -> None:
        """Map HMM states to regime interpretations based on feature characteristics"""
        states = self.model.predict(features.values)
        
        for state in range(self.n_components):
            mask = states == state
            state_features = features[mask]
            
            if len(state_features) == 0:
                continue
            
            mean_return = state_features['returns_1d'].mean() if 'returns_1d' in state_features else 0
            mean_vol = state_features['vol_21d'].mean() if 'vol_21d' in state_features else 0
            mean_vix = state_features['vix'].mean() if 'vix' in state_features else 0
            
            # Simple heuristic mapping
            if mean_return > 0 and mean_vol < np.percentile(features['vol_21d'], 33):
                self.state_to_regime_mapping[state] = Regime.BULL_TREND
            elif mean_return < 0 and mean_vol > np.percentile(features['vol_21d'], 67):
                self.state_to_regime_mapping[state] = Regime.BEAR_TREND
            elif mean_vol > np.percentile(features['vol_21d'], 67):
                self.state_to_regime_mapping[state] = Regime.HIGH_VOL
            elif mean_vol < np.percentile(features['vol_21d'], 33):
                self.state_to_regime_mapping[state] = Regime.LOW_VOL
            else:
                self.state_to_regime_mapping[state] = Regime.SIDEWAYS
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get state transition matrix"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.transmat_
    
    def get_state_persistence(self) -> Dict[str, float]:
        """Get average persistence (in periods) for each regime"""
        transmat = self.get_transition_matrix()
        persistence = {}
        
        for state, regime in self.state_to_regime_mapping.items():
            # Expected time in state = 1 / (1 - P(state -> state))
            diag_prob = transmat[state, state]
            persistence[regime.value] = 1 / (1 - diag_prob) if diag_prob < 1 else float('inf')
        
        return persistence

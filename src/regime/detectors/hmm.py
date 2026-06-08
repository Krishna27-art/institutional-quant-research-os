"""
HMM Regime Detector - Hidden Markov Model for regime detection
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    hmm = None
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    PANIC = "panic"


class HMMDetector:
    """Robust HMM-based regime detector with scaling, correlation control, and PCA."""
    
    def __init__(self, n_states: int = 4, covariance_type: str = "diag", n_iter: int = 500, tol: int = 1e-4, regularization: float = 1e-6):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.regularization = regularization
        self.model = None
        self.scaler = StandardScaler()
        self.state_names = ['bear', 'sideways', 'bull', 'high_vol']
        self.state_labels: dict[int, str] = {}
        self.to_drop = []
        self.pca = None
        self.is_fitted = False
        
    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        if 'close' in df.columns:
            features['ret'] = np.log(df['close'] / df['close'].shift(1))
            features['vol'] = features['ret'].rolling(20).std() * np.sqrt(252)
        else:
            # Fallback if raw returns/vol are already present
            features['ret'] = df.get('returns_1d', pd.Series(0.0, index=df.index))
            features['vol'] = df.get('vol_21d', pd.Series(0.15, index=df.index))

        if 'high' in df.columns and 'low' in df.columns:
            features['range'] = np.log(df['high'] / df['low'])
            features['range_vol'] = features['range'].rolling(20).std()
        else:
            features['range'] = 0.0
            features['range_vol'] = 0.0
            
        if 'volume' in df.columns:
            features['log_vol'] = np.log(df['volume'] / df['volume'].shift(1))
        else:
            features['log_vol'] = 0.0

        for col in ['pct_above_20ema', 'pct_above_50ema', 'pct_above_200ema', 'new_highs_20d', 'new_lows_20d', 'avg_correlation', 'vix', 'breadth']:
            if col in df.columns:
                features[col] = df[col]
        
        return features.replace([np.inf, -np.inf], np.nan).dropna()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        return self._feature_frame(df).values
        
    def _transform_features(self, X: np.ndarray, is_fit: bool = False) -> np.ndarray:
        if is_fit:
            X_scaled = self.scaler.fit_transform(X)
            corr_matrix = np.abs(np.corrcoef(X_scaled.T))
            np.fill_diagonal(corr_matrix, 0)
            to_drop = set()
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    if corr_matrix[i, j] > 0.95:
                        to_drop.add(j)
            self.to_drop = list(to_drop)
            if self.to_drop:
                X_scaled = np.delete(X_scaled, self.to_drop, axis=1)
                
            if X_scaled.shape[1] > 5:
                self.pca = PCA(n_components=0.95, svd_solver='full')
                X_scaled = self.pca.fit_transform(X_scaled)
            else:
                self.pca = None
        else:
            X_scaled = self.scaler.transform(X)
            if hasattr(self, 'to_drop') and self.to_drop:
                valid_drops = [d for d in self.to_drop if d < X_scaled.shape[1]]
                if valid_drops:
                    X_scaled = np.delete(X_scaled, valid_drops, axis=1)
            if hasattr(self, 'pca') and self.pca is not None:
                X_scaled = self.pca.transform(X_scaled)
                
        return X_scaled + self.regularization
    
    def fit(self, df: pd.DataFrame):
        X = self._prepare_features(df)
        if len(X) < 100:
            warnings.warn(f"Need at least 100 observations, got {len(X)}. Falling back to rule-based classification.")
            self.model = None
            self.is_fitted = True
            return self
        
        n_states = 3 if len(X) < 500 else self.n_states
        X_scaled = self._transform_features(X, is_fit=True)
        
        best_score = -np.inf
        best_model = None
        
        if not HMM_AVAILABLE or hmm is None:
            self.model = None
            self.is_fitted = True
            return self

        for seed in range(5):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=42 + seed,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    model.fit(X_scaled)
                    score = model.score(X_scaled)
                    if score > best_score:
                        best_score = score
                        best_model = model
            except Exception:
                continue
        
        if best_model is None:
            self.model = None
        else:
            self.model = best_model
            self.n_states = n_states
            self._reorder_states(df)
            
        self.is_fitted = True
        return self
        
    def _reorder_states(self, df: pd.DataFrame):
        features = self._feature_frame(df)
        if features.empty or self.model is None:
            self.state_labels = {i: self.state_names[min(i, len(self.state_names) - 1)] for i in range(self.n_states)}
            return

        X = self._prepare_features(df)
        X_scaled = self._transform_features(X, is_fit=False)
        states = self.model.predict(X_scaled)
        stats_by_state = []
        for state in range(self.n_states):
            mask = states == state
            if not mask.any():
                stats_by_state.append((state, 0.0, -np.inf))
                continue
            stats_by_state.append((
                state,
                float(features.loc[mask, 'ret'].mean()),
                float(features.loc[mask, 'vol'].mean()),
            ))

        labels: dict[int, str] = {}
        if self.n_states == 1:
            labels[stats_by_state[0][0]] = 'sideways'
        else:
            high_vol_state = max(stats_by_state, key=lambda item: item[2])[0]
            labels[high_vol_state] = 'high_vol'
            remaining = [item for item in stats_by_state if item[0] != high_vol_state]
            remaining_sorted = sorted(remaining, key=lambda item: item[1])
            labels[remaining_sorted[0][0]] = 'bear'
            if len(remaining_sorted) > 1:
                labels[remaining_sorted[-1][0]] = 'bull'
            for state, _, _ in remaining_sorted[1:-1]:
                labels[state] = 'sideways'

        self.state_labels = labels
        ordered_labels = []
        for label in ['bear', 'sideways', 'bull', 'high_vol']:
            if label in labels.values():
                ordered_labels.append(label)
        self.state_names = ordered_labels
    
    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        if self.model is None:
            # Rule-based fallback
            features = self._feature_frame(df)
            regimes = []
            for idx, row in features.iterrows():
                vol = row.get('vol', 0.0)
                ret_5d = features.loc[:idx, 'ret'].tail(5).sum()
                if vol >= 0.25:
                    regimes.append('high_vol')
                elif ret_5d >= 0.01:
                    regimes.append('bull')
                elif ret_5d <= -0.01:
                    regimes.append('bear')
                else:
                    regimes.append('sideways')
            return pd.Series(regimes, index=features.index)

        X = self._prepare_features(df)
        X_scaled = self._transform_features(X, is_fit=False)
        states = self.model.predict(X_scaled)
        regime_names = [self.state_labels.get(int(s), 'sideways') for s in states]
        idx = self._feature_frame(df).index
        return pd.Series(regime_names, index=idx)

    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Expose prediction method for Ensemble compatibility."""
        regimes = self.predict_regime(df)
        probs = self.regime_probabilities(df)
        return regimes, probs
    
    def regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self._feature_frame(df)
        if self.model is None:
            probs_list = []
            regimes = self.predict_regime(df)
            for idx, reg in regimes.items():
                prob = {'bear': 0.1, 'sideways': 0.1, 'bull': 0.1, 'high_vol': 0.1}
                prob[reg] = 0.7
                total = sum(prob.values())
                prob = {k: v/total for k, v in prob.items()}
                probs_list.append(prob)
            return pd.DataFrame(probs_list, index=features.index)

        X = self._prepare_features(df)
        X_scaled = self._transform_features(X, is_fit=False)
        probs = self.model.predict_proba(X_scaled)
        raw_cols = [self.state_labels.get(i, f"state_{i}") for i in range(self.n_states)]
        probs_df = pd.DataFrame(probs, columns=raw_cols, index=features.index)
        return probs_df.T.groupby(level=0).sum().T
    
    def confidence(self, df: pd.DataFrame) -> float:
        probs_df = self.regime_probabilities(df)
        if probs_df.empty:
            return 0.5
        latest = probs_df.iloc[-1]
        entropy = - (latest * np.log(latest + 1e-9)).sum()
        max_entropy = np.log(max(len(latest), 2))
        confidence = 1 - (entropy / max_entropy)
        return float(np.clip(confidence, 0.0, 1.0))

    def get_transition_matrix(self) -> np.ndarray:
        if self.model is None:
            return np.eye(self.n_states)
        return self.model.transmat_

    def get_state_persistence(self) -> Dict[str, float]:
        if self.model is None:
            return {name: 10.0 for name in self.state_names}
        transmat = self.get_transition_matrix()
        persistence = {}
        for state, label in self.state_labels.items():
            diag_prob = transmat[state, state]
            persistence[label] = 1 / (1 - diag_prob) if diag_prob < 1 else float('inf')
        return persistence


RobustHMMRegime = HMMDetector

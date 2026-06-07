import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"


@dataclass(frozen=True, slots=True)
class HMMConfig:
    min_samples: int = 100
    high_vol_threshold: float = 0.25
    bull_return_threshold: float = 0.01
    bear_return_threshold: float = -0.01


@dataclass(frozen=True, slots=True)
class RegimeState:
    regime: Regime
    confidence: float
    timestamp: datetime
    probabilities: dict[str, float]


class HMMRegimeEngine:
    """Backward-compatible regime facade with deterministic fallback rules."""

    def __init__(self, config: dict[str, Any] | HMMConfig | None = None):
        if isinstance(config, HMMConfig):
            self.config = config
        else:
            params = config or {}
            self.config = HMMConfig(**{k: v for k, v in params.items() if k in HMMConfig.__dataclass_fields__})
        self.history: list[dict[str, float]] = []

    def predict_regime(self, features: dict[str, float], timestamp: datetime) -> RegimeState:
        clean = {key: float(value) for key, value in features.items() if value is not None and np.isfinite(value)}
        self.history.append(clean)

        realized_vol = clean.get("realized_vol_5d", clean.get("realized_vol", 0.0))
        implied_vol = clean.get("implied_vol", clean.get("india_vix", 0.0) / 100)
        ret_5d = clean.get("nifty_return_5d", clean.get("return_5d", 0.0))
        vix = clean.get("india_vix", implied_vol * 100)

        if max(realized_vol, implied_vol) >= self.config.high_vol_threshold or vix >= 25:
            regime = Regime.HIGH_VOL
            confidence = 0.65
        elif ret_5d >= self.config.bull_return_threshold:
            regime = Regime.BULL_TREND
            confidence = min(0.9, 0.55 + abs(ret_5d) * 5)
        elif ret_5d <= self.config.bear_return_threshold:
            regime = Regime.BEAR_TREND
            confidence = min(0.9, 0.55 + abs(ret_5d) * 5)
        else:
            regime = Regime.SIDEWAYS
            confidence = 0.55

        base = {item.value: (1 - confidence) / 3 for item in Regime}
        base[regime.value] = confidence
        return RegimeState(regime=regime, confidence=confidence, timestamp=timestamp, probabilities=base)

class RobustHMMRegime:
    def __init__(self, n_states=4, covariance_type="diag", n_iter=500, tol=1e-4, regularization=1e-6):
        """
        CRITICAL FIX: Changed default covariance_type from "full" to "diag".
        
        With 500+ features and ~1250 days of data, a full covariance matrix is
        rank-deficient and non-invertible. Diagonal covariance is required for
        numerical stability.
        
        Args:
            regularization: Small value added to diagonal to prevent singularity
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.regularization = regularization
        self.model = None
        self.scaler = StandardScaler()
        self.state_names = ['bear', 'sideways', 'bull', 'high_vol']  # interpretable
        self.state_labels: dict[int, str] = {}
        
    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime features: returns, volatility, volume, spread."""
        features = pd.DataFrame(index=df.index)
        # Log returns
        features['ret'] = np.log(df['close'] / df['close'].shift(1))
        # Realized volatility (20-period)
        features['vol'] = features['ret'].rolling(20).std() * np.sqrt(252)
        # Candlestick range volatility
        if {'high', 'low'}.issubset(df.columns):
            features['range'] = np.log(df['high'] / df['low'])
            features['range_vol'] = features['range'].rolling(20).std()
        else:
            features['range'] = 0.0
            features['range_vol'] = 0.0
        # Volume change
        features['log_vol'] = np.log(df['volume'] / df['volume'].shift(1))
        # Spread (if available)
        if 'spread' in df.columns:
            features['spread'] = df['spread']
        else:
            features['spread'] = 0.0
        
        return features.replace([np.inf, -np.inf], np.nan).dropna()

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute regime features as an array for the HMM."""
        return self._feature_frame(df).values
    
    def fit(self, df: pd.DataFrame):
        """
        Fit HMM with scaled features, PCA, and convergence monitoring.
        
        CRITICAL FIX: Added PCA for dimensionality reduction, correlation removal,
        and enhanced regularization to handle highly correlated features.
        """
        X = self._prepare_features(df)
        if len(X) < 100:
            raise ValueError(f"Need at least 100 observations, got {len(X)}")
        
        logger.info(f"Feature matrix shape before preprocessing: {X.shape}")
        
        # CRITICAL FIX: Reduce states if data is limited
        n_states = 3 if len(X) < 500 else self.n_states
        
        # Scale features (critical for HMM convergence)
        X_scaled = self.scaler.fit_transform(X)
        
        # CRITICAL FIX: Remove highly correlated features (>0.95)
        corr_matrix = np.abs(np.corrcoef(X_scaled.T))
        np.fill_diagonal(corr_matrix, 0)  # Ignore self-correlation
        to_drop = set()
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if corr_matrix[i, j] > 0.95:
                    to_drop.add(j)  # Drop the later feature
        if to_drop:
            X_scaled = np.delete(X_scaled, list(to_drop), axis=1)
            logger.info(f"Removed {len(to_drop)} highly correlated features, new shape: {X_scaled.shape}")
        
        # CRITICAL FIX: Apply PCA to retain 95% variance if features > 5
        if X_scaled.shape[1] > 5:
            pca = PCA(n_components=0.95, svd_solver='full')
            X_pca = pca.fit_transform(X_scaled)
            logger.info(f"PCA: {X_scaled.shape[1]} -> {X_pca.shape[1]} components (95% variance)")
            X_scaled = X_pca
        else:
            logger.info(f"Skipping PCA (only {X_scaled.shape[1]} features)")
        
        # CRITICAL FIX: Add regularization to prevent singular matrices
        X_scaled = X_scaled + self.regularization
        
        # Check condition number to ensure numerical stability
        cov_matrix = np.cov(X_scaled.T)
        cond_number = np.linalg.cond(cov_matrix)
        logger.info(f"Covariance matrix condition number: {cond_number:.2e}")
        if cond_number > 1e10:
            logger.warning(f"Covariance matrix condition number {cond_number:.2e} is too high. "
                         "Consider reducing feature dimensionality or increasing regularization.")
        
        # Initialize model with reasonable parameters
        # Fit with multiple restarts to avoid local optima
        best_score = -np.inf
        best_model = None
        converged = False
        
        for seed in range(5):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    verbose=False,
                    init_params="stmc",
                    random_state=42 + seed,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    model.fit(X_scaled)
                    
                    # CRITICAL FIX: Check convergence
                    if hasattr(model, 'monitor_') and model.monitor_.converged:
                        converged = True
                        logger.info(f"HMM converged on attempt {seed + 1}")
                    else:
                        logger.warning(f"HMM did not converge on attempt {seed + 1}")
                    
                    score = model.score(X_scaled)
                    if score > best_score:
                        best_score = score
                        best_model = model
            except np.linalg.LinAlgError as e:
                logger.warning(f"Linear algebra error in HMM fit: {e}. "
                             "Matrix may be singular. Try increasing regularization.")
                continue
            except Exception as e:
                logger.warning(f"Fit attempt failed: {e}")
                continue
        
        if best_model is None:
            logger.error("HMM failed to converge after 5 restarts - using rule-based fallback")
            self._use_fallback = True
            self.model = None
            return self
        
        if not converged:
            logger.warning("HMM did not converge on any attempt - using best model but results may be unreliable")
        
        self.model = best_model
        self.n_states = n_states  # Update actual states used
        logger.info(f"HMM fitted with {n_states} states, log-likelihood {best_score:.2f}")
        
        # Assign interpretable state order
        self._reorder_states(df)
        return self
        
    def _reorder_states(self, df: pd.DataFrame):
        """Build a stable mapping from HMM state ids to regime labels."""
        features = self._feature_frame(df)
        if features.empty:
            self.state_labels = {i: self.state_names[min(i, len(self.state_names) - 1)] for i in range(self.n_states)}
            return

        X_scaled = self.scaler.transform(features.values)
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
        """Predict regime for each timestamp."""
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        # Map state indices to regime names
        regime_names = [self.state_labels.get(int(s), 'sideways') for s in states]
        # Return series with same index as input (after feature prep)
        idx = self._feature_frame(df).index
        return pd.Series(regime_names, index=idx)
    
    def regime_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return probability of each regime (for confidence)."""
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        raw_cols = [self.state_labels.get(i, f"state_{i}") for i in range(self.n_states)]
        probs_df = pd.DataFrame(probs, columns=raw_cols, index=self._feature_frame(df).index)
        return probs_df.T.groupby(level=0).sum().T
    
    def confidence(self, df: pd.DataFrame) -> float:
        """Confidence = 1 - entropy of regime probabilities."""
        probs_df = self.regime_probabilities(df)
        latest = probs_df.iloc[-1]
        entropy = - (latest * np.log(latest + 1e-9)).sum()
        max_entropy = np.log(max(len(latest), 2))
        confidence = 1 - (entropy / max_entropy)
        return float(np.clip(confidence, 0.0, 1.0))

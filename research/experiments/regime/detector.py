"""
Regime Detection Engine
Implements HMM + Change Point Detection ensemble for market regime classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from hmmlearn import hmm
import ruptures as rpt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from arch import arch_model

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime types"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    PANIC = "panic"
    EUPHORIA = "euphoria"
    LIQUIDITY_EXPANSION = "liquidity_expansion"
    LIQUIDITY_CONTRACTION = "liquidity_contraction"


@dataclass
class RegimeState:
    """Regime state information"""
    regime: RegimeType
    probability: float
    confidence: float
    features: Dict[str, float]
    timestamp: pd.Timestamp


class HMMRegimeDetector:
    """Hidden Markov Model for regime detection"""
    
    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize HMM regime detector
        
        Args:
            n_components: Number of hidden states
            covariance_type: Type of covariance matrix
            n_iter: Number of EM iterations
            random_state: Random seed
        """
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.state_mapping = {}  # Map HMM states to regime types
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for HMM"""
        features = []
        
        # Returns at multiple horizons
        close = df['close'].values
        for window in [1, 5, 10, 20]:
            if len(close) > window:
                ret = np.log(close[-1] / close[-window-1])
                features.append(ret)
        
        # Volatility
        if len(close) > 20:
            vol = pd.Series(close).pct_change().rolling(20).std()[-1]
            features.append(vol)
        
        # Price vs MA
        if len(close) > 200:
            ma200 = close[-200:].mean()
            features.append((close[-1] - ma200) / ma200)
        
        if len(close) > 50:
            ma50 = close[-50:].mean()
            features.append((close[-1] - ma50) / ma50)
        
        # Volume (if available)
        if 'volume' in df.columns:
            volume = df['volume'].values
            if len(volume) > 20:
                vol_ratio = volume[-1] / volume[-20:].mean()
                features.append(vol_ratio)
        
        # High-Low range
        if 'high' in df.columns and 'low' in df.columns:
            high = df['high'].values
            low = df['low'].values
            if len(high) > 1 and len(low) > 1:
                hl_range = (high[-1] - low[-1]) / close[-1]
                features.append(hl_range)
        
        return np.array(features).reshape(1, -1)
    
    def fit(self, df: pd.DataFrame, min_samples: int = 252) -> bool:
        """
        Fit HMM model on historical data
        
        Args:
            df: Historical price data
            min_samples: Minimum samples required
        """
        if len(df) < min_samples:
            logger.warning(f"Insufficient data for HMM fitting: {len(df)} < {min_samples}")
            return False
        
        try:
            # Extract features for all time points
            feature_matrix = []
            for i in range(min_samples, len(df)):
                features = self.extract_features(df.iloc[:i])
                feature_matrix.append(features[0])
            
            feature_matrix = np.array(feature_matrix)
            
            # Scale features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Fit HMM
            self.model.fit(feature_matrix_scaled)
            self.is_fitted = True
            
            # Map states to regimes based on emission parameters
            self._map_states_to_regimes(feature_matrix_scaled)
            
            logger.info(f"HMM fitted successfully with {self.n_components} states")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            return False
    
    def _map_states_to_regimes(self, features: np.ndarray):
        """Map HMM states to regime types based on emission parameters"""
        # Get state means
        state_means = self.model.means_
        
        # Sort states by mean return (proxy for trend)
        sorted_indices = np.argsort(state_means[:, 0])  # Sort by first feature (1-day return)
        
        # Map to regimes
        self.state_mapping = {
            sorted_indices[0]: RegimeType.BEAR_TREND,
            sorted_indices[1]: RegimeType.LOW_VOLATILITY,
            sorted_indices[2]: RegimeType.SIDEWAYS,
            sorted_indices[3]: RegimeType.BULL_TREND,
            sorted_indices[4]: RegimeType.HIGH_VOLATILITY
        }
        
        logger.info(f"State mapping: {self.state_mapping}")
    
    def predict(self, df: pd.DataFrame) -> Optional[RegimeType]:
        """Predict current regime"""
        if not self.is_fitted:
            logger.warning("HMM not fitted")
            return None
        
        try:
            features = self.extract_features(df)
            features_scaled = self.scaler.transform(features)
            
            # Get state probabilities
            state_probs = self.model.predict_proba(features_scaled)[0]
            
            # Get most likely state
            state = np.argmax(state_probs)
            
            # Map to regime
            regime = self.state_mapping.get(state, RegimeType.SIDEWAYS)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return RegimeType.SIDEWAYS
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[Optional[RegimeType], float]:
        """Predict regime with confidence score"""
        if not self.is_fitted:
            return None, 0.0
        
        try:
            features = self.extract_features(df)
            features_scaled = self.scaler.transform(features)
            
            # Get state probabilities
            state_probs = self.model.predict_proba(features_scaled)[0]
            
            # Get most likely state
            state = np.argmax(state_probs)
            confidence = state_probs[state]
            
            # Map to regime
            regime = self.state_mapping.get(state, RegimeType.SIDEWAYS)
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error predicting regime with confidence: {e}")
            return RegimeType.SIDEWAYS, 0.0


class ChangePointDetector:
    """Change point detection for structural breaks"""
    
    def __init__(self, model: str = "l2", min_size: int = 20):
        """
        Initialize change point detector
        
        Args:
            model: Cost function model (l2, rbf, linear)
            min_size: Minimum segment size
        """
        self.model = model
        self.min_size = min_size
        self.detector = rpt.Binseg(model=model, min_size=min_size)
    
    def detect_change_points(self, returns: np.ndarray, n_bkps: int = 5) -> List[int]:
        """
        Detect change points in returns
        
        Args:
            returns: Return series
            n_bkps: Number of change points to detect
        """
        try:
            # Detect change points
            self.detector.fit(returns)
            change_points = self.detector.predict(n_bkps=n_bkps)
            
            return change_points[:-1]  # Exclude last point (end of series)
            
        except Exception as e:
            logger.error(f"Error detecting change points: {e}")
            return []
    
    def has_recent_change(self, returns: np.ndarray, window: int = 20) -> bool:
        """Check if there's a recent change point"""
        change_points = self.detect_change_points(returns, n_bkps=3)
        
        if not change_points:
            return False
        
        # Check if any change point is in the recent window
        recent_change = any(cp >= len(returns) - window for cp in change_points)
        return recent_change


class GARCHVolatilityDetector:
    """GARCH-based volatility regime detection"""
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH detector
        
        Args:
            p: GARCH p parameter
            q: GARCH q parameter
        """
        self.p = p
        self.q = q
        self.model = None
        self.is_fitted = False
    
    def fit(self, returns: pd.Series) -> bool:
        """Fit GARCH model"""
        try:
            # Scale returns (multiply by 100 for percentage)
            returns_scaled = returns * 100
            
            # Fit GARCH(1,1)
            self.model = arch_model(returns_scaled, vol='Garch', p=self.p, q=self.q)
            self.model = self.model.fit(disp='off')
            self.is_fitted = True
            
            logger.info("GARCH model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting GARCH: {e}")
            return False
    
    def predict_volatility(self, returns: pd.Series) -> Optional[float]:
        """Predict conditional volatility"""
        if not self.is_fitted:
            return None
        
        try:
            # Get forecast
            forecast = self.model.forecast(horizon=1)
            vol = np.sqrt(forecast.variance.values[-1, :][0]) / 100  # Convert back
            return vol
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return None
    
    def get_volatility_regime(self, returns: pd.Series) -> str:
        """Classify volatility regime"""
        if not self.is_fitted:
            return "medium"
        
        try:
            vol = self.predict_volatility(returns)
            
            if vol is None:
                return "medium"
            
            # Classify based on historical percentiles
            historical_vol = returns.rolling(20).std()
            
            if vol < historical_vol.quantile(0.33):
                return "low"
            elif vol > historical_vol.quantile(0.67):
                return "high"
            else:
                return "medium"
                
        except Exception as e:
            logger.error(f"Error getting volatility regime: {e}")
            return "medium"


class RegimeDetectorEnsemble:
    """Ensemble regime detector combining HMM, Change Point, and GARCH"""
    
    def __init__(
        self,
        hmm_components: int = 5,
        cp_model: str = "l2",
        garch_p: int = 1,
        garch_q: int = 1
    ):
        """
        Initialize ensemble detector
        
        Args:
            hmm_components: Number of HMM states
            cp_model: Change point model
            garch_p: GARCH p parameter
            garch_q: GARCH q parameter
        """
        self.hmm_detector = HMMRegimeDetector(n_components=hmm_components)
        self.cp_detector = ChangePointDetector(model=cp_model)
        self.garch_detector = GARCHVolatilityDetector(p=garch_p, q=garch_q)
        
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, min_samples: int = 252) -> bool:
        """Fit all detectors"""
        success = True
        
        # Fit HMM
        if not self.hmm_detector.fit(df, min_samples):
            success = False
        
        # Fit GARCH
        if len(df) > 100:
            returns = df['close'].pct_change().dropna()
            if not self.garch_detector.fit(returns):
                success = False
        
        self.is_fitted = success
        return success
    
    def detect_regime(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect current regime using ensemble
        
        Args:
            df: Current market data
        """
        if not self.is_fitted:
            logger.warning("Ensemble not fitted, returning default regime")
            return RegimeState(
                regime=RegimeType.SIDEWAYS,
                probability=0.5,
                confidence=0.0,
                features={},
                timestamp=pd.Timestamp.now()
            )
        
        # Get HMM prediction
        hmm_regime, hmm_confidence = self.hmm_detector.predict_with_confidence(df)
        
        # Get change point info
        returns = df['close'].pct_change().dropna()
        recent_change = self.cp_detector.has_recent_change(returns.values, window=20)
        
        # Get volatility regime
        vol_regime = self.garch_detector.get_volatility_regime(returns)
        
        # Ensemble decision
        final_regime = self._combine_predictions(
            hmm_regime, hmm_confidence, recent_change, vol_regime
        )
        
        # Calculate ensemble confidence
        ensemble_confidence = self._calculate_confidence(
            hmm_confidence, recent_change
        )
        
        # Extract features for logging
        features = {
            'hmm_regime': hmm_regime.value if hmm_regime else 'unknown',
            'hmm_confidence': hmm_confidence,
            'recent_change': recent_change,
            'vol_regime': vol_regime
        }
        
        return RegimeState(
            regime=final_regime,
            probability=hmm_confidence,
            confidence=ensemble_confidence,
            features=features,
            timestamp=pd.Timestamp.now()
        )
    
    def _combine_predictions(
        self,
        hmm_regime: Optional[RegimeType],
        hmm_confidence: float,
        recent_change: bool,
        vol_regime: str
    ) -> RegimeType:
        """Combine predictions from all detectors"""
        
        # If recent structural break, be conservative
        if recent_change:
            if vol_regime == "high":
                return RegimeType.PANIC
            elif vol_regime == "low":
                return RegimeType.SIDEWAYS
            else:
                return RegimeType.SIDEWAYS
        
        # If HMM confidence is high, trust it
        if hmm_confidence > 0.7 and hmm_regime:
            return hmm_regime
        
        # Otherwise, combine with volatility regime
        if vol_regime == "high":
            if hmm_regime in [RegimeType.BEAR_TREND, RegimeType.BULL_TREND]:
                return RegimeType.HIGH_VOLATILITY
            return RegimeType.HIGH_VOLATILITY
        elif vol_regime == "low":
            if hmm_regime == RegimeType.BULL_TREND:
                return RegimeType.EUPHORIA
            return RegimeType.LOW_VOLATILITY
        
        # Default to HMM prediction
        return hmm_regime if hmm_regime else RegimeType.SIDEWAYS
    
    def _calculate_confidence(self, hmm_confidence: float, recent_change: bool) -> float:
        """Calculate ensemble confidence"""
        
        # Reduce confidence if recent change
        if recent_change:
            return hmm_confidence * 0.5
        
        return hmm_confidence
    
    def get_regime_history(self, df: pd.DataFrame, window: int = 252) -> List[RegimeState]:
        """Get regime history for backtesting"""
        history = []
        
        for i in range(window, len(df)):
            subset = df.iloc[:i]
            regime_state = self.detect_regime(subset)
            history.append(regime_state)
        
        return history


class RegimeBasedWeights:
    """Regime-specific alpha weights"""
    
    WEIGHTS = {
        RegimeType.BULL_TREND: {
            'momentum': 0.50,
            'mean_reversion': 0.05,
            'volatility': 0.15,
            'options': 0.10,
            'microstructure': 0.10,
            'factor': 0.10
        },
        RegimeType.BEAR_TREND: {
            'momentum': 0.40,
            'mean_reversion': 0.10,
            'volatility': 0.20,
            'options': 0.15,
            'microstructure': 0.05,
            'factor': 0.10
        },
        RegimeType.SIDEWAYS: {
            'momentum': 0.10,
            'mean_reversion': 0.50,
            'volatility': 0.15,
            'options': 0.15,
            'microstructure': 0.05,
            'factor': 0.05
        },
        RegimeType.HIGH_VOLATILITY: {
            'momentum': 0.20,
            'mean_reversion': 0.10,
            'volatility': 0.40,
            'options': 0.20,
            'microstructure': 0.05,
            'factor': 0.05
        },
        RegimeType.LOW_VOLATILITY: {
            'momentum': 0.30,
            'mean_reversion': 0.20,
            'volatility': 0.10,
            'options': 0.10,
            'microstructure': 0.20,
            'factor': 0.10
        },
        RegimeType.PANIC: {
            'momentum': 0.00,
            'mean_reversion': 0.20,
            'volatility': 0.30,
            'options': 0.30,
            'microstructure': 0.10,
            'factor': 0.10
        },
        RegimeType.EUPHORIA: {
            'momentum': 0.20,
            'mean_reversion': 0.20,
            'volatility': 0.10,
            'options': 0.10,
            'microstructure': 0.20,
            'factor': 0.20
        }
    }
    
    @classmethod
    def get_weights(cls, regime: RegimeType) -> Dict[str, float]:
        """Get alpha weights for a regime"""
        return cls.WEIGHTS.get(regime, cls.WEIGHTS[RegimeType.SIDEWAYS])

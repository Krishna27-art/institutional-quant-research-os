"""
Hybrid Regime Detection: HMM + VVG Classifier
For NIFTY/BANKNIFTY and Indian equity markets.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats

logger = logging.getLogger(__name__)


class Regime(Enum):
    BULL_TRENDING = "bull_trending"
    BULL_VOLATILE = "bull_volatile"
    BEAR_TRENDING = "bear_trending"
    BEAR_VOLATILE = "bear_volatile"
    SIDEWAYS_CALM = "sideways_calm"
    SIDEWAYS_VOLATILE = "sideways_volatile"
    CRISIS = "crisis"


class StrategyType(Enum):
    ORB = "orb"
    VWAP_TREND = "vwap_trend"
    VWAP_REVERSION = "vwap_reversion"
    GCN_MOMENTUM = "gcn_momentum"
    GCN_REVERSION = "gcn_reversion"
    RISK_OFF = "risk_off"


@dataclass
class RegimeState:
    """Current regime state with confidence and strategy recommendations."""
    regime: Regime
    probability: float
    hmm_state: int
    vvg_label: str
    recommended_strategies: List[StrategyType]
    position_multiplier: float
    volatility_estimate: float


class HMMRegimeDetector:
    """
    Hidden Markov Model for long-term regime detection.
    Uses daily log-returns to identify Bull/Bear/Sideways states.

    Based on research showing HMM outperforms simple moving average
    classification for regime identification in emerging markets.
    """

    def __init__(
        self,
        n_components: int = 3,
        lookback: int = 252,
        covariance_type: str = "full",
        n_iterations: int = 200,
        random_state: int = 42
    ):
        self.n_components = n_components
        self.lookback = lookback
        self.covariance_type = covariance_type
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self._state_mapping: Dict[int, Regime] = {}
        self._is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for HMM.
        Features: log returns, squared returns (vol proxy),
        realized volatility.
        """
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

        features = np.column_stack([
            returns.values,
            returns.values ** 2,
            returns.rolling(21).std().dropna().values,
        ])

        # Remove NaN rows
        features = features[~np.isnan(features).any(axis=1)]

        return features

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit HMM on historical daily data.

        Args:
            df: Daily OHLCV DataFrame for NIFTY/BANKNIFTY
        """
        features = self._prepare_features(df)

        if len(features) < self.lookback:
            logger.warning(
                f"Insufficient data for HMM: {len(features)} < {self.lookback}"
            )
            features = features  # Use what we have

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iterations,
            random_state=self.random_state,
            tol=1e-6,
        )

        try:
            self.model.fit(features)
            self._map_states(df)
            self._is_fitted = True
            logger.info(
                f"HMM fitted with {self.n_components} states "
                f"on {len(features)} samples"
            )
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            raise

        return self

    def _map_states(self, df: pd.DataFrame) -> None:
        """
        Map HMM states to semantic regimes based on
        mean return and volatility of each state.
        """
        if not self.model:
            return

        means = self.model.means_[:, 0]  # Mean return for each state
        sorted_indices = np.argsort(means)  # Low to high returns

        # Map: lowest return = Bear, middle = Sideways, highest = Bull
        if self.n_components == 3:
            self._state_mapping = {
                sorted_indices[0]: Regime.BEAR_TRENDING,
                sorted_indices[1]: Regime.SIDEWAYS_CALM,
                sorted_indices[2]: Regime.BULL_TRENDING,
            }
        elif self.n_components == 5:
            self._state_mapping = {
                sorted_indices[0]: Regime.CRISIS,
                sorted_indices[1]: Regime.BEAR_VOLATILE,
                sorted_indices[2]: Regime.SIDEWAYS_VOLATILE,
                sorted_indices[3]: Regime.BULL_VOLATILE,
                sorted_indices[4]: Regime.BULL_TRENDING,
            }
        else:
            # Generic mapping
            for i, idx in enumerate(sorted_indices):
                if i < len(sorted_indices) // 2:
                    self._state_mapping[idx] = Regime.BEAR_TRENDING
                elif i == len(sorted_indices) // 2:
                    self._state_mapping[idx] = Regime.SIDEWAYS_CALM
                else:
                    self._state_mapping[idx] = Regime.BULL_TRENDING

        logger.info(f"State mapping: {self._state_mapping}")

    def predict(self, df: pd.DataFrame) -> Tuple[Regime, np.ndarray]:
        """
        Predict current regime and state probabilities.

        Returns:
            (current_regime, state_probabilities)
        """
        if not self._is_fitted:
            raise RuntimeError("HMM not fitted. Call fit() first.")

        features = self._prepare_features(df)

        try:
            state_probs = self.model.predict_proba(features[-1].reshape(1, -1))[0]
            current_state = np.argmax(state_probs)
            regime = self._state_mapping.get(current_state, Regime.SIDEWAYS_CALM)

            return regime, state_probs

        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return Regime.SIDEWAYS_CALM, np.ones(self.n_components) / self.n_components

    def get_expected_duration(self, state: int) -> float:
        """Expected duration (in days) of each regime state."""
        if not self.model:
            return 0.0
        return 1.0 / (1.0 - self.model.transmat_[state, state])


class VVGClassifier:
    """
    Volatility-Volume-Gap Classifier for intraday regime detection.

    Determines whether the current intraday session favors:
    - Sequential/trending patterns (good for ORB, VWAP trend)
    - Mean-reverting patterns (good for VWAP reversion)
    - Choppy/noise (avoid trading)

    Based on research showing that gap size + opening volume
    predicts intraday regime with >60% accuracy.
    """

    def __init__(
        self,
        gap_threshold: float = 0.005,
        volume_percentile: int = 70,
        volatility_percentile: int = 60,
        lookback: int = 60
    ):
        self.gap_threshold = gap_threshold
        self.volume_percentile = volume_percentile
        self.volatility_percentile = volatility_percentile
        self.lookback = lookback
        self._historical_stats = {}

    def _compute_gap(self, df: pd.DataFrame) -> float:
        """
        Compute opening gap as percentage of previous close.
        Positive = gap up, Negative = gap down.
        """
        if len(df) < 2:
            return 0.0
        prev_close = df["Close"].iloc[-2]
        today_open = df["Open"].iloc[-1]
        return (today_open - prev_close) / prev_close

    def _compute_opening_volume_rank(self, df: pd.DataFrame) -> float:
        """
        Rank today's opening volume vs historical.
        Returns percentile rank [0, 1].
        """
        if len(df) < self.lookback:
            return 0.5

        current_vol = df["Volume"].iloc[-1]
        historical_vols = df["Volume"].iloc[-self.lookback:-1].values
        return stats.percentileofscore(historical_vols, current_vol) / 100

    def _compute_volatility_rank(self, df: pd.DataFrame) -> float:
        """
        Rank current volatility vs historical.
        Uses 5-day realized volatility.
        """
        if len(df) < self.lookback + 5:
            return 0.5

        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        current_vol = returns.tail(5).std()
        historical_vols = pd.Series([
            returns.iloc[i:i+5].std()
            for i in range(len(returns) - self.lookback, len(returns) - 5)
        ])

        if len(historical_vols) == 0 or historical_vols.std() == 0:
            return 0.5

        return stats.percentileofscore(historical_vols, current_vol) / 100

    def classify(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """
        Classify current intraday regime.

        Returns:
            (regime_label, features_dict)
        """
        gap = self._compute_gap(df)
        vol_rank = self._compute_opening_volume_rank(df)
        volat_rank = self._compute_volatility_rank(df)

        features = {
            "gap": gap,
            "gap_abs": abs(gap),
            "volume_rank": vol_rank,
            "volatility_rank": volat_rank,
        }

        # Classification logic
        large_gap = abs(gap) > self.gap_threshold
        high_volume = vol_rank > self.volume_percentile / 100
        high_volatility = volat_rank > self.volatility_percentile / 100
        gap_direction = "up" if gap > 0 else "down"

        if large_gap and high_volume:
            label = "trending_strong"
        elif large_gap and not high_volume:
            label = "trending_weak"
        elif not large_gap and high_volume and high_volatility:
            label = "choppy_volatile"
        elif not large_gap and not high_volume:
            label = "choppy_calm"
        elif high_volume and not large_gap:
            label = "rotation"
        else:
            label = "trending_moderate"

        features["label"] = label
        features["gap_direction"] = gap_direction

        return label, features


class HybridRegimeEngine:
    """
    Combines HMM (long-term) and VVG (intraday) classifiers
    into a unified regime state with strategy recommendations.
    """

    # Regime -> Strategy mapping
    REGIME_STRATEGY_MAP = {
        (Regime.BULL_TRENDING, "trending_strong"): [
            StrategyType.ORB, StrategyType.VWAP_TREND,
            StrategyType.GCN_MOMENTUM
        ],
        (Regime.BULL_TRENDING, "trending_moderate"): [
            StrategyType.VWAP_TREND, StrategyType.GCN_MOMENTUM
        ],
        (Regime.BULL_TRENDING, "choppy_volatile"): [
            StrategyType.VWAP_REVERSION, StrategyType.GCN_REVERSION
        ],
        (Regime.BULL_TRENDING, "choppy_calm"): [
            StrategyType.VWAP_REVERSION
        ],
        (Regime.BEAR_TRENDING, "trending_strong"): [
            StrategyType.ORB, StrategyType.VWAP_TREND
        ],
        (Regime.BEAR_TRENDING, "choppy_volatile"): [
            StrategyType.RISK_OFF
        ],
        (Regime.SIDEWAYS_CALM, "choppy_calm"): [
            StrategyType.VWAP_REVERSION
        ],
        (Regime.SIDEWAYS_VOLATILE, "_"): [  # Any intraday
            StrategyType.RISK_OFF
        ],
        (Regime.CRISIS, "_"): [
            StrategyType.RISK_OFF
        ],
    }

    def __init__(self, config: dict):
        self.config = config
        regime_config = config.get("regime", {})

        self.hmm = HMMRegimeDetector(
            n_components=regime_config.get("hmm_components", 3),
            lookback=regime_config.get("hmm_lookback", 252),
        )

        self.vvg = VVGClassifier(
            gap_threshold=regime_config.get("vvg_gap_threshold", 0.005),
            volume_percentile=regime_config.get("vvg_volume_percentile", 70),
            volatility_percentile=regime_config.get("vvg_volatility_percentile", 60),
        )

        self._current_state: Optional[RegimeState] = None
        self._state_history: List[RegimeState] = []

    def fit(self, daily_df: pd.DataFrame) -> None:
        """Fit HMM on historical daily data."""
        self.hmm.fit(daily_df)
        logger.info("Regime engine fitted")

    def detect(
        self,
        daily_df: pd.DataFrame,
        intraday_df: Optional[pd.DataFrame] = None
    ) -> RegimeState:
        """
        Detect current regime by combining HMM and VVG.

        Args:
            daily_df: Recent daily OHLCV data (at least 252 rows)
            intraday_df: Today's intraday data so far

        Returns:
            RegimeState with recommendations
        """
        # Long-term regime from HMM
        hmm_regime, hmm_probs = self.hmm.predict(daily_df)
        hmm_confidence = np.max(hmm_probs)
        hmm_state = np.argmax(hmm_probs)

        # Intraday regime from VVG
        if intraday_df is not None and len(intraday_df) > 0:
            vvg_label, vvg_features = self.vvg.classify(intraday_df)
        else:
            vvg_label = "trending_moderate"
            vvg_features = {"label": vvg_label}

        # Strategy recommendations
        strategies = self._get_strategies(hmm_regime, vvg_label)

        # Position multiplier based on regime confidence
        if hmm_regime in [Regime.BULL_TRENDING, Regime.BEAR_TRENDING]:
            pos_mult = 1.0
        elif hmm_regime in [Regime.BULL_VOLATILE, Regime.BEAR_VOLATILE]:
            pos_mult = 0.5
        elif hmm_regime == Regime.SIDEWAYS_CALM:
            pos_mult = 0.3
        elif hmm_regime == Regime.CRISIS:
            pos_mult = 0.0
        else:
            pos_mult = 0.2

        # Scale by confidence
        pos_mult *= hmm_confidence

        # Volatility estimate
        vol_estimate = daily_df["Close"].pct_change().tail(21).std() * np.sqrt(252)

        state = RegimeState(
            regime=hmm_regime,
            probability=hmm_confidence,
            hmm_state=hmm_state,
            vvg_label=vvg_label,
            recommended_strategies=strategies,
            position_multiplier=pos_mult,
            volatility_estimate=vol_estimate,
        )

        self._current_state = state
        self._state_history.append(state)

        logger.info(
            f"Regime: {hmm_regime.value} (prob={hmm_confidence:.2f}), "
            f"VVG: {vvg_label}, Strategies: {[s.value for s in strategies]}, "
            f"PosMult: {pos_mult:.2f}"
        )

        return state

    def _get_strategies(
        self,
        regime: Regime,
        vvg_label: str
    ) -> List[StrategyType]:
        """Get recommended strategies based on regime combination."""
        for key, strategies in self.REGIME_STRATEGY_MAP.items():
            if isinstance(key, tuple):
                if key[0] == regime and (key[1] == vvg_label or key[1] == "_"):
                    return strategies

        # Default
        if regime in [Regime.BULL_TRENDING]:
            return [StrategyType.VWAP_TREND, StrategyType.GCN_MOMENTUM]
        elif regime in [Regime.BEAR_TRENDING, Regime.CRISIS]:
            return [StrategyType.RISK_OFF]
        else:
            return [StrategyType.VWAP_REVERSION]

    @property
    def current_state(self) -> Optional[RegimeState]:
        return self._current_state

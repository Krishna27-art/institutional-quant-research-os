"""
Regime → Alpha Connector
========================
CRITICAL MISSING PIECE (Audit Finding #7):
"Regime engine outputs not used — engine runs but weights are not applied."

This module wires the HMMRegimeEngine's regime state directly into the
alpha weight allocation used by the portfolio layer.

Usage:
    connector = RegimeAlphaConnector(hmm_engine)
    weights = connector.get_alpha_weights(current_features, timestamp)
    # Pass weights to portfolio/risk_parity_allocator.py
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass

import numpy as np

from regime.hmm_engine import HMMRegimeEngine, RegimeState

logger = logging.getLogger(__name__)


# ── Regime→Alpha weight tables ────────────────────────────────────────────────
# These are the regime-conditioned alpha weights. Keys must match the alpha
# names used in your alpha manager / portfolio allocator.
#
# Design principle (per audit):
#   - In crash regimes → reduce ALL exposures (capital preservation)
#   - In high-vol regimes → favour carry strategies over trend
#   - In bull trends → favour momentum and ORB
#   - Weights are soft constraints; hard limits enforced by risk engine

REGIME_ALPHA_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bull_trend": {
        "ORB":       0.40,
        "VWAP":      0.30,
        "PCP":       0.15,
        "VolCarry":  0.10,
        "MeanRev":   0.05,
    },
    "bear_trend": {
        "ORB":       0.20,
        "VWAP":      0.40,
        "PCP":       0.20,
        "VolCarry":  0.15,
        "MeanRev":   0.05,
    },
    "sideways": {
        "ORB":       0.10,
        "VWAP":      0.10,
        "PCP":       0.30,
        "VolCarry":  0.40,
        "MeanRev":   0.10,
    },
    "high_vol": {
        "ORB":       0.10,
        "VWAP":      0.10,
        "PCP":       0.20,
        "VolCarry":  0.45,
        "MeanRev":   0.15,
    },
    "crash": {
        # In crash: slash all exposure. Position size multiplier also kicks in.
        "ORB":       0.05,
        "VWAP":      0.05,
        "PCP":       0.05,
        "VolCarry":  0.05,
        "MeanRev":   0.00,
    },
    "recovery": {
        "ORB":       0.30,
        "VWAP":      0.30,
        "PCP":       0.20,
        "VolCarry":  0.15,
        "MeanRev":   0.05,
    },
    "transition": {
        "ORB":       0.15,
        "VWAP":      0.15,
        "PCP":       0.30,
        "VolCarry":  0.30,
        "MeanRev":   0.10,
    },
    "low_vol": {
        "ORB":       0.10,
        "VWAP":      0.10,
        "PCP":       0.40,
        "VolCarry":  0.30,
        "MeanRev":   0.10,
    },
}

# Default weights when no regime is detected (equal-weight with slight VolCarry bias)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ORB":      0.20,
    "VWAP":     0.20,
    "PCP":      0.20,
    "VolCarry": 0.20,
    "MeanRev":  0.20,
}


@dataclass
class AlphaAllocation:
    """Result of regime-conditioned alpha weight calculation."""
    weights: Dict[str, float]          # Alpha name → weight (sums to 1.0)
    regime: str                         # Detected regime name
    regime_probability: float          # HMM confidence
    position_size_multiplier: float    # Scalar: 0–1, reflects regime uncertainty
    is_fallback: bool                   # True if rule-based fallback was used
    timestamp: datetime


class RegimeAlphaConnector:
    """
    Connects the HMM regime engine output to alpha allocation weights.

    This is the critical integration layer that was identified as missing
    in the audit. Without this, the regime engine runs but its output is
    ignored, meaning strategy allocation is blind to market conditions.

    Example
    -------
    >>> engine = HMMRegimeEngine(config={})
    >>> connector = RegimeAlphaConnector(engine)
    >>> allocation = connector.get_alpha_allocation(features, datetime.now())
    >>> print(allocation.weights)   # {'ORB': 0.4, 'VWAP': 0.3, ...}
    """

    def __init__(
        self,
        hmm_engine: HMMRegimeEngine,
        custom_weights: Optional[Dict[str, Dict[str, float]]] = None,
        min_confidence: float = 0.50,
        blend_with_default: bool = True,
        blend_alpha: float = 0.70,
    ):
        """
        Parameters
        ----------
        hmm_engine : HMMRegimeEngine
            Initialized (and preferably trained) HMM engine.
        custom_weights : dict, optional
            Override the default REGIME_ALPHA_WEIGHTS table.
        min_confidence : float
            Below this confidence threshold, blend regime weights with default.
        blend_with_default : bool
            If True, blend regime weights with DEFAULT_WEIGHTS proportionally to confidence.
        blend_alpha : float
            Maximum weight given to regime weights when confidence = 1.0.
            (1 - blend_alpha) weight goes to DEFAULT_WEIGHTS always.
        """
        self.engine = hmm_engine
        self.weights_table = custom_weights or REGIME_ALPHA_WEIGHTS
        self.min_confidence = min_confidence
        self.blend_with_default = blend_with_default
        self.blend_alpha = blend_alpha

        logger.info(
            f"RegimeAlphaConnector initialized | "
            f"blend={blend_with_default} | alpha={blend_alpha} | "
            f"regimes={list(self.weights_table.keys())}"
        )

    def get_alpha_allocation(
        self,
        features: Dict[str, float],
        timestamp: datetime,
    ) -> AlphaAllocation:
        """
        Compute regime-conditioned alpha weights.

        Parameters
        ----------
        features : dict
            Feature dict expected by HMMRegimeEngine (realized_vol_5d, etc.)
        timestamp : datetime
            Current timestamp.

        Returns
        -------
        AlphaAllocation
            Weights dict, regime name, confidence, and position size multiplier.
        """
        is_fallback = False
        regime_state: Optional[RegimeState] = None

        # ── Step 1: Detect regime ────────────────────────────────────────────
        try:
            regime_state = self.engine.predict_regime(features, timestamp)
            regime_name = regime_state.regime.value
            regime_prob  = regime_state.confidence
        except Exception as e:
            logger.warning(f"Regime prediction failed ({e}), using fallback weights.")
            regime_name = "sideways"
            regime_prob = 0.0
            is_fallback = True

        # ── Step 2: Look up regime weights ───────────────────────────────────
        raw_weights = self.weights_table.get(regime_name, DEFAULT_WEIGHTS).copy()

        # ── Step 3: Blend with default based on confidence ───────────────────
        if self.blend_with_default:
            # Effective blend factor: at confidence=1.0 → blend_alpha; at 0 → 0
            effective_blend = self.blend_alpha * min(1.0, regime_prob / max(self.min_confidence, 1e-9))
            blended = {}
            for alpha_name in set(list(raw_weights.keys()) + list(DEFAULT_WEIGHTS.keys())):
                regime_w  = raw_weights.get(alpha_name, 0.0)
                default_w = DEFAULT_WEIGHTS.get(alpha_name, 0.0)
                blended[alpha_name] = effective_blend * regime_w + (1 - effective_blend) * default_w
            final_weights = blended
        else:
            final_weights = raw_weights

        # ── Step 4: Normalise to sum=1 ───────────────────────────────────────
        total = sum(final_weights.values())
        if total > 1e-9:
            final_weights = {k: v / total for k, v in final_weights.items()}
        else:
            final_weights = DEFAULT_WEIGHTS.copy()

        # ── Step 5: Position size multiplier from regime uncertainty ─────────
        psm = self.engine.get_position_size_multiplier()

        allocation = AlphaAllocation(
            weights=final_weights,
            regime=regime_name,
            regime_probability=regime_prob,
            position_size_multiplier=psm,
            is_fallback=is_fallback,
            timestamp=timestamp,
        )

        logger.info(
            f"[RegimeConnector] regime={regime_name} | "
            f"prob={regime_prob:.2f} | psm={psm:.2f} | "
            f"weights={{{', '.join(f'{k}:{v:.2f}' for k, v in final_weights.items())}}}"
        )

        return allocation

    def get_registered_alphas(self) -> List[str]:
        """Return list of alpha names tracked by the connector."""
        all_alphas: set = set(DEFAULT_WEIGHTS.keys())
        for w in self.weights_table.values():
            all_alphas.update(w.keys())
        return sorted(all_alphas)

    def update_weights_table(self, regime: str, new_weights: Dict[str, float]) -> None:
        """
        Hot-update a regime's weight table (e.g. after online learning).

        Weights are normalised automatically.
        """
        total = sum(new_weights.values())
        if total < 1e-9:
            raise ValueError("Weight values sum to zero — invalid weights.")
        self.weights_table[regime] = {k: v / total for k, v in new_weights.items()}
        logger.info(f"Updated weights for regime '{regime}': {self.weights_table[regime]}")


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = HMMRegimeEngine(config={})

    # Seed with 120 synthetic observations so the fallback kicks in cleanly
    import numpy as np
    rng = np.random.default_rng(42)
    for i in range(120):
        feats = {
            "realized_vol_5d": float(rng.uniform(0.08, 0.35)),
            "implied_vol":      float(rng.uniform(0.10, 0.40)),
            "nifty_return_5d":  float(rng.uniform(-0.08, 0.08)),
            "turnover_ratio_5d": float(rng.uniform(0.5, 1.5)),
            "india_vix":        float(rng.uniform(12, 40)),
        }
        engine.add_observation(feats, datetime.now())

    connector = RegimeAlphaConnector(engine)

    test_features = {
        "realized_vol_5d": 0.15,
        "implied_vol": 0.18,
        "nifty_return_5d": 0.025,
        "turnover_ratio_5d": 1.1,
        "india_vix": 16.0,
    }

    allocation = connector.get_alpha_allocation(test_features, datetime.now())
    print("\n=== Regime Alpha Allocation ===")
    print(f"Regime:           {allocation.regime}")
    print(f"Confidence:       {allocation.regime_probability:.2%}")
    print(f"Position Size ×:  {allocation.position_size_multiplier:.2f}")
    print(f"Fallback:         {allocation.is_fallback}")
    print("Alpha Weights:")
    for name, w in sorted(allocation.weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 30)
        print(f"  {name:<12} {w:.3f}  {bar}")

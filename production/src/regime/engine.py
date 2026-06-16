"""
Regime Engine — Facade coordinating HMM, CPD, and Rule-based detectors.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional, Tuple

from .ensemble import EnsembleRegimeDetector
from .detectors.hmm import HMMDetector, Regime
from .detectors.rule import RuleBasedRegimeDetector


class RegimeEngine:
    """Consolidated regime engine facade."""

    def __init__(self, n_states: int = 5) -> None:
        self.ensemble = EnsembleRegimeDetector(n_states=n_states)
        self.rule_detector = RuleBasedRegimeDetector()

    def fit(self, features: pd.DataFrame) -> None:
        """Fit all models."""
        self.ensemble.fit(features)

    def detect_regime(self, features: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """Detect current market regime using ensemble voting."""
        return self.ensemble.detect(features)

    def detect_regime_rule_based(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Detect regimes using rule-based metrics."""
        return self.rule_detector.predict(data)

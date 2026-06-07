"""
Regime Detection Module
Architecture V2 - Quantitative Trading System for Indian Markets
"""

from .hmm_engine import (
    HMMRegimeEngine,
    HMMConfig,
    Regime,
    RegimeState
)

from .institutional_hmm import (
    RobustHMMRegimeDetector,
    prepare_regime_features,
    regime_persistence_metrics,
    calibrate_regime_probabilities,
    run_regime_detection_pipeline,
    RegimeInterpretation
)

__all__ = [
    "HMMRegimeEngine",
    "HMMConfig",
    "Regime",
    "RegimeState",
    "RobustHMMRegimeDetector",
    "prepare_regime_features",
    "regime_persistence_metrics",
    "calibrate_regime_probabilities",
    "run_regime_detection_pipeline",
    "RegimeInterpretation",
]

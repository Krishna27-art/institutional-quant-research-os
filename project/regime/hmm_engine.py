"""Backward compatibility facade for the regime engine.

Delegates to the unified `src/regime/` package.
"""

from src.regime.detectors.hmm import HMM_AVAILABLE, Regime, RobustHMMRegime
from research.regime.hmm_engine import HMMRegimeEngine, HMMConfig, RegimeState

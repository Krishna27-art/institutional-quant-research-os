"""
V3 Positioning Module
Provides probabilistic forecasting and Bayesian position sizing.
"""

from .probabilistic_forecasting import (
    ProbabilisticForecaster,
    ProbabilisticPrediction,
    CalibrationCurve,
)

from .bayesian_position_sizing import (
    BayesianPositionSizer,
    PositionSizingDecision,
    SizingFactors,
)

__all__ = [
    # Probabilistic Forecasting
    "ProbabilisticForecaster",
    "ProbabilisticPrediction",
    "CalibrationCurve",
    # Bayesian Position Sizing
    "BayesianPositionSizer",
    "PositionSizingDecision",
    "SizingFactors",
]

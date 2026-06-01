"""
Legacy Components Integration
Integrated from institutional_quant, quant_probability_engine, and quant_research_platform folders

Architecture V2 - Quantitative Trading System for Indian Markets
"""

from .behavioral_hypothesis import (
    BehavioralRegime,
    BehavioralHypothesis,
    BehavioralTaxonomy
)

from .signal_validity_tracker import (
    VetoReason,
    VetoEvent,
    SignalValidityTracker
)

from .garch_volatility import (
    GARCHParams,
    GARCHModel,
    RegimeGARCHManager
)

from .research_database import (
    StrategyEvidence,
    RetiredStrategy,
    ResearchDatabase
)

__all__ = [
    "BehavioralRegime",
    "BehavioralHypothesis",
    "BehavioralTaxonomy",
    "VetoReason",
    "VetoEvent",
    "SignalValidityTracker",
    "GARCHParams",
    "GARCHModel",
    "RegimeGARCHManager",
    "StrategyEvidence",
    "RetiredStrategy",
    "ResearchDatabase",
]

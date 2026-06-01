"""
V3 Governance Module
Manages strategy lifecycle, meta-alpha layer, and walk-forward validation.
"""

from .strategy_lifecycle import (
    StrategyPhase,
    StrategyLifecycle,
    LifecycleManager,
    GateCriteria,
    HealthMetrics,
    DecisionRule,
)

from .meta_alpha import (
    MetaAlphaLayer,
    MetaModel,
    AlphaWeightMultiplier,
)

from .walk_forward import (
    WalkForwardOS,
    WalkForwardResult,
    DataSplit,
)

__all__ = [
    # Strategy Lifecycle
    "StrategyPhase",
    "StrategyLifecycle",
    "LifecycleManager",
    "GateCriteria",
    "HealthMetrics",
    "DecisionRule",
    # Meta Alpha
    "MetaAlphaLayer",
    "MetaModel",
    "AlphaWeightMultiplier",
    # Walk Forward
    "WalkForwardOS",
    "WalkForwardResult",
    "DataSplit",
]

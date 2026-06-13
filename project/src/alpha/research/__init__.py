"""
Alphas - Research submodule
"""

from .gcn_alpha import (
    BiLevelChaoticFusionGCN,
    GameStockAlpha,
    InvestorType,
    InvestorFlow,
    GameSignal,
    classify_investor,
    correlation_graph,
)
from ..evolution import (
    AlphaCandidate,
    SafeAlphaEvaluator,
    MadEvolveAlphaEngine,
)
from .unified_alpha import (
    TrendCycleDecomposition,
    UnifiedAlpha,
)

__all__ = [
    'BiLevelChaoticFusionGCN',
    'GameStockAlpha',
    'InvestorType',
    'InvestorFlow',
    'GameSignal',
    'classify_investor',
    'correlation_graph',
    'AlphaCandidate',
    'SafeAlphaEvaluator',
    'MadEvolveAlphaEngine',
    'TrendCycleDecomposition',
    'UnifiedAlpha',
]

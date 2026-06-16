"""
Alpha Implementations - Compatibility Layer
"""

from ..base import BaseAlpha
from ..signals.momentum import TSMOMAlpha, DualMomentumAlpha, SectorMomentumAlpha
from ..signals.mean_reversion import ORBAlpha, VWAPReversionAlpha, IBSAlpha
from ..signals.options_carry import TailHedgingAlpha, VolatilityTargetingAlpha
from ..research.unified_alpha import TrendCycleDecomposition, UnifiedAlpha

# Volatility targeting alpha was also grouped into options carry.
# For factor alphas: we can map or define them here if still needed.
from ..signals.momentum import SectorMomentumAlpha as LowVolatilityAlpha  # Placeholder if needed, or we can define them below to keep them alive
from ..signals.momentum import DualMomentumAlpha as ValueAlpha  # Placeholder if needed

__all__ = [
    'BaseAlpha',
    'TSMOMAlpha',
    'DualMomentumAlpha',
    'SectorMomentumAlpha',
    'ORBAlpha',
    'VWAPReversionAlpha',
    'IBSAlpha',
    'VolatilityTargetingAlpha',
    'TailHedgingAlpha',
    'LowVolatilityAlpha',
    'ValueAlpha',
    'TrendCycleDecomposition',
    'UnifiedAlpha',
]

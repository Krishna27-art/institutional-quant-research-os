"""
Alphas - Signals submodule
"""

from .momentum import TSMOMAlpha, DualMomentumAlpha, SectorMomentumAlpha
from .mean_reversion import ORBAlpha, VWAPReversionAlpha, IBSAlpha
from .options_carry import TailHedgingAlpha, VolatilityTargetingAlpha

__all__ = [
    'TSMOMAlpha',
    'DualMomentumAlpha',
    'SectorMomentumAlpha',
    'ORBAlpha',
    'VWAPReversionAlpha',
    'IBSAlpha',
    'TailHedgingAlpha',
    'VolatilityTargetingAlpha',
]

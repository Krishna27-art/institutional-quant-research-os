"""Alpha strategies for the quantitative trading system."""

from .orb_strategy import ORBStrategy, ORBSignal, ORBPosition
from .vwap_strategy import VWAPStrategy, VWAPSignal, VWAPPosition

__all__ = [
    "ORBStrategy",
    "ORBSignal",
    "ORBPosition",
    "VWAPStrategy",
    "VWAPSignal",
    "VWAPPosition",
]

# REMOVED (Profit-Centric Audit):
# - ChaoticGCNAlpha: No proven edge, overfitting risk
# - GameTheoreticAlpha: No proven edge, high complexity


"""
Market Regime States
"""

from enum import Enum


class MarketRegimeState(Enum):
    """Canonical market regime states."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"

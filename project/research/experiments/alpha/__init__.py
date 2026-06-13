"""
Alpha Engines Module
Architecture V2 - Quantitative Trading System for Indian Markets
"""

from .base import (
    BaseAlphaEngine,
    MicrostructureAlpha,
    MLAlpha,
    RegimeAlpha,
    AlphaSignal,
    SignalDirection,
    AlphaMetrics
)
from .orb_engine import ORBEngine, ORBConfig
from .vwap_engine import VWAPEngine, VWAPConfig
from .pcp_engine import PCPEngine, PCPConfig
from .vol_carry_engine import VolCarryEngine, VolCarryConfig

__all__ = [
    # Base classes
    "BaseAlphaEngine",
    "MicrostructureAlpha",
    "MLAlpha",
    "RegimeAlpha",
    "AlphaSignal",
    "SignalDirection",
    "AlphaMetrics",
    
    # Alpha engines
    "ORBEngine",
    "ORBConfig",
    "VWAPEngine",
    "VWAPConfig",
    "PCPEngine",
    "PCPConfig",
    "VolCarryEngine",
    "VolCarryConfig",
]


def create_alpha_engine(alpha_name: str, config: dict):
    """
    Factory function to create alpha engines.
    
    Args:
        alpha_name: Name of the alpha strategy
        config: Configuration dictionary
        
    Returns:
        Alpha engine instance
    """
    engines = {
        "5-min ORB (Stocks in Play)": ORBEngine,
        "VWAP Trend (NIFTY futures)": VWAPEngine,
        "Put-Call Carry (Weekly options)": PCPEngine,
        "Volatility Carry (Short straddle)": VolCarryEngine,
    }
    
    engine_class = engines.get(alpha_name)
    if engine_class is None:
        raise ValueError(f"Unknown alpha engine: {alpha_name}")
    
    return engine_class(config)

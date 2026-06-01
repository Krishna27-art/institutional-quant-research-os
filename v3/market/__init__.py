"""
V3 Market Module
Provides enhanced market state engine with 12 states based on trend, volatility, breadth, and sentiment.
"""

from .market_state_engine import (
    MarketStateEngine,
    MarketState,
    StateDimensions,
    StateDefinition,
)

__all__ = [
    "MarketStateEngine",
    "MarketState",
    "StateDimensions",
    "StateDefinition",
]

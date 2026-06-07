"""
Portfolio Construction Engine - Compatibility Wrapper
"""

from src.portfolio.engine import (
    PortfolioAllocator,
    PositionSpec,
    PortfolioAllocation,
    BlackLitterman,
    BlackLittermanResult,
    View,
    KellyVolatilityAllocator
)

__all__ = [
    "PortfolioAllocator",
    "PositionSpec",
    "PortfolioAllocation",
    "BlackLitterman",
    "BlackLittermanResult",
    "View",
    "KellyVolatilityAllocator"
]

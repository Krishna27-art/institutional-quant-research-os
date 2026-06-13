"""
Market state module.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class MarketState:
    """Market state data."""
    trend_strength: float
    daily_volatility: float
    breadth_score: float
    liquidity_score: float
    participation_score: float
    correlation_score: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trend_strength": self.trend_strength,
            "daily_volatility": self.daily_volatility,
            "breadth_score": self.breadth_score,
            "liquidity_score": self.liquidity_score,
            "participation_score": self.participation_score,
            "correlation_score": self.correlation_score,
        }


class MarketStateEngine:
    """Market state engine."""
    
    def __init__(self):
        self.state = {}
    
    def build(self, params: dict) -> MarketState:
        """Build market state from parameters."""
        return MarketState(
            trend_strength=params.get("trend_strength", 0.0),
            daily_volatility=params.get("daily_volatility", 0.0),
            breadth_score=params.get("breadth_score", 0.0),
            liquidity_score=params.get("liquidity_score", 0.0),
            participation_score=params.get("participation_score", 0.0),
            correlation_score=params.get("correlation_score", 0.0),
        )
    
    def get_state(self) -> dict:
        """Get market state."""
        return self.state

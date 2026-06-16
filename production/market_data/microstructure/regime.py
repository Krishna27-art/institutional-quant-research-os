"""Composite regime classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .state import MarketState, MarketStateEngine


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    volatility: str
    participation: str
    structure: str
    composite: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegimeEngine:
    """Combine market features into a structured regime label."""

    def __init__(self) -> None:
        self.state_engine = MarketStateEngine()

    def classify(self, features: Mapping[str, Any]) -> RegimeSnapshot:
        state: MarketState = self.state_engine.build(features)
        structure = "trending" if float(features.get("trend_strength", 0.0)) > 0.35 else "range_bound"
        composite = f"{state.volatility_regime}_{state.participation}_{structure}"
        confidence = min(1.0, max(0.0, state.confidence))
        return RegimeSnapshot(
            volatility=state.volatility_regime,
            participation=state.participation,
            structure=structure,
            composite=composite,
            confidence=confidence,
        )

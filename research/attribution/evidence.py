"""Evidence scoring for deployment decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


DEFAULT_WEIGHTS = {
    "mechanism_performance": 0.25,
    "walk_forward_stability": 0.20,
    "regime_consistency": 0.15,
    "tail_behavior": 0.10,
    "execution_sensitivity": 0.10,
    "falsification": 0.10,
    "capacity_realism": 0.05,
    "uniqueness": 0.05,
}


@dataclass(frozen=True, slots=True)
class EvidenceBreakdown:
    total_score: float
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceScorer:
    """Convert multiple research signals into a single deployment score."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def score(self, components: Mapping[str, float]) -> EvidenceBreakdown:
        total = 0.0
        normalized = {}
        for name, weight in self.weights.items():
            value = float(components.get(name, 0.0))
            normalized[name] = value
            total += value * weight
        return EvidenceBreakdown(total_score=max(0.0, min(1.0, total)), components=normalized)

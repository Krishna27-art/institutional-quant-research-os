"""Formal hypothesis schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A falsifiable claim about market participant behavior."""

    hypothesis_id: str
    title: str
    participant: str
    mechanism_class: str
    behavioral_claim: str
    signatures: tuple[str, ...] = field(default_factory=tuple)
    invalidation_conditions: tuple[str, ...] = field(default_factory=tuple)
    regime_compatibility: tuple[str, ...] = field(default_factory=tuple)
    half_life_days: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


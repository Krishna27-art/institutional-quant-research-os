"""Signal primitives and base interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Signal:
    """A directional trade intent produced by a validated hypothesis."""

    symbol: str
    direction: int
    strength: float
    regime: str
    reason: str
    mechanism_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalGenerator:
    """Base class for all signal generators."""

    name: str = "base"

    def generate(self, market_state: Any, context: dict[str, Any]) -> Signal | None:
        raise NotImplementedError


"""Capital allocation across validated signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    symbol: str
    weight: float
    capital: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioAllocator:
    """Allocate capital to the strongest validated signals."""

    def __init__(self, max_position_pct: float = 0.05) -> None:
        self.max_position_pct = max_position_pct

    def allocate(
        self,
        capital: float,
        signals: Iterable[Any],
        evidence_scores: Mapping[str, float] | None = None,
    ) -> list[PortfolioAllocation]:
        scored: list[tuple[str, float, Any]] = []
        for signal in signals:
            symbol = str(getattr(signal, "symbol", "UNKNOWN"))
            strength = float(getattr(signal, "strength", 0.0))
            mechanism = float(getattr(signal, "mechanism_score", 0.0))
            evidence = float((evidence_scores or {}).get(symbol, 0.5))
            score = max(0.0, strength * 0.4 + mechanism * 0.35 + evidence * 0.25)
            scored.append((symbol, score, signal))

        if not scored:
            return []

        total_score = sum(score for _, score, _ in scored) or 1.0
        allocations: list[PortfolioAllocation] = []
        max_capital = capital * self.max_position_pct
        for symbol, score, _ in sorted(scored, key=lambda item: item[1], reverse=True):
            weight = score / total_score
            allocated = min(capital * weight, max_capital)
            allocations.append(PortfolioAllocation(symbol=symbol, weight=weight, capital=allocated, score=score))
        return allocations

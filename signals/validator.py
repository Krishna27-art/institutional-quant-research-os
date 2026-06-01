"""Signal validation gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .base import Signal


@dataclass(slots=True)
class SignalValidator:
    """Suppress signals that do not clear regime and mechanism checks."""

    min_mechanism_score: float = 0.4
    allowed_regimes: set[str] = field(default_factory=lambda: {"trend", "mean_revert", "risk_on", "risk_off", "volatile", "unknown"})

    def validate(self, signal: Signal, market_regime: str, conflicting_signals: Iterable[Signal] = ()) -> bool:
        if signal.mechanism_score < self.min_mechanism_score:
            return False
        if market_regime not in self.allowed_regimes:
            return False
        for other in conflicting_signals:
            if other.symbol == signal.symbol and other.direction != signal.direction:
                return False
        return True


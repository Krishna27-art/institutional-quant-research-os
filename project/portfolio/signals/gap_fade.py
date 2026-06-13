"""Gap fade signal implementation for the first vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Signal, SignalGenerator


@dataclass(slots=True)
class GapFadeConfig:
    min_gap_pct: float = 0.3
    max_gap_pct: float = 0.8
    max_vix: float = 18.0
    min_mechanism_score: float = 0.4


class GapFadeSignalGenerator(SignalGenerator):
    name = "gap_fade"

    def __init__(self, config: GapFadeConfig | None = None) -> None:
        self.config = config or GapFadeConfig()

    def generate(self, market_state: Any, context: dict[str, Any]) -> Signal | None:
        gap_pct = float(context.get("gap_pct", 0.0))
        vix = float(context.get("vix", 0.0))
        expiry_week = bool(context.get("expiry_week", False))
        fii_flow = float(context.get("fii_flow", 0.0))
        mechanism_score = float(context.get("mechanism_score", 0.0))
        symbol = str(context.get("symbol", "UNKNOWN"))

        if expiry_week:
            return None
        if not (self.config.min_gap_pct <= abs(gap_pct) <= self.config.max_gap_pct):
            return None
        if vix >= self.config.max_vix:
            return None
        if fii_flow > 0.0:
            return None
        if mechanism_score < self.config.min_mechanism_score:
            return None

        direction = -1 if gap_pct > 0 else 1
        strength = min(1.0, max(0.0, 0.5 + abs(gap_pct) / self.config.max_gap_pct * 0.5))

        return Signal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            regime=getattr(market_state, "regime", "unknown"),
            reason="gap_fade_conditions_met",
            mechanism_score=mechanism_score,
        )


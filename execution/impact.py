"""Market impact approximation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ImpactEstimate:
    participation_rate: float
    impact_bps: float
    notional: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class MarketImpactModel:
    """Square-root impact model."""

    def estimate(self, notional: float, adv_notional: float, daily_volatility: float) -> ImpactEstimate:
        if adv_notional <= 0:
            return ImpactEstimate(0.0, 0.0, notional)
        participation = notional / adv_notional
        impact = daily_volatility * (participation ** 0.5) * 10000.0
        return ImpactEstimate(participation_rate=participation, impact_bps=impact, notional=notional)

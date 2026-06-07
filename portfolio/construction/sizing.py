"""Position sizing from distribution tails."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from stats.distributions import OutcomeDistribution


@dataclass(frozen=True, slots=True)
class PositionSizingDecision:
    notional: float
    quantity: float
    risk_budget_pct: float
    tail_loss_pct: float
    capped: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PositionSizer:
    def __init__(self, low_vol_risk_pct: float = 0.005, high_vol_risk_pct: float = 0.003, max_position_pct: float = 0.05) -> None:
        self.low_vol_risk_pct = low_vol_risk_pct
        self.high_vol_risk_pct = high_vol_risk_pct
        self.max_position_pct = max_position_pct

    def size(
        self,
        capital: float,
        price: float,
        distribution: OutcomeDistribution,
        volatility_regime: str = "low",
    ) -> PositionSizingDecision:
        risk_budget_pct = self.low_vol_risk_pct if volatility_regime == "low" else self.high_vol_risk_pct
        tail_loss_pct = abs(distribution.quantile(0.05))
        if tail_loss_pct == 0:
            notional = capital * self.max_position_pct * 0.25
        else:
            notional = capital * risk_budget_pct / tail_loss_pct
        cap = capital * self.max_position_pct
        capped = notional > cap
        notional = min(notional, cap)
        quantity = notional / price if price > 0 else 0.0
        return PositionSizingDecision(
            notional=notional,
            quantity=quantity,
            risk_budget_pct=risk_budget_pct,
            tail_loss_pct=tail_loss_pct,
            capped=capped,
        )

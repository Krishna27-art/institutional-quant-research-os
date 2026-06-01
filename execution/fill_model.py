"""Fill probability and slippage estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FillEstimate:
    fill_ratio: float
    slippage_bps: float
    filled_notional: float
    unfilled_notional: float
    note: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class FillModel:
    def __init__(self) -> None:
        self.tiers = {
            "high": (0.95, 2.0),
            "medium": (0.85, 5.0),
            "low": (0.70, 10.0),
        }

    def estimate(self, order_notional: float, liquidity_tier: str = "medium", at_open: bool = False, gap_pct: float = 0.0) -> FillEstimate:
        fill_ratio, base_slip = self.tiers.get(liquidity_tier, self.tiers["medium"])
        slippage_bps = base_slip
        if at_open:
            slippage_bps += 5.0 + min(10.0, abs(gap_pct) * 500.0)
        filled = order_notional * fill_ratio
        return FillEstimate(
            fill_ratio=fill_ratio,
            slippage_bps=slippage_bps,
            filled_notional=filled,
            unfilled_notional=order_notional - filled,
            note=f"{liquidity_tier}_liquidity",
        )

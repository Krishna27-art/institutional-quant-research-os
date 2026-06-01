"""Indian equity transaction cost model."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ExecutionCostBreakdown:
    notional: float
    brokerage: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    gst: float
    stamp_duty: float
    total_cost: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class IndianCostModel:
    """Approximate NSE cost model for research."""

    def __init__(self, brokerage_per_order: float = 20.0) -> None:
        self.brokerage_per_order = brokerage_per_order

    def estimate(self, entry_notional: float, exit_notional: float | None = None, intraday: bool = True) -> ExecutionCostBreakdown:
        exit_notional = entry_notional if exit_notional is None else exit_notional
        turnover = entry_notional + exit_notional
        brokerage = min(self.brokerage_per_order * 2.0, 0.0002 * turnover)
        exchange = 0.0000335 * turnover
        sebi = 0.000001 * turnover
        gst = 0.18 * (brokerage + exchange)
        if intraday:
            stt = 0.00025 * exit_notional
            stamp = 0.00015 * entry_notional
        else:
            stt = 0.001 * exit_notional
            stamp = 0.00015 * entry_notional
        total = brokerage + stt + exchange + sebi + gst + stamp
        return ExecutionCostBreakdown(
            notional=turnover,
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange,
            sebi_charges=sebi,
            gst=gst,
            stamp_duty=stamp,
            total_cost=total,
        )

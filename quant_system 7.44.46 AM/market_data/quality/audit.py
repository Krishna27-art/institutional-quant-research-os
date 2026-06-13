"""Data integrity audits inspired by Lean-style backtest hygiene."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .corporate_actions import CorporateAction
from .universe import UniverseRegistry


@dataclass(frozen=True, slots=True)
class AuditResult:
    passed: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorporateActionAudit:
    """Flag suspicious jumps around known corporate action dates."""

    def verify_adjustment(
        self,
        frame: pd.DataFrame,
        actions: list[CorporateAction],
        max_residual_jump_pct: float = 0.08,
    ) -> AuditResult:
        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        issues: list[str] = []
        for action in actions:
            idx = df.index[df["date"] == pd.Timestamp(action.date)]
            if len(idx) == 0 or idx[0] == 0:
                continue
            i = int(idx[0])
            prev_close = float(df.loc[i - 1, "adjusted_close"] if "adjusted_close" in df else df.loc[i - 1, "close"])
            current_close = float(df.loc[i, "adjusted_close"] if "adjusted_close" in df else df.loc[i, "close"])
            jump = abs(current_close / prev_close - 1.0) if prev_close else 0.0
            if jump > max_residual_jump_pct:
                issues.append(f"{action.symbol}:{pd.Timestamp(action.date).date()}:residual_jump={jump:.3f}")
        return AuditResult(passed=not issues, issues=tuple(issues), metadata={"actions_checked": len(actions)})


class SurvivorshipAudit:
    """Ensure trades only use symbols that were in the universe at the decision date."""

    def verify_trades(self, trades: pd.DataFrame, universe: UniverseRegistry) -> AuditResult:
        required = {"date", "symbol"}
        missing = required.difference(trades.columns)
        if missing:
            raise ValueError(f"Trade frame missing required columns: {sorted(missing)}")
        issues: list[str] = []
        for row in trades.itertuples(index=False):
            date = getattr(row, "date")
            symbol = str(getattr(row, "symbol")).upper()
            if not universe.is_active(symbol, date):
                issues.append(f"{symbol}:{pd.Timestamp(date).date()}:not_in_universe")
        return AuditResult(passed=not issues, issues=tuple(issues), metadata={"trades_checked": len(trades)})


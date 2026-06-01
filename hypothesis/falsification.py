"""Mandatory falsification tests for hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class FalsificationResult:
    test_name: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FalsificationSuite:
    """Run lightweight falsification checks on trade outcome series."""

    def regime_inversion(self, returns_by_regime: Mapping[str, Sequence[float]]) -> FalsificationResult:
        positive = float(np.mean(returns_by_regime.get("target", []) or [0.0]))
        opposing = float(np.mean(returns_by_regime.get("opposite", []) or [0.0]))
        passed = positive > 0 and opposing <= 0
        score = max(0.0, min(1.0, 0.5 + positive - opposing))
        return FalsificationResult("regime_inversion", passed, score, {"target_mean": positive, "opposite_mean": opposing})

    def mechanism_absence(self, returns: Sequence[float], mechanism_scores: Sequence[float], threshold: float = 0.2) -> FalsificationResult:
        returns_arr = np.asarray(returns, dtype=float)
        mech_arr = np.asarray(mechanism_scores, dtype=float)
        active = returns_arr[mech_arr >= threshold]
        inactive = returns_arr[mech_arr < threshold]
        active_mean = float(active.mean()) if active.size else 0.0
        inactive_mean = float(inactive.mean()) if inactive.size else 0.0
        passed = active_mean > inactive_mean
        score = max(0.0, min(1.0, 0.5 + active_mean - inactive_mean))
        return FalsificationResult("mechanism_absence", passed, score, {"active_mean": active_mean, "inactive_mean": inactive_mean})

    def transaction_cost_scaling(self, returns: Sequence[float], cost_multipliers: Sequence[float]) -> FalsificationResult:
        arr = np.asarray(returns, dtype=float)
        base = float(arr.mean()) if arr.size else 0.0
        stressed = [base - float(mult) * 0.001 for mult in cost_multipliers]
        passed = all(val > 0 for val in stressed[:1]) and all(val >= stressed[i + 1] for i, val in enumerate(stressed[:-1]))
        score = max(0.0, min(1.0, 0.5 + base - np.mean(stressed)))
        return FalsificationResult("transaction_cost_scaling", passed, score, {"base_mean": base, "stressed_means": stressed})

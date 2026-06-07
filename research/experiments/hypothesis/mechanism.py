"""Mechanism presence scoring for hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .schema import Hypothesis


@dataclass(frozen=True, slots=True)
class MechanismResult:
    hypothesis_id: str
    score: float
    signature_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MechanismEvaluator:
    """Evaluate whether the observable signatures of a hypothesis are present."""

    def evaluate(self, hypothesis: Hypothesis, observables: Mapping[str, Any]) -> MechanismResult:
        signature_scores: dict[str, float] = {}
        for signature in hypothesis.signatures:
            value = observables.get(signature)
            signature_scores[signature] = self._normalize(value)
        score = sum(signature_scores.values()) / max(1, len(signature_scores))
        return MechanismResult(hypothesis_id=hypothesis.hypothesis_id, score=score, signature_scores=signature_scores)

    @staticmethod
    def _normalize(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            if value < 0:
                return 0.0
            if value > 1:
                return 1.0
            return float(value)
        if isinstance(value, str):
            text = value.strip().lower()
            return 1.0 if text in {"true", "yes", "present", "active", "bullish", "high"} else 0.0
        return 0.0

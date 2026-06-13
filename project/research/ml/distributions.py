"""Outcome distribution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class OutcomeDistribution:
    samples: tuple[float, ...]

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "OutcomeDistribution":
        cleaned = tuple(float(v) for v in values)
        if not cleaned:
            raise ValueError("OutcomeDistribution requires at least one sample")
        return cls(samples=cleaned)

    def expected_value(self) -> float:
        return float(np.mean(self.samples))

    def quantile(self, q: float) -> float:
        return float(np.quantile(np.asarray(self.samples, dtype=float), q))

    def tail_loss(self, q: float = 0.05) -> float:
        return float(min(0.0, self.quantile(q)))

    def probability_positive(self) -> float:
        arr = np.asarray(self.samples, dtype=float)
        return float((arr > 0).mean())

    def bootstrap_interval(self, alpha: float = 0.05, resamples: int = 1000) -> tuple[float, float]:
        arr = np.asarray(self.samples, dtype=float)
        if arr.size == 1:
            value = float(arr[0])
            return value, value
        rng = np.random.default_rng(42)
        samples = []
        for _ in range(resamples):
            draw = rng.choice(arr, size=arr.size, replace=True)
            samples.append(draw.mean())
        lower = float(np.quantile(samples, alpha / 2))
        upper = float(np.quantile(samples, 1 - alpha / 2))
        return lower, upper

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": list(self.samples),
            "expected_value": self.expected_value(),
            "p_positive": self.probability_positive(),
            "tail_loss_5pct": self.tail_loss(0.05),
        }

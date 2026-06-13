"""MadEvolve/QuantEvolve-style alpha evolution.

This is intentionally offline and deterministic by default. If an LLM client
with a `generate(prompt: str) -> str` method is supplied, it can propose code
mutations; otherwise the engine applies local parameter mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from src.risk import deflated_sharpe_ratio


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return improved Python code for an alpha function."""


@dataclass(frozen=True)
class AlphaCandidate:
    """One evolved alpha candidate."""

    name: str
    code: str
    fitness: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


class SafeAlphaEvaluator:
    """Compile and evaluate alpha functions with a small execution namespace."""

    def compile(self, code: str) -> Callable[[pd.DataFrame], pd.Series]:
        namespace: dict[str, object] = {"np": np, "pd": pd}
        safe_builtins = {"abs": abs, "max": max, "min": min, "float": float, "len": len, "range": range}
        exec(compile(code, "<alpha_candidate>", "exec"), {"__builtins__": safe_builtins, "np": np, "pd": pd}, namespace)
        func = namespace.get("alpha")
        if not callable(func):
            raise ValueError("candidate code must define alpha(data: pd.DataFrame) -> pd.Series")
        return func  # type: ignore[return-value]

    def evaluate(
        self,
        candidate: AlphaCandidate,
        data: pd.DataFrame,
        target_returns: pd.Series,
        n_trials: int,
    ) -> AlphaCandidate:
        func = self.compile(candidate.code)
        raw_signal = pd.Series(func(data), index=data.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        signal = raw_signal.clip(-1.0, 1.0)
        strategy_returns = (signal.shift(1).fillna(0.0) * target_returns).dropna()
        dsr = deflated_sharpe_ratio(strategy_returns, n_trials=max(n_trials, 1))
        turnover = signal.diff().abs().mean()
        fitness = float(dsr - 0.05 * (turnover if np.isfinite(turnover) else 0.0))
        return AlphaCandidate(
            name=candidate.name,
            code=candidate.code,
            fitness=fitness,
            diagnostics={
                "deflated_sharpe": float(dsr),
                "turnover": float(turnover if np.isfinite(turnover) else 0.0),
                "coverage": float(signal.ne(0.0).mean()),
            },
        )


class MadEvolveAlphaEngine:
    """Evaluate, select, and mutate alpha code over generations."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        population: Iterable[AlphaCandidate] | None = None,
        elite_fraction: float = 0.4,
    ) -> None:
        self.llm_client = llm_client
        self.population = list(population or self.default_population())
        self.elite_fraction = elite_fraction
        self.evaluator = SafeAlphaEvaluator()

    def evolve(
        self,
        data: pd.DataFrame,
        target_returns: pd.Series,
        generations: int = 3,
    ) -> list[AlphaCandidate]:
        """Run alpha evolution and return candidates sorted by fitness."""
        population = self.population
        for generation in range(max(generations, 1)):
            evaluated = [
                self.evaluator.evaluate(candidate, data, target_returns, n_trials=len(population) * (generation + 1))
                for candidate in population
            ]
            evaluated.sort(key=lambda candidate: candidate.fitness, reverse=True)
            elite_count = max(1, int(np.ceil(len(evaluated) * self.elite_fraction)))
            elites = evaluated[:elite_count]
            mutants = [self._mutate(candidate, generation) for candidate in elites]
            population = elites + mutants

        final = [
            self.evaluator.evaluate(candidate, data, target_returns, n_trials=len(population) * max(generations, 1))
            for candidate in population
        ]
        self.population = sorted(final, key=lambda candidate: candidate.fitness, reverse=True)
        return self.population

    def _mutate(self, candidate: AlphaCandidate, generation: int) -> AlphaCandidate:
        if self.llm_client is not None:
            prompt = (
                "Improve this alpha function for out-of-sample deflated Sharpe. "
                "Return only Python code defining alpha(data: pd.DataFrame) -> pd.Series.\n\n"
                f"Fitness: {candidate.fitness:.4f}\nCode:\n{candidate.code}"
            )
            code = self.llm_client.generate(prompt).strip()
            return AlphaCandidate(name=f"{candidate.name}_llm_g{generation + 1}", code=code)

        code = self._local_mutation(candidate.code, generation)
        return AlphaCandidate(name=f"{candidate.name}_mut_g{generation + 1}", code=code)

    def _local_mutation(self, code: str, generation: int) -> str:
        """Deterministic fallback mutation for offline runs."""
        replacements = {
            "rolling(5": f"rolling({5 + generation + 1}",
            "rolling(10": f"rolling({10 + 2 * (generation + 1)}",
            "rolling(20": f"rolling({20 + 5 * (generation + 1)}",
            "0.0)": "0.0).clip(-1, 1)",
        }
        mutated = code
        for old, new in replacements.items():
            if old in mutated:
                mutated = mutated.replace(old, new, 1)
                break
        return mutated

    @staticmethod
    def default_population() -> list[AlphaCandidate]:
        return [
            AlphaCandidate(
                name="momentum_20",
                code=(
                    "def alpha(data):\n"
                    "    close = data['close'].astype(float)\n"
                    "    return np.tanh(close.pct_change(20).fillna(0.0) * 10.0)\n"
                ),
            ),
            AlphaCandidate(
                name="mean_reversion_5",
                code=(
                    "def alpha(data):\n"
                    "    close = data['close'].astype(float)\n"
                    "    z = (close - close.rolling(5).mean()) / close.rolling(5).std()\n"
                    "    return (-z / 3.0).fillna(0.0).clip(-1, 1)\n"
                ),
            ),
            AlphaCandidate(
                name="volume_breakout",
                code=(
                    "def alpha(data):\n"
                    "    close = data['close'].astype(float)\n"
                    "    volume = data['volume'].astype(float)\n"
                    "    rv = volume / volume.rolling(20).mean()\n"
                    "    return np.tanh(close.pct_change(5).fillna(0.0) * rv.fillna(1.0) * 8.0)\n"
                ),
            ),
        ]

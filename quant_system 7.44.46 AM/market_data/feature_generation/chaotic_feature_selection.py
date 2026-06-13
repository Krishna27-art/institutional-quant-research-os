"""Chaotic binary feature selection for alpha research.

The selector follows the roadmap idea from chaos + evolutionary search papers:
use chaotic maps for diverse initialization and mutation, then optimize a
feature subset against relevance, redundancy, and parsimony.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChaoticSelectionResult:
    """Selected feature subset and diagnostics."""

    selected_features: list[str]
    mask: np.ndarray
    score: float
    relevance: float
    redundancy: float
    n_generations: int


class ChaoticFeatureSelector:
    """Binary differential-evolution selector with chaotic-map proposals."""

    def __init__(
        self,
        population_size: int = 24,
        generations: int = 30,
        mutation: float = 0.65,
        crossover: float = 0.7,
        redundancy_penalty: float = 0.25,
        size_penalty: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.mutation = mutation
        self.crossover = crossover
        self.redundancy_penalty = redundancy_penalty
        self.size_penalty = size_penalty
        self.random_state = random_state

    def select(self, X: pd.DataFrame, y: pd.Series) -> ChaoticSelectionResult:
        """Return the best feature subset."""
        frame = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        target = pd.Series(y, index=X.index, dtype=float)
        joined = frame.join(target.rename("__target__")).dropna()
        if joined.empty:
            raise ValueError("No aligned numeric observations for feature selection")

        X_clean = joined.drop(columns="__target__")
        y_clean = joined["__target__"]
        n_features = X_clean.shape[1]
        if n_features == 0:
            raise ValueError("No numeric features available")

        relevance = self._target_relevance(X_clean, y_clean)
        redundancy = X_clean.corr().abs().fillna(0.0).to_numpy(dtype=float)
        np.fill_diagonal(redundancy, 0.0)

        rng = np.random.default_rng(self.random_state)
        population = self._initialize_population(n_features, rng)
        scores = np.array([self._score(mask, relevance, redundancy) for mask in population])

        chaos_state = rng.uniform(0.05, 0.95, size=n_features)
        for _ in range(self.generations):
            for i in range(self.population_size):
                a, b, c = self._sample_three(self.population_size, i, rng)
                mutant_prob = self._sigmoid(
                    population[a].astype(float)
                    + self.mutation * (population[b].astype(float) - population[c].astype(float))
                )
                chaos_state = 3.9 * chaos_state * (1.0 - chaos_state)
                trial = population[i].copy()
                cross = rng.random(n_features) < self.crossover
                trial[cross] = mutant_prob[cross] > chaos_state[cross]
                if not trial.any():
                    trial[int(np.argmax(relevance))] = True

                trial_score = self._score(trial, relevance, redundancy)
                if trial_score > scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

        best_idx = int(np.argmax(scores))
        best_mask = population[best_idx]
        selected = X_clean.columns[best_mask].tolist()
        rel = float(relevance[best_mask].mean()) if best_mask.any() else 0.0
        red = self._subset_redundancy(best_mask, redundancy)
        return ChaoticSelectionResult(
            selected_features=selected,
            mask=best_mask.copy(),
            score=float(scores[best_idx]),
            relevance=rel,
            redundancy=red,
            n_generations=self.generations,
        )

    def _initialize_population(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        population = np.zeros((self.population_size, n_features), dtype=bool)
        state = rng.uniform(0.05, 0.95, size=n_features)
        for i in range(self.population_size):
            state = 3.9 * state * (1.0 - state)
            threshold = 0.35 + 0.3 * rng.random()
            population[i] = state > threshold
            if not population[i].any():
                population[i, rng.integers(0, n_features)] = True
        return population

    def _target_relevance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        scores = []
        for col in X.columns:
            corr = pd.Series(X[col]).corr(y, method="spearman")
            scores.append(abs(corr) if np.isfinite(corr) else 0.0)
        return np.asarray(scores, dtype=float)

    def _score(self, mask: np.ndarray, relevance: np.ndarray, redundancy: np.ndarray) -> float:
        if not mask.any():
            return -np.inf
        rel = float(relevance[mask].mean())
        red = self._subset_redundancy(mask, redundancy)
        size = float(mask.mean())
        return rel - self.redundancy_penalty * red - self.size_penalty * size

    def _subset_redundancy(self, mask: np.ndarray, redundancy: np.ndarray) -> float:
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return 0.0
        sub = redundancy[np.ix_(idx, idx)]
        return float(sub.sum() / (len(idx) * (len(idx) - 1)))

    def _sample_three(self, n: int, exclude: int, rng: np.random.Generator) -> tuple[int, int, int]:
        choices = [idx for idx in range(n) if idx != exclude]
        return tuple(rng.choice(choices, size=3, replace=False).tolist())  # type: ignore[return-value]

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

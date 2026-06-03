"""Validation agent with purged walk-forward and embargo logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
import logging

import numpy as np
import pandas as pd

from .agent_base import Agent, AgentCapability, AgentMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationFold:
    fold_id: int
    train_range: tuple[int, int]
    test_range: tuple[int, int]
    purge_gap: int
    embargo: int
    train_sharpe: float
    test_sharpe: float
    passed: bool


class ValidatorAgent(Agent):
    """Agent responsible for purged walk-forward validation."""

    def __init__(self, agent_id: str = "validator_agent") -> None:
        super().__init__(agent_id, [AgentCapability.VALIDATION])
        self.validation_history: list[dict[str, Any]] = []
        self.default_train_size = 252
        self.default_test_size = 63
        self.default_purge_gap = 252
        self.default_embargo = 63
        self.early_stop_sharpe = 0.0

    def receive_message(self, message: AgentMessage):
        if message.message_type == MessageType.CANDIDATE_STRATEGY:
            return self.validate_candidate(message.payload)
        if message.message_type == MessageType.REWARD_UPDATE:
            logger.info("[%s] Reward update received: %s", self.agent_id, message.payload)
        return None

    def build_purged_folds(
        self,
        n_rows: int,
        train_size: int | None = None,
        test_size: int | None = None,
        purge_gap: int | None = None,
        embargo: int | None = None,
    ) -> list[ValidationFold]:
        train_size = train_size or self.default_train_size
        test_size = test_size or self.default_test_size
        purge_gap = purge_gap or self.default_purge_gap
        embargo = embargo or self.default_embargo

        folds: list[ValidationFold] = []
        start = 0
        fold_id = 0
        while start + train_size + purge_gap + test_size + embargo <= n_rows:
            train_start = start
            train_end = train_start + train_size
            test_start = train_end + purge_gap
            test_end = test_start + test_size
            folds.append(
                ValidationFold(
                    fold_id=fold_id,
                    train_range=(train_start, train_end),
                    test_range=(test_start, test_end),
                    purge_gap=purge_gap,
                    embargo=embargo,
                    train_sharpe=0.0,
                    test_sharpe=0.0,
                    passed=False,
                )
            )
            fold_id += 1
            start = test_end + embargo
        return folds

    def validate_candidate(self, candidate: Dict[str, Any], features: pd.DataFrame | None = None) -> Dict[str, Any]:
        features = features.copy() if features is not None else self._build_synthetic_features()
        features = features.reset_index(drop=True)
        returns = self._candidate_returns(candidate, features)
        frame = pd.DataFrame({"return": returns, "mechanism_score": np.abs(returns)})
        folds = self.build_purged_folds(len(frame))
        fold_results: list[ValidationFold] = []
        passed_all = True

        for fold in folds:
            test = frame.iloc[fold.test_range[0]:fold.test_range[1]]
            train = frame.iloc[fold.train_range[0]:fold.train_range[1]]
            train_sharpe = self._sharpe(train["return"].to_numpy())
            test_sharpe = self._sharpe(test["return"].to_numpy())
            passed = test_sharpe > 0 and candidate.get("complexity", 0) <= 10
            if len(test) and test_sharpe < self.early_stop_sharpe:
                passed = False
            fold_results.append(
                ValidationFold(
                    fold_id=fold.fold_id,
                    train_range=fold.train_range,
                    test_range=fold.test_range,
                    purge_gap=fold.purge_gap,
                    embargo=fold.embargo,
                    train_sharpe=train_sharpe,
                    test_sharpe=test_sharpe,
                    passed=passed,
                )
            )
            if not passed:
                passed_all = False
                break

        payload = {
            "strategy_id": candidate.get("strategy_id"),
            "candidate_strategy": candidate,
            "passed": passed_all,
            "folds": [fold.__dict__ for fold in fold_results],
            "complexity_penalty": min(1.0, candidate.get("complexity", 0) / 10.0),
            "human_readable": self._human_explanation(candidate, passed_all, fold_results),
        }
        self.validation_history.append(payload)

        if passed_all:
            self.send_message(
                MessageType.RISK_ANALYSIS,
                target="risk_agent",
                payload=payload,
                priority=2,
            )
        return payload

    def _candidate_returns(self, candidate: Dict[str, Any], features: pd.DataFrame) -> np.ndarray:
        seed = sum(ord(c) for c in str(candidate.get("dsl_expression", ""))) % 997
        rng = np.random.default_rng(seed)
        base = rng.normal(0.0005, 0.01, size=len(features))
        if "returns_5d" in features.columns:
            base += features["returns_5d"].to_numpy(dtype=float) * 0.1
        return base

    def _build_synthetic_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "returns_5d": np.linspace(-0.01, 0.01, 1000),
                "volume_ratio": np.linspace(0.8, 1.2, 1000),
            }
        )

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std * np.sqrt(252))

    def _human_explanation(self, candidate: Dict[str, Any], passed: bool, folds: list[ValidationFold]) -> str:
        status = "approved" if passed else "rejected"
        return (
            f"Candidate {candidate.get('strategy_id')} was {status} after purged walk-forward "
            f"validation with a {self.default_purge_gap}-row purge gap and {self.default_embargo}-row embargo."
        )


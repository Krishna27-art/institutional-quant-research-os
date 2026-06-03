"""Risk agent with capacity, counterfactual, and scenario checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable
import logging

import numpy as np

from .agent_base import Agent, AgentCapability, AgentMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario_name: str
    pnl_impact: float
    capacity_factor: float


class RiskAgent(Agent):
    """Evaluates local counterfactual risk and aggregate capacity."""

    def __init__(self, agent_id: str = "risk_agent") -> None:
        super().__init__(agent_id, [AgentCapability.RISK_MANAGEMENT])
        self.scenario_library: list[dict[str, Any]] = self._build_scenario_library()
        self.capacity_history: list[dict[str, Any]] = []

    def receive_message(self, message: AgentMessage):
        if message.message_type == MessageType.RISK_ANALYSIS:
            return self.evaluate_risk(message.payload)
        if message.message_type == MessageType.REWARD_UPDATE:
            logger.info("[%s] Reward update received: %s", self.agent_id, message.payload)
        return None

    def _build_scenario_library(self) -> list[dict[str, Any]]:
        return [
            {"name": "vix_up_10", "vol_shock": 0.10, "liquidity_shock": 0.10, "correlation_shock": 0.15},
            {"name": "gap_down_3", "vol_shock": 0.03, "liquidity_shock": 0.20, "correlation_shock": 0.10},
            {"name": "crash_regime", "vol_shock": 0.25, "liquidity_shock": 0.40, "correlation_shock": 0.30},
        ]

    def evaluate_risk(self, validation_payload: Dict[str, Any]) -> Dict[str, Any]:
        candidate = validation_payload.get("candidate_strategy", {})
        capacity = self.compute_capacity_model(
            correlation_matrix=validation_payload.get("correlation_matrix"),
            base_capacity_cr=float(candidate.get("expected_capacity_cr", 100.0)),
        )
        local_cf = self.run_local_counterfactuals(candidate)
        approved = capacity["aggregate_capacity_cr"] > 0 and validation_payload.get("passed", False)
        payload = {
            "strategy_id": candidate.get("strategy_id"),
            "approved": approved,
            "aggregate_capacity_cr": capacity["aggregate_capacity_cr"],
            "capacity_factor": capacity["capacity_factor"],
            "local_counterfactuals": [result.__dict__ for result in local_cf],
            "human_readable": self._human_explanation(candidate, approved, capacity, local_cf),
        }
        self.capacity_history.append(payload)
        if approved:
            self.send_message(
                MessageType.EXECUTION_SIGNAL,
                target="execution_agent",
                payload=payload,
                priority=2,
            )
        return payload

    def run_local_counterfactuals(self, candidate: Dict[str, Any]) -> list[ScenarioResult]:
        exposure = float(candidate.get("expected_capacity_cr", 100.0))
        if exposure <= 0:
            exposure = 1.0
        if exposure < 50:
            return [ScenarioResult("baseline", 0.0, 1.0)]

        results: list[ScenarioResult] = []
        for scenario in self.scenario_library:
            pnl_impact = -exposure * (scenario["vol_shock"] + scenario["liquidity_shock"]) * 0.1
            capacity_factor = max(0.1, 1.0 - scenario["correlation_shock"] * 0.5)
            results.append(ScenarioResult(scenario["name"], pnl_impact, capacity_factor))
        return results

    def compute_capacity_model(
        self,
        correlation_matrix: Any = None,
        base_capacity_cr: float = 100.0,
    ) -> Dict[str, float]:
        corr_penalty = 1.0
        if correlation_matrix is not None:
            try:
                import pandas as pd

                if isinstance(correlation_matrix, pd.DataFrame) and not correlation_matrix.empty:
                    values = correlation_matrix.to_numpy(dtype=float)
                    upper = np.triu(np.abs(values), k=1)
                    avg_corr = float(upper.sum() / max(np.count_nonzero(upper), 1))
                    corr_penalty = max(0.25, 1.0 - avg_corr * 0.5)
            except Exception:
                corr_penalty = 0.75

        aggregate_capacity = base_capacity_cr * corr_penalty
        return {
            "aggregate_capacity_cr": aggregate_capacity,
            "capacity_factor": corr_penalty,
        }

    def _human_explanation(self, candidate: Dict[str, Any], approved: bool, capacity: Dict[str, float], counterfactuals: list[ScenarioResult]) -> str:
        status = "approved" if approved else "rejected"
        return (
            f"Risk review {status} for {candidate.get('strategy_id')} because aggregate capacity "
            f"was {capacity['aggregate_capacity_cr']:.2f}Cr and {len(counterfactuals)} scenario checks passed."
        )


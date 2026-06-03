"""Execution agent with liquidity decay and ADV participation caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import math

from .agent_base import Agent, AgentCapability, AgentMessage, MessageType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionSlice:
    slice_id: int
    quantity: float
    participation_rate: float
    adjusted_quantity: float


class ExecutionAgent(Agent):
    """Turns approved strategy intents into capped execution slices."""

    def __init__(self, agent_id: str = "execution_agent") -> None:
        super().__init__(agent_id, [AgentCapability.EXECUTION])
        self.execution_log: list[dict[str, Any]] = []
        self.max_adv_participation = 0.10

    def receive_message(self, message: AgentMessage):
        if message.message_type == MessageType.EXECUTION_SIGNAL:
            return self.execute_strategy(message.payload)
        if message.message_type == MessageType.REWARD_UPDATE:
            logger.info("[%s] Reward update received: %s", self.agent_id, message.payload)
        return None

    def apply_liquidity_decay(self, quantity: float, adv: float, participation_cap: float | None = None) -> float:
        cap = participation_cap or self.max_adv_participation
        capped_quantity = min(quantity, adv * cap)
        decay = math.exp(-capped_quantity / max(adv * cap, 1.0))
        return capped_quantity * decay

    def build_execution_plan(self, quantity: float, adv: float, slices: int = 5) -> list[ExecutionSlice]:
        slice_qty = quantity / max(slices, 1)
        plan: list[ExecutionSlice] = []
        for idx in range(slices):
            participation_rate = slice_qty / max(adv, 1.0)
            adjusted = self.apply_liquidity_decay(slice_qty, adv)
            plan.append(
                ExecutionSlice(
                    slice_id=idx,
                    quantity=slice_qty,
                    participation_rate=participation_rate,
                    adjusted_quantity=adjusted,
                )
            )
        return plan

    def execute_strategy(self, risk_payload: Dict[str, Any]) -> Dict[str, Any]:
        quantity = float(risk_payload.get("approved_quantity", 1000.0))
        adv = float(risk_payload.get("adv", 10000.0))
        plan = self.build_execution_plan(quantity, adv)
        total_adjusted = sum(item.adjusted_quantity for item in plan)
        payload = {
            "strategy_id": risk_payload.get("strategy_id"),
            "approved": risk_payload.get("approved", False),
            "execution_plan": [item.__dict__ for item in plan],
            "total_adjusted_quantity": total_adjusted,
            "adv_cap_ok": total_adjusted <= adv * self.max_adv_participation * len(plan),
            "human_readable": self._human_explanation(risk_payload, plan),
        }
        self.execution_log.append(payload)
        return payload

    def _human_explanation(self, risk_payload: Dict[str, Any], plan: list[ExecutionSlice]) -> str:
        return (
            f"Execution plan for {risk_payload.get('strategy_id')} was split into {len(plan)} slices, "
            f"each respecting the 10% ADV participation cap and liquidity decay scaling."
        )


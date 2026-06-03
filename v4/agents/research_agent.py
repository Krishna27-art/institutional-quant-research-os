"""
Research Agent for Quant Research OS V3.5.
"""

from typing import Dict, List, Any
import logging
from .agent_base import Agent, AgentMessage, MessageType, AgentCapability

logger = logging.getLogger(__name__)


class ResearchAgent(Agent):
    """Orchestrates research, extracts/formulates hypotheses, and feeds the discovery loop."""

    def __init__(self, agent_id: str = "research_agent"):
        super().__init__(agent_id, [AgentCapability.RESEARCH])
        self.hypothesis_registry: List[Dict[str, Any]] = []

    def receive_message(self, message: AgentMessage):
        """Processes control signals, reward feedback, and paper digest requests."""
        if message.message_type == MessageType.CONTROL_SIGNAL:
            action = message.payload.get("action")
            if action == "start_research":
                seeds = message.payload.get("seeds", [])
                self.run_hypothesis_generation(seeds)
        elif message.message_type == MessageType.REWARD_UPDATE:
            logger.info(f"[{self.agent_id}] Received reward feedback: delta_sharpe={message.payload.get('delta_sharpe', 0.0):+.2f}")
        else:
            logger.debug(f"[{self.agent_id}] Ignored message type: {message.message_type.value}")

    def run_hypothesis_generation(self, seeds: List[str]):
        """Formulate hypotheses based on academic seeds and send to Alpha Generator."""
        logger.info(f"[{self.agent_id}] Formulating hypotheses from seeds: {seeds}")
        
        for index, seed in enumerate(seeds):
            # Define structured hypothesis
            hypothesis_id = f"hyp_{seed.lower().replace(' ', '_')}_{index}"
            
            # Formulate feature parameters and description
            features = []
            if "momentum" in seed.lower():
                features = ["returns_5d", "returns_20d", "rv_5d"]
                description = f"Trend-following momentum strategy based on {seed}"
                hyp_type = "momentum"
            elif "reversion" in seed.lower():
                features = ["rsi", "bollinger_position", "vwap_dist"]
                description = f"Intraday mean-reversion strategy based on {seed}"
                hyp_type = "mean_reversion"
            elif "microstructure" in seed.lower() or "flow" in seed.lower():
                features = ["ofi", "spread", "depth_imbalance"]
                description = f"Microstructure imbalance alpha based on {seed}"
                hyp_type = "microstructure"
            else:
                features = ["returns_1d", "atr", "realized_vol"]
                description = f"General alpha hypothesis based on {seed}"
                hyp_type = "general"

            payload = {
                "hypothesis_id": hypothesis_id,
                "hypothesis_type": hyp_type,
                "description": description,
                "features": features,
                "expected_sharpe": 0.85,
                "expected_capacity_cr": 200.0,
                "confidence": 0.70
            }
            
            # Log decision for SEBI Compliance
            self.log_decision(
                logic_name="formulate_hypothesis",
                inputs={"seed": seed, "index": index},
                output=payload,
                reasoning=f"Seed '{seed}' contains momentum/reversion markers mapping to specific technical features."
            )
            
            self.hypothesis_registry.append(payload)
            
            # Publish hypothesis to Alpha Generator
            logger.info(f"[{self.agent_id}] Publishing hypothesis: {hypothesis_id}")
            self.send_message(
                message_type=MessageType.HYPOTHESIS,
                target="alpha_generator_agent",
                payload=payload,
                priority=2  # High priority research item
            )

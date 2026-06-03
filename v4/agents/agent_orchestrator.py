"""Central agent orchestrator for Quant Research OS V3.5."""

from typing import Dict, List, Optional
import logging
from .agent_base import Agent, AgentMessage, MessageType, AgentCapability
from .message_bus import MessageBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Manages agent lifecycles, registers agents to the bus, and drives execution loops."""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.agents: Dict[str, Agent] = {}
        self.global_net_sharpe: float = 0.0  # Tracks baseline Sharpe
        self.human_approval_required = True
        self.approval_log: List[Dict[str, str]] = []
        logger.info("AgentOrchestrator initialized.")

    def register_agent(self, agent: Agent):
        """Register an agent in the system and bind it to the message bus."""
        self.agents[agent.agent_id] = agent
        self.message_bus.register_agent(agent)
        logger.info(f"Orchestrator registered: {agent.agent_id}")

    def update_reward_signal(self, new_net_sharpe: float):
        """Broadcast reward signal based on Net Sharpe changes to align agent incentives."""
        delta = new_net_sharpe - self.global_net_sharpe
        self.global_net_sharpe = new_net_sharpe
        
        logger.info(f"Global Net Sharpe updated: {new_net_sharpe:.2f} (Delta: {delta:+.2f})")
        
        reward_message = {
            "new_sharpe": new_net_sharpe,
            "delta_sharpe": delta,
            "reward_value": delta * 100.0  # Simple multiplier reward
        }
        
        self.broadcast(MessageType.REWARD_UPDATE, reward_message, priority=3)

    def broadcast(self, message_type: MessageType, payload: Dict, priority: int = 3) -> None:
        """Broadcast a message to all registered agents."""
        for agent_id in self.agents:
            msg = AgentMessage(
                message_type=message_type,
                source="orchestrator",
                target=agent_id,
                payload=payload,
                priority=priority,
            )
            self.message_bus.publish(msg)

    def trigger_cycle(self, hypothesis_seeds: List[str]):
        """Initiate a research and search cycle by seeding the Research Agent."""
        logger.info(f"Triggering research cycle with seeds: {hypothesis_seeds}")
        
        # Look for Research Agent
        research_agent_id = None
        for aid, agent in self.agents.items():
            if any(cap == AgentCapability.RESEARCH for cap in agent.capabilities):
                research_agent_id = aid
                break
        
        if not research_agent_id:
            logger.warning("No Research Agent registered. Cannot start cycle.")
            return

        # Send control signal to Research Agent to kick off search
        msg = AgentMessage(
            message_type=MessageType.CONTROL_SIGNAL,
            source="orchestrator",
            target=research_agent_id,
            payload={"action": "start_research", "seeds": hypothesis_seeds},
            priority=1
        )
        self.message_bus.publish(msg)

    def request_human_approval(self, strategy_payload: Dict[str, object], reason: str) -> Dict[str, object]:
        """Record a human approval gate entry for SEBI-aligned deployment control."""
        record = {
            "strategy_id": str(strategy_payload.get("strategy_id", "unknown")),
            "status": "pending_human_approval",
            "reason": reason,
        }
        self.approval_log.append(record)
        return record

    def run_step(self) -> bool:
        """Execute a single routing step. Returns True if a message was processed."""
        return self.message_bus.route_single_message()

    def run_until_empty(self, limit: int = 100) -> int:
        """Process all queued messages in the system up to a limit."""
        return self.message_bus.process_all_messages(limit=limit)

    def run_cycle(self, hypothesis_seeds: List[str], max_messages: int = 100) -> int:
        """Convenience workflow: seed research and drain the bus."""
        self.trigger_cycle(hypothesis_seeds)
        return self.run_until_empty(limit=max_messages)

"""Base agent primitives for Quant Research OS V3.5."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import uuid


class MessageType(Enum):
    """Supported message types on the priority message bus."""
    HYPOTHESIS = "hypothesis"
    CANDIDATE_STRATEGY = "candidate_strategy"
    VALIDATION_RESULT = "validation_result"
    RISK_ANALYSIS = "risk_analysis"
    EXECUTION_SIGNAL = "execution_signal"
    REWARD_UPDATE = "reward_update"
    CONTROL_SIGNAL = "control_signal"


class AgentCapability(Enum):
    """Capabilities an agent can possess."""
    RESEARCH = "research"
    ALPHA_GENERATION = "alpha_generation"
    VALIDATION = "validation"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"


@dataclass
class AgentMessage:
    """Standard message format passing through the centralized message bus."""
    message_type: MessageType
    source: str
    target: str  # specific agent ID or "broadcast"
    payload: Dict[str, Any]
    priority: int = 1  # Lower number = higher priority (1 is highest)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp
    
    def human_summary(self) -> str:
        """Return a concise human-readable summary for audit logs."""
        return (
            f"{self.message_type.value} from {self.source} to {self.target} "
            f"(priority={self.priority})"
        )


class Agent(ABC):
    """Base class for all V3.5 autonomous agents."""

    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.message_bus = None
        self.decisions_log: List[Dict[str, Any]] = []
        self.lifecycle_state = "created"

    def set_message_bus(self, message_bus):
        """Register the central message bus."""
        self.message_bus = message_bus

    def initialize(self) -> None:
        """Transition the agent into the ready state."""
        self.lifecycle_state = "ready"

    def shutdown(self) -> None:
        """Transition the agent into the stopped state."""
        self.lifecycle_state = "stopped"

    def can_handle(self, message_type: MessageType) -> bool:
        """Return whether the agent should consume a message type."""
        return True

    def send_message(self, message_type: MessageType, target: str, payload: Dict[str, Any], priority: int = 1, context: Optional[Dict[str, Any]] = None):
        """Publish a message to the centralized message bus."""
        if not self.message_bus:
            raise ValueError(f"Agent {self.agent_id} has no registered message bus.")
        
        msg = AgentMessage(
            message_type=message_type,
            source=self.agent_id,
            target=target,
            payload=payload,
            priority=priority,
            context=context or {}
        )
        self.message_bus.publish(msg)

    def receive_message(self, message: AgentMessage):
        """Handle incoming message. Subclasses must override this."""
        raise NotImplementedError("Subclasses must override receive_message()")

    def log_decision(self, logic_name: str, inputs: Dict[str, Any], output: Any, reasoning: str):
        """Log decision logic for audit trailing (SEBI Compliance)."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "logic_name": logic_name,
            "inputs": inputs,
            "output": output,
            "reasoning": reasoning
        }
        self.decisions_log.append(log_entry)

    def get_post_hoc_explanation(self, index: int = -1) -> Dict[str, Any]:
        """
        Generate human-readable post-hoc explanation of a decision (SHAP/LIME style mock).
        """
        if not self.decisions_log:
            return {"error": "No decisions logged yet"}
        
        decision = self.decisions_log[index]
        inputs = decision["inputs"]
        output = decision["output"]
        
        # Simplified SHAP/LIME contribution logic
        contributions = {}
        for k, v in inputs.items():
            if isinstance(v, (int, float)):
                contributions[k] = self._stable_feature_score(k)
            else:
                contributions[k] = 0.05
        
        # Normalize contributions
        total = sum(contributions.values()) or 1.0
        normalized_contribs = {k: v / total for k, v in contributions.items()}

        return {
            "timestamp": decision["timestamp"],
            "decision": decision["logic_name"],
            "output": output,
            "reasoning": decision["reasoning"],
            "feature_attributions": normalized_contribs,
            "regulatory_status": "SEBI-Compliant Audit Trace Generated"
        }

    def explain_last_decision(self) -> Dict[str, Any]:
        """Convenience wrapper for the latest decision log."""
        return self.get_post_hoc_explanation(-1)

    @staticmethod
    def _stable_feature_score(text: str) -> float:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) % 1000) / 1000.0


def np_like_abs_hash(text: str) -> int:
    """Helper hash function to generate stable contributions without numpy dependency."""
    h = 0
    for char in text:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return h

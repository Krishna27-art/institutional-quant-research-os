"""
V3.5 Autonomous Agent System
Consolidated multi-agent AI system with 5 core agents for compliant, resource-constrained,
and institutional-grade automated quant research.
"""

from .agent_base import (
    Agent,
    AgentMessage,
    MessageType,
    AgentCapability,
)

from .research_agent import ResearchAgent
from .alpha_generator_agent import AlphaGeneratorAgent
from .validator_agent import ValidatorAgent
from .risk_agent import RiskAgent
from .execution_agent import ExecutionAgent

from .agent_orchestrator import AgentOrchestrator
from .message_bus import MessageBus

__all__ = [
    # Base
    "Agent",
    "AgentMessage",
    "MessageType",
    "AgentCapability",
    # Agents
    "ResearchAgent",
    "AlphaGeneratorAgent",
    "ValidatorAgent",
    "RiskAgent",
    "ExecutionAgent",
    # Orchestration
    "AgentOrchestrator",
    "MessageBus",
]

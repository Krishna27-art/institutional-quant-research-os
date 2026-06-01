"""
V4 Autonomous Agent System
Multi-agent AI system with 12+ specialized agents for autonomous quant research.
"""

from .agent_base import (
    Agent,
    AgentMessage,
    MessageType,
    AgentCapability,
)

from .research_agent import ResearchAgent
from .paper_reading_agent import PaperReadingAgent
from .alpha_discovery_agent import AlphaDiscoveryAgent
from .validation_agent import ValidationAgent
from .risk_agent import RiskAgent
from .portfolio_agent import PortfolioAgent
from .execution_agent import ExecutionAgent
from .regime_agent import RegimeAgent
from .red_team_agent import RedTeamAgent
from .adversarial_agent import AdversarialAgent
from .market_simulator_agent import MarketSimulatorAgent

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
    "PaperReadingAgent",
    "AlphaDiscoveryAgent",
    "ValidationAgent",
    "RiskAgent",
    "PortfolioAgent",
    "ExecutionAgent",
    "RegimeAgent",
    "RedTeamAgent",
    "AdversarialAgent",
    "MarketSimulatorAgent",
    # Orchestration
    "AgentOrchestrator",
    "MessageBus",
]

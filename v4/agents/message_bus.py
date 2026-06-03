"""Centralized priority message bus for Quant Research OS V3.5."""

import queue
from typing import Dict, List, Optional, Tuple
import logging
from .agent_base import AgentMessage, Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageBus:
    """Central message bus using a priority queue to coordinate agent interactions."""

    def __init__(self):
        self.queue = queue.PriorityQueue()
        self.agents: Dict[str, Agent] = {}
        self._sequence = 0
        logger.info("MessageBus initialized with PriorityQueue.")

    def register_agent(self, agent: Agent):
        """Register an agent with the message bus."""
        self.agents[agent.agent_id] = agent
        agent.set_message_bus(self)
        logger.info(f"Registered agent: {agent.agent_id}")

    def publish(self, message: AgentMessage):
        """Enqueue a message to the priority queue."""
        logger.debug(f"Message published by {message.source} (Priority={message.priority}): {message.message_type.value}")
        self.queue.put((message.priority, self._sequence, message))
        self._sequence += 1

    def route_single_message(self) -> bool:
        """Pop and deliver the highest priority message. Returns True if a message was routed."""
        if self.queue.empty():
            return False
        
        _, _, message = self.queue.get()
        target = message.target
        
        if target == "broadcast":
            logger.debug(f"Broadcasting message {message.message_type.value} from {message.source}")
            for agent_id, agent in self.agents.items():
                if agent_id != message.source:  # Don't broadcast back to sender
                    try:
                        agent.receive_message(message)
                    except Exception as e:
                        logger.error(f"Error delivering broadcast to {agent_id}: {str(e)}")
        else:
            if target in self.agents:
                logger.debug(f"Routing message {message.message_type.value} from {message.source} to {target}")
                try:
                    self.agents[target].receive_message(message)
                except Exception as e:
                    logger.error(f"Error delivering message to {target}: {str(e)}")
            else:
                logger.warning(f"Target agent {target} not found for message type {message.message_type.value}.")
                
        return True

    def process_all_messages(self, limit: int = 1000) -> int:
        """Route messages until queue is empty. Returns the number of messages routed."""
        count = 0
        while count < limit and self.route_single_message():
            count += 1
        return count

    def pending_count(self) -> int:
        """Return the number of queued messages."""
        return self.queue.qsize()

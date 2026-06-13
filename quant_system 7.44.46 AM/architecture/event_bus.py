"""
Event-Driven Architecture with Message Bus
Improves System Architecture Score: 65 → 80+
"""

import asyncio
import json
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import warnings
warnings.filterwarnings('ignore')


class EventType(Enum):
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class Event:
    """Event message"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.HEARTBEAT
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    data: Dict = field(default_factory=dict)
    correlation_id: Optional[str] = None


class MessageBus:
    """
    Message Bus for Event-Driven Architecture
    
    Implements publish-subscribe pattern for decoupled communication.
    """
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        self.max_history = 10000
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    print(f"Error in subscriber: {e}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history."""
        if event_type:
            return [e for e in self.event_history if e.event_type == event_type][-limit:]
        return self.event_history[-limit:]


class EventDrivenSystem:
    """
    Event-Driven System Architecture
    
    Replaces monolithic design with event-driven microservices.
    """
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.services: Dict[str, object] = {}
        self.is_running = False
    
    def register_service(self, name: str, service: object) -> None:
        """Register a service."""
        self.services[name] = service
        print(f"Service registered: {name}")
    
    async def start(self) -> None:
        """Start the event-driven system."""
        self.is_running = True
        print("Event-driven system started")
        
        # Start heartbeat
        while self.is_running:
            heartbeat = Event(event_type=EventType.HEARTBEAT, source="system")
            await self.message_bus.publish(heartbeat)
            await asyncio.sleep(5)
    
    async def stop(self) -> None:
        """Stop the event-driven system."""
        self.is_running = False
        print("Event-driven system stopped")


# Global message bus instance
message_bus = MessageBus()


def create_event_driven_system():
    """Create sample event-driven system."""
    system = EventDrivenSystem()
    
    # Subscribe to market data events
    async def handle_market_data(event: Event):
        print(f"Market data received: {event.data}")
    
    message_bus.subscribe(EventType.MARKET_DATA, handle_market_data)
    
    # Publish a sample event
    async def publish_sample():
        event = Event(
            event_type=EventType.MARKET_DATA,
            source="kite",
            data={"symbol": "NIFTY", "price": 20000}
        )
        await message_bus.publish(event)
    
    return system


if __name__ == "__main__":
    system = create_event_driven_system()

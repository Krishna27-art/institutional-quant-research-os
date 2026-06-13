"""
Institutional-Grade Event Bus for Layered Architecture

This module implements a production-ready event bus that enables proper
event-driven communication between architectural layers.

Key Features:
- Type-safe event definitions
- Event filtering and routing
- Event persistence for audit trail
- Event replay capabilities
- Circuit breaking for failing subscribers
- Dead letter queue for failed events
- Metrics and monitoring
- Support for both sync and async subscribers

Architecture Integration:
- Infrastructure Layer: Event bus foundation
- Data Layer: Publishes DATA_UPDATED events
- Feature Layer: Publishes FEATURES_COMPUTED events
- Research Layer: Publishes SIGNAL_GENERATED events
- Portfolio Layer: Publishes PORTFOLIO_CONSTRUCTED events
- Execution Layer: Publishes ORDER_EXECUTED events
"""

import asyncio
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any, Set
from uuid import uuid4
import queue
import time

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for layered architecture communication."""
    
    # Data Layer Events
    DATA_UPDATED = "data.updated"
    DATA_VALIDATED = "data.validated"
    DATA_STALE = "data.stale"
    DATA_ERROR = "data.error"
    
    # Feature Layer Events
    FEATURES_COMPUTED = "features.computed"
    FEATURES_STORED = "features.stored"
    FEATURES_ERROR = "features.error"
    
    # Research Layer Events
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_VALIDATED = "signal.validated"
    MODEL_TRAINED = "model.trained"
    RESEARCH_ERROR = "research.error"
    
    # Portfolio Layer Events
    PORTFOLIO_CONSTRUCTED = "portfolio.constructed"
    RISK_CHECK = "risk.check"
    ALLOCATION_UPDATED = "allocation.updated"
    PORTFOLIO_ERROR = "portfolio.error"
    
    # Execution Layer Events
    ORDER_SUBMITTED = "order.submitted"
    ORDER_EXECUTED = "order.executed"
    ORDER_CANCELLED = "order.cancelled"
    POSITION_UPDATED = "position.updated"
    EXECUTION_ERROR = "execution.error"
    
    # System Events
    HEARTBEAT = "system.heartbeat"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker.tripped"
    SYSTEM_ERROR = "system.error"


@dataclass
class Event:
    """Event message with metadata."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = EventType.HEARTBEAT
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"
    source_layer: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None  # For event chains
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "source_layer": self.source_layer,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            source_layer=data.get("source_layer", "unknown"),
            data=data["data"],
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            metadata=data.get("metadata", {})
        )


class EventFilter(ABC):
    """Abstract base class for event filters."""
    
    @abstractmethod
    def matches(self, event: Event) -> bool:
        """Check if event matches filter."""
        pass


class EventTypeFilter(EventFilter):
    """Filter events by type."""
    
    def __init__(self, event_types: List[EventType]):
        self.event_types = set(event_types)
    
    def matches(self, event: Event) -> bool:
        return event.event_type in self.event_types


class SourceLayerFilter(EventFilter):
    """Filter events by source layer."""
    
    def __init__(self, layers: List[str]):
        self.layers = set(layers)
    
    def matches(self, event: Event) -> bool:
        return event.source_layer in self.layers


class DataFilter(EventFilter):
    """Filter events by data content."""
    
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
    
    def matches(self, event: Event) -> bool:
        return event.data.get(self.key) == self.value


class CircuitBreaker:
    """Circuit breaker for failing subscribers."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_calls = 0
        self._lock = threading.Lock()
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self.state == "half_open":
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "closed"
                    self.failure_count = 0
                    self.half_open_calls = 0
                    logger.info("Circuit breaker closed after successful recovery")
            elif self.state == "closed":
                self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                # Check if timeout has elapsed
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed > self.timeout_seconds:
                        self.state = "half_open"
                        self.half_open_calls = 0
                        logger.info("Circuit breaker moved to half-open state")
                        return True
                return False
            elif self.state == "half_open":
                return self.half_open_calls < self.half_open_max_calls
            return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "half_open_calls": self.half_open_calls
            }


class DeadLetterQueue:
    """Queue for events that failed to be processed."""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.max_size = max_size
    
    def put(self, event: Event, error: Exception) -> None:
        """Add failed event to dead letter queue."""
        try:
            self.queue.put({
                "event": event,
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            }, block=False)
            logger.warning(f"Event {event.event_id} added to dead letter queue")
        except queue.Full:
            logger.error("Dead letter queue full, dropping event")
    
    def get(self) -> Optional[Dict[str, Any]]:
        """Get next failed event from queue."""
        try:
            return self.queue.get(block=False)
        except queue.Empty:
            return None
    
    def size(self) -> int:
        """Get queue size."""
        return self.queue.qsize()


class EventBus:
    """
    Institutional-grade event bus for layered architecture.
    
    Features:
    - Type-safe event publishing and subscribing
    - Event filtering
    - Circuit breaking for failing subscribers
    - Dead letter queue for failed events
    - Event persistence for audit trail
    - Event replay capabilities
    - Metrics and monitoring
    """
    
    def __init__(
        self,
        enable_persistence: bool = True,
        db_path: str = "data/event_bus.db",
        max_history: int = 10000
    ):
        """
        Args:
            enable_persistence: Whether to persist events to database
            db_path: Path to event database
            max_history: Maximum number of events to keep in memory
        """
        self.subscribers: Dict[EventType, List[tuple]] = {}  # event_type -> [(callback, filter, circuit_breaker)]
        self.event_history: List[Event] = []
        self.max_history = max_history
        self.enable_persistence = enable_persistence
        self.db_path = db_path
        self.dead_letter_queue = DeadLetterQueue()
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            "events_published": 0,
            "events_delivered": 0,
            "events_failed": 0,
            "subscriber_errors": 0
        }
        
        # Initialize persistence if enabled
        if enable_persistence:
            self._init_persistence()
        
        logger.info("EventBus initialized with institutional-grade features")
    
    def _init_persistence(self) -> None:
        """Initialize event persistence database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_layer TEXT NOT NULL,
                    data TEXT NOT NULL,
                    correlation_id TEXT,
                    causation_id TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON events(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_layer 
                ON events(source_layer)
            """)
            conn.commit()
        
        logger.info(f"Event persistence initialized at {self.db_path}")
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable,
        filter: Optional[EventFilter] = None,
        enable_circuit_breaker: bool = True
    ) -> None:
        """
        Subscribe to events with optional filtering and circuit breaking.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published
            filter: Optional event filter
            enable_circuit_breaker: Whether to enable circuit breaking
        """
        circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append((callback, filter, circuit_breaker))
        
        logger.info(f"Subscribed to {event_type.value} with filter={filter}, circuit_breaker={enable_circuit_breaker}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Unsubscribe from events."""
        with self._lock:
            if event_type in self.subscribers:
                self.subscribers[event_type] = [
                    (cb, f, cbk) for cb, f, cbk in self.subscribers[event_type]
                    if cb != callback
                ]
        
        logger.info(f"Unsubscribed from {event_type.value}")
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Update metrics
        self.metrics["events_published"] += 1
        
        # Add to history
        with self._lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
        
        # Persist event if enabled
        if self.enable_persistence:
            self._persist_event(event)
        
        # Notify subscribers
        if event.event_type in self.subscribers:
            for callback, event_filter, circuit_breaker in self.subscribers[event.event_type]:
                # Apply filter
                if event_filter and not event_filter.matches(event):
                    continue
                
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.can_execute():
                    logger.warning(f"Circuit breaker blocking callback for {event.event_type.value}")
                    continue
                
                try:
                    # Execute callback
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                    
                    # Record success
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    self.metrics["events_delivered"] += 1
                    
                except Exception as e:
                    logger.error(f"Error in subscriber for {event.event_type.value}: {e}")
                    
                    # Record failure
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    # Add to dead letter queue
                    self.dead_letter_queue.put(event, e)
                    
                    self.metrics["events_failed"] += 1
                    self.metrics["subscriber_errors"] += 1
    
    def _persist_event(self, event: Event) -> None:
        """Persist event to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO events 
                    (event_id, event_type, timestamp, source, source_layer, data, 
                     correlation_id, causation_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.source,
                    event.source_layer,
                    json.dumps(event.data),
                    event.correlation_id,
                    event.causation_id,
                    json.dumps(event.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist event: {e}")
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        source_layer: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Get event history from memory."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source_layer:
            events = [e for e in events if e.source_layer == source_layer]
        
        return events[-limit:]
    
    def replay_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Event]:
        """
        Replay events from persistence.
        
        Args:
            event_type: Optional event type filter
            start_time: Optional start time filter
            end_time: Optional end time filter
        
        Returns:
            List of replayed events
        """
        if not self.enable_persistence:
            logger.warning("Event persistence not enabled, cannot replay")
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM events WHERE 1=1"
                params = []
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                query += " ORDER BY timestamp"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    events.append(Event(
                        event_id=row[0],
                        event_type=EventType(row[1]),
                        timestamp=datetime.fromisoformat(row[2]),
                        source=row[3],
                        source_layer=row[4],
                        data=json.loads(row[5]),
                        correlation_id=row[6],
                        causation_id=row[7],
                        metadata=json.loads(row[8]) if row[8] else {}
                    ))
                
                logger.info(f"Replayed {len(events)} events from persistence")
                return events
                
        except Exception as e:
            logger.error(f"Failed to replay events: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        with self._lock:
            return {
                **self.metrics,
                "subscriber_count": sum(len(subs) for subs in self.subscribers.values()),
                "event_history_size": len(self.event_history),
                "dead_letter_queue_size": self.dead_letter_queue.size(),
                "circuit_breaker_states": self._get_circuit_breaker_states()
            }
    
    def _get_circuit_breaker_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers."""
        states = {}
        for event_type, subscribers in self.subscribers.items():
            for i, (callback, event_filter, circuit_breaker) in enumerate(subscribers):
                if circuit_breaker:
                    key = f"{event_type.value}_subscriber_{i}"
                    states[key] = circuit_breaker.get_state()
        return states
    
    def process_dead_letter_queue(self) -> int:
        """
        Attempt to reprocess events from dead letter queue.
        
        Returns:
            Number of events successfully reprocessed
        """
        reprocessed = 0
        max_attempts = 100
        
        for _ in range(max_attempts):
            item = self.dead_letter_queue.get()
            if not item:
                break
            
            event = item["event"]
            try:
                # Republish the event
                asyncio.create_task(self.publish(event))
                reprocessed += 1
                logger.info(f"Successfully reprocessed event {event.event_id}")
            except Exception as e:
                logger.error(f"Failed to reprocess event {event.event_id}: {e}")
                # Put back in queue
                self.dead_letter_queue.put(event, Exception(str(e)))
        
        logger.info(f"Reprocessed {reprocessed} events from dead letter queue")
        return reprocessed


# Singleton instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the singleton event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


if __name__ == "__main__":
    # Test the event bus
    print("Testing Institutional-Grade Event Bus...")
    
    bus = EventBus(enable_persistence=False)  # Disable persistence for testing
    
    # Subscribe to events
    def handle_data_updated(event: Event):
        print(f"Data updated: {event.data}")
    
    bus.subscribe(EventType.DATA_UPDATED, handle_data_updated)
    
    # Publish an event
    async def test():
        event = Event(
            event_type=EventType.DATA_UPDATED,
            source="data_layer",
            source_layer="data",
            data={"symbol": "RELIANCE", "price": 2500.0}
        )
        await bus.publish(event)
        
        # Get metrics
        metrics = bus.get_metrics()
        print(f"Event bus metrics: {metrics}")
    
    asyncio.run(test())
    print("Event bus test completed successfully")

"""
Event-driven engine for market research and paper trading.
"""

from __future__ import annotations

from typing import Any, Callable
from datetime import datetime, timezone
from pathlib import Path
import logging

from .events import EventBus, Event, EventType


logger = logging.getLogger(__name__)


class EventDrivenEngine:
    """
    Event-driven engine that processes market events through a pipeline.
    """
    
    def __init__(self, journal_path: Path | None = None) -> None:
        self.event_bus = EventBus()
        self.is_running = False
        self._handlers: dict[str, Callable] = {}
        self.journal_path = journal_path
    
    @classmethod
    def with_journal(cls, journal_path: Path) -> "EventDrivenEngine":
        """Create engine with journal path."""
        return cls(journal_path=journal_path)
    
    def start(self) -> None:
        """Start the event engine."""
        self.is_running = True
        logger.info("EventDrivenEngine started")
    
    def stop(self) -> None:
        """Stop the event engine."""
        self.is_running = False
        logger.info("EventDrivenEngine stopped")
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe to an event type."""
        self.event_bus.subscribe(event_type, handler)
    
    def emit(self, event_type: EventType, payload: dict[str, Any]) -> Event:
        """Emit an event."""
        event = Event(event_type=event_type, payload=payload)
        return self.event_bus.emit(event)
    
    def run_bars(self, bars: list[dict[str, Any]]) -> None:
        """Process bar data."""
        for bar in bars:
            self.emit(EventType.BAR, bar)
    
    def replay(self, journal: Any) -> list[Event]:
        """Replay events from journal."""
        return self.get_history()
    
    def get_history(self) -> list[Event]:
        """Get event history."""
        return self.event_bus.history
    
    def reset(self) -> None:
        """Reset the engine."""
        self.event_bus.reset()

"""Small deterministic event bus for research and paper trading."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, DefaultDict


class EventType(str, Enum):
    MARKET_OPEN = "market_open"
    BAR = "bar"
    MARKET_STATE = "market_state"
    SIGNAL = "signal"
    VALIDATION = "validation"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    MARKET_CLOSE = "market_close"


@dataclass(frozen=True, slots=True)
class Event:
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }


EventHandler = Callable[[Event], None]


class EventBus:
    """Synchronous event dispatcher with explicit event types."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[EventType, list[EventHandler]] = defaultdict(list)
        self.history: list[Event] = []

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def emit(self, event: Event) -> Event:
        stored = event if event.sequence is not None else replace(event, sequence=len(self.history))
        self.history.append(stored)
        for handler in list(self._handlers[stored.event_type]):
            handler(stored)
        return stored

    def reset(self) -> None:
        self.history.clear()

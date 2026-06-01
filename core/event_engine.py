"""Exchange-like event loop for daily/intraday research runs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .events import Event, EventBus, EventType
from research_os.replay import ReplayJournal


class EventDrivenEngine:
    """Minimal event loop that keeps strategy, risk, and execution decoupled."""

    def __init__(self, bus: EventBus | None = None, journal: ReplayJournal | None = None) -> None:
        self.bus = bus or EventBus()
        self.journal = journal

    @classmethod
    def with_journal(cls, journal_path: str | Path) -> "EventDrivenEngine":
        return cls(journal=ReplayJournal(journal_path))

    def emit(self, event_type: EventType, payload: dict[str, Any] | None = None) -> Event:
        event = self.bus.emit(Event(event_type, dict(payload or {})))
        if self.journal is not None:
            self.journal.append_event(event)
        return event

    def run_bars(self, bars: Iterable[dict[str, Any]]) -> list[Event]:
        self.emit(EventType.MARKET_OPEN)
        for bar in bars:
            self.emit(EventType.BAR, dict(bar))
        self.emit(EventType.MARKET_CLOSE)
        return list(self.bus.history)

    def replay(self, journal: ReplayJournal) -> list[Event]:
        self.bus.reset()
        for replay_event in journal.load():
            event_type = EventType(replay_event.event_type)
            self.bus.emit(
                Event(
                    event_type=event_type,
                    payload=dict(replay_event.payload),
                    timestamp=replay_event.timestamp or "",
                    sequence=replay_event.sequence,
                )
            )
        return list(self.bus.history)

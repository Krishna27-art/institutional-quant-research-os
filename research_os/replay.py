"""Deterministic replay journal inspired by event-sourced trading engines."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    sequence: int
    event_type: str
    timestamp: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayJournal:
    """Append-only replay log for research decisions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_events (
                    sequence INTEGER PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT,
                    payload_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(replay_events)").fetchall()}
            if "timestamp" not in columns:
                conn.execute("ALTER TABLE replay_events ADD COLUMN timestamp TEXT")

    def append(self, event_type: str, payload: dict[str, Any], timestamp: str | None = None) -> ReplayEvent:
        with sqlite3.connect(self.db_path) as conn:
            next_seq = conn.execute("SELECT COALESCE(MAX(sequence), -1) + 1 FROM replay_events").fetchone()[0]
            event = ReplayEvent(sequence=int(next_seq), event_type=event_type, timestamp=timestamp, payload=payload)
            conn.execute(
                "INSERT INTO replay_events(sequence, event_type, timestamp, payload_json, fingerprint) VALUES (?, ?, ?, ?, ?)",
                (event.sequence, event.event_type, event.timestamp, json.dumps(event.payload, sort_keys=True), event.fingerprint()),
            )
        return event

    def append_event(self, event: Any) -> ReplayEvent:
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        event_type = str(data["event_type"])
        timestamp = data.get("timestamp")
        payload = dict(data.get("payload", {}))
        return self.append(event_type=event_type, payload=payload, timestamp=timestamp)

    def load(self) -> list[ReplayEvent]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT sequence, event_type, timestamp, payload_json FROM replay_events ORDER BY sequence").fetchall()
        return [
            ReplayEvent(sequence=int(seq), event_type=event_type, timestamp=timestamp, payload=json.loads(payload_json))
            for seq, event_type, timestamp, payload_json in rows
        ]

    def verify(self) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT sequence, event_type, timestamp, payload_json, fingerprint FROM replay_events ORDER BY sequence").fetchall()
        for seq, event_type, timestamp, payload_json, fingerprint in rows:
            event = ReplayEvent(sequence=int(seq), event_type=event_type, timestamp=timestamp, payload=json.loads(payload_json))
            if event.fingerprint() != fingerprint:
                return False
        return True

    def fingerprints(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            return [row[0] for row in conn.execute("SELECT fingerprint FROM replay_events ORDER BY sequence").fetchall()]

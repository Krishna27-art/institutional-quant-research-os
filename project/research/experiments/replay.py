"""Deterministic replay journal for events and decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from core.events import Event


@dataclass(frozen=True, slots=True)
class ReplayEntry:
    sequence: int
    event_type: str
    timestamp: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReplayJournal:
    """Append-only journal for deterministic event replay."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._has_fingerprint = False
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS replay_events (
                    sequence INTEGER PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL
                )
                """
            )
            columns = [row[1] for row in conn.execute("PRAGMA table_info(replay_events)").fetchall()]
            self._has_fingerprint = "fingerprint" in columns
            conn.commit()

    @staticmethod
    def _fingerprint(sequence: int, event_type: str, timestamp: str, payload: dict[str, Any]) -> str:
        material = json.dumps(
            {
                "sequence": sequence,
                "event_type": event_type,
                "timestamp": timestamp,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def append_event(self, event: Event) -> ReplayEntry:
        sequence = 0 if event.sequence is None else int(event.sequence)
        entry = ReplayEntry(
            sequence=sequence,
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            payload=dict(event.payload),
        )
        fingerprint = self._fingerprint(entry.sequence, entry.event_type, entry.timestamp, entry.payload)
        with self._connect() as conn:
            if self._has_fingerprint:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO replay_events (sequence, event_type, timestamp, payload_json, fingerprint)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entry.sequence,
                        entry.event_type,
                        entry.timestamp,
                        json.dumps(entry.payload, sort_keys=True),
                        fingerprint,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO replay_events (sequence, event_type, timestamp, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entry.sequence, entry.event_type, entry.timestamp, json.dumps(entry.payload, sort_keys=True)),
                )
            conn.commit()
        return entry

    def append(self, event_type: str, payload: dict[str, Any], timestamp: str) -> ReplayEntry:
        existing = self.load()
        entry = ReplayEntry(
            sequence=len(existing),
            event_type=event_type,
            timestamp=timestamp,
            payload=dict(payload),
        )
        fingerprint = self._fingerprint(entry.sequence, entry.event_type, entry.timestamp, entry.payload)
        with self._connect() as conn:
            if self._has_fingerprint:
                conn.execute(
                    """
                    INSERT INTO replay_events (sequence, event_type, timestamp, payload_json, fingerprint)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entry.sequence,
                        entry.event_type,
                        entry.timestamp,
                        json.dumps(entry.payload, sort_keys=True),
                        fingerprint,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO replay_events (sequence, event_type, timestamp, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entry.sequence, entry.event_type, entry.timestamp, json.dumps(entry.payload, sort_keys=True)),
                )
            conn.commit()
        return entry

    def load(self) -> list[ReplayEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sequence, event_type, timestamp, payload_json FROM replay_events ORDER BY sequence ASC"
            ).fetchall()
        return [
            ReplayEntry(
                sequence=int(sequence),
                event_type=str(event_type),
                timestamp=str(timestamp),
                payload=json.loads(payload_json),
            )
            for sequence, event_type, timestamp, payload_json in rows
        ]

    def verify(self) -> bool:
        entries = self.load()
        return all(entry.sequence == idx for idx, entry in enumerate(entries))

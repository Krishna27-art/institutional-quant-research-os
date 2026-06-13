"""Immutable experiment tracking backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    hypothesis_id: str
    data_fingerprint: str
    created_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    parent_experiment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    """Simple append-only store for research experiments."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    data_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    parent_experiment_id TEXT
                )
                """
            )
            conn.commit()

    def save(self, record: ExperimentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, hypothesis_id, data_fingerprint, created_at,
                    params_json, metrics_json, parent_experiment_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.hypothesis_id,
                    record.data_fingerprint,
                    record.created_at,
                    json.dumps(record.params, sort_keys=True),
                    json.dumps(record.metrics, sort_keys=True),
                    record.parent_experiment_id,
                ),
            )
            conn.commit()


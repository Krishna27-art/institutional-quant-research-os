"""Immutable experiment tracking."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    hypothesis_id: str
    data_fingerprint: str
    created_at: str
    params: dict[str, Any]
    metrics: dict[str, float]
    parent_experiment_id: str | None = None

    @staticmethod
    def fingerprint_payload(payload: Mapping[str, Any]) -> str:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
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

    def save(self, record: ExperimentRecord) -> None:
        with sqlite3.connect(self.path) as conn:
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


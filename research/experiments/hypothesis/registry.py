"""Persistent hypothesis registry."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import Hypothesis


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    hypothesis_id: str
    status: str
    updated_at: str
    payload_json: str


class HypothesisRegistry:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def upsert(self, hypothesis: Hypothesis, status: str = "proposed", updated_at: str | None = None) -> None:
        payload = json.dumps(hypothesis.to_dict(), sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hypotheses(hypothesis_id, status, updated_at, payload_json)
                VALUES (?, ?, COALESCE(?, datetime('now')), ?)
                """,
                (hypothesis.hypothesis_id, status, updated_at, payload),
            )

    def update_status(self, hypothesis_id: str, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE hypotheses
                SET status = ?, updated_at = datetime('now')
                WHERE hypothesis_id = ?
                """,
                (status, hypothesis_id),
            )

    def list(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM hypotheses ORDER BY updated_at DESC", conn)

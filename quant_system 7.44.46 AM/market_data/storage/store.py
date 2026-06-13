"""SQLite-backed data store for cleaned research datasets."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class StoredFrameMeta:
    symbol: str
    table_name: str
    rows: int
    validation_score: float | None = None
    updated_at: str | None = None


class DataStore:
    """Persist validated research data locally."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    validation_score REAL,
                    updated_at TEXT,
                    PRIMARY KEY(symbol, table_name)
                )
                """
            )

    def save_frame(self, symbol: str, frame: pd.DataFrame, table_name: str = "ohlcv") -> StoredFrameMeta:
        table = self._table_name(symbol, table_name)
        df = frame.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table, conn, if_exists="replace", index=False)
            conn.execute(
                """
                INSERT OR REPLACE INTO data_metadata(symbol, table_name, rows, validation_score, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (symbol.upper(), table_name, len(df), None),
            )
        return StoredFrameMeta(symbol.upper(), table_name, len(df))

    def load_frame(self, symbol: str, table_name: str = "ohlcv") -> pd.DataFrame:
        table = self._table_name(symbol, table_name)
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f'SELECT * FROM "{table}" ORDER BY date', conn)

    def set_validation_score(self, symbol: str, table_name: str, score: float) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE data_metadata
                SET validation_score = ?, updated_at = datetime('now')
                WHERE symbol = ? AND table_name = ?
                """,
                (score, symbol.upper(), table_name),
            )

    def metadata(self, symbol: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM data_metadata"
        params: tuple[Any, ...] = ()
        if symbol is not None:
            query += " WHERE symbol = ?"
            params = (symbol.upper(),)
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    @staticmethod
    def _table_name(symbol: str, table_name: str) -> str:
        safe_symbol = symbol.upper().replace("-", "_")
        safe_table = table_name.strip().lower().replace("-", "_")
        return f"{safe_symbol}_{safe_table}"

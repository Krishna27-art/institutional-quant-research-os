"""
Prediction storage system for audit trail and evaluation.

CRITICAL FIX: Added database persistence for predictions to enable
tracking and performance measurement.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import sqlite3
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Single prediction record."""
    symbol: str
    direction: str  # 'long' or 'short'
    confidence: float
    target_price: float
    stop_loss: float
    entry_price: float
    timestamp: datetime
    strategy: str
    realized_return: Optional[float] = None
    is_correct: Optional[bool] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    id: Optional[int] = None


from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "src" / "data" / "prediction_registry.db")

class PredictionStorage:
    """Database storage for predictions with evaluation tracking."""
    
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=1000")
        except Exception:
            pass
        return conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                predicted_return REAL DEFAULT 0.0,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                horizon_minutes INTEGER NOT NULL DEFAULT 390,
                target_price REAL DEFAULT 0.0,
                stop_loss REAL DEFAULT 0.0,
                exit_price REAL,
                realized_return REAL,
                exit_timestamp TEXT,
                is_correct INTEGER,
                ic_contribution REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol ON predictions(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy ON predictions(strategy)
        """)
        
        # Dynamic migration: ensure newer columns exist
        cursor.execute("PRAGMA table_info(predictions)")
        cols = [col[1] for col in cursor.fetchall()]
        if "target_price" not in cols:
            cursor.execute("ALTER TABLE predictions ADD COLUMN target_price REAL DEFAULT 0.0")
        if "stop_loss" not in cols:
            cursor.execute("ALTER TABLE predictions ADD COLUMN stop_loss REAL DEFAULT 0.0")
        if "is_correct" not in cols:
            cursor.execute("ALTER TABLE predictions ADD COLUMN is_correct INTEGER")

        conn.commit()
        conn.close()
        logger.info(f"Prediction storage initialized and migrated at {self.db_path}")
    
    def store_prediction(self, prediction: Prediction) -> int:
        """Store a new prediction."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                symbol, direction, confidence, target_price, stop_loss,
                entry_price, timestamp, strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.symbol,
            prediction.direction,
            prediction.confidence,
            prediction.target_price,
            prediction.stop_loss,
            prediction.entry_price,
            prediction.timestamp.isoformat(),
            prediction.strategy
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Stored prediction {prediction_id} for {prediction.symbol}")
        return prediction_id
    
    def update_outcome(self, prediction_id: int, exit_price: float, exit_timestamp: datetime):
        """Update prediction with actual outcome."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get original prediction
        cursor.execute("SELECT entry_price, direction, target_price, stop_loss FROM predictions WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            logger.warning(f"Prediction {prediction_id} not found")
            return
        
        entry_price, direction, target_price, stop_loss = row
        
        # Calculate realized return
        if direction == 'long':
            realized_return = (exit_price - entry_price) / entry_price
            is_correct = exit_price >= target_price
        else:  # short
            realized_return = (entry_price - exit_price) / entry_price
            is_correct = exit_price <= target_price
        
        cursor.execute("""
            UPDATE predictions
            SET exit_price = ?, exit_timestamp = ?, realized_return = ?, is_correct = ?
            WHERE id = ?
        """, (
            exit_price,
            exit_timestamp.isoformat(),
            realized_return,
            1 if is_correct else 0,
            prediction_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated prediction {prediction_id} with outcome: return={realized_return:.2%}, correct={is_correct}")
    
    def get_predictions(self, symbol: Optional[str] = None, strategy: Optional[str] = None, 
                      limit: int = 100) -> List[Prediction]:
        """Retrieve predictions with optional filters."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT id, symbol, direction, confidence, target_price, stop_loss,
                   entry_price, timestamp, strategy, realized_return, is_correct,
                   exit_price, exit_timestamp
            FROM predictions
        """
        params = []
        
        if symbol or strategy:
            conditions = []
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append(Prediction(
                symbol=row[1],
                direction=row[2],
                confidence=row[3],
                target_price=row[4],
                stop_loss=row[5],
                entry_price=row[6],
                timestamp=datetime.fromisoformat(row[7]),
                strategy=row[8],
                realized_return=row[9],
                is_correct=bool(row[10]) if row[10] is not None else None,
                exit_price=row[11],
                exit_timestamp=datetime.fromisoformat(row[12]) if row[12] else None,
                id=row[0]
            ))
        
        return predictions
    
    def get_performance_metrics(self, strategy: Optional[str] = None) -> dict:
        """Calculate prediction performance metrics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Query total, pending, and realized counts
        count_query = "SELECT COUNT(*), SUM(CASE WHEN exit_price IS NULL THEN 1 ELSE 0 END), SUM(CASE WHEN exit_price IS NOT NULL THEN 1 ELSE 0 END), MAX(timestamp) FROM predictions"
        count_params = []
        if strategy:
            count_query += " WHERE strategy = ?"
            count_params.append(strategy)
            
        cursor.execute(count_query, count_params)
        total_all, pending_all, realized_all, last_time = cursor.fetchone()
        
        total_all = total_all or 0
        pending_all = pending_all or 0
        realized_all = realized_all or 0
        
        # Query detailed metrics for realized predictions
        perf_query = """
            SELECT 
                COUNT(*) as total_realized,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(realized_return) as avg_return,
                AVG(CASE WHEN direction = 'long' THEN realized_return END) as avg_long_return,
                AVG(CASE WHEN direction = 'short' THEN realized_return END) as avg_short_return
            FROM predictions
            WHERE realized_return IS NOT NULL
        """
        perf_params = []
        if strategy:
            perf_query += " AND strategy = ?"
            perf_params.append(strategy)
            
        cursor.execute(perf_query, perf_params)
        total_realized, correct, avg_return, avg_long_return, avg_short_return = cursor.fetchone()
        conn.close()
        
        total_realized = total_realized or 0
        correct = correct or 0
        win_rate = correct / total_realized if total_realized > 0 else 0.0
        
        return {
            "total_predictions": total_all,
            "pending": pending_all,
            "realized": realized_all,
            "accuracy": win_rate,
            "win_rate": win_rate,
            "realized_predictions": realized_all,
            "pending_predictions": pending_all,
            "last_prediction_time": last_time or "None",
            "avg_return": avg_return or 0.0,
            "avg_long_return": avg_long_return or 0.0,
            "avg_short_return": avg_short_return or 0.0
        }

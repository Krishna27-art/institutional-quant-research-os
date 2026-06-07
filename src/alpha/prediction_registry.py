"""
Prediction Registry — track every prediction, its outcome, and compute IC.

This replaces the basic alpha/prediction_storage.py with a real registry that:
1. Stores every prediction with entry metadata
2. Resolves outcomes when exit prices arrive
3. Computes rolling Information Coefficient (IC) per strategy
4. Auto-flags strategies with IC < min_ic_threshold for demotion
5. Provides per-strategy and aggregate performance reports

The registry uses SQLite for simplicity (no external DB dependency).
Migrate to PostgreSQL when the system goes live.
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Default config ──────────────────────────────────────────────────
DEFAULT_DB_PATH = "data/prediction_registry.db"
MIN_IC_THRESHOLD = 0.02  # Demote strategies with IC below this
IC_LOOKBACK_DAYS = 90    # Rolling window for IC computation
MIN_PREDICTIONS_FOR_IC = 30  # Need at least this many resolved predictions


@dataclass
class PredictionRecord:
    """A single prediction with outcome tracking."""
    symbol: str
    strategy: str
    direction: str           # 'long' or 'short'
    predicted_return: float  # Expected return magnitude
    confidence: float        # Model confidence [0, 1]
    entry_price: float
    timestamp: datetime
    horizon_minutes: int = 390  # Default: 1 trading day (6.5h)

    # Filled after outcome resolution
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    ic_contribution: Optional[float] = None  # rank IC for this prediction
    id: Optional[int] = None


@dataclass
class StrategyReport:
    """Performance report for a strategy."""
    strategy: str
    total_predictions: int
    resolved_predictions: int
    pending_predictions: int
    hit_rate: float          # % of correct direction calls
    mean_ic: float           # Mean rank IC
    rolling_ic: float        # IC over last IC_LOOKBACK_DAYS
    avg_return: float        # Average realized return
    sharpe: float            # Annualized Sharpe of realized returns
    is_active: bool          # False if IC < threshold
    last_prediction: Optional[datetime] = None


class PredictionRegistry:
    """
    Central prediction registry with IC tracking.

    Thread-safe via connection-per-call pattern (SQLite limitation).
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        min_ic: float = MIN_IC_THRESHOLD,
        ic_lookback_days: int = IC_LOOKBACK_DAYS,
    ):
        self.db_path = db_path
        self.min_ic = min_ic
        self.ic_lookback_days = ic_lookback_days
        self._lock = threading.Lock()

        # Ensure parent dir exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Database setup ──────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    predicted_return REAL NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL DEFAULT 390,
                    exit_price REAL,
                    realized_return REAL,
                    exit_timestamp TEXT,
                    ic_contribution REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_strategy
                ON predictions(strategy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_symbol
                ON predictions(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_timestamp
                ON predictions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_resolved
                ON predictions(exit_price)
            """)

            # Strategy demotion log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_demotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    ic_value REAL,
                    demoted_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()
            logger.info(f"PredictionRegistry initialized at {self.db_path}")

    # ── Core operations ─────────────────────────────────────────────

    def record_prediction(self, pred: PredictionRecord) -> int:
        """Store a new prediction. Returns the prediction ID."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    symbol, strategy, direction, predicted_return,
                    confidence, entry_price, timestamp, horizon_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred.symbol,
                pred.strategy,
                pred.direction,
                pred.predicted_return,
                pred.confidence,
                pred.entry_price,
                pred.timestamp.isoformat(),
                pred.horizon_minutes,
            ))

            pred_id = cursor.lastrowid
            conn.commit()
            conn.close()

        logger.debug(
            f"Recorded prediction #{pred_id}: {pred.strategy} {pred.direction} "
            f"{pred.symbol} @ {pred.entry_price:.2f}"
        )
        return pred_id

    def resolve_prediction(
        self,
        prediction_id: int,
        exit_price: float,
        exit_timestamp: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Resolve a prediction with actual outcome.

        Returns the realized return, or None if prediction not found.
        """
        exit_ts = exit_timestamp or datetime.now()

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT entry_price, direction, predicted_return FROM predictions WHERE id = ?",
                (prediction_id,),
            )
            row = cursor.fetchone()
            if not row:
                conn.close()
                logger.warning(f"Prediction #{prediction_id} not found")
                return None

            entry_price, direction, predicted_return = row

            # Compute realized return
            if direction == "long":
                realized = (exit_price - entry_price) / entry_price
            else:
                realized = (entry_price - exit_price) / entry_price

            # IC contribution: correlation between predicted and realized
            # For a single prediction, this is the sign agreement
            ic_contrib = 1.0 if (predicted_return * realized > 0) else -1.0

            cursor.execute("""
                UPDATE predictions
                SET exit_price = ?, realized_return = ?, exit_timestamp = ?, ic_contribution = ?
                WHERE id = ?
            """, (exit_price, realized, exit_ts.isoformat(), ic_contrib, prediction_id))

            conn.commit()
            conn.close()

        logger.debug(
            f"Resolved prediction #{prediction_id}: realized={realized:.4f}, ic={ic_contrib}"
        )
        return realized

    def resolve_expired(self, current_prices: Dict[str, float]) -> int:
        """
        Resolve all expired predictions using current prices.

        A prediction is expired if:
        - It has no exit_price yet
        - Its timestamp + horizon has passed

        Returns number of predictions resolved.
        """
        now = datetime.now()
        resolved_count = 0

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, symbol, entry_price, direction, predicted_return,
                       timestamp, horizon_minutes
                FROM predictions
                WHERE exit_price IS NULL
            """)

            rows = cursor.fetchall()

            for row in rows:
                pred_id, symbol, entry_price, direction, predicted_return, ts_str, horizon = row
                pred_time = datetime.fromisoformat(ts_str)
                expiry = pred_time + timedelta(minutes=horizon)

                if now >= expiry and symbol in current_prices:
                    exit_price = current_prices[symbol]

                    if direction == "long":
                        realized = (exit_price - entry_price) / entry_price
                    else:
                        realized = (entry_price - exit_price) / entry_price

                    ic_contrib = 1.0 if (predicted_return * realized > 0) else -1.0

                    cursor.execute("""
                        UPDATE predictions
                        SET exit_price = ?, realized_return = ?,
                            exit_timestamp = ?, ic_contribution = ?
                        WHERE id = ?
                    """, (exit_price, realized, now.isoformat(), ic_contrib, pred_id))

                    resolved_count += 1

            conn.commit()
            conn.close()

        if resolved_count > 0:
            logger.info(f"Auto-resolved {resolved_count} expired predictions")

        return resolved_count

    # ── IC computation ──────────────────────────────────────────────

    def compute_ic(
        self,
        strategy: Optional[str] = None,
        lookback_days: Optional[int] = None,
    ) -> float:
        """
        Compute rank Information Coefficient for a strategy.

        IC = Spearman correlation between predicted_return and realized_return
        across resolved predictions in the lookback window.

        Returns 0.0 if insufficient data.
        """
        lookback = lookback_days or self.ic_lookback_days
        cutoff = (datetime.now() - timedelta(days=lookback)).isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()

        query = """
            SELECT predicted_return, realized_return
            FROM predictions
            WHERE realized_return IS NOT NULL
              AND timestamp >= ?
        """
        params = [cutoff]

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < MIN_PREDICTIONS_FOR_IC:
            return 0.0

        predicted = np.array([r[0] for r in rows])
        realized = np.array([r[1] for r in rows])

        # Spearman rank correlation
        corr, _ = stats.spearmanr(predicted, realized)

        if np.isnan(corr):
            return 0.0

        return float(corr)

    def get_strategy_ic_map(self) -> Dict[str, float]:
        """Compute IC for every strategy that has enough data."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT strategy FROM predictions
            WHERE realized_return IS NOT NULL
        """)
        strategies = [row[0] for row in cursor.fetchall()]
        conn.close()

        return {s: self.compute_ic(strategy=s) for s in strategies}

    def check_demotions(self) -> List[str]:
        """
        Check which strategies should be demoted (IC < threshold).

        Returns list of strategy names that should be deactivated.
        """
        ic_map = self.get_strategy_ic_map()
        demoted = []

        for strategy, ic in ic_map.items():
            if ic < self.min_ic and ic != 0.0:  # 0.0 means insufficient data
                demoted.append(strategy)
                self._log_demotion(strategy, ic)
                logger.warning(
                    f"Strategy '{strategy}' DEMOTED: IC={ic:.4f} < threshold={self.min_ic}"
                )

        return demoted

    def _log_demotion(self, strategy: str, ic: float) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO strategy_demotions (strategy, reason, ic_value)
            VALUES (?, ?, ?)
        """, (strategy, f"IC {ic:.4f} < {self.min_ic}", ic))
        conn.commit()
        conn.close()

    # ── Reporting ───────────────────────────────────────────────────

    def get_strategy_report(self, strategy: str) -> StrategyReport:
        """Generate performance report for a strategy."""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Counts
        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE strategy = ?", (strategy,)
        )
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE strategy = ? AND exit_price IS NOT NULL",
            (strategy,),
        )
        resolved = cursor.fetchone()[0]

        # Hit rate
        cursor.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE strategy = ? AND realized_return IS NOT NULL AND realized_return > 0
        """, (strategy,))
        correct = cursor.fetchone()[0]

        # Returns
        cursor.execute("""
            SELECT realized_return FROM predictions
            WHERE strategy = ? AND realized_return IS NOT NULL
            ORDER BY timestamp
        """, (strategy,))
        returns = [r[0] for r in cursor.fetchall()]

        # Last prediction
        cursor.execute("""
            SELECT MAX(timestamp) FROM predictions WHERE strategy = ?
        """, (strategy,))
        last_ts = cursor.fetchone()[0]

        conn.close()

        # Compute metrics
        hit_rate = correct / resolved if resolved > 0 else 0.0
        avg_return = float(np.mean(returns)) if returns else 0.0
        mean_ic = self.compute_ic(strategy=strategy, lookback_days=365)
        rolling_ic = self.compute_ic(strategy=strategy)

        # Sharpe
        if len(returns) >= 2:
            ret_arr = np.array(returns)
            sharpe = float(np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(252))
        else:
            sharpe = 0.0

        is_active = rolling_ic >= self.min_ic or rolling_ic == 0.0

        return StrategyReport(
            strategy=strategy,
            total_predictions=total,
            resolved_predictions=resolved,
            pending_predictions=total - resolved,
            hit_rate=hit_rate,
            mean_ic=mean_ic,
            rolling_ic=rolling_ic,
            avg_return=avg_return,
            sharpe=sharpe,
            is_active=is_active,
            last_prediction=datetime.fromisoformat(last_ts) if last_ts else None,
        )

    def get_all_reports(self) -> List[StrategyReport]:
        """Generate reports for all strategies."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT strategy FROM predictions")
        strategies = [r[0] for r in cursor.fetchall()]
        conn.close()

        return [self.get_strategy_report(s) for s in strategies]

    def get_summary(self) -> Dict:
        """Quick summary of the prediction registry."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE exit_price IS NULL")
        pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE exit_price IS NOT NULL")
        resolved = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT strategy) FROM predictions")
        n_strategies = cursor.fetchone()[0]

        conn.close()

        ic_map = self.get_strategy_ic_map()
        demoted = [s for s, ic in ic_map.items() if ic < self.min_ic and ic != 0.0]

        return {
            "total_predictions": total,
            "pending": pending,
            "resolved": resolved,
            "strategies_tracked": n_strategies,
            "strategy_ics": ic_map,
            "demoted_strategies": demoted,
        }


# ── Module-level singleton ──────────────────────────────────────────
_registry: Optional[PredictionRegistry] = None


def get_prediction_registry() -> PredictionRegistry:
    """Get the singleton prediction registry."""
    global _registry
    if _registry is None:
        _registry = PredictionRegistry()
    return _registry

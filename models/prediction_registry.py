"""
Prediction Registry for Accountability

This module implements a prediction registry to track all predictions
for accountability and accuracy tracking, as required by institutional systems.

Key Features:
- Prediction logging with metadata
- Prediction vs actual tracking
- Model attribution
- Feature hash for reproducibility
- Prediction confidence tracking
- Performance analytics by model

Based on Audit Report Priority 1: Research Quality
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)


class PredictionStatus(Enum):
    """Prediction status."""
    PENDING = "pending"
    REALIZED = "realized"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class PredictionRecord:
    """Prediction record."""
    prediction_id: str
    model_id: str
    symbol: str
    prediction_time: datetime
    prediction_value: float  # -1 to 1
    confidence: float  # 0 to 1
    features_hash: str
    target_time: Optional[datetime] = None
    actual_value: Optional[float] = None
    prediction_error: Optional[float] = None
    status: PredictionStatus = PredictionStatus.PENDING
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def calculate_error(self) -> Optional[float]:
        """Calculate prediction error if actual value is known."""
        if self.actual_value is None:
            return None
        return self.prediction_value - self.actual_value
    
    def is_correct(self, threshold: float = 0.0) -> Optional[bool]:
        """Check if prediction was correct (directional)."""
        if self.actual_value is None:
            return None
        return (self.prediction_value > threshold) == (self.actual_value > threshold)


class PredictionRegistry:
    """
    Prediction registry for accountability.
    
    This class tracks all predictions and their outcomes for
    model performance analysis and accountability.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize prediction registry.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path) if db_path else Path(__file__).parent / "predictions.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        logger.info(f"PredictionRegistry initialized at {self.db_path}")
    
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prediction_time TIMESTAMP NOT NULL,
                prediction_value REAL NOT NULL,
                confidence REAL NOT NULL,
                features_hash TEXT NOT NULL,
                target_time TIMESTAMP,
                actual_value REAL,
                prediction_error REAL,
                status TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_id ON predictions(model_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol ON predictions(symbol)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_time ON predictions(prediction_time)
        """)
        
        conn.commit()
        conn.close()
    
    def log_prediction(
        self,
        model_id: str,
        symbol: str,
        prediction_value: float,
        confidence: float,
        features: Dict[str, float],
        target_time: Optional[datetime] = None,
        metadata: Dict = None
    ) -> str:
        """
        Log a prediction.
        
        Args:
            model_id: Model identifier
            symbol: Stock symbol
            prediction_value: Prediction value (-1 to 1)
            confidence: Confidence score (0 to 1)
            features: Feature dictionary used for prediction
            target_time: Target time for prediction realization
            metadata: Additional metadata
            
        Returns:
            Prediction ID
        """
        # Generate prediction ID
        prediction_id = str(hashlib.uuid4())
        
        # Generate features hash
        features_str = json.dumps(features, sort_keys=True)
        features_hash = hashlib.md5(features_str.encode()).hexdigest()
        
        # Convert metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else "{}"
        
        # Insert into database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                prediction_id, model_id, symbol, prediction_time,
                prediction_value, confidence, features_hash,
                target_time, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id, model_id, symbol, datetime.now(),
            prediction_value, confidence, features_hash,
            target_time, PredictionStatus.PENDING.value, metadata_json
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged prediction {prediction_id} for {symbol}")
        return prediction_id
    
    def realize_prediction(
        self,
        prediction_id: str,
        actual_value: float,
        status: PredictionStatus = PredictionStatus.REALIZED
    ) -> None:
        """
        Mark a prediction as realized with actual value.
        
        Args:
            prediction_id: Prediction identifier
            actual_value: Actual realized value
            status: Prediction status
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get prediction value
        cursor.execute(
            "SELECT prediction_value FROM predictions WHERE prediction_id = ?",
            (prediction_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"Prediction {prediction_id} not found")
            conn.close()
            return
        
        prediction_value = result[0]
        prediction_error = prediction_value - actual_value
        
        # Update prediction
        cursor.execute("""
            UPDATE predictions
            SET actual_value = ?, prediction_error = ?, status = ?
            WHERE prediction_id = ?
        """, (actual_value, prediction_error, status.value, prediction_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Realized prediction {prediction_id} with error {prediction_error:.4f}")
    
    def get_predictions(
        self,
        model_id: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[PredictionStatus] = None,
        limit: int = 1000
    ) -> List[PredictionRecord]:
        """
        Get predictions with optional filters.
        
        Args:
            model_id: Filter by model ID
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of records
            
        Returns:
            List of prediction records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY prediction_time DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        records = []
        for row in rows:
            record = PredictionRecord(
                prediction_id=row[0],
                model_id=row[1],
                symbol=row[2],
                prediction_time=pd.to_datetime(row[3]),
                prediction_value=row[4],
                confidence=row[5],
                features_hash=row[6],
                target_time=pd.to_datetime(row[7]) if row[7] else None,
                actual_value=row[8],
                prediction_error=row[9],
                status=PredictionStatus(row[10]),
                metadata=json.loads(row[11]) if row[11] else {}
            )
            records.append(record)
        
        return records
    
    def get_model_performance(self, model_id: str) -> Dict:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Performance metrics dictionary
        """
        predictions = self.get_predictions(model_id=model_id, status=PredictionStatus.REALIZED)
        
        if not predictions:
            return {}
        
        # Calculate metrics
        errors = [p.prediction_error for p in predictions if p.prediction_error is not None]
        correct = sum(1 for p in predictions if p.is_correct())
        total = len(predictions)
        
        metrics = {
            'total_predictions': total,
            'realized_predictions': len(predictions),
            'accuracy': correct / total if total > 0 else 0.0,
            'mean_absolute_error': np.mean([abs(e) for e in errors]) if errors else 0.0,
            'mean_squared_error': np.mean([e**2 for e in errors]) if errors else 0.0,
            'mean_error': np.mean(errors) if errors else 0.0,
            'std_error': np.std(errors) if errors else 0.0
        }
        
        return metrics
    
    def get_symbol_performance(self, symbol: str) -> Dict:
        """
        Get performance metrics for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Performance metrics dictionary
        """
        predictions = self.get_predictions(symbol=symbol, status=PredictionStatus.REALIZED)
        
        if not predictions:
            return {}
        
        errors = [p.prediction_error for p in predictions if p.prediction_error is not None]
        correct = sum(1 for p in predictions if p.is_correct())
        total = len(predictions)
        
        metrics = {
            'total_predictions': total,
            'accuracy': correct / total if total > 0 else 0.0,
            'mean_absolute_error': np.mean([abs(e) for e in errors]) if errors else 0.0,
            'mean_error': np.mean(errors) if errors else 0.0
        }
        
        return metrics
    
    def cleanup_old_predictions(self, days: int = 90) -> int:
        """
        Clean up old predictions.
        
        Args:
            days: Delete predictions older than this many days
            
        Returns:
            Number of predictions deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "DELETE FROM predictions WHERE prediction_time < ?",
            (cutoff_date,)
        )
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted {deleted} old predictions")
        return deleted
    
    def export_predictions(self, output_path: str) -> None:
        """
        Export predictions to CSV.
        
        Args:
            output_path: Output file path
        """
        predictions = self.get_predictions(limit=10000)
        
        data = []
        for p in predictions:
            data.append({
                'prediction_id': p.prediction_id,
                'model_id': p.model_id,
                'symbol': p.symbol,
                'prediction_time': p.prediction_time,
                'prediction_value': p.prediction_value,
                'confidence': p.confidence,
                'features_hash': p.features_hash,
                'target_time': p.target_time,
                'actual_value': p.actual_value,
                'prediction_error': p.prediction_error,
                'status': p.status.value
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(df)} predictions to {output_path}")
    
    def print_summary(self) -> None:
        """Print prediction registry summary."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = ?", (PredictionStatus.PENDING.value,))
        pending = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = ?", (PredictionStatus.REALIZED.value,))
        realized = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT model_id) FROM predictions")
        models = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM predictions")
        symbols = cursor.fetchone()[0]
        
        conn.close()
        
        print("\n" + "="*60)
        print("PREDICTION REGISTRY SUMMARY")
        print("="*60)
        print(f"\nTotal Predictions: {total}")
        print(f"Pending: {pending}")
        print(f"Realized: {realized}")
        print(f"Unique Models: {models}")
        print(f"Unique Symbols: {symbols}")
        print("\n" + "="*60)


# Singleton instance
_prediction_registry = None

def get_prediction_registry() -> PredictionRegistry:
    """Get the singleton prediction registry instance."""
    global _prediction_registry
    if _prediction_registry is None:
        _prediction_registry = PredictionRegistry()
    return _prediction_registry


if __name__ == "__main__":
    # Test the prediction registry
    print("Testing Prediction Registry...")
    
    registry = PredictionRegistry()
    
    # Log some predictions
    for i in range(10):
        registry.log_prediction(
            model_id="model_v1",
            symbol="RELIANCE",
            prediction_value=np.random.uniform(-1, 1),
            confidence=np.random.uniform(0.5, 0.9),
            features={'returns': np.random.randn(), 'volume': np.random.randn()}
        )
    
    # Realize some predictions
    predictions = registry.get_predictions(limit=5)
    for p in predictions:
        registry.realize_prediction(p.prediction_id, np.random.uniform(-1, 1))
    
    # Print summary
    registry.print_summary()

"""
Truth Database - Single Source of Truth for Market Data

This module implements the single source of truth for all market data in the system.
All downstream consumers (feature store, research, execution) must read from this database.

Key Principles:
1. Only validated data enters the truth database
2. Immutable audit trail - all writes are logged
3. Single interface - all consumers use the same API
4. Data lineage - track source and validation metadata
5. Versioning - maintain data versions for reproducibility
6. Fast reads - optimized for query performance

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│              Data Validation Pipeline                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Truth Database                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Validated    │  │ Audit Trail  │  │ Data         │          │
│  │ Market Data  │  │              │  │ Lineage      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Downstream Consumers                            │
│  (Feature Store, Research Layer, Execution Engine)               │
└─────────────────────────────────────────────────────────────────┘
"""

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class DataMetadata:
    """Metadata for stored data."""
    symbol: str
    source: str
    validation_status: str
    validated_at: datetime
    data_type: str
    interval: str
    record_count: int
    checksum: str
    version: int = 1


class TruthDatabase:
    """
    Single source of truth for validated market data.
    
    This database stores only data that has passed validation.
    All downstream consumers must read from this database.
    """
    
    def __init__(self, db_path: str = "data/truth_database.db"):
        """
        Initialize truth database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_db()
        
        logger.info(f"TruthDatabase initialized at {db_path}")
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            
            # Main data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    data_type TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    source TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, interval, version)
                )
            """)
            
            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    validated_at TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audit trail table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    record_count INTEGER,
                    source TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    details TEXT
                )
            """)
            
            # Data lineage table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    transformation_steps TEXT,
                    validation_results TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol 
                ON market_data(symbol)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
                ON market_data(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON market_data(symbol, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_symbol 
                ON data_metadata(symbol)
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Truth database schema initialized")
    
    @contextmanager
    def _get_conn(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data integrity."""
        # Use hash of DataFrame values for checksum
        return str(hash(pd.util.hash_pandas_object(data)))
    
    def write_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        source: str,
        validation_status: str,
        data_type: str = "ohlcv",
        interval: str = "1min",
        validation_results: Optional[List[Dict]] = None
    ) -> bool:
        """
        Write validated data to truth database.
        
        Args:
            symbol: Stock/index symbol
            data: Validated DataFrame
            source: Data source
            validation_status: Status from validation pipeline
            data_type: Type of data (ohlcv, tick, etc.)
            interval: Data interval
            validation_results: List of validation result dictionaries
        
        Returns:
            True if write successful, False otherwise
        """
        if data.empty:
            logger.warning(f"Cannot write empty data for {symbol}")
            return False
        
        try:
            with self._lock:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    # Get current version
                    cursor.execute("""
                        SELECT MAX(version) FROM market_data 
                        WHERE symbol = ? AND interval = ?
                    """, (symbol, interval))
                    result = cursor.fetchone()
                    current_version = result[0] if result[0] else 0
                    new_version = current_version + 1
                    
                    # Prepare data for insertion
                    records = []
                    for idx, row in data.iterrows():
                        timestamp_str = idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        records.append((
                            symbol,
                            timestamp_str,
                            float(row.get('open', 0)),
                            float(row.get('high', 0)),
                            float(row.get('low', 0)),
                            float(row.get('close', 0)),
                            int(row.get('volume', 0)),
                            data_type,
                            interval,
                            source,
                            new_version
                        ))
                    
                    # Insert data
                    cursor.executemany("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timestamp, open, high, low, close, volume, 
                         data_type, interval, source, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)
                    
                    # Write metadata
                    checksum = self._calculate_checksum(data)
                    cursor.execute("""
                        INSERT INTO data_metadata 
                        (symbol, source, validation_status, validated_at, 
                         data_type, interval, record_count, checksum, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        source,
                        validation_status,
                        datetime.now().isoformat(),
                        data_type,
                        interval,
                        len(data),
                        checksum,
                        new_version
                    ))
                    
                    # Write audit trail
                    cursor.execute("""
                        INSERT INTO audit_trail 
                        (action, symbol, record_count, source, details)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        "write",
                        symbol,
                        len(data),
                        source,
                        f"version={new_version}, validation_status={validation_status}"
                    ))
                    
                    # Write data lineage
                    if validation_results:
                        import json
                        cursor.execute("""
                            INSERT INTO data_lineage 
                            (symbol, source, validation_results)
                            VALUES (?, ?, ?)
                        """, (
                            symbol,
                            source,
                            json.dumps(validation_results)
                        ))
                    
                    logger.info(
                        f"Wrote {len(data)} records for {symbol} to truth database "
                        f"(version {new_version}, source: {source})"
                    )
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to write data for {symbol}: {e}")
            return False
    
    def read_data(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: str = "1min",
        version: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read data from truth database.
        
        Args:
            symbol: Stock/index symbol
            start: Start datetime (optional)
            end: End datetime (optional)
            interval: Data interval
            version: Specific version to read (default: latest)
        
        Returns:
            DataFrame with market data
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = ? AND interval = ?
                """
                params = [symbol, interval]
                
                if version is not None:
                    query += " AND version = ?"
                    params.append(version)
                else:
                    # Get latest version
                    query += " AND version = (SELECT MAX(version) FROM market_data WHERE symbol = ? AND interval = ?)"
                    params.extend([symbol, interval])
                
                if start is not None:
                    query += " AND timestamp >= ?"
                    params.append(start.isoformat())
                
                if end is not None:
                    query += " AND timestamp <= ?"
                    params.append(end.isoformat())
                
                query += " ORDER BY timestamp"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    logger.debug(f"No data found for {symbol} in truth database")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in rows])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                logger.debug(f"Read {len(df)} records for {symbol} from truth database")
                return df
                
        except Exception as e:
            logger.error(f"Failed to read data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_timestamp(self, symbol: str, interval: str = "1min") -> Optional[datetime]:
        """Get the latest timestamp for a symbol."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(timestamp) FROM market_data
                    WHERE symbol = ? AND interval = ?
                """, (symbol, interval))
                result = cursor.fetchone()
                if result and result[0]:
                    return pd.to_datetime(result[0])
                return None
        except Exception as e:
            logger.error(f"Failed to get latest timestamp for {symbol}: {e}")
            return None
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols in truth database."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT symbol FROM market_data")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def get_metadata(self, symbol: str, interval: str = "1min") -> Optional[DataMetadata]:
        """Get metadata for a symbol."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM data_metadata
                    WHERE symbol = ? AND interval = ?
                    ORDER BY version DESC
                    LIMIT 1
                """, (symbol, interval))
                row = cursor.fetchone()
                if row:
                    return DataMetadata(
                        symbol=row['symbol'],
                        source=row['source'],
                        validation_status=row['validation_status'],
                        validated_at=pd.to_datetime(row['validated_at']),
                        data_type=row['data_type'],
                        interval=row['interval'],
                        record_count=row['record_count'],
                        checksum=row['checksum'],
                        version=row['version']
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get metadata for {symbol}: {e}")
            return None
    
    def get_audit_trail(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit trail entries."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM audit_trail"
                params = []
                
                if symbol:
                    query += " WHERE symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return []
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of truth database."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                # Count total records
                cursor.execute("SELECT COUNT(*) FROM market_data")
                total_records = cursor.fetchone()[0]
                
                # Count unique symbols
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data")
                unique_symbols = cursor.fetchone()[0]
                
                # Count by source
                cursor.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM market_data 
                    GROUP BY source
                """)
                by_source = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Count by validation status
                cursor.execute("""
                    SELECT validation_status, COUNT(*) as count 
                    FROM data_metadata 
                    GROUP BY validation_status
                """)
                by_status = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_records": total_records,
                    "unique_symbols": unique_symbols,
                    "by_source": by_source,
                    "by_validation_status": by_status
                }
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {}
    
    def cleanup_old_versions(self, keep_versions: int = 3) -> int:
        """
        Clean up old versions of data, keeping only the latest N versions.
        
        Args:
            keep_versions: Number of versions to keep
        
        Returns:
            Number of records deleted
        """
        try:
            with self._lock:
                with self._get_conn() as conn:
                    cursor = conn.cursor()
                    
                    # Get versions to keep
                    cursor.execute("""
                        SELECT symbol, interval, MAX(version) as max_version
                        FROM market_data
                        GROUP BY symbol, interval
                    """)
                    
                    deleted_count = 0
                    for row in cursor.fetchall():
                        symbol, interval, max_version = row
                        min_version_to_keep = max_version - keep_versions + 1
                        
                        if min_version_to_keep > 0:
                            cursor.execute("""
                                DELETE FROM market_data
                                WHERE symbol = ? AND interval = ? AND version < ?
                            """, (symbol, interval, min_version_to_keep))
                            deleted_count += cursor.rowcount
                    
                    conn.commit()
                    logger.info(f"Cleaned up {deleted_count} old version records")
                    return deleted_count
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old versions: {e}")
            return 0


# Singleton instance
_truth_db: Optional[TruthDatabase] = None


def get_truth_database() -> TruthDatabase:
    """Get the singleton truth database instance."""
    global _truth_db
    if _truth_db is None:
        _truth_db = TruthDatabase()
    return _truth_db


if __name__ == "__main__":
    # Test the truth database
    print("Testing Truth Database...")
    
    db = TruthDatabase(":memory:")  # Use in-memory database for testing
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    test_data = pd.DataFrame({
        'open': np.random.uniform(1000, 1100, 100),
        'high': np.random.uniform(1100, 1200, 100),
        'low': np.random.uniform(900, 1000, 100),
        'close': np.random.uniform(1000, 1100, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Write data
    success = db.write_data(
        'RELIANCE',
        test_data,
        'yahoo',
        'valid',
        'ohlcv',
        '1min'
    )
    print(f"Write success: {success}")
    
    # Read data
    read_data = db.read_data('RELIANCE', interval='1min')
    print(f"Read {len(read_data)} records")
    
    # Get summary
    summary = db.get_summary()
    print(f"Database summary: {summary}")
    
    print("Truth Database test completed successfully")

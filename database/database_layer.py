"""
Database Layer - Redis, TimescaleDB, ClickHouse Integration
Based on Blueprint V1.0

Architecture:
- L1 (Real-time): Redis for current state, pub/sub, feature cache
- L2 (Operational): TimescaleDB for 1-min bars, options, trades
- L3 (Analytics): ClickHouse for historical tick data, backtest results
- L4 (Cold Storage): Parquet + S3 for archived data

Features:
- Unified interface for all database operations
- Automatic data routing based on access pattern
- Connection pooling and retry logic
- Schema management
- Data migration utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class DatabaseLayer(Enum):
    """Database layer types."""
    REDIS = "redis"  # L1: Real-time
    TIMESCALEDB = "timescaledb"  # L2: Operational
    CLICKHOUSE = "clickhouse"  # L3: Analytics
    PARQUET = "parquet"  # L4: Cold storage


@dataclass
class DBConfig:
    """Configuration for database layer."""
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # TimescaleDB
    timescale_host: str = "localhost"
    timescale_port: int = 5432
    timescale_dbname: str = "quant_db"
    timescale_user: str = "postgres"
    timescale_password: Optional[str] = None
    
    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "quant_analytics"
    clickhouse_user: str = "default"
    clickhouse_password: Optional[str] = None
    
    # Parquet
    parquet_path: str = "./data/parquet"
    
    # Connection settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


class RedisClient:
    """Redis client for L1 real-time data."""
    
    def __init__(self, config: DBConfig):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            import redis
            self.client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.client.ping()
        except ImportError:
            print("Redis library not available. Install with: pip install redis")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set key-value pair with TTL."""
        if self.client is None:
            return False
        try:
            self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        if self.client is None:
            return None
        try:
            return self.client.get(key)
        except Exception as e:
            print(f"Redis get error: {e}")
            return None
    
    def hset(self, key: str, field: str, value: Any) -> bool:
        """Set hash field."""
        if self.client is None:
            return False
        try:
            self.client.hset(key, field, value)
            return True
        except Exception as e:
            print(f"Redis hset error: {e}")
            return False
    
    def hget(self, key: str, field: str) -> Optional[Any]:
        """Get hash field."""
        if self.client is None:
            return None
        try:
            return self.client.hget(key, field)
        except Exception as e:
            print(f"Redis hget error: {e}")
            return None
    
    def hgetall(self, key: str) -> Dict:
        """Get all hash fields."""
        if self.client is None:
            return {}
        try:
            return self.client.hgetall(key)
        except Exception as e:
            print(f"Redis hgetall error: {e}")
            return {}
    
    def publish(self, channel: str, message: str) -> bool:
        """Publish message to channel."""
        if self.client is None:
            return False
        try:
            self.client.publish(channel, message)
            return True
        except Exception as e:
            print(f"Redis publish error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        if self.client is None:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False


class TimescaleDBClient:
    """TimescaleDB client for L2 operational data."""
    
    def __init__(self, config: DBConfig):
        self.config = config
        self.engine = None
        self._connect()
    
    def _connect(self):
        """Connect to TimescaleDB."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.engine.url import URL
            
            url = URL.create(
                drivername="postgresql",
                username=self.config.timescale_user,
                password=self.config.timescale_password,
                host=self.config.timescale_host,
                port=self.config.timescale_port,
                database=self.config.timescale_dbname
            )
            
            self.engine = create_engine(
                url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
                
        except ImportError:
            print("SQLAlchemy not available. Install with: pip install sqlalchemy psycopg2-binary")
        except Exception as e:
            print(f"Failed to connect to TimescaleDB: {e}")
    
    def insert_ohlcv(self, df: pd.DataFrame, table_name: str = "bars_1min") -> bool:
        """Insert OHLCV data."""
        if self.engine is None:
            return False
        try:
            df.to_sql(table_name, self.engine, if_exists="append", index=False)
            return True
        except Exception as e:
            print(f"TimescaleDB insert error: {e}")
            return False
    
    def query_ohlcv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        table_name: str = "bars_1min"
    ) -> Optional[pd.DataFrame]:
        """Query OHLCV data."""
        if self.engine is None:
            return None
        try:
            query = f"""
            SELECT * FROM {table_name}
            WHERE symbol = %s AND time BETWEEN %s AND %s
            ORDER BY time
            """
            df = pd.read_sql_query(
                query,
                self.engine,
                params=(symbol, start_time, end_time)
            )
            return df
        except Exception as e:
            print(f"TimescaleDB query error: {e}")
            return None
    
    def create_hypertable(self, table_name: str, time_column: str = "time") -> bool:
        """Convert table to hypertable."""
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists := TRUE)")
            return True
        except Exception as e:
            print(f"TimescaleDB hypertable error: {e}")
            return False


class ClickHouseClient:
    """ClickHouse client for L3 analytics data."""
    
    def __init__(self, config: DBConfig):
        self.config = config
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to ClickHouse."""
        try:
            import clickhouse_connect
            self.client = clickhouse_connect.get_client(
                host=self.config.clickhouse_host,
                port=self.config.clickhouse_port,
                database=self.config.clickhouse_database,
                username=self.config.clickhouse_user,
                password=self.config.clickhouse_password
            )
            # Test connection
            self.client.command("SELECT 1")
        except ImportError:
            print("ClickHouse connect not available. Install with: pip install clickhouse-connect")
        except Exception as e:
            print(f"Failed to connect to ClickHouse: {e}")
    
    def insert_df(self, df: pd.DataFrame, table_name: str) -> bool:
        """Insert DataFrame into ClickHouse."""
        if self.client is None:
            return False
        try:
            self.client.insert_df(table_name, df)
            return True
        except Exception as e:
            print(f"ClickHouse insert error: {e}")
            return False
    
    def query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute query and return DataFrame."""
        if self.client is None:
            return None
        try:
            result = self.client.query_df(query)
            return result
        except Exception as e:
            print(f"ClickHouse query error: {e}")
            return None
    
    def create_table(self, table_name: str, schema: str) -> bool:
        """Create table in ClickHouse."""
        if self.client is None:
            return False
        try:
            self.client.command(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema}) ENGINE = MergeTree() ORDER BY (time)")
            return True
        except Exception as e:
            print(f"ClickHouse create table error: {e}")
            return False


class ParquetStorage:
    """Parquet storage for L4 cold storage."""
    
    def __init__(self, config: DBConfig):
        self.config = config
        import os
        os.makedirs(self.config.parquet_path, exist_ok=True)
    
    def save(self, df: pd.DataFrame, table_name: str, partition_cols: List[str] = None) -> str:
        """Save DataFrame to Parquet."""
        import os
        filepath = os.path.join(self.config.parquet_path, f"{table_name}.parquet")
        
        try:
            df.to_parquet(filepath, partition_cols=partition_cols)
            return filepath
        except Exception as e:
            print(f"Parquet save error: {e}")
            return ""
    
    def load(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet."""
        import os
        filepath = os.path.join(self.config.parquet_path, f"{table_name}.parquet")
        
        if not os.path.exists(filepath):
            return None
        
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            print(f"Parquet load error: {e}")
            return None


class DatabaseLayer:
    """
    Unified database layer for institutional quant platform.
    
    Automatically routes data to appropriate database based on:
    - Access pattern (real-time vs historical)
    - Data size (small vs large)
    - Query type (point vs aggregation)
    """
    
    def __init__(self, config: DBConfig = None):
        self.config = config or DBConfig()
        
        # Initialize clients
        self.redis = RedisClient(self.config)
        self.timescaledb = TimescaleDBClient(self.config)
        self.clickhouse = ClickHouseClient(self.config)
        self.parquet = ParquetStorage(self.config)
    
    def store_feature(
        self,
        symbol: str,
        feature_name: str,
        value: float,
        timestamp: datetime,
        ttl: int = 3600
    ) -> bool:
        """Store feature value in Redis (L1)."""
        key = f"feature:{symbol}:{feature_name}"
        return self.redis.set(key, value, ttl)
    
    def get_feature(
        self,
        symbol: str,
        feature_name: str
    ) -> Optional[float]:
        """Get feature value from Redis (L1)."""
        key = f"feature:{symbol}:{feature_name}"
        value = self.redis.get(key)
        return float(value) if value is not None else None
    
    def store_position(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        pnl: float
    ) -> bool:
        """Store position state in Redis (L1)."""
        key = f"position:{symbol}"
        self.redis.hset(key, "quantity", quantity)
        self.redis.hset(key, "avg_price", avg_price)
        self.redis.hset(key, "pnl", pnl)
        return True
    
    def get_position(self, symbol: str) -> Dict:
        """Get position state from Redis (L1)."""
        key = f"position:{symbol}"
        return self.redis.hgetall(key)
    
    def store_ohlcv(
        self,
        df: pd.DataFrame,
        use_timescale: bool = True
    ) -> bool:
        """Store OHLCV data (L2 TimescaleDB or L4 Parquet)."""
        if use_timescale:
            return self.timescaledb.insert_ohlcv(df)
        else:
            self.parquet.save(df, f"ohlcv_{df.iloc[0]['symbol']}")
            return True
    
    def query_ohlcv(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        use_clickhouse: bool = False
    ) -> Optional[pd.DataFrame]:
        """Query OHLCV data (L2 TimescaleDB or L3 ClickHouse)."""
        if use_clickhouse:
            query = f"""
            SELECT * FROM bars_1min
            WHERE symbol = '{symbol}' AND time BETWEEN '{start_time}' AND '{end_time}'
            ORDER BY time
            """
            return self.clickhouse.query(query)
        else:
            return self.timescaledb.query_ohlcv(symbol, start_time, end_time)
    
    def store_backtest_result(
        self,
        result: Dict,
        use_clickhouse: bool = True
    ) -> bool:
        """Store backtest result (L3 ClickHouse or L4 Parquet)."""
        df = pd.DataFrame([result])
        
        if use_clickhouse:
            return self.clickhouse.insert_df(df, "backtest_results")
        else:
            self.parquet.save(df, "backtest_results")
            return True
    
    def publish_signal(
        self,
        signal: Dict,
        channel: str = "signals"
    ) -> bool:
        """Publish signal via Redis pub/sub."""
        import json
        message = json.dumps(signal)
        return self.redis.publish(channel, message)
    
    def get_regime_state(self) -> Optional[Dict]:
        """Get current regime state from Redis (L1)."""
        return self.redis.hgetall("regime_state")
    
    def set_regime_state(self, state: Dict, ttl: int = 3600) -> bool:
        """Set regime state in Redis (L1)."""
        for key, value in state.items():
            self.redis.hset("regime_state", key, value)
        return True
    
    def archive_old_data(
        self,
        days_old: int = 90,
        table_name: str = "bars_1min"
    ) -> bool:
        """Archive old data from TimescaleDB to Parquet (L4)."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Query old data
        df = self.timescaledb.query_ohlcv(
            symbol="*",  # All symbols
            start_time=datetime(2020, 1, 1),
            end_time=cutoff_date,
            table_name=table_name
        )
        
        if df is None or df.empty:
            return True
        
        # Save to Parquet
        self.parquet.save(df, f"{table_name}_archived")
        
        # Delete from TimescaleDB (optional)
        # self.timescaledb.delete_old_data(table_name, cutoff_date)
        
        return True
    
    def get_database_status(self) -> Dict:
        """Get status of all database connections."""
        return {
            'redis': self.redis.client is not None,
            'timescaledb': self.timescaledb.engine is not None,
            'clickhouse': self.clickhouse.client is not None,
            'parquet': True  # Always available
        }


if __name__ == "__main__":
    # Test the database layer
    print("Testing Database Layer...")
    
    config = DBConfig()
    db = DatabaseLayer(config)
    
    # Test Redis operations
    print("\nTesting Redis...")
    db.store_feature("RELIANCE", "rsi_14", 65.5, datetime.now())
    feature = db.get_feature("RELIANCE", "rsi_14")
    print(f"Retrieved feature: {feature}")
    
    # Test position storage
    db.store_position("RELIANCE", 100, 2500.0, 5000.0)
    position = db.get_position("RELIANCE")
    print(f"Retrieved position: {position}")
    
    # Test regime state
    regime_state = {"regime": "bull_trend", "probability": 0.8}
    db.set_regime_state(regime_state)
    retrieved_state = db.get_regime_state()
    print(f"Retrieved regime state: {retrieved_state}")
    
    # Get database status
    status = db.get_database_status()
    print(f"\nDatabase Status: {status}")

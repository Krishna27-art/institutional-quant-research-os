"""
Database Architecture Layer for Architecture V2
Redis cache + ClickHouse analytics + PostgreSQL metadata

Architecture:
- Hot cache (last 24 hours): Redis (in-memory, 5ms)
- Real-time ingest: Redis Streams (one stream per symbol)
- Historical 1-min bars: ClickHouse (partition by symbol, time)
- Raw ticks: Parquet on S3 (partitioned by symbol/year/month)
- Features & signals: ClickHouse + Redis (latest only)
- Research: DuckDB on local Parquet files
"""

import redis
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_cache_duration_hours: int = 24
    
    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_database: str = "quant_trading"
    clickhouse_user: str = "default"
    clickhouse_password: Optional[str] = None
    
    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "quant_metadata"
    postgres_user: str = "postgres"
    postgres_password: Optional[str] = None
    
    # Parquet archive
    parquet_archive_path: str = "./data/archive"


class RedisCache:
    """
    Redis cache for hot data (last 24 hours).
    
    Use cases:
    - Latest market data per symbol
    - Feature vectors
    - Signal cache
    - Session state
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=True
        )
        self.cache_duration = timedelta(hours=config.redis_cache_duration_hours)
    
    def set_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Cache latest market data for symbol."""
        key = f"market:{symbol}"
        value = json.dumps(data)
        return self.client.setex(key, self.cache_duration, value)
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached market data for symbol."""
        key = f"market:{symbol}"
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_features(self, symbol: str, features: Dict[str, float]) -> bool:
        """Cache feature vector for symbol."""
        key = f"features:{symbol}"
        value = json.dumps(features)
        return self.client.setex(key, self.cache_duration, value)
    
    def get_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get cached features for symbol."""
        key = f"features:{symbol}"
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_signal(self, signal_id: str, signal: Dict[str, Any]) -> bool:
        """Cache signal."""
        key = f"signal:{signal_id}"
        value = json.dumps(signal)
        return self.client.setex(key, self.cache_duration, value)
    
    def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get cached signal."""
        key = f"signal:{signal_id}"
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_regime(self, regime: str, regime_data: Dict[str, Any]) -> bool:
        """Cache current regime."""
        key = "regime:current"
        value = json.dumps(regime_data)
        return self.client.setex(key, self.cache_duration, value)
    
    def get_regime(self) -> Optional[Dict[str, Any]]:
        """Get cached regime."""
        key = "regime:current"
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def publish_to_stream(self, stream_name: str, data: Dict[str, Any]) -> str:
        """Publish data to Redis Stream."""
        value = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                 for k, v in data.items()}
        return self.client.xadd(stream_name, value)
    
    def read_from_stream(self, stream_name: str, count: int = 10) -> List[Dict[str, Any]]:
        """Read data from Redis Stream."""
        try:
            entries = self.client.xread({stream_name: '0'}, count=count)
            result = []
            for stream, messages in entries:
                for message_id, data in messages:
                    result.append({
                        'id': message_id,
                        'data': {k: json.loads(v) if self._is_json(v) else v 
                                 for k, v in data.items()}
                    })
            return result
        except Exception as e:
            print(f"Error reading from stream: {e}")
            return []
    
    def _is_json(self, value: str) -> bool:
        """Check if string is JSON."""
        try:
            json.loads(value)
            return True
        except:
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern."""
        keys = self.client.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0
    
    def flush_cache(self) -> bool:
        """Flush all cache (use with caution)."""
        return self.client.flushdb()


class ClickHouseManager:
    """
    ClickHouse manager for historical analytics.
    
    Schema:
    - minute_bars: 1-minute OHLCV data
    - features: Feature vectors
    - signals: Alpha signals
    - trades: Executed trades
    - pnl: Performance metrics
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client = None
    
    def connect(self):
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
            print("Connected to ClickHouse")
        except ImportError:
            print("clickhouse-connect not installed. Install with: pip install clickhouse-connect")
        except Exception as e:
            print(f"Error connecting to ClickHouse: {e}")
    
    def create_schema(self):
        """Create database schema."""
        if not self.client:
            return
        
        # Create database if not exists
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.config.clickhouse_database}")
        
        # Create minute_bars table
        self.client.command("""
            CREATE TABLE IF NOT EXISTS minute_bars (
                date Date,
                datetime DateTime,
                symbol String,
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume UInt64,
                vwap Float64,
                trades UInt32
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(date)
            ORDER BY (symbol, datetime)
        """)
        
        # Create features table
        self.client.command("""
            CREATE TABLE IF NOT EXISTS features (
                datetime DateTime,
                symbol String,
                feature_vector Array(Float32),
                relative_volume Float32,
                vwap_distance Float32,
                realized_vol Float32,
                iv Float32,
                pcr Float32
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(datetime)
            ORDER BY (symbol, datetime)
        """)
        
        # Create signals table
        self.client.command("""
            CREATE TABLE IF NOT EXISTS signals (
                datetime DateTime,
                symbol String,
                alpha_name String,
                direction String,
                confidence Float32,
                expected_return Float32,
                regime String
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(datetime)
            ORDER BY (alpha_name, symbol, datetime)
        """)
        
        # Create trades table
        self.client.command("""
            CREATE TABLE IF NOT EXISTS trades (
                datetime DateTime,
                symbol String,
                side String,
                quantity UInt32,
                price Float64,
                commission Float64,
                strategy String
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(datetime)
            ORDER BY (symbol, datetime)
        """)
        
        # Create pnl table
        self.client.command("""
            CREATE TABLE IF NOT EXISTS pnl (
                date Date,
                strategy String,
                realized_pnl Float64,
                unrealized_pnl Float64,
                trades_count UInt32,
                sharpe_ratio Float32,
                max_drawdown Float32
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(date)
            ORDER BY (strategy, date)
        """)
        
        print("ClickHouse schema created")
    
    def insert_minute_bars(self, data: List[Dict[str, Any]]):
        """Insert minute bars."""
        if not self.client:
            return
        
        self.client.insert('minute_bars', data)
    
    def insert_features(self, data: List[Dict[str, Any]]):
        """Insert features."""
        if not self.client:
            return
        
        self.client.insert('features', data)
    
    def insert_signals(self, data: List[Dict[str, Any]]):
        """Insert signals."""
        if not self.client:
            return
        
        self.client.insert('signals', data)
    
    def insert_trades(self, data: List[Dict[str, Any]]):
        """Insert trades."""
        if not self.client:
            return
        
        self.client.insert('trades', data)
    
    def query_minute_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Query minute bars for symbol."""
        if not self.client:
            return pd.DataFrame()
        
        query = f"""
            SELECT * FROM minute_bars
            WHERE symbol = '{symbol}'
            AND datetime BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY datetime
        """
        
        result = self.client.query(query)
        return result.result_df
    
    def query_features(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Query features for symbol."""
        if not self.client:
            return pd.DataFrame()
        
        query = f"""
            SELECT * FROM features
            WHERE symbol = '{symbol}'
            AND datetime BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY datetime
        """
        
        result = self.client.query(query)
        return result.result_df
    
    def query_signals(
        self,
        alpha_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Query signals for alpha."""
        if not self.client:
            return pd.DataFrame()
        
        query = f"""
            SELECT * FROM signals
            WHERE alpha_name = '{alpha_name}'
            AND datetime BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY datetime
        """
        
        result = self.client.query(query)
        return result.result_df


class PostgreSQLManager:
    """
    PostgreSQL manager for metadata.
    
    Schema:
    - symbols: Symbol universe
    - strategies: Strategy configurations
    - experiments: Research experiments
    - users: User management
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
    
    def connect(self):
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            self.connection = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            print("Connected to PostgreSQL")
        except ImportError:
            print("psycopg2 not installed. Install with: pip install psycopg2-binary")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
    
    def create_schema(self):
        """Create database schema."""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        
        # Create symbols table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) UNIQUE NOT NULL,
                name VARCHAR(100),
                sector VARCHAR(50),
                exchange VARCHAR(20),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create strategies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                type VARCHAR(50),
                config JSONB,
                is_enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                name VARCHAR(200) NOT NULL,
                description TEXT,
                config JSONB,
                status VARCHAR(20) DEFAULT 'pending',
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                results JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        cursor.close()
        print("PostgreSQL schema created")
    
    def insert_symbol(self, symbol: str, name: str = None, sector: str = None, exchange: str = "NSE"):
        """Insert symbol."""
        if not self.connection:
            return
        
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO symbols (symbol, name, sector, exchange) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (symbol) DO UPDATE SET name = EXCLUDED.name, sector = EXCLUDED.sector",
            (symbol, name, sector, exchange)
        )
        self.connection.commit()
        cursor.close()
    
    def get_symbols(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get symbols."""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        if active_only:
            cursor.execute("SELECT * FROM symbols WHERE is_active = TRUE")
        else:
            cursor.execute("SELECT * FROM symbols")
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        
        return [dict(zip(columns, row)) for row in rows]


class DatabaseManager:
    """
    Unified database manager for Architecture V2.
    
    Coordinates Redis, ClickHouse, and PostgreSQL.
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.redis = RedisCache(config)
        self.clickhouse = ClickHouseManager(config)
        self.postgres = PostgreSQLManager(config)
    
    def initialize(self):
        """Initialize all database connections and schemas."""
        print("Initializing database architecture...")
        
        # Connect to ClickHouse and create schema
        self.clickhouse.connect()
        self.clickhouse.create_schema()
        
        # Connect to PostgreSQL and create schema
        self.postgres.connect()
        self.postgres.create_schema()
        
        # Test Redis connection
        try:
            self.redis.client.ping()
            print("Redis connection successful")
        except Exception as e:
            print(f"Redis connection failed: {e}")
        
        print("Database architecture initialized")
    
    def store_market_data(self, symbol: str, data: Dict[str, Any], persist_to_clickhouse: bool = False):
        """Store market data in cache and optionally persist."""
        # Cache in Redis
        self.redis.set_market_data(symbol, data)
        
        # Publish to stream for real-time processing
        self.redis.publish_to_stream(f"market:{symbol}", data)
        
        # Persist to ClickHouse if enabled
        if persist_to_clickhouse:
            clickhouse_data = [{
                'date': datetime.now().date(),
                'datetime': datetime.now(),
                'symbol': symbol,
                'open': data.get('open', 0),
                'high': data.get('high', 0),
                'low': data.get('low', 0),
                'close': data.get('close', 0),
                'volume': data.get('volume', 0),
                'vwap': data.get('vwap', 0),
                'trades': data.get('trades', 0)
            }]
            self.clickhouse.insert_minute_bars(clickhouse_data)
    
    def get_market_data(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get market data from cache or ClickHouse."""
        if use_cache:
            data = self.redis.get_market_data(symbol)
            if data:
                return data
        
        # Fallback to ClickHouse (implement query logic)
        return None
    
    def store_features(self, symbol: str, features: Dict[str, float], persist_to_clickhouse: bool = False):
        """Store features in cache and optionally persist."""
        self.redis.set_features(symbol, features)
        
        if persist_to_clickhouse:
            clickhouse_data = [{
                'datetime': datetime.now(),
                'symbol': symbol,
                'feature_vector': list(features.values()),
                'relative_volume': features.get('relative_volume', 0),
                'vwap_distance': features.get('vwap_distance_pct', 0),
                'realized_vol': features.get('realized_volatility_5d', 0),
                'iv': features.get('implied_volatility', 0),
                'pcr': features.get('put_call_ratio', 0)
            }]
            self.clickhouse.insert_features(clickhouse_data)
    
    def store_signal(self, signal: Dict[str, Any], persist_to_clickhouse: bool = False):
        """Store signal in cache and optionally persist."""
        signal_id = f"{signal['symbol']}_{signal['timestamp'].isoformat()}"
        self.redis.set_signal(signal_id, signal)
        
        if persist_to_clickhouse:
            clickhouse_data = [{
                'datetime': signal['timestamp'],
                'symbol': signal['symbol'],
                'alpha_name': signal.get('alpha_name', 'unknown'),
                'direction': signal['direction'].value,
                'confidence': signal['confidence'],
                'expected_return': signal['expected_return'],
                'regime': signal.get('regime', 'unknown')
            }]
            self.clickhouse.insert_signals(clickhouse_data)
    
    def close(self):
        """Close all database connections."""
        if self.clickhouse.client:
            self.clickhouse.client.close()
        if self.postgres.connection:
            self.postgres.connection.close()
        self.redis.client.close()
        print("Database connections closed")

"""
Database initialization script for TimescaleDB and ClickHouse.

CRITICAL FIX: Initialize database schemas and create tables for data persistence.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Initialize and setup TimescaleDB and ClickHouse databases."""
    
    def __init__(self, timescale_host: str = "localhost", timescale_port: int = 5432,
                 clickhouse_host: str = "localhost", clickhouse_port: int = 8123):
        self.timescale_host = timescale_host
        self.timescale_port = timescale_port
        self.clickhouse_host = clickhouse_host
        self.clickhouse_port = clickhouse_port
    
    def initialize_timescale(self, dbname: str = "quant_os", user: str = "postgres", 
                            password: str = "") -> bool:
        """Initialize TimescaleDB with required schemas and tables."""
        try:
            import psycopg2
        except ImportError:
            logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
            return False
        
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(
                host=self.timescale_host,
                port=self.timescale_port,
                user=user,
                password=password,
                dbname="postgres"  # Connect to default db first
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Create database if it doesn't exist
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{dbname}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {dbname}")
                logger.info(f"Created TimescaleDB database: {dbname}")
            
            conn.close()
            
            # Connect to the new database
            conn = psycopg2.connect(
                host=self.timescale_host,
                port=self.timescale_port,
                user=user,
                password=password,
                dbname=dbname
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Create extension if not exists
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            
            # Create market_data table with hypertable
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION
                )
            """)
            
            # Convert to hypertable if not already
            try:
                cursor.execute("SELECT 1 FROM _timescaledb_catalog.hypertable WHERE table_name = 'market_data'")
                is_hyper = cursor.fetchone()
            except Exception:
                is_hyper = False
            
            if not is_hyper:
                try:
                    cursor.execute("SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE)")
                except Exception as e:
                    logger.warning(f"Could not convert market_data to hypertable: {e}")
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol 
                ON market_data (symbol, timestamp DESC)
            """)
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id VARCHAR(255) PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    commission DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    exit_time TIMESTAMPTZ,
                    pnl DOUBLE PRECISION,
                    status TEXT
                )
            """)
            
            # Convert to hypertable
            try:
                cursor.execute("SELECT 1 FROM _timescaledb_catalog.hypertable WHERE table_name = 'trades'")
                is_hyper = cursor.fetchone()
            except Exception:
                is_hyper = False
            
            if not is_hyper:
                try:
                    cursor.execute("SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE)")
                except Exception as e:
                    logger.warning(f"Could not convert trades to hypertable: {e}")
            
            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    current_price DOUBLE PRECISION,
                    pnl DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            conn.close()
            logger.info("TimescaleDB initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"TimescaleDB initialization failed: {e}")
            return False
    
    def initialize_clickhouse(self, database: str = "quant_os") -> bool:
        """Initialize ClickHouse with required schemas and tables."""
        try:
            import clickhouse_connect
        except ImportError:
            logger.error("clickhouse-connect not installed. Run: pip install clickhouse-connect")
            return False
        
        try:
            client = clickhouse_connect.get_client(
                host=self.clickhouse_host,
                port=self.clickhouse_port,
                database="default"
            )
            
            # Create database if it doesn't exist
            client.command(f"CREATE DATABASE IF NOT EXISTS {database}")
            
            # Switch to the database
            client = clickhouse_connect.get_client(
                host=self.clickhouse_host,
                port=self.clickhouse_port,
                database=database
            )
            
            # Create market_data table
            client.command("""
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp DateTime,
                    symbol String,
                    open Float64,
                    high Float64,
                    low Float64,
                    close Float64,
                    volume Float64
                ) ENGINE = MergeTree()
                ORDER BY (symbol, timestamp)
            """)
            
            # Create trades table
            client.command("""
                CREATE TABLE IF NOT EXISTS trades (
                    id UInt64,
                    timestamp DateTime,
                    symbol String,
                    direction String,
                    quantity Int32,
                    price Float64,
                    commission Float64
                ) ENGINE = MergeTree()
                ORDER BY (timestamp, symbol)
            """)
            
            # Create analytics table
            client.command("""
                CREATE TABLE IF NOT EXISTS analytics (
                    timestamp DateTime,
                    metric_name String,
                    metric_value Float64,
                    tags Map(String, String)
                ) ENGINE = MergeTree()
                ORDER BY (metric_name, timestamp)
            """)
            
            logger.info("ClickHouse initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ClickHouse initialization failed: {e}")
            return False
    
    def initialize_all(self) -> dict:
        """Initialize both TimescaleDB and ClickHouse."""
        results = {
            "timescale": self.initialize_timescale(),
            "clickhouse": self.initialize_clickhouse()
        }
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initializer = DatabaseInitializer()
    results = initializer.initialize_all()
    print("Database initialization results:", results)

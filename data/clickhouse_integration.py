"""
ClickHouse Integration for Analytics
Based on Architecture V2 agent debate consensus

Key findings from research:
- ClickHouse wins for analytics (1 billion rows/sec on single node)
- Replaces TimescaleDB for historical 1-min bars
- Partition by symbol and time
- Store last 30 days hot, 2 years warm, 10+ years cold
- Redis for hot cache, ClickHouse for analytics, Parquet on S3 for archive

Architecture V2 - Quantitative Trading System for Indian Markets
Phase 1: ClickHouse for analytics, Redis for hot cache
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import clickhouse_connect


@dataclass
class ClickHouseConfig:
    """Configuration for ClickHouse connection"""
    host: str = "localhost"
    port: int = 8123
    database: str = "quantdb"
    user: str = "default"
    password: str = ""
    
    # Table configuration
    minute_bars_table: str = "minute_bars"
    features_table: str = "features"
    signals_table: str = "signals"


class ClickHouseManager:
    """
    ClickHouse Manager for time-series analytics.
    
    Architecture V2 Data Storage:
    - Hot cache (last 24 hours): Redis (in-memory, 5ms)
    - Real-time ingest: Redis Streams → batch insert to ClickHouse every minute
    - Historical 1-min bars: ClickHouse (partition by symbol, time)
    - Raw ticks: Parquet on S3 (partitioned by symbol/year/month)
    - Features & signals: ClickHouse + Redis (latest only)
    - Research: DuckDB on local Parquet files
    
    Key Features:
    - 1 billion rows/sec on single node
    - Column-oriented storage
    - Efficient compression
    - Partition by symbol and time
    """
    
    def __init__(self, config: ClickHouseConfig):
        self.config = config
        self.client = None
    
    def connect(self) -> None:
        """Establish connection to ClickHouse."""
        self.client = clickhouse_connect.get_client(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
        print(f"Connected to ClickHouse at {self.config.host}:{self.config.port}")
    
    def disconnect(self) -> None:
        """Close connection to ClickHouse."""
        if self.client:
            self.client.close()
            print("Disconnected from ClickHouse")
    
    def create_database(self) -> None:
        """Create database if not exists."""
        query = f"CREATE DATABASE IF NOT EXISTS {self.config.database}"
        self.client.command(query)
        print(f"Database {self.config.database} created/verified")
    
    def create_minute_bars_table(self) -> None:
        """Create minute bars table with partitioning."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.minute_bars_table} (
            date Date,
            symbol String,
            timestamp DateTime,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume UInt64,
            vwap Float64,
            trades UInt32
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (symbol, timestamp)
        SETTINGS index_granularity = 8192
        """
        self.client.command(query)
        print(f"Table {self.config.minute_bars_table} created")
    
    def create_features_table(self) -> None:
        """Create features table for ML pipeline."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.features_table} (
            timestamp DateTime,
            symbol String,
            feature_name String,
            feature_value Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (symbol, timestamp, feature_name)
        SETTINGS index_granularity = 8192
        """
        self.client.command(query)
        print(f"Table {self.config.features_table} created")
    
    def create_signals_table(self) -> None:
        """Create signals table for alpha outputs."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.config.database}.{self.config.signals_table} (
            timestamp DateTime,
            symbol String,
            strategy String,
            signal Float64,
            confidence Float64
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (symbol, timestamp, strategy)
        SETTINGS index_granularity = 8192
        """
        self.client.command(query)
        print(f"Table {self.config.signals_table} created")
    
    def insert_minute_bars(self, data: pd.DataFrame) -> None:
        """
        Insert minute bars data into ClickHouse.
        
        Args:
            data: DataFrame with OHLCV data
        """
        # Add date column for partitioning
        data = data.copy()
        data['date'] = data.index.date
        
        # Insert data
        self.client.insert_df(
            f"{self.config.database}.{self.config.minute_bars_table}",
            data
        )
        print(f"Inserted {len(data)} minute bars into ClickHouse")
    
    def query_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Query minute bars from ClickHouse.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        query = f"""
        SELECT timestamp, symbol, open, high, low, close, volume, vwap, trades
        FROM {self.config.database}.{self.config.minute_bars_table}
        WHERE symbol = '{symbol}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY timestamp
        """
        
        result = self.client.query_df(query)
        result.set_index('timestamp', inplace=True)
        
        return result
    
    def query_aggregated_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        aggregation: str = "5 minute"
    ) -> pd.DataFrame:
        """
        Query aggregated bars (e.g., 5-minute, 15-minute, hourly).
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            aggregation: Aggregation interval
            
        Returns:
            DataFrame with aggregated OHLCV data
        """
        query = f"""
        SELECT
            toStartOfInterval(timestamp, INTERVAL {aggregation}) AS bucket,
            symbol,
            argMin(open, timestamp) AS open,
            max(high) AS high,
            min(low) AS low,
            argMax(close, timestamp) AS close,
            sum(volume) AS volume,
            sum(volume * vwap) / sum(volume) AS vwap,
            sum(trades) AS trades
        FROM {self.config.database}.{self.config.minute_bars_table}
        WHERE symbol = '{symbol}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        GROUP BY bucket, symbol
        ORDER BY bucket
        """
        
        result = self.client.query_df(query)
        result.set_index('bucket', inplace=True)
        
        return result
    
    def query_latest_bars(self, symbol: str, n_bars: int = 100) -> pd.DataFrame:
        """
        Query latest n bars for a symbol.
        
        Args:
            symbol: Stock symbol
            n_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with latest OHLCV data
        """
        query = f"""
        SELECT timestamp, symbol, open, high, low, close, volume, vwap, trades
        FROM {self.config.database}.{self.config.minute_bars_table}
        WHERE symbol = '{symbol}'
        ORDER BY timestamp DESC
        LIMIT {n_bars}
        """
        
        result = self.client.query_df(query)
        result.set_index('timestamp', inplace=True)
        result.sort_index(inplace=True)
        
        return result
    
    def insert_features(self, features: pd.DataFrame) -> None:
        """
        Insert features into ClickHouse.
        
        Args:
            features: DataFrame with timestamp, symbol, feature_name, feature_value
        """
        self.client.insert_df(
            f"{self.config.database}.{self.config.features_table}",
            features
        )
        print(f"Inserted {len(features)} feature rows into ClickHouse")
    
    def query_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Query features for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            feature_names: Optional list of feature names to filter
            
        Returns:
            DataFrame with features
        """
        feature_filter = ""
        if feature_names:
            feature_names_str = "', '".join(feature_names)
            feature_filter = f"AND feature_name IN ('{feature_names_str}')"
        
        query = f"""
        SELECT timestamp, symbol, feature_name, feature_value
        FROM {self.config.database}.{self.config.features_table}
        WHERE symbol = '{symbol}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
          {feature_filter}
        ORDER BY timestamp, feature_name
        """
        
        result = self.client.query_df(query)
        
        # Pivot to wide format
        if not result.empty:
            result = result.pivot(index='timestamp', columns='feature_name', values='feature_value')
        
        return result
    
    def insert_signals(self, signals: pd.DataFrame) -> None:
        """
        Insert signals into ClickHouse.
        
        Args:
            signals: DataFrame with timestamp, symbol, strategy, signal, confidence
        """
        self.client.insert_df(
            f"{self.config.database}.{self.config.signals_table}",
            signals
        )
        print(f"Inserted {len(signals)} signal rows into ClickHouse")
    
    def query_signals(
        self,
        strategy: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Query signals for a strategy.
        
        Args:
            strategy: Strategy name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with signals
        """
        query = f"""
        SELECT timestamp, symbol, strategy, signal, confidence
        FROM {self.config.database}.{self.config.signals_table}
        WHERE strategy = '{strategy}'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY timestamp
        """
        
        result = self.client.query_df(query)
        result.set_index('timestamp', inplace=True)
        
        return result
    
    def get_table_stats(self, table_name: str) -> Dict[str, any]:
        """
        Get table statistics.
        
        Args:
            table_name: Table name
            
        Returns:
            Dictionary with table statistics
        """
        query = f"""
        SELECT
            count() as total_rows,
            min(date) as min_date,
            max(date) as max_date,
            uniqExact(symbol) as unique_symbols
        FROM {self.config.database}.{table_name}
        """
        
        result = self.client.query(query)
        
        return {
            'total_rows': result[0][0],
            'min_date': result[0][1],
            'max_date': result[0][2],
            'unique_symbols': result[0][3]
        }
    
    def print_table_stats(self, table_name: str) -> None:
        """Print table statistics."""
        stats = self.get_table_stats(table_name)
        
        print("\n" + "="*60)
        print(f"CLICKHOUSE TABLE STATISTICS: {table_name}")
        print("="*60)
        print(f"Total Rows: {stats['total_rows']:,}")
        print(f"Date Range: {stats['min_date']} to {stats['max_date']}")
        print(f"Unique Symbols: {stats['unique_symbols']}")
        print("="*60)


def run_sample_integration():
    """Run sample ClickHouse integration."""
    config = ClickHouseConfig(
        host="localhost",
        port=8123,
        database="quantdb",
        user="default",
        password=""
    )
    
    manager = ClickHouseManager(config)
    
    try:
        # Connect
        manager.connect()
        
        # Create database and tables
        manager.create_database()
        manager.create_minute_bars_table()
        manager.create_features_table()
        manager.create_signals_table()
        
        # Create sample minute bars data
        dates = pd.date_range("2023-01-01", periods=100, freq="min")
        dates = dates[dates.indexer_between_time('9:15', '15:30')]
        
        np.random.seed(42)
        prices = 20000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, len(dates)))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(50000, 200000, len(dates)),
            'vwap': prices,
            'trades': np.random.randint(100, 500, len(dates))
        })
        
        # Insert data
        manager.insert_minute_bars(data)
        
        # Query data
        result = manager.query_latest_bars('NIFTY', n_bars=10)
        print(f"\nQueried {len(result)} bars")
        print(result.head())
        
        # Print table stats
        manager.print_table_stats(config.minute_bars_table)
        
    finally:
        manager.disconnect()


if __name__ == "__main__":
    run_sample_integration()

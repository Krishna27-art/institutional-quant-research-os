"""
TimescaleDB Integration for Time-Series Data
Based on research recommendations for Indian markets

Key findings from research:
- TimescaleDB wins for Indian markets
- Native time-series support
- PostgreSQL compatibility
- Continuous aggregates
- 90% space reduction with compression
- Hypertables for automatic partitioning

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_values


@dataclass
class TimescaleConfig:
    """Configuration for TimescaleDB connection"""
    host: str = "localhost"
    port: int = 5432
    database: str = "quantdb"
    user: str = "postgres"
    password: str = "password"
    
    # Hypertable configuration
    chunk_time_interval: str = "1 day"
    
    # Continuous aggregates
    five_min_agg: bool = True
    fifteen_min_agg: bool = True
    hourly_agg: bool = True
    daily_agg: bool = True


class TimescaleDBManager:
    """
    TimescaleDB Manager for time-series data storage and retrieval.
    
    Architecture:
    - Hot Path: Redis + TimescaleDB (real-time)
    - Warm Path: TimescaleDB + ClickHouse (recent history)
    - Cold Path: Parquet on S3 (historical)
    
    Key Features:
    - Native time-series support
    - Continuous aggregates
    - 90% compression
    - Hypertables for partitioning
    - PostgreSQL compatibility
    """
    
    def __init__(self, config: TimescaleConfig):
        self.config = config
        self.conn = None
    
    def connect(self) -> None:
        """Establish connection to TimescaleDB."""
        self.conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
        print(f"Connected to TimescaleDB at {self.config.host}:{self.config.port}")
    
    def disconnect(self) -> None:
        """Close connection to TimescaleDB."""
        if self.conn:
            self.conn.close()
            print("Disconnected from TimescaleDB")
    
    def create_hypertable(
        self,
        table_name: str,
        time_column: str = "time",
        if_not_exists: bool = True
    ) -> None:
        """
        Convert a table to a hypertable.
        
        Args:
            table_name: Name of the table
            time_column: Time column name
            if_not_exists: Skip if hypertable already exists
        """
        with self.conn.cursor() as cur:
            # Check if hypertable exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM timescaledb_information.hypertables
                    WHERE hypertable_name = %s
                )
            """, (table_name,))
            
            exists = cur.fetchone()[0]
            
            if exists and if_not_exists:
                print(f"Hypertable {table_name} already exists")
                return
            
            # Create hypertable
            cur.execute("""
                SELECT create_hypertable(%s, %s, 
                    chunk_time_interval => %s,
                    if_not_exists => %s)
            """, (table_name, time_column, self.config.chunk_time_interval, if_not_exists))
            
            self.conn.commit()
            print(f"Hypertable {table_name} created successfully")
    
    def create_minute_bars_table(self) -> None:
        """Create minute bars table with hypertable."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS minute_bars (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open DECIMAL(10,2),
                    high DECIMAL(10,2),
                    low DECIMAL(10,2),
                    close DECIMAL(10,2),
                    volume BIGINT,
                    vwap DECIMAL(12,2),
                    trades INT,
                    PRIMARY KEY (time, symbol)
                )
            """)
            
            self.conn.commit()
        
        # Convert to hypertable
        self.create_hypertable("minute_bars", "time")
    
    def create_continuous_aggregates(self) -> None:
        """Create continuous aggregates for different timeframes."""
        with self.conn.cursor() as cur:
            # 5-minute aggregate
            if self.config.five_min_agg:
                cur.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS five_min_bars
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('5 minutes', time) AS bucket,
                           symbol,
                           FIRST(open, time) AS open,
                           MAX(high) AS high,
                           MIN(low) AS low,
                           LAST(close, time) AS close,
                           SUM(volume) AS volume,
                           SUM(volume * vwap) / SUM(volume) AS vwap,
                           SUM(trades) AS trades
                    FROM minute_bars
                    GROUP BY bucket, symbol
                """)
                
                # Set refresh policy
                cur.execute("""
                    SELECT add_continuous_aggregate_policy('five_min_bars',
                        start_offset => INTERVAL '1 hour',
                        end_offset => INTERVAL '1 minute',
                        schedule_interval => INTERVAL '5 minutes',
                        if_not_exists => TRUE)
                """)
            
            # 15-minute aggregate
            if self.config.fifteen_min_agg:
                cur.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS fifteen_min_bars
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('15 minutes', time) AS bucket,
                           symbol,
                           FIRST(open, time) AS open,
                           MAX(high) AS high,
                           MIN(low) AS low,
                           LAST(close, time) AS close,
                           SUM(volume) AS volume,
                           SUM(volume * vwap) / SUM(volume) AS vwap,
                           SUM(trades) AS trades
                    FROM minute_bars
                    GROUP BY bucket, symbol
                """)
                
                cur.execute("""
                    SELECT add_continuous_aggregate_policy('fifteen_min_bars',
                        start_offset => INTERVAL '3 hours',
                        end_offset => INTERVAL '15 minutes',
                        schedule_interval => INTERVAL '15 minutes',
                        if_not_exists => TRUE)
                """)
            
            # Hourly aggregate
            if self.config.hourly_agg:
                cur.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_bars
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('1 hour', time) AS bucket,
                           symbol,
                           FIRST(open, time) AS open,
                           MAX(high) AS high,
                           MIN(low) AS low,
                           LAST(close, time) AS close,
                           SUM(volume) AS volume,
                           SUM(volume * vwap) / SUM(volume) AS vwap,
                           SUM(trades) AS trades
                    FROM minute_bars
                    GROUP BY bucket, symbol
                """)
                
                cur.execute("""
                    SELECT add_continuous_aggregate_policy('hourly_bars',
                        start_offset => INTERVAL '1 day',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE)
                """)
            
            # Daily aggregate
            if self.config.daily_agg:
                cur.execute("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS daily_bars
                    WITH (timescaledb.continuous) AS
                    SELECT time_bucket('1 day', time) AS bucket,
                           symbol,
                           FIRST(open, time) AS open,
                           MAX(high) AS high,
                           MIN(low) AS low,
                           LAST(close, time) AS close,
                           SUM(volume) AS volume,
                           SUM(volume * vwap) / SUM(volume) AS vwap,
                           SUM(trades) AS trades
                    FROM minute_bars
                    GROUP BY bucket, symbol
                """)
                
                cur.execute("""
                    SELECT add_continuous_aggregate_policy('daily_bars',
                        start_offset => INTERVAL '1 month',
                        end_offset => INTERVAL '1 day',
                        schedule_interval => INTERVAL '1 day',
                        if_not_exists => TRUE)
                """)
            
            self.conn.commit()
            print("Continuous aggregates created successfully")
    
    def insert_minute_bars(self, data: pd.DataFrame) -> None:
        """
        Insert minute bars data into TimescaleDB.
        
        Args:
            data: DataFrame with OHLCV data
        """
        with self.conn.cursor() as cur:
            # Prepare data for insertion
            records = []
            for idx, row in data.iterrows():
                records.append((
                    idx,
                    row.get('symbol', 'NIFTY'),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    row.get('vwap', row['close']),
                    row.get('trades', 0)
                ))
            
            # Batch insert
            execute_values(
                cur,
                """
                INSERT INTO minute_bars 
                (time, symbol, open, high, low, close, volume, vwap, trades)
                VALUES %s
                ON CONFLICT (time, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    vwap = EXCLUDED.vwap,
                    trades = EXCLUDED.trades
                """,
                records
            )
            
            self.conn.commit()
            print(f"Inserted {len(records)} minute bars")
    
    def query_minute_bars(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        agg_interval: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query minute bars from TimescaleDB.
        
        Args:
            symbol: Stock symbol
            start_time: Start time
            end_time: End time
            agg_interval: Optional aggregation interval (e.g., '5 minutes', '1 hour')
            
        Returns:
            DataFrame with OHLCV data
        """
        with self.conn.cursor() as cur:
            if agg_interval:
                # Query from continuous aggregate
                view_name = self._get_view_name(agg_interval)
                cur.execute(f"""
                    SELECT bucket AS time,
                           symbol,
                           open, high, low, close, volume, vwap
                    FROM {view_name}
                    WHERE symbol = %s
                      AND bucket >= %s
                      AND bucket <= %s
                    ORDER BY bucket
                """, (symbol, start_time, end_time))
            else:
                # Query from raw table
                cur.execute("""
                    SELECT time, symbol, open, high, low, close, volume, vwap
                    FROM minute_bars
                    WHERE symbol = %s
                      AND time >= %s
                      AND time <= %s
                    ORDER BY time
                """, (symbol, start_time, end_time))
            
            columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap']
            rows = cur.fetchall()
            
            df = pd.DataFrame(rows, columns=columns)
            df.set_index('time', inplace=True)
            
            return df
    
    def _get_view_name(self, interval: str) -> str:
        """Get view name based on aggregation interval."""
        interval_map = {
            '5 minutes': 'five_min_bars',
            '15 minutes': 'fifteen_min_bars',
            '1 hour': 'hourly_bars',
            '1 day': 'daily_bars'
        }
        return interval_map.get(interval, 'minute_bars')
    
    def query_latest_bars(self, symbol: str, n_bars: int = 100) -> pd.DataFrame:
        """
        Query latest n bars for a symbol.
        
        Args:
            symbol: Stock symbol
            n_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with latest OHLCV data
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT time, symbol, open, high, low, close, volume, vwap
                FROM minute_bars
                WHERE symbol = %s
                ORDER BY time DESC
                LIMIT %s
            """, (symbol, n_bars))
            
            columns = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap']
            rows = cur.fetchall()
            
            df = pd.DataFrame(rows, columns=columns)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            return df
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics for hypertables."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT hypertable_name,
                       total_size,
                       compressed_size,
                       compression_ratio
                FROM hypertable_compression_stats
            """)
            
            rows = cur.fetchall()
            
            stats = {}
            for row in rows:
                stats[row[0]] = {
                    'total_size': row[1],
                    'compressed_size': row[2],
                    'compression_ratio': row[3]
                }
            
            return stats
    
    def print_compression_stats(self) -> None:
        """Print compression statistics."""
        stats = self.get_compression_stats()
        
        print("\n" + "="*60)
        print("TIMESCALEDB COMPRESSION STATISTICS")
        print("="*60)
        
        for table, stat in stats.items():
            print(f"\nTable: {table}")
            print(f"  Total Size: {stat['total_size']:,} bytes")
            print(f"  Compressed Size: {stat['compressed_size']:,} bytes")
            print(f"  Compression Ratio: {stat['compression_ratio']:.2%}")
        
        print("="*60)


def run_sample_integration():
    """Run sample TimescaleDB integration."""
    config = TimescaleConfig(
        host="localhost",
        port=5432,
        database="quantdb",
        user="postgres",
        password="password"
    )
    
    manager = TimescaleDBManager(config)
    
    try:
        # Connect
        manager.connect()
        
        # Create tables
        manager.create_minute_bars_table()
        
        # Create continuous aggregates
        manager.create_continuous_aggregates()
        
        # Insert sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="min")
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': 20000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, 100)),
            'high': 20000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, 100)) * 1.001,
            'low': 20000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, 100)) * 0.999,
            'close': 20000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, 100)),
            'volume': np.random.randint(50000, 200000, 100),
            'symbol': 'NIFTY'
        }, index=dates)
        
        manager.insert_minute_bars(sample_data)
        
        # Query data
        result = manager.query_latest_bars('NIFTY', n_bars=10)
        print(f"\nQueried {len(result)} bars")
        print(result.head())
        
        # Print compression stats
        manager.print_compression_stats()
        
    finally:
        manager.disconnect()


if __name__ == "__main__":
    run_sample_integration()

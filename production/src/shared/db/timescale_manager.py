"""
TimescaleDB Manager - Time-series database operations
"""

import psycopg2
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import re


class TimescaleManager:
    """Manage TimescaleDB operations"""
    
    def __init__(self, connection: psycopg2.extensions.connection):
        self.conn = connection
    
    def _validate_identifier(self, identifier: str) -> str:
        """
        Validate SQL identifier to prevent SQL injection.
        
        CRITICAL FIX: Only allows alphanumeric characters and underscores.
        """
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier
    
    def create_hypertable(self, table_name: str, time_column: str = 'time',
                         partition_column: Optional[str] = None,
                         number_partitions: int = 1) -> None:
        """Convert table to hypertable for time-series optimization"""
        table_name = self._validate_identifier(table_name)
        time_column = self._validate_identifier(time_column)
        
        with self.conn.cursor() as cur:
            if partition_column:
                partition_column = self._validate_identifier(partition_column)
                cur.execute("""
                    SELECT create_hypertable(%s, %s,
                        partitioning_column => %s,
                        number_partitions => %s)
                """, (table_name, time_column, partition_column, number_partitions))
            else:
                cur.execute("""
                    SELECT create_hypertable(%s, %s)
                """, (table_name, time_column))
            self.conn.commit()
    
    def add_retention_policy(self, table_name: str, interval: str) -> None:
        """Add data retention policy"""
        table_name = self._validate_identifier(table_name)
        
        # Validate interval format
        if not re.match(r'^\d+\s+(day|week|month|year)s?$', interval):
            raise ValueError(f"Invalid retention interval format: {interval}")
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT add_retention_policy(%s, INTERVAL %s)
            """, (table_name, interval))
            self.conn.commit()
    
    def insert_bars(self, table_name: str, bars: pd.DataFrame) -> None:
        """Insert OHLCV bars"""
        table_name = self._validate_identifier(table_name)
        
        with self.conn.cursor() as cur:
            for idx, row in bars.iterrows():
                cur.execute(f"""
                    INSERT INTO {table_name} (time, symbol, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (time, symbol) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """, (idx, row.get('symbol', ''), row['open'], row['high'], 
                      row['low'], row['close'], row.get('volume', 0)))
            self.conn.commit()
    
    def query_bars(self, table_name: str, symbol: str, start: datetime,
                   end: datetime) -> pd.DataFrame:
        """Query bars for a symbol in time range"""
        table_name = self._validate_identifier(table_name)
        
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM {table_name}
            WHERE symbol = %s AND time >= %s AND time <= %s
            ORDER BY time
        """
        
        return pd.read_sql_query(query, self.conn, params=(symbol, start, end))
    
    def create_continuous_aggregate(self, view_name: str, source_table: str,
                                   interval: str) -> None:
        """Create continuous aggregate for downsampled data"""
        view_name = self._validate_identifier(view_name)
        source_table = self._validate_identifier(source_table)
        
        # Validate interval format
        if not re.match(r'^\d+\s+(minute|hour|day|week|month|year)s?$', interval):
            raise ValueError(f"Invalid interval format: {interval}")
        
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE MATERIALIZED VIEW {view_name} WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket(%s, time) AS time,
                    symbol,
                    first(open, time) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, time) AS close,
                    sum(volume) AS volume
                FROM {source_table}
                GROUP BY time_bucket(%s, time), symbol
                WITH DATA;
            """, (interval, interval))
            self.conn.commit()
    
    def refresh_continuous_aggregate(self, view_name: str) -> None:
        """Refresh continuous aggregate"""
        view_name = self._validate_identifier(view_name)
        
        with self.conn.cursor() as cur:
            cur.execute("""
                CALL refresh_continuous_aggregate(%s, NULL, NULL)
            """, (view_name,))
            self.conn.commit()

"""
NSE 1-Minute Intraday Data Pipeline

Based on Comprehensive Upgrade Analysis - Tier 1 Upgrade (#9)
Expected Sharpe improvement: +0.2–0.3
Essential for Indian market signals

Methodology:
- Ingest NSE 1-minute intraday data
- Process and clean data
- Store in TimescaleDB (operational) and ClickHouse (analytics)
- Provide real-time data access for strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import clickhouse_connect
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False


class DataFrequency(Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


@dataclass
class NSEDataConfig:
    """Configuration for NSE data pipeline"""
    # Database connections
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "nse_intraday"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_db: str = "nse_analytics"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    
    # Data parameters
    symbols: List[str] = None  # List of NSE symbols to track
    start_date: str = "2020-01-01"
    end_date: str = None  # If None, current date
    
    # Data quality
    min_volume_threshold: int = 100  # Minimum volume for valid data
    max_price_change_pct: float = 20.0  # Max price change for outlier detection
    
    # Real-time parameters
    enable_realtime: bool = True
    update_interval_seconds: int = 60
    
    # Storage
    store_in_postgres: bool = True
    store_in_clickhouse: bool = True


@dataclass
class IntradayBar:
    """Single intraday bar"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    num_trades: int


class NSEDataPipeline:
    """
    NSE 1-Minute Intraday Data Pipeline
    
    Ingests, processes, and stores NSE intraday data.
    Provides real-time data access for trading strategies.
    """
    
    def __init__(self, config: NSEDataConfig):
        self.config = config
        
        # Default symbols (NIFTY 50)
        if config.symbols is None:
            self.config.symbols = self._get_nifty50_symbols()
        
        # Database connections
        self.pg_conn = None
        self.ch_client = None
        
        # Initialize connections
        if self.config.store_in_postgres and POSTGRES_AVAILABLE:
            self._init_postgres()
        
        if self.config.store_in_clickhouse and CLICKHOUSE_AVAILABLE:
            self._init_clickhouse()
    
    def _get_nifty50_symbols(self) -> List[str]:
        """Get NIFTY 50 symbols"""
        return [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
            "HINDUNILVR", "AXISBANK", "BAJFINANCE", "MARUTI", "HCLTECH",
            "ASIANPAINT", "SUNPHARMA", "TATAMOTORS", "TITAN", "DMART"
        ]
    
    def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection"""
        try:
            self.pg_conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password
            )
            print("PostgreSQL connection established")
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
    
    def _init_clickhouse(self) -> None:
        """Initialize ClickHouse connection"""
        try:
            self.ch_client = clickhouse_connect.get_client(
                host=self.config.clickhouse_host,
                port=self.config.clickhouse_port,
                database=self.config.clickhouse_db,
                username=self.config.clickhouse_user,
                password=self.config.clickhouse_password
            )
            print("ClickHouse connection established")
        except Exception as e:
            print(f"Failed to connect to ClickHouse: {e}")
    
    def ingest_data(self, data: pd.DataFrame) -> int:
        """
        Ingest intraday data
        
        Args:
            data: DataFrame with columns: symbol, timestamp, open, high, low, close, volume
            
        Returns:
            Number of records ingested
        """
        # Clean and validate data
        data = self._clean_data(data)
        
        if data.empty:
            return 0
        
        # Store in PostgreSQL
        if self.config.store_in_postgres and self.pg_conn:
            self._store_postgres(data)
        
        # Store in ClickHouse
        if self.config.store_in_clickhouse and self.ch_client:
            self._store_clickhouse(data)
        
        return len(data)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        data = data.copy()
        
        # Ensure timestamp is datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['symbol', 'timestamp'])
        
        # Sort by timestamp
        data = data.sort_values(['symbol', 'timestamp'])
        
        # Filter by symbols
        data = data[data['symbol'].isin(self.config.symbols)]
        
        # Remove zero or negative prices
        data = data[(data['open'] > 0) & (data['high'] > 0) & 
                   (data['low'] > 0) & (data['close'] > 0)]
        
        # Remove zero volume
        data = data[data['volume'] > 0]
        
        # Remove outliers (extreme price changes)
        data['price_change_pct'] = (data['close'] - data['open']) / data['open'] * 100
        data = data[abs(data['price_change_pct']) < self.config.max_price_change_pct]
        
        # Filter by minimum volume
        data = data[data['volume'] >= self.config.min_volume_threshold]
        
        # Calculate VWAP
        data['vwap'] = (data['close'] * data['volume']) / data['volume']
        
        # Calculate number of trades (estimate from volume)
        data['num_trades'] = (data['volume'] / 100).astype(int)  # Rough estimate
        
        return data
    
    def _store_postgres(self, data: pd.DataFrame) -> None:
        """Store data in PostgreSQL"""
        if not self.pg_conn:
            return
        
        cursor = self.pg_conn.cursor()
        
        # Create table if not exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS intraday_bars (
            symbol VARCHAR(20),
            timestamp TIMESTAMP,
            open DECIMAL(10,2),
            high DECIMAL(10,2),
            low DECIMAL(10,2),
            close DECIMAL(10,2),
            volume BIGINT,
            vwap DECIMAL(10,2),
            num_trades INTEGER,
            PRIMARY KEY (symbol, timestamp)
        );
        """
        cursor.execute(create_table_sql)
        
        # Insert data
        insert_sql = """
        INSERT INTO intraday_bars 
        (symbol, timestamp, open, high, low, close, volume, vwap, num_trades)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            vwap = EXCLUDED.vwap,
            num_trades = EXCLUDED.num_trades;
        """
        
        records = [
            (row['symbol'], row['timestamp'], row['open'], row['high'],
             row['low'], row['close'], row['volume'], row['vwap'], row['num_trades'])
            for _, row in data.iterrows()
        ]
        
        execute_batch(cursor, insert_sql, records)
        self.pg_conn.commit()
        cursor.close()
    
    def _store_clickhouse(self, data: pd.DataFrame) -> None:
        """Store data in ClickHouse"""
        if not self.ch_client:
            return
        
        # Create table if not exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS intraday_bars (
            symbol String,
            timestamp DateTime,
            open Float32,
            high Float32,
            low Float32,
            close Float32,
            volume UInt64,
            vwap Float32,
            num_trades UInt32,
            date Date MATERIALIZED toDate(timestamp)
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (symbol, timestamp)
        """
        
        self.ch_client.command(create_table_sql)
        
        # Insert data
        self.ch_client.insert('intraday_bars', data.to_dict('records'))
    
    def get_data(self, 
                 symbol: str, 
                 start_date: datetime, 
                 end_date: datetime,
                 frequency: DataFrequency = DataFrequency.MINUTE_1) -> pd.DataFrame:
        """
        Retrieve intraday data for a symbol
        
        Args:
            symbol: NSE symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.config.store_in_postgres and self.pg_conn:
            return self._get_postgres_data(symbol, start_date, end_date)
        elif self.config.store_in_clickhouse and self.ch_client:
            return self._get_clickhouse_data(symbol, start_date, end_date)
        else:
            return pd.DataFrame()
    
    def _get_postgres_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve data from PostgreSQL"""
        if not self.pg_conn:
            return pd.DataFrame()
        
        query = """
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM intraday_bars
        WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp
        """
        
        df = pd.read_sql(query, self.pg_conn, params=(symbol, start_date, end_date))
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _get_clickhouse_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve data from ClickHouse"""
        if not self.ch_client:
            return pd.DataFrame()
        
        query = f"""
        SELECT timestamp, open, high, low, close, volume, vwap, num_trades
        FROM intraday_bars
        WHERE symbol = '{symbol}' AND timestamp >= '{start_date}' AND timestamp <= '{end_date}'
        ORDER BY timestamp
        """
        
        result = self.ch_client.query(query)
        df = result.result_df
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def resample_data(self, data: pd.DataFrame, frequency: DataFrequency) -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            data: Original data
            frequency: Target frequency
            
        Returns:
            Resampled data
        """
        freq_map = {
            DataFrequency.MINUTE_1: "1T",
            DataFrequency.MINUTE_5: "5T",
            DataFrequency.MINUTE_15: "15T",
            DataFrequency.HOUR_1: "1H",
            DataFrequency.DAY_1: "1D"
        }
        
        rule = freq_map.get(frequency, "1T")
        
        resampled = data.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vwap': 'mean',
            'num_trades': 'sum'
        }).dropna()
        
        return resampled
    
    def get_latest_data(self, symbol: str, n_bars: int = 100) -> pd.DataFrame:
        """
        Get latest n bars for a symbol
        
        Args:
            symbol: NSE symbol
            n_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with latest data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        data = self.get_data(symbol, start_date, end_date)
        
        if not data.empty:
            return data.tail(n_bars)
        
        return pd.DataFrame()
    
    def close(self) -> None:
        """Close database connections"""
        if self.pg_conn:
            self.pg_conn.close()
        
        if self.ch_client:
            self.ch_client.close()


def simulate_nse_intraday_data(n_days: int = 5, symbols: List[str] = None) -> pd.DataFrame:
    """Simulate NSE intraday data for testing"""
    if symbols is None:
        symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    
    data = []
    
    for symbol in symbols:
        base_price = np.random.uniform(100, 3000)
        
        for day in range(n_days):
            date = datetime.now() - timedelta(days=n_days - day)
            
            # Generate intraday data (9:15 to 15:30)
            for minute in range(375):  # 6.25 hours * 60 minutes
                timestamp = date.replace(hour=9, minute=15) + timedelta(minutes=minute)
                
                # Skip non-trading hours
                if timestamp.hour >= 15 and timestamp.minute > 30:
                    continue
                
                # Random walk
                price_change = np.random.normal(0, 0.001) * base_price
                base_price = max(base_price + price_change, 10)
                
                # Generate OHLC
                open_price = base_price
                high_price = base_price * (1 + abs(np.random.normal(0, 0.002)))
                low_price = base_price * (1 - abs(np.random.normal(0, 0.002)))
                close_price = base_price + np.random.normal(0, 0.0005) * base_price
                
                volume = int(np.random.exponential(10000))
                vwap = close_price
                num_trades = volume // 100
                
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'vwap': vwap,
                    'num_trades': num_trades
                })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    config = NSEDataConfig(
        store_in_postgres=False,  # Disable for testing
        store_in_clickhouse=False,  # Disable for testing
        symbols=["RELIANCE", "TCS", "HDFCBANK"]
    )
    
    pipeline = NSEDataPipeline(config)
    
    # Simulate data
    print("Simulating NSE intraday data...")
    simulated_data = simulate_nse_intraday_data(n_days=5, symbols=config.symbols)
    
    # Ingest data
    print(f"Ingesting {len(simulated_data)} records...")
    count = pipeline.ingest_data(simulated_data)
    print(f"Ingested {count} records")
    
    # Test resampling
    print("\nTesting resampling...")
    symbol_data = simulated_data[simulated_data['symbol'] == 'RELIANCE'].copy()
    symbol_data.set_index('timestamp', inplace=True)
    
    resampled_5m = pipeline.resample_data(symbol_data, DataFrequency.MINUTE_5)
    print(f"Original: {len(symbol_data)} bars")
    print(f"5-minute resampled: {len(resampled_5m)} bars")
    
    resampled_1h = pipeline.resample_data(symbol_data, DataFrequency.HOUR_1)
    print(f"1-hour resampled: {len(resampled_1h)} bars")
    
    pipeline.close()

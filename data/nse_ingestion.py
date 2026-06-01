"""
NSE/BSE Data Ingestion Module
Historical tick data ingestion for 2020-2024

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class IngestionConfig:
    """Configuration for data ingestion"""
    # Data sources
    use_nselib: bool = True
    use_yfinance: bool = True
    
    # Date range
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    
    # Symbols
    nifty_50_symbols: List[str] = None
    top_100_stocks: List[str] = None
    
    # Data frequency
    frequency: str = "tick"  # tick, 1min, 5min
    
    # Storage
    parquet_path: str = "./data/raw/ticks"
    minute_bars_path: str = "./data/processed/minute_bars"
    
    # Batch size
    batch_size: int = 1000


class NSEDataIngestion:
    """
    NSE/BSE data ingestion using nselib and yfinance.
    
    Features:
    - Historical tick data download
    - Corporate actions handling
    - Survivorship bias correction
    - Parquet storage with partitioning
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        
        # Default NIFTY 50 symbols
        if config.nifty_50_symbols is None:
            self.nifty_50_symbols = self._get_nifty_50_symbols()
        else:
            self.nifty_50_symbols = config.nifty_50_symbols
        
        # Default top 100 stocks
        if config.top_100_stocks is None:
            self.top_100_stocks = self.nifty_50_symbols  # Start with NIFTY 50
        else:
            self.top_100_stocks = config.top_100_stocks
    
    def _get_nifty_50_symbols(self) -> List[str]:
        """Get NIFTY 50 stock symbols."""
        # Common NIFTY 50 stocks
        return [
            "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
            "AXISBANK", "HCLTECH", "ASIANPAINT", "MARUTI", "SUNPHARMA",
            "TCS", "TATAMOTORS", "POWERGRID", "TITAN", "BAJFINANCE",
            "DMART", "WIPRO", "ULTRACEMCO", "NTPC", "LICI",
            "HDFCLIFE", "TATACONSUM", "NESTLEIND", "BAJAJFINSV", "JSWSTEEL",
            "DIVISLAB", "DRREDDY", "TATASTEEL", "ADANIENT", "SBILIFE",
            "M&M", "GRASIM", "CIPLA", "COALINDIA", "PNB",
            "BPCL", "ONGC", "SHREECEM", "TATAMOTORS", "HEROMOTOCO",
            "EICHERMOT", "BRITANNIA", "UPL", "HAL", "ACC"
        ]
    
    async def download_historical_data_nselib(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data using nselib.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import nselib
            from nselib import derivatives
            
            # Get historical data
            data = nselib.get_price_history(
                symbol,
                from_date=start_date,
                to_date=end_date,
                period_type="daily"
            )
            
            if data is not None and not data.empty:
                # Standardize columns
                data.columns = [col.lower() for col in data.columns]
                return data
            
            return None
        
        except ImportError:
            print("nselib not installed. Install with: pip install nselib")
            return None
        except Exception as e:
            print(f"Error downloading {symbol} from nselib: {e}")
            return None
    
    async def download_historical_data_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data using yfinance.
        
        Args:
            symbol: Stock symbol (with .NS suffix)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
            
            # Add .NS suffix for NSE stocks
            if not symbol.endswith(".NS"):
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                prepost=False
            )
            
            if data is not None and not data.empty:
                # Standardize columns
                data.columns = [col.lower() for col in data.columns]
                return data
            
            return None
        
        except Exception as e:
            print(f"Error downloading {symbol} from yfinance: {e}")
            return None
    
    def aggregate_to_minute_bars(
        self,
        tick_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate tick data to 1-minute bars.
        
        Args:
            tick_data: Tick-level data
            
        Returns:
            DataFrame with 1-minute OHLCV bars
        """
        if tick_data is None or tick_data.empty:
            return pd.DataFrame()
        
        # Resample to 1-minute bars
        minute_bars = tick_data.resample('1T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return minute_bars
    
    def save_to_parquet(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_type: str = "minute_bars"
    ) -> bool:
        """
        Save data to Parquet format with partitioning.
        
        Args:
            data: DataFrame to save
            symbol: Stock symbol
            data_type: Type of data (minute_bars, ticks)
            
        Returns:
            True if successful
        """
        try:
            if data_type == "minute_bars":
                path = f"{self.config.minute_bars_path}/{symbol}"
            else:
                path = f"{self.config.parquet_path}/{symbol}"
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(data)
            
            # Write to Parquet with partitioning by year and month
            pq.write_to_dataset(
                table,
                root_path=path,
                partition_cols=['year', 'month'] if 'year' in data.columns else None
            )
            
            return True
        
        except Exception as e:
            print(f"Error saving {symbol} to Parquet: {e}")
            return False
    
    async def ingest_symbol(
        self,
        symbol: str,
        use_source: str = "nselib"
    ) -> bool:
        """
        Ingest data for a single symbol.
        
        Args:
            symbol: Stock symbol
            use_source: Data source (nselib or yfinance)
            
        Returns:
            True if successful
        """
        print(f"Ingesting {symbol} using {use_source}...")
        
        # Download data
        if use_source == "nselib":
            data = await self.download_historical_data_nselib(
                symbol,
                self.config.start_date,
                self.config.end_date
            )
        else:
            data = await self.download_historical_data_yfinance(
                symbol,
                self.config.start_date,
                self.config.end_date
            )
        
        if data is None or data.empty:
            print(f"No data retrieved for {symbol}")
            return False
        
        # Add date columns for partitioning
        data['date'] = data.index.date
        data['year'] = data.index.year
        data['month'] = data.index.month
        
        # Save to Parquet
        success = self.save_to_parquet(data, symbol, "minute_bars")
        
        if success:
            print(f"Successfully ingested {symbol}: {len(data)} records")
        
        return success
    
    async def ingest_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        use_source: str = "nselib"
    ) -> Dict[str, bool]:
        """
        Ingest data for all symbols.
        
        Args:
            symbols: List of symbols (default: top 100)
            use_source: Data source (nselib or yfinance)
            
        Returns:
            Dictionary mapping symbol to success status
        """
        if symbols is None:
            symbols = self.top_100_stocks
        
        results = {}
        
        for symbol in symbols:
            success = await self.ingest_symbol(symbol, use_source)
            results[symbol] = success
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        # Summary
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\nIngestion complete: {successful}/{total} symbols successful")
        
        return results
    
    def load_minute_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load minute bars from Parquet storage.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with minute bars
        """
        try:
            path = f"{self.config.minute_bars_path}/{symbol}"
            
            # Read from Parquet
            dataset = pq.ParquetDataset(path)
            table = dataset.read()
            df = table.to_pandas()
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            return df
        
        except Exception as e:
            print(f"Error loading {symbol} from Parquet: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with available data."""
        import os
        
        symbols = []
        
        if os.path.exists(self.config.minute_bars_path):
            for item in os.listdir(self.config.minute_bars_path):
                if os.path.isdir(os.path.join(self.config.minute_bars_path, item)):
                    symbols.append(item)
        
        return symbols


class CorporateActionsHandler:
    """
    Handle corporate actions for survivorship bias correction.
    
    Actions handled:
    - Splits
    - Bonuses
    - Dividends
    - Mergers/Acquisitions
    """
    
    def __init__(self):
        self.actions_cache: Dict[str, List[Dict]] = {}
    
    def fetch_corporate_actions(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        Fetch corporate actions for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of corporate actions
        """
        try:
            import nselib
            from nselib import corporate_actions
            
            actions = nselib.get_corporate_actions(
                symbol,
                from_date=start_date,
                to_date=end_date
            )
            
            if actions is not None:
                self.actions_cache[symbol] = actions.to_dict('records')
                return self.actions_cache[symbol]
            
            return []
        
        except ImportError:
            print("nselib not installed")
            return []
        except Exception as e:
            print(f"Error fetching corporate actions for {symbol}: {e}")
            return []
    
    def adjust_for_splits(
        self,
        data: pd.DataFrame,
        split_ratio: float,
        split_date: datetime
    ) -> pd.DataFrame:
        """
        Adjust historical prices for stock splits.
        
        Args:
            data: Price data
            split_ratio: Split ratio (e.g., 2 for 2:1 split)
            split_date: Date of split
            
        Returns:
            Adjusted price data
        """
        # Adjust prices before split date
        mask = data.index < split_date
        
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                data.loc[mask, col] = data.loc[mask, col] / split_ratio
        
        # Adjust volume (multiply by split ratio)
        if 'volume' in data.columns:
            data.loc[mask, 'volume'] = data.loc[mask, 'volume'] * split_ratio
        
        return data
    
    def adjust_for_dividends(
        self,
        data: pd.DataFrame,
        dividend_amount: float,
        ex_date: datetime
    ) -> pd.DataFrame:
        """
        Adjust prices for dividends.
        
        Args:
            data: Price data
            dividend_amount: Dividend amount
            ex_date: Ex-dividend date
            
        Returns:
            Adjusted price data
        """
        # Adjust prices before ex-date
        mask = data.index < ex_date
        
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                data.loc[mask, col] = data.loc[mask, col] - dividend_amount
        
        return data


async def main():
    """Main function to run data ingestion."""
    config = IngestionConfig(
        start_date="2020-01-01",
        end_date="2024-12-31",
        frequency="1min"
    )
    
    ingestion = NSEDataIngestion(config)
    
    # Ingest NIFTY 50 symbols
    print("Starting NIFTY 50 data ingestion...")
    results = await ingestion.ingest_all_symbols(
        symbols=ingestion.nifty_50_symbols,
        use_source="yfinance"  # Fallback to yfinance
    )
    
    print(f"\nIngestion results:")
    for symbol, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {symbol}")


if __name__ == "__main__":
    asyncio.run(main())

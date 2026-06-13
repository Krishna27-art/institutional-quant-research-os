"""
NSE Data Fetcher using nselib

This module provides real NSE market data using the nselib library,
replacing synthetic data with actual market prices.

Based on repository analysis: nselib is the best library for Indian markets
with direct NSE bhavcopy, FII/DII flows, options chain, and corporate actions.

Key Features:
- Real equity prices from NSE
- Historical data with bhavcopy
- Index data (NIFTY, BANKNIFTY, etc.)
- FII/DII flows for market breadth signals
- Options chain data
- Corporate actions handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

try:
    import nselib
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False
    logger.warning("nselib not installed. Run: pip install nselib")


class NSEDataFetcher:
    """
    Fetch real market data from NSE using nselib.
    
    This replaces synthetic data generation with actual market data,
    fixing the issue of wrong prices in the dashboard.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        cache_hours: int = 1
    ):
        """
        Initialize NSE data fetcher.
        
        Args:
            cache_dir: Directory for caching data
            use_cache: Whether to use cached data
            cache_hours: Cache validity in hours
        """
        self.use_cache = use_cache
        self.cache_hours = cache_hours
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NIFTY50 universe (will be fetched dynamically)
        self.nifty50_symbols = self._get_nifty50_universe()
        
        if not NSELIB_AVAILABLE:
            logger.error("nselib not available. Install with: pip install nselib")
    
    def _get_cache_path(self, symbol: str, data_type: str) -> Path:
        """Get cache file path for a symbol and data type."""
        return self.cache_dir / f"{symbol}_{data_type}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not self.use_cache or not cache_path.exists():
            return False
        
        cache_age = time.time() - cache_path.stat().st_mtime
        return cache_age < (self.cache_hours * 3600)
    
    def _get_nifty50_universe(self) -> List[str]:
        """
        Get NIFTY50 universe from NSE.
        
        Returns:
            List of NIFTY50 symbols
        """
        if not NSELIB_AVAILABLE:
            # Fallback to hardcoded list
            return [
                'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN',
                'BHARTIARTL', 'KOTAKBANK', 'LT', 'HINDUNILVR', 'AXISBANK',
                'BAJFINANCE', 'TCS', 'MARUTI', 'ASIANPAINT', 'WIPRO',
                'HCLTECH', 'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND',
                'TATAMOTORS', 'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA',
                'TATASTEEL', 'JSWSTEEL', 'BAJAJ-AUTO', 'M&M', 'DRREDDY',
                'CIPLA', 'ADANIENT', 'GRASIM', 'EICHERMOT', 'HEROMOTOCO'
            ]
        
        try:
            # Try to fetch from NSE
            from nselib import get_index_constituents
            constituents = get_index_constitents(index='NIFTY 50')
            symbols = [c['symbol'] for c in constituents]
            logger.info(f"Fetched {len(symbols)} NIFTY50 symbols from NSE")
            return symbols
        except Exception as e:
            logger.warning(f"Failed to fetch NIFTY50 from NSE: {e}. Using fallback list.")
            return self._get_nifty50_universe()
    
    def get_equity_price_history(
        self,
        symbol: str,
        period: str = '1y',
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get historical price data for an equity.
        
        Args:
            symbol: Stock symbol
            period: Time period ('1d', '1w', '1m', '3m', '6m', '1y')
            from_date: Start date (overrides period)
            to_date: End date (overrides period)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not NSELIB_AVAILABLE:
            raise RuntimeError("nselib not available. Install with: pip install nselib")
        
        cache_path = self._get_cache_path(symbol, 'equity_history')
        
        if self._is_cache_valid(cache_path):
            logger.debug(f"Using cached data for {symbol}")
            return pd.read_parquet(cache_path)
        
        try:
            from nselib import get_price_history
            
            # Determine date range
            if from_date is None:
                if period == '1d':
                    from_date = date.today()
                elif period == '1w':
                    from_date = date.today() - timedelta(days=7)
                elif period == '1m':
                    from_date = date.today() - timedelta(days=30)
                elif period == '3m':
                    from_date = date.today() - timedelta(days=90)
                elif period == '6m':
                    from_date = date.today() - timedelta(days=180)
                elif period == '1y':
                    from_date = date.today() - timedelta(days=365)
                else:
                    from_date = date.today() - timedelta(days=365)
            
            if to_date is None:
                to_date = date.today()
            
            # Fetch from NSE
            data = get_price_history(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                equity_segment='EQ'
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'VWAP': 'vwap',
                'VOLUME': 'volume',
                'TIMESTAMP': 'date'
            })
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.warning(f"Missing column {col} for {symbol}")
            
            # Set date as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            
            # Sort by date
            data = data.sort_index()
            
            # Cache the data
            if self.use_cache:
                data.to_parquet(cache_path)
            
            logger.info(f"Fetched {len(data)} rows for {symbol} from {from_date} to {to_date}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch equity history for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_data(
        self,
        index: str = 'NIFTY 50',
        period: str = '1y'
    ) -> pd.DataFrame:
        """
        Get index data.
        
        Args:
            index: Index name ('NIFTY 50', 'NIFTY BANK', 'NIFTY FIN SERVICE')
            period: Time period
            
        Returns:
            DataFrame with index OHLCV data
        """
        if not NSELIB_AVAILABLE:
            raise RuntimeError("nselib not available. Install with: pip install nselib")
        
        cache_path = self._get_cache_path(index.replace(' ', '_'), 'index_data')
        
        if self._is_cache_valid(cache_path):
            logger.debug(f"Using cached index data for {index}")
            return pd.read_parquet(cache_path)
        
        try:
            from nselib import get_index_history
            
            # Determine date range
            if period == '1d':
                from_date = date.today()
            elif period == '1w':
                from_date = date.today() - timedelta(days=7)
            elif period == '1m':
                from_date = date.today() - timedelta(days=30)
            elif period == '3m':
                from_date = date.today() - timedelta(days=90)
            elif period == '6m':
                from_date = date.today() - timedelta(days=180)
            elif period == '1y':
                from_date = date.today() - timedelta(days=365)
            else:
                from_date = date.today() - timedelta(days=365)
            
            to_date = date.today()
            
            # Fetch from NSE
            data = get_index_history(
                index_name=index,
                from_date=from_date,
                to_date=to_date
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No data returned for index {index}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close',
                'VWAP': 'vwap',
                'TIMESTAMP': 'date'
            })
            
            # Set date as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.set_index('date')
            
            # Sort by date
            data = data.sort_index()
            
            # Cache the data
            if self.use_cache:
                data.to_parquet(cache_path)
            
            logger.info(f"Fetched {len(data)} rows for index {index}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch index data for {index}: {e}")
            return pd.DataFrame()
    
    def get_fii_dii_flows(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get FII/DII flows data.
        
        This is a valuable India-specific signal for market breadth.
        
        Args:
            from_date: Start date
            to_date: End date
            
        Returns:
            DataFrame with FII/DII flow data
        """
        if not NSELIB_AVAILABLE:
            raise RuntimeError("nselib not available. Install with: pip install nselib")
        
        try:
            from nselib import get_fii_dii
            
            if from_date is None:
                from_date = date.today() - timedelta(days=30)
            if to_date is None:
                to_date = date.today()
            
            data = get_fii_dii(from_date=from_date, to_date=to_date)
            
            if data is None or len(data) == 0:
                logger.warning("No FII/DII data returned")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(data)} rows of FII/DII data")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch FII/DII data: {e}")
            return pd.DataFrame()
    
    def get_universe_data(
        self,
        symbols: Optional[List[str]] = None,
        period: str = '1y'
    ) -> pd.DataFrame:
        """
        Get data for entire universe.
        
        Args:
            symbols: List of symbols (if None, uses NIFTY50)
            period: Time period
            
        Returns:
            DataFrame with all symbols as columns (multi-index)
        """
        if symbols is None:
            symbols = self.nifty50_symbols
        
        all_data = []
        
        for symbol in symbols:
            try:
                data = self.get_equity_price_history(symbol, period=period)
                if len(data) > 0:
                    data = data.copy()
                    data['symbol'] = symbol
                    all_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        if not all_data:
            logger.error("No data fetched for any symbol")
            return pd.DataFrame()
        
        combined = pd.concat(all_data)
        logger.info(f"Fetched data for {len(all_data)} symbols, {len(combined)} total rows")
        return combined
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or None if unavailable
        """
        try:
            data = self.get_equity_price_history(symbol, period='1d')
            if len(data) > 0:
                return float(data['close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None
    
    def get_universe_snapshot(self) -> Dict[str, Dict]:
        """
        Get current snapshot of universe prices.
        
        Returns:
            Dictionary with symbol -> {price, change, volume}
        """
        snapshot = {}
        
        for symbol in self.nifty50_symbols:
            try:
                data = self.get_equity_price_history(symbol, period='5d')
                if len(data) >= 2:
                    current = float(data['close'].iloc[-1])
                    prev = float(data['close'].iloc[-2])
                    change = (current - prev) / prev * 100
                    volume = float(data['volume'].iloc[-1])
                    
                    snapshot[symbol] = {
                        'price': current,
                        'change': change,
                        'volume': volume
                    }
            except Exception as e:
                logger.warning(f"Failed to get snapshot for {symbol}: {e}")
                continue
        
        logger.info(f"Got snapshot for {len(snapshot)} symbols")
        return snapshot


def get_nse_fetcher(
    cache_dir: Optional[str] = None,
    use_cache: bool = True
) -> NSEDataFetcher:
    """
    Factory function to get an NSE data fetcher.
    
    Args:
        cache_dir: Directory for caching
        use_cache: Whether to use cache
        
    Returns:
        NSEDataFetcher instance
    """
    return NSEDataFetcher(
        cache_dir=cache_dir,
        use_cache=use_cache
    )


if __name__ == "__main__":
    # Test the NSE data fetcher
    print("Testing NSE Data Fetcher...")
    
    try:
        fetcher = get_nse_fetcher(use_cache=True)
        
        # Test single equity
        print("\nFetching RELIANCE data...")
        reliance_data = fetcher.get_equity_price_history('RELIANCE', period='1m')
        if len(reliance_data) > 0:
            print(f"Fetched {len(reliance_data)} rows for RELIANCE")
            print(f"Latest price: ₹{reliance_data['close'].iloc[-1]:.2f}")
            print(f"Date range: {reliance_data.index[0]} to {reliance_data.index[-1]}")
        
        # Test index data
        print("\nFetching NIFTY 50 data...")
        nifty_data = fetcher.get_index_data('NIFTY 50', period='1m')
        if len(nifty_data) > 0:
            print(f"Fetched {len(nifty_data)} rows for NIFTY 50")
            print(f"Latest level: {nifty_data['close'].iloc[-1]:.2f}")
        
        # Test universe snapshot
        print("\nFetching universe snapshot...")
        snapshot = fetcher.get_universe_snapshot()
        print(f"Got snapshot for {len(snapshot)} symbols")
        for symbol, data in list(snapshot.items())[:5]:
            print(f"  {symbol}: ₹{data['price']:.2f} ({data['change']:+.2f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Install nselib with: pip install nselib")

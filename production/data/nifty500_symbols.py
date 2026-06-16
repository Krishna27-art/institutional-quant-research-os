"""
NIFTY500 Symbol Loader

This module loads the complete list of NIFTY500 constituents from NSE,
expanding from NIFTY50 to the full 500-stock universe for comprehensive
market coverage and alpha generation.

Key Features:
- Fetches complete NIFTY500 list from NSE API
- Sector breakdown and classification
- Caching mechanism for performance
- Fallback list for API failures
- Market cap filtering

Based on Blueprint Week 1-2: Foundation & Data Quality
"""

import pandas as pd
import requests
from typing import List, Optional, Dict
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class NIFTY500Loader:
    """
    Load and manage NIFTY500 constituent symbols.
    
    This class fetches the official NIFTY500 list from NSE and provides
    methods to access the complete universe of stocks.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize NIFTY500 loader.
        
        Args:
            cache_dir: Directory for caching symbol lists
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "nifty500"
        else:
            cache_dir = Path(cache_dir)
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "nifty500_symbols.json"
        self._nifty500_list = None
    
    def get_nifty500_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        Get NIFTY500 symbols.
        
        Args:
            force_refresh: Force refresh from API
            
        Returns:
            List of NIFTY500 symbols
        """
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self._nifty500_list = data['symbols']
                    logger.info(f"Loaded {len(self._nifty500_list)} NIFTY500 symbols from cache")
                    return self._nifty500_list
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Fetch from NSE
        symbols = self._fetch_from_nse()
        
        # Cache the results
        if symbols:
            self._cache_symbols(symbols)
        
        return symbols
    
    def _fetch_from_nse(self) -> List[str]:
        """
        Fetch NIFTY500 symbols from NSE API.
        
        Returns:
            List of symbols
        """
        try:
            # NSE NIFTY500 API endpoint
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                symbols = [item['symbol'] for item in data['data']]
                logger.info(f"Fetched {len(symbols)} NIFTY500 symbols from NSE")
                return symbols
            else:
                logger.warning("No data returned from NSE API")
                return self._get_fallback_list()
                
        except Exception as e:
            logger.error(f"Failed to fetch NIFTY500 from NSE: {e}")
            return self._get_fallback_list()
    
    def _get_fallback_list(self) -> List[str]:
        """
        Get fallback list of major NIFTY500 stocks.
        
        Returns:
            Fallback symbol list
        """
        # Major NIFTY500 stocks as fallback
        fallback = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
            'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LICI',
            'AXISBANK', 'LT', 'HINDUNILVR', 'BAJFINANCE', 'MARUTI',
            'TATAMOTORS', 'SUNPHARMA', 'TITAN', 'NTPC', 'WIPRO',
            'HCLTECH', 'ASIANPAINT', 'ULTRACEMCO', 'NESTLEIND', 'POWERGRID',
            'TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'ONGC', 'GAIL',
            'M&M', 'BAJAJFINSV', 'DABUR', 'BRITANNIA', 'DIVISLAB',
            'DRREDDY', 'CIPLA', 'SUNPHARMA', 'AUROPHARMA', 'LUPIN',
            'TATAPOWER', 'ADANIPORTS', 'ADANIENT', 'GRASIM', 'ACC',
            'AMBUJACEM', 'UPL', 'SHREECEM', 'ZEEL', 'MOTHERSUMI',
            'TATACONSUM', 'EICHERMOT', 'HEROMOTOCO', 'MARUTI', 'M&M'
        ]
        
        logger.warning(f"Using fallback list with {len(fallback)} symbols")
        return fallback
    
    def _cache_symbols(self, symbols: List[str]) -> None:
        """
        Cache symbols to file.
        
        Args:
            symbols: Symbol list to cache
        """
        try:
            data = {
                'symbols': symbols,
                'count': len(symbols),
                'last_updated': pd.Timestamp.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
                
            logger.info(f"Cached {len(symbols)} NIFTY500 symbols")
            
        except Exception as e:
            logger.warning(f"Failed to cache symbols: {e}")
    
    def get_sector_breakdown(self) -> Dict[str, List[str]]:
        """
        Get sector breakdown of NIFTY500 stocks.
        
        Returns:
            Dictionary mapping sectors to symbol lists
        """
        symbols = self.get_nifty500_symbols()
        
        # Simplified sector classification based on naming patterns
        sectors = {
            'IT': [],
            'BANKING': [],
            'PHARMA': [],
            'AUTO': [],
            'ENERGY': [],
            'METALS': [],
            'CONSUMER': [],
            'INFRASTRUCTURE': [],
            'OTHER': []
        }
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            
            if any(x in symbol_upper for x in ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECH']):
                sectors['IT'].append(symbol)
            elif any(x in symbol_upper for x in ['BANK', 'ICICI', 'HDFC', 'KOTAK', 'AXIS', 'SBI', 'IDFC']):
                sectors['BANKING'].append(symbol)
            elif any(x in symbol_upper for x in ['PHARMA', 'SUN', 'CIPLA', 'DRREDDY', 'LUPIN', 'AURO']):
                sectors['PHARMA'].append(symbol)
            elif any(x in symbol_upper for x in ['M&M', 'MARUTI', 'TATA', 'HERO', 'EICHER', 'MOTHER']):
                sectors['AUTO'].append(symbol)
            elif any(x in symbol_upper for x in ['ONGC', 'GAIL', 'POWER', 'NTPC', 'COAL', 'TATAPOWER']):
                sectors['ENERGY'].append(symbol)
            elif any(x in symbol_upper for x in ['STEEL', 'JSW', 'TATASTEEL', 'HINDALCO', 'COAL']):
                sectors['METALS'].append(symbol)
            elif any(x in symbol_upper for x in ['ITC', 'HIND', 'DABUR', 'BRITANNIA', 'NESTLE', 'TITAN']):
                sectors['CONSUMER'].append(symbol)
            elif any(x in symbol_upper for x in ['L&T', 'ULTRA', 'ACC', 'AMBUJA', 'GRASIM', 'SHREE']):
                sectors['INFRASTRUCTURE'].append(symbol)
            else:
                sectors['OTHER'].append(symbol)
        
        return sectors
    
    def get_large_cap_stocks(self, min_market_cap: float = 1000) -> List[str]:
        """
        Get large-cap stocks from NIFTY500.
        
        Args:
            min_market_cap: Minimum market cap in crores
            
        Returns:
            List of large-cap symbols
        """
        # In production, this would fetch actual market cap data
        # For now, return top 100 by index weight
        symbols = self.get_nifty500_symbols()
        
        # Return top 100 as large-cap approximation
        return symbols[:100]


# Singleton instance
_nifty500_loader = None


def get_nifty500_symbols(force_refresh: bool = False) -> List[str]:
    """
    Get NIFTY500 symbols (singleton function).
    
    Args:
        force_refresh: Force refresh from API
        
    Returns:
        List of NIFTY500 symbols
    """
    global _nifty500_loader
    if _nifty500_loader is None:
        _nifty500_loader = NIFTY500Loader()
    return _nifty500_loader.get_nifty500_symbols(force_refresh)


if __name__ == "__main__":
    # Test NIFTY500 loader
    print("Testing NIFTY500 Symbol Loader...")
    
    loader = NIFTY500Loader()
    symbols = loader.get_nifty500_symbols()
    
    print(f"\nTotal NIFTY500 symbols: {len(symbols)}")
    print(f"First 10 symbols: {symbols[:10]}")
    
    # Get sector breakdown
    sectors = loader.get_sector_breakdown()
    print(f"\nSector Breakdown:")
    for sector, stocks in sectors.items():
        print(f"{sector}: {len(stocks)} stocks")
    
    print("\nNIFTY500 Symbol Loader test completed.")

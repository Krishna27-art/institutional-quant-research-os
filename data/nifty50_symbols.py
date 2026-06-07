"""
NIFTY50 Symbol Loader

This module loads the complete list of NIFTY50 constituents from NSE,
replacing the hardcoded list of 5 stocks with the full universe.

Based on Audit Report Priority 0: Critical - Week 1-2
"""

import pandas as pd
import requests
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class NIFTY50Loader:
    """
    Load and manage NIFTY50 constituent symbols.
    
    This class fetches the official NIFTY50 list from NSE and provides
    methods to access the complete universe of stocks.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize NIFTY50 loader.
        
        Args:
            cache_dir: Directory to cache the symbol list
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "nifty50_symbols.json"
        
        self._symbols: Optional[List[str]] = None
    
    def get_symbols(self, force_refresh: bool = False, as_of_date: Optional[datetime] = None) -> List[str]:
        """
        Get NIFTY50 symbols.
        
        Args:
            force_refresh: Force refresh from NSE instead of using cache
            as_of_date: Date to get NIFTY50 constituents for (avoids survivorship bias)
            
        Returns:
            List of NIFTY50 stock symbols
        """
        if self._symbols is None or force_refresh:
            self._symbols = self._load_symbols(force_refresh)
            
        symbols = list(self._symbols)
        if as_of_date is None:
            return symbols
            
        # Point-in-time adjustments
        # Adjust constituents backwards from today (approx June 2026)
        dt = pd.Timestamp(as_of_date)
        
        # September 2024 changes: TRENT, BEL added; LTIMINDTREE, UPL/DIVISLAB removed
        if dt < pd.Timestamp("2024-09-30"):
            for s in ["TRENT", "BEL"]:
                if s in symbols: symbols.remove(s)
            for s in ["LTIMINDTREE", "DIVISLAB"]:
                if s not in symbols: symbols.append(s)
                
        # March 2024 changes: SHRIRAM FINANCE added; UPL removed
        if dt < pd.Timestamp("2024-03-28"):
            if "SHRIRAMFIN" in symbols: symbols.remove("SHRIRAMFIN")
            if "UPL" not in symbols: symbols.append("UPL")
            
        # July 2023 changes: HDFC merged, LTIMINDTREE added
        if dt < pd.Timestamp("2023-07-13"):
            if "LTIMINDTREE" in symbols: symbols.remove("LTIMINDTREE")
            if "HDFC" not in symbols: symbols.append("HDFC")
            
        # September 2022 changes: ADANI ENTERPRISES added; SHREE CEMENT removed
        if dt < pd.Timestamp("2022-09-30"):
            if "ADANIENT" in symbols: symbols.remove("ADANIENT")
            if "SHREECEM" not in symbols: symbols.append("SHREECEM")
            
        # March 2022 changes: APOLLO HOSPITALS added; INDIAN OIL CORP removed
        if dt < pd.Timestamp("2022-03-31"):
            if "APOLLOHOSP" in symbols: symbols.remove("APOLLOHOSP")
            if "IOC" not in symbols: symbols.append("IOC")
            
        # March 2021 changes: TATA CONSUMER PRODUCTS added; GAIL removed
        if dt < pd.Timestamp("2021-03-31"):
            if "TATACONSUM" in symbols: symbols.remove("TATACONSUM")
            if "GAIL" not in symbols: symbols.append("GAIL")
            
        return symbols
    
    def _load_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        Load symbols from cache or fetch from NSE.
        
        Args:
            force_refresh: Force refresh from NSE
            
        Returns:
            List of NIFTY50 stock symbols
        """
        # Try to load from cache first
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                    cache_time = pd.to_datetime(cached_data['timestamp'])
                    # Cache is valid for 24 hours
                    if (pd.Timestamp.now() - cache_time).total_seconds() < 86400:
                        logger.info(f"Loaded NIFTY50 symbols from cache ({len(cached_data['symbols'])} symbols)")
                        return cached_data['symbols']
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Fetch from NSE
        symbols = self._fetch_from_nse()
        
        # Save to cache
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'symbols': symbols
                }, f)
            logger.info(f"Cached NIFTY50 symbols ({len(symbols)} symbols)")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
        
        return symbols
    
    def _fetch_from_nse(self) -> List[str]:
        """
        Fetch NIFTY50 symbols from NSE.
        
        Returns:
            List of NIFTY50 stock symbols
        """
        try:
            # NSE provides CSV download for index constituents
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info("Fetching NIFTY50 symbols from NSE...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract symbols from the response
            if 'data' in data:
                symbols = [item['symbol'] for item in data['data']]
                logger.info(f"Fetched {len(symbols)} NIFTY50 symbols from NSE")
                return symbols
            else:
                logger.error("Unexpected response format from NSE")
                return self._get_fallback_symbols()
                
        except Exception as e:
            logger.error(f"Failed to fetch from NSE: {e}")
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self) -> List[str]:
        """
        Get fallback list of NIFTY50 symbols.
        
        This is a hardcoded list of major NIFTY50 stocks used as fallback
        when the NSE API is unavailable.
        
        Returns:
            List of NIFTY50 stock symbols
        """
        logger.warning("Using fallback NIFTY50 symbol list")
        
        fallback_symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
            "LT", "AXISBANK", "BAJFINANCE", "MARUTI", "HCLTECH",
            "ASIANPAINT", "SUNPHARMA", "TITAN", "DMART", "ULTRATECH",
            "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "WIPRO",
            "ONGC", "JSWSTEEL", "M&M", "ADANIENT", "NESTLEIND",
            "CIPLA", "COALINDIA", "BRITANNIA", "DRREDDY", "TATACONSUM",
            "GRASIM", "SHREECEM", "HEROMOTOCO", "BAJAJFINSV", "DIVISLAB",
            "APOLLOHOSP", "EICHERMOT", "HINDALCO", "TechM", "UPL"
        ]
        
        logger.info(f"Using fallback list with {len(fallback_symbols)} symbols")
        return fallback_symbols
    
    def get_symbol_details(self) -> pd.DataFrame:
        """
        Get detailed information about NIFTY50 symbols.
        
        Returns:
            DataFrame with symbol details
        """
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                logger.info(f"Fetched details for {len(df)} NIFTY50 symbols")
                return df
            else:
                logger.error("Unexpected response format from NSE")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch symbol details: {e}")
            return pd.DataFrame()
    
    def get_sector_breakdown(self) -> Dict[str, List[str]]:
        """
        Get sector-wise breakdown of NIFTY50 symbols.
        
        Returns:
            Dictionary mapping sectors to symbol lists
        """
        details = self.get_symbol_details()
        
        if details.empty or 'industry' not in details.columns:
            return {}
        
        sector_map = {}
        for _, row in details.iterrows():
            sector = row.get('industry', 'Unknown')
            symbol = row['symbol']
            
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(symbol)
        
        return sector_map


# Singleton instance
_nifty50_loader = None

def get_nifty50_loader() -> NIFTY50Loader:
    """Get the singleton NIFTY50 loader instance."""
    global _nifty50_loader
    if _nifty50_loader is None:
        _nifty50_loader = NIFTY50Loader()
    return _nifty50_loader


def get_nifty50_symbols(force_refresh: bool = False, as_of_date: Optional[datetime] = None) -> List[str]:
    """
    Convenience function to get NIFTY50 symbols.
    
    Args:
        force_refresh: Force refresh from NSE
        as_of_date: Date to get NIFTY50 constituents for
        
    Returns:
        List of NIFTY50 stock symbols
    """
    loader = get_nifty50_loader()
    return loader.get_symbols(force_refresh, as_of_date)


if __name__ == "__main__":
    # Test the NIFTY50 loader
    print("Testing NIFTY50 Loader...")
    
    loader = NIFTY50Loader()
    
    # Get symbols
    symbols = loader.get_symbols()
    print(f"Loaded {len(symbols)} NIFTY50 symbols")
    print(f"First 10 symbols: {symbols[:10]}")
    
    # Get sector breakdown
    sectors = loader.get_sector_breakdown()
    if sectors:
        print(f"\nSector Breakdown:")
        for sector, sector_symbols in sectors.items():
            print(f"  {sector}: {len(sector_symbols)} symbols")

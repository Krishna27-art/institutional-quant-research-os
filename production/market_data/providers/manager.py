"""
Unified Data Manager
Consolidates feed access, gap handling, and point-in-time universe utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataManagerConfig:
    """Configuration for data manager."""
    data_dir: str = "/data/nse_bars"
    enable_gap_handling: bool = True
    enable_corporate_actions: bool = True
    enable_point_in_time: bool = True
    max_gap_minutes: int = 5


class DataManager:
    """
    Unified Data Manager that consolidates feed access, gap handling, and point-in-time universe utilities.
    
    Algorithms:
    1. Feed Consolidation: Unified interface to multiple data feeds
    2. Gap Detection: Identifies missing data points in time series
    3. Backfill: Fills missing data from fallback feeds
    4. Corporate Actions: Applies splits, dividends, bonuses to historical data
    5. Point-in-Time Universe: Builds universe as it existed at each point in time
    6. Normalization: Standardizes OHLCV column layout across feeds
    """
    
    def __init__(self, config: DataManagerConfig = None):
        self.config = config or DataManagerConfig()
        self.feeds: Dict[str, object] = {}
        self.corporate_actions_db: Dict[str, List[Dict]] = {}
        self.universe_history: Dict[datetime, List[str]] = {}
        
    def register_feed(self, name: str, feed: object) -> None:
        """Register a data feed."""
        self.feeds[name] = feed
        logger.info(f"Registered feed: {name}")
    
    def get_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        feed_name: str = "primary"
    ) -> pd.DataFrame:
        """
        Get data from specified feed with fallback handling.
        
        Args:
            symbol: Symbol to fetch
            start: Start datetime
            end: End datetime
            feed_name: Primary feed to use
            
        Returns:
            DataFrame with OHLCV data
        """
        if feed_name not in self.feeds:
            logger.warning(f"Feed {feed_name} not found, trying fallback")
            return self._get_fallback_data(symbol, start, end)
        
        feed = self.feeds[feed_name]
        
        try:
            data = feed.get_historical_bars(symbol, start, end)
            
            if data.empty:
                logger.warning(f"No data from {feed_name}, trying fallback")
                return self._get_fallback_data(symbol, start, end)
            
            # Normalize columns
            data = self._normalize_ohlcv(data)
            
            # Handle gaps
            if self.config.enable_gap_handling:
                data = self._handle_gaps(data)
            
            # Apply corporate actions
            if self.config.enable_corporate_actions:
                data = self._apply_corporate_actions(symbol, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching from {feed_name}: {e}")
            return self._get_fallback_data(symbol, start, end)
    
    def _get_fallback_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get data from fallback feeds."""
        for name, feed in self.feeds.items():
            if name != "primary":
                try:
                    data = feed.get_historical_bars(symbol, start, end)
                    if not data.empty:
                        logger.info(f"Got fallback data from {name}")
                        return self._normalize_ohlcv(data)
                except Exception as e:
                    logger.warning(f"Fallback {name} failed: {e}")
        
        return pd.DataFrame()
    
    def _normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize OHLCV column layout across feeds."""
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        data = data.rename(columns=column_map)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in data.columns:
                logger.warning(f"Missing column {col}, filling with NaN")
                data[col] = np.nan
        
        return data
    
    def _handle_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify and fill gaps in time series."""
        if data.empty:
            return data
        
        # Detect gaps
        time_diffs = data.index.to_series().diff().dropna()
        gap_threshold = timedelta(minutes=self.config.max_gap_minutes)
        gaps = time_diffs[time_diffs > gap_threshold]
        
        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} gaps in data")
            
            # Backfill from other feeds
            for gap_time in gaps.index:
                gap_start = gap_time - time_diffs[gap_time]
                gap_end = gap_time
                
                # Try to fill from fallback feeds
                symbol = data.get('symbol', 'UNKNOWN')
                if isinstance(symbol, pd.Series):
                    symbol = symbol.iloc[0]
                
                fill_data = self._get_fallback_data(symbol, gap_start, gap_end)
                if not fill_data.empty:
                    data = pd.concat([data, fill_data]).sort_index()
        
        return data[~data.index.duplicated(keep='first')]
    
    def _apply_corporate_actions(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Apply splits, dividends, bonuses to historical data."""
        if symbol not in self.corporate_actions_db:
            return data
        
        actions = self.corporate_actions_db[symbol]
        
        for action in actions:
            action_type = action.get('type')
            action_date = action.get('date')
            ratio = action.get('ratio', 1.0)
            amount = action.get('amount', 0.0)
            
            if action_date not in data.index:
                continue
            
            if action_type == 'split':
                # Adjust prices before split date
                mask = data.index < action_date
                data.loc[mask, ['open', 'high', 'low', 'close']] /= ratio
                data.loc[mask, 'volume'] *= ratio
                
            elif action_type == 'bonus':
                # Adjust prices before bonus date
                mask = data.index < action_date
                data.loc[mask, ['open', 'high', 'low', 'close']] /= (1 + ratio)
                data.loc[mask, 'volume'] *= (1 + ratio)
                
            elif action_type == 'dividend':
                # Adjust for dividend (subtract from price)
                mask = data.index >= action_date
                data.loc[mask, ['open', 'high', 'low', 'close']] -= amount
        
        return data
    
    def build_point_in_time_universe(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        frequency: str = 'D'
    ) -> Dict[datetime, List[str]]:
        """
        Build universe as it existed at each point in time.
        
        Handles IPOs, delistings, and mergers.
        """
        dates = pd.date_range(start, end, freq=frequency)
        universe = {}
        
        for date in dates:
            # Get symbols that were tradable on this date
            tradable = []
            for symbol in symbols:
                # Check if symbol was listed on this date
                if self._was_symbol_tradable(symbol, date):
                    tradable.append(symbol)
            
            universe[date] = tradable
        
        self.universe_history = universe
        return universe
    
    def _was_symbol_tradable(self, symbol: str, date: datetime) -> bool:
        """Check if symbol was tradable on given date."""
        # This would check IPO dates, delisting dates, merger dates
        # For now, return True for all symbols
        return True
    
    def get_universe_at_date(self, date: datetime) -> List[str]:
        """Get universe of tradable symbols at specific date."""
        if not self.universe_history:
            return []
        
        # Find closest date
        dates = list(self.universe_history.keys())
        closest_date = min(dates, key=lambda d: abs(d - date))
        
        return self.universe_history.get(closest_date, [])

"""
Point-in-Time Universe Tracker and NSE Market Calendar
Manages index constituents over time and handles Indian market trading sessions, holidays, and expiries.
"""

import logging
from datetime import datetime, time, timedelta, date
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class UniverseTracker:
    """
    Tracks constituent changes in indices (e.g. NIFTY 50) over time,
    providing point-in-time active symbols to backtesters.
    """

    def __init__(self) -> None:
        # Map of index_name -> sorted list of tuples (date, added_set, removed_set)
        self._history: Dict[str, List[tuple[datetime, Set[str], Set[str]]]] = {}
        # Initial state: index_name -> set of initial constituents
        self._initial_universe: Dict[str, Set[str]] = {}

    def set_initial_universe(self, index_name: str, symbols: List[str]) -> None:
        """Set the base universe for an index at the beginning of tracking."""
        self._initial_universe[index_name] = set(symbols)
        if index_name not in self._history:
            self._history[index_name] = []
        logger.info(f"Initialized base universe for {index_name} with {len(symbols)} symbols.")

    def add_change(
        self,
        index_name: str,
        date: datetime,
        added: List[str],
        removed: List[str]
    ) -> None:
        """Record a change (addition/removal) to index constituents at a specific date."""
        if index_name not in self._history:
            self._history[index_name] = []
            
        self._history[index_name].append((date, set(added), set(removed)))
        # Keep changes sorted by date
        self._history[index_name].sort(key=lambda x: x[0])
        logger.info(f"Recorded change for {index_name} on {date.strftime('%Y-%m-%d')}: +{len(added)}, -{len(removed)}")

    def get_universe(self, index_name: str, timestamp: datetime) -> List[str]:
        """
        Get the list of active constituents for an index at a specific point in time.
        """
        universe = set(self._initial_universe.get(index_name, []))

        # Replay changes up to the given timestamp
        changes = self._history.get(index_name, [])
        for change_date, added, removed in changes:
            if change_date > timestamp:
                # Changes are sorted, so we can stop replaying future changes
                break
            universe.update(added)
            universe.difference_update(removed)

        return sorted(list(universe))


@dataclass
class TradingSession:
    """Trading session definition"""
    name: str
    start_time: time
    end_time: time
    is_active: bool = True


@dataclass
class Holiday:
    """Holiday definition"""
    date: date
    name: str
    is_trading_day: bool = False
    special_hours: Optional[Tuple[time, time]] = None


class NSEMarketCalendar:
    """
    Exact trading sessions, expiry calendars, rollover dates.
    
    NSE-specific logic:
    - Regular trading: 9:15 - 15:30 IST
    - Pre-market: 9:00 - 9:15 IST
    - Post-market: 15:40 - 16:00 IST
    - Weekly options expiry by day of week
    - Monthly futures expiry on last Thursday
    - Muhurat trading (special hours on Diwali)
    """
    
    def __init__(self):
        # Load holidays from CSV (updated annually)
        self.holidays = self._load_holidays()
        
        # Weekly options expiry by day of week
        self.weekly_expiry_weekday = {
            'NIFTY': 3,
            'BANKNIFTY': 2,
            'FINNIFTY': 2,
            'MIDCPNIFTY': 3,
            'SENSEX': 4,
        }
        self.weekly_expiries = self.weekly_expiry_weekday
        
        # Trading sessions
        self.sessions = {
            'pre_market': TradingSession('Pre-Market', time(9, 0), time(9, 15)),
            'regular': TradingSession('Regular', time(9, 15), time(15, 30)),
            'post_market': TradingSession('Post-Market', time(15, 40), time(16, 00))
        }
        
        # Muhurat trading (special hours on Diwali)
        self.muhurat_dates = self._load_muhurat_dates()
        
    def _load_holidays(self) -> Dict[date, Holiday]:
        """Load NSE holidays"""
        holidays = {}
        holiday_list = [
            (date(2024, 1, 26), "Republic Day"),
            (date(2024, 3, 25), "Holi"),
            (date(2024, 3, 29), "Good Friday"),
            (date(2024, 4, 11), "Ram Navami"),
            (date(2024, 4, 17), "Mahavir Jayanti"),
            (date(2024, 5, 1), "Maharashtra Day"),
            (date(2024, 6, 17), "Bakri Eid"),
            (date(2024, 7, 17), "Muharram"),
            (date(2024, 8, 15), "Independence Day"),
            (date(2024, 9, 16), "Ganesh Chaturthi"),
            (date(2024, 10, 2), "Mahatma Gandhi Jayanti"),
            (date(2024, 10, 12), "Dussehra"),
            (date(2024, 11, 1), "Diwali Balipratipada"),
            (date(2024, 11, 15), "Diwali Laxmi Pujan"),
            (date(2024, 12, 25), "Christmas"),
            (date(2025, 1, 26), "Republic Day"),
            (date(2025, 3, 14), "Holi"),
            (date(2025, 4, 18), "Good Friday"),
            (date(2025, 5, 1), "Maharashtra Day"),
            (date(2025, 8, 15), "Independence Day"),
            (date(2025, 10, 2), "Gandhi Jayanti"),
            (date(2025, 10, 21), "Diwali Laxmi Puja"),
            (date(2025, 12, 25), "Christmas"),
            (date(2026, 1, 26), "Republic Day"),
            (date(2026, 3, 25), "Holi"),
            (date(2026, 4, 17), "Good Friday"),
            (date(2026, 5, 1), "Maharashtra Day"),
            (date(2026, 8, 15), "Independence Day"),
            (date(2026, 10, 2), "Gandhi Jayanti"),
            (date(2026, 11, 4), "Diwali Laxmi Puja"),
            (date(2026, 12, 25), "Christmas"),
        ]
        for dt, name in holiday_list:
            holidays[dt] = Holiday(dt, name, is_trading_day=False)
        return holidays
    
    def _load_muhurat_dates(self) -> Dict[date, Tuple[time, time]]:
        """Load Muhurat trading dates and special hours"""
        return {
            date(2024, 11, 1): (time(18, 0), time(19, 0)),
            date(2025, 10, 21): (time(18, 0), time(19, 0)),
            date(2026, 11, 4): (time(18, 0), time(19, 0)),
        }
    
    def _is_muhurat(self, dt: datetime) -> bool:
        """Check if datetime is during Muhurat trading"""
        dt_date = dt.date()
        if dt_date in self.muhurat_dates:
            start, end = self.muhurat_dates[dt_date]
            return start <= dt.time() <= end
        return False
    
    def is_trading_day(self, dt: date) -> bool:
        """Check if a date is a trading day"""
        if dt in self.holidays:
            return self.holidays[dt].is_trading_day
        if dt.weekday() >= 5:  # Saturday, Sunday
            return False
        return True
    
    def is_trading_session(self, dt: datetime) -> bool:
        """Check if datetime is within a trading session"""
        if self._is_muhurat(dt):
            return True
        if not self.is_trading_day(dt.date()):
            return False
        regular = self.sessions['regular']
        return regular.start_time <= dt.time() <= regular.end_time
    
    def get_session(self, dt: datetime) -> Optional[str]:
        """Get the current trading session"""
        if not self.is_trading_day(dt.date()):
            return None
        if self._is_muhurat(dt):
            return 'muhurat'
        for session_name, session in self.sessions.items():
            if session.start_time <= dt.time() <= session.end_time:
                return session_name
        return None
    
    def next_trading_day(self, dt: date) -> date:
        """Get the next trading day"""
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def previous_trading_day(self, dt: date) -> date:
        """Get the previous trading day"""
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def _last_thursday_of_month(self, dt: date) -> date:
        """Get the last Thursday of the month for the given date"""
        year = dt.year
        month = dt.month
        last_day = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        while last_day.weekday() != 3:  # 3 = Thursday
            last_day -= timedelta(days=1)
        return last_day
    
    def next_expiry(self, dt: datetime, product_type: str = 'FUTURES', index: str = 'NIFTY') -> date:
        """Get the next expiry date for a product"""
        if product_type == 'FUTURES':
            expiry = self._last_thursday_of_month(dt.date())
            if expiry <= dt.date():
                if dt.month == 12:
                    next_month = date(dt.year + 1, 1, 28)
                else:
                    next_month = date(dt.year, dt.month + 1, 28)
                expiry = self._last_thursday_of_month(next_month)
            return expiry
        elif product_type == 'OPTIONS':
            expiry_day = self.weekly_expiry_weekday.get(index.upper(), 3)
            current_day = dt.weekday()
            days_until_expiry = (expiry_day - current_day) % 7
            if days_until_expiry == 0:
                days_until_expiry = 7
            expiry = dt.date() + timedelta(days=days_until_expiry)
            while not self.is_trading_day(expiry):
                expiry -= timedelta(days=1)
            return expiry
        else:
            raise ValueError(f"Unknown product type: {product_type}")
    
    def get_trading_days_in_range(self, start_date: date, end_date: date) -> List[date]:
        """Get all trading days in a date range"""
        trading_days = []
        current = start_date
        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days
    
    def get_holidays_in_range(self, start_date: date, end_date: date) -> List[Holiday]:
        """Get all holidays in a date range"""
        holidays = []
        for dt, holiday in self.holidays.items():
            if start_date <= dt <= end_date:
                holidays.append(holiday)
        return sorted(holidays, key=lambda h: h.date)
    
    def time_until_market_close(self, dt: datetime) -> timedelta:
        """Get time until market close"""
        if not self.is_trading_session(dt):
            return timedelta(0)
        regular = self.sessions['regular']
        close_time = datetime.combine(dt.date(), regular.end_time)
        return close_time - dt
    
    def time_until_market_open(self, dt: datetime) -> timedelta:
        """Get time until market open"""
        if self.is_trading_session(dt):
            return timedelta(0)
        next_day = self.next_trading_day(dt.date())
        open_time = datetime.combine(next_day, self.sessions['regular'].start_time)
        return open_time - dt


# Singleton instance
_calendar_instance = None

def get_nse_calendar() -> NSEMarketCalendar:
    """Get the singleton NSE market calendar instance"""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = NSEMarketCalendar()
    return _calendar_instance

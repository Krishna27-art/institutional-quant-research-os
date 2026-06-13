"""
NSE Market Calendar with exact trading sessions, expiry calendars, rollover dates.

This implements production-grade market calendar for Indian markets with:
- Exact trading session times
- Holiday calendar (NSE official holidays)
- Weekly options expiry by day of week
- Monthly futures expiry (last Thursday)
- Muhurat trading special hours
- Pre-market and post-market sessions
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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
        # Weekday codes: Monday=0 ... Friday=4. Update when exchange circulars change.
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
        """Load NSE holidays from CSV or hardcoded list"""
        # In production, load from CSV file
        # For now, use a sample list of 2024 holidays
        holidays = {}
        
        # Multi-year NSE holidays. In production this should be refreshed from
        # the official exchange holiday file during annual ops setup.
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
        # Muhurat trading typically occurs on Diwali evening
        # Special hours: 18:00 - 19:00 IST
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
        """
        Check if datetime is within a trading session.
        
        Returns True if:
        - It's a trading day
        - Time is within regular hours (9:15 - 15:30)
        - OR it's Muhurat trading hours
        """
        # Check Muhurat trading first
        if self._is_muhurat(dt):
            return True

        if not self.is_trading_day(dt.date()):
            return False
        
        # Check regular trading session
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
        
        # Start from the last day of the month
        last_day = date(year, month + 1, 1) - timedelta(days=1)
        
        # Find the last Thursday
        while last_day.weekday() != 3:  # 3 = Thursday
            last_day -= timedelta(days=1)
        
        return last_day
    
    def next_expiry(self, dt: datetime, product_type: str = 'FUTURES', index: str = 'NIFTY') -> date:
        """
        Get the next expiry date for a product.
        
        Args:
            dt: Current datetime
            product_type: 'FUTURES' or 'OPTIONS'
            index: Index name (for weekly options)
            
        Returns:
            Next expiry date
        """
        if product_type == 'FUTURES':
            # Monthly futures expire on last Thursday
            expiry = self._last_thursday_of_month(dt.date())
            if expiry <= dt.date():
                # Move to next month
                next_month = dt.replace(day=28) + timedelta(days=4)  # Go to next month
                expiry = self._last_thursday_of_month(next_month.date())
            return expiry
        
        elif product_type == 'OPTIONS':
            # Weekly options expiry by day of week
            # Map index to expiry day
            expiry_day = self.weekly_expiry_weekday.get(index.upper(), 3)
            
            # Find next occurrence of expiry_day
            current_day = dt.weekday()
            days_until_expiry = (expiry_day - current_day) % 7
            if days_until_expiry == 0:
                days_until_expiry = 7  # If today is expiry day, go to next week
            
            expiry = dt.date() + timedelta(days=days_until_expiry)
            
            # Skip if expiry is a holiday
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
        
        # Find next trading day
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

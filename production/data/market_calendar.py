"""
Market Calendar - NSE Trading Hours, Holidays, and Weekly Off

This module provides market calendar functionality to prevent trading on:
- NSE holidays
- Weekly off days
- Non-trading hours
- Exchange maintenance periods

Based on Jane Street Gap Analysis Priority #3: Market Calendar Implementation
"""

import pandas as pd
from datetime import datetime, time, timedelta
from typing import List, Optional, Set
from src.data.universe_tracker import NSE_HOLIDAYS
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# NSE Trading Hours (IST)
MARKET_OPEN = time(9, 15)  # 9:15 AM
MARKET_CLOSE = time(15, 30)  # 3:30 PM
PRE_OPEN_START = time(9, 0)  # 9:00 AM
PRE_OPEN_END = time(9, 8)   # 9:08 AM
POST_CLOSE_END = time(15, 40)  # 3:40 PM

# Weekly off (NSE is closed on Saturdays and Sundays)
WEEKLY_OFF_DAYS = {5, 6}  # Saturday=5, Sunday=6 (Monday=0)

# NSE Holidays dictionary removed to prevent duplication (bug #4)


class MarketCalendar:
    """
    Market calendar for NSE trading hours and holidays.
    
    This class provides methods to check if the market is open,
    if a given timestamp is during trading hours, and to get
    the next trading day.
    """
    
    def __init__(self, holidays: Optional[Set[str]] = None):
        """
        Initialize market calendar.
        
        Args:
            holidays: Set of holiday dates in 'YYYY-MM-DD' format.
                      If None, uses default NSE holidays.
        """
        self.holidays = holidays or self._get_all_holidays()
        self.weekly_off_days = WEEKLY_OFF_DAYS
        self.market_open = MARKET_OPEN
        self.market_close = MARKET_CLOSE
        
    def _get_all_holidays(self) -> Set[str]:
        """Get all NSE holidays from the calendar."""
        return {d.strftime('%Y-%m-%d') for d in NSE_HOLIDAYS}
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day.
        
        Args:
            date: Datetime to check
            
        Returns:
            True if it's a trading day, False otherwise
        """
        # Check if it's a weekly off (Saturday/Sunday)
        if date.weekday() in self.weekly_off_days:
            return False
        
        # Check if it's a holiday
        date_str = date.strftime('%Y-%m-%d')
        if date_str in self.holidays:
            return False
        
        return True
    
    def is_trading_time(self, timestamp: datetime) -> bool:
        """
        Check if a given timestamp is during trading hours.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            True if it's during trading hours, False otherwise
        """
        # First check if it's a trading day
        if not self.is_trading_day(timestamp):
            return False
        
        # Check if it's during trading hours
        current_time = timestamp.time()
        return self.market_open <= current_time <= self.market_close
    
    def is_pre_open(self, timestamp: datetime) -> bool:
        """Check if timestamp is during pre-open session."""
        if not self.is_trading_day(timestamp):
            return False
        current_time = timestamp.time()
        return PRE_OPEN_START <= current_time <= PRE_OPEN_END
    
    def is_post_close(self, timestamp: datetime) -> bool:
        """Check if timestamp is during post-close session."""
        if not self.is_trading_day(timestamp):
            return False
        current_time = timestamp.time()
        return self.market_close < current_time <= POST_CLOSE_END
    
    def get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day after a given date.
        
        Args:
            date: Starting date
            
        Returns:
            Next trading day datetime
        """
        next_day = date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def get_previous_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before a given date.
        
        Args:
            date: Starting date
            
        Returns:
            Previous trading day datetime
        """
        prev_day = date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def get_trading_days_in_range(self, start: datetime, end: datetime) -> List[datetime]:
        """
        Get all trading days in a date range.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of trading day datetimes
        """
        trading_days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        return trading_days
    
    def get_market_status(self, timestamp: Optional[datetime] = None) -> str:
        """
        Get current market status.
        
        Args:
            timestamp: Datetime to check (defaults to now)
            
        Returns:
            Market status string: 'OPEN', 'CLOSED', 'PRE_OPEN', 'POST_CLOSE'
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.is_pre_open(timestamp):
            return 'PRE_OPEN'
        elif self.is_trading_time(timestamp):
            return 'OPEN'
        elif self.is_post_close(timestamp):
            return 'POST_CLOSE'
        else:
            return 'CLOSED'
    
    def add_holiday(self, date: str):
        """
        Add a custom holiday to the calendar.
        
        Args:
            date: Holiday date in 'YYYY-MM-DD' format
        """
        self.holidays.add(date)
        logger.info(f"Added holiday: {date}")
    
    def remove_holiday(self, date: str):
        """
        Remove a holiday from the calendar.
        
        Args:
            date: Holiday date in 'YYYY-MM-DD' format
        """
        if date in self.holidays:
            self.holidays.remove(date)
            logger.info(f"Removed holiday: {date}")
    
    def get_holidays_for_year(self, year: int) -> List[str]:
        """
        Get all holidays for a specific year.
        
        Args:
            year: Year to get holidays for
            
        Returns:
            List of holiday dates in 'YYYY-MM-DD' format
        """
        return sorted([h for h in self.holidays if h.startswith(str(year))])
    
    def save_calendar(self, filepath: str):
        """
        Save current calendar configuration to file.
        
        Args:
            filepath: Path to save calendar
        """
        calendar_data = {
            'holidays': sorted(list(self.holidays)),
            'weekly_off_days': list(self.weekly_off_days),
            'market_open': self.market_open.strftime('%H:%M'),
            'market_close': self.market_close.strftime('%H:%M'),
        }
        
        with open(filepath, 'w') as f:
            json.dump(calendar_data, f, indent=2)
        
        logger.info(f"Saved calendar to {filepath}")
    
    def load_calendar(self, filepath: str):
        """
        Load calendar configuration from file.
        
        Args:
            filepath: Path to load calendar from
        """
        with open(filepath, 'r') as f:
            calendar_data = json.load(f)
        
        self.holidays = set(calendar_data['holidays'])
        self.weekly_off_days = set(calendar_data['weekly_off_days'])
        self.market_open = datetime.strptime(calendar_data['market_open'], '%H:%M').time()
        self.market_close = datetime.strptime(calendar_data['market_close'], '%H:%M').time()
        
        logger.info(f"Loaded calendar from {filepath}")


# Singleton instance
_market_calendar = None

def get_market_calendar(holidays: Optional[Set[str]] = None) -> MarketCalendar:
    """Get the singleton market calendar instance."""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = MarketCalendar(holidays)
    return _market_calendar


if __name__ == "__main__":
    # Test the market calendar
    print("Testing Market Calendar...")
    
    calendar = MarketCalendar()
    
    # Test current time
    now = datetime.now()
    print(f"Current time: {now}")
    print(f"Market status: {calendar.get_market_status(now)}")
    print(f"Is trading day: {calendar.is_trading_day(now)}")
    print(f"Is trading time: {calendar.is_trading_time(now)}")
    
    # Test specific dates
    test_dates = [
        datetime(2024, 1, 26, 10, 0),  # Republic Day (holiday)
        datetime(2024, 1, 27, 10, 0),  # Saturday (weekly off)
        datetime(2024, 1, 22, 10, 0),  # Monday (trading day)
        datetime(2024, 1, 22, 8, 0),   # Before market open
        datetime(2024, 1, 22, 16, 0),  # After market close
    ]
    
    print("\nTesting specific dates:")
    for test_date in test_dates:
        status = calendar.get_market_status(test_date)
        is_trading = calendar.is_trading_day(test_date)
        print(f"{test_date}: Status={status}, TradingDay={is_trading}")
    
    # Test holidays for 2024
    print(f"\nNSE Holidays 2024: {len(calendar.get_holidays_for_year(2024))}")
    print(f"First 5 holidays: {calendar.get_holidays_for_year(2024)[:5]}")

"""
Market Calendar - Single Source of Truth for NSE Trading Hours, Holidays, and Sessions.
"""

from datetime import datetime, time, date, timezone
from typing import Set, Optional
from zoneinfo import ZoneInfo

# Timezone Contract: All market time logic operates in IST.
IST = ZoneInfo("Asia/Kolkata")

# NSE Trading Hours (IST)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 15)  # Pre-open officially ends at 9:15 when regular market opens
POST_CLOSE_END = time(16, 0)

# Weekly off (NSE is closed on Saturdays and Sundays)
# Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
WEEKLY_OFF_DAYS = {5, 6}

# Consolidated NSE Holidays (2024-2026)
NSE_HOLIDAYS: Set[date] = {
    # 2024
    date(2024, 1, 26),  # Republic Day
    date(2024, 3, 25),  # Holi
    date(2024, 3, 29),  # Good Friday
    date(2024, 4, 11),  # Ram Navami
    date(2024, 4, 17),  # Mahavir Jayanti
    date(2024, 5, 1),   # Maharashtra Day
    date(2024, 6, 17),  # Bakri Id
    date(2024, 7, 17),  # Muharram
    date(2024, 8, 15),  # Independence Day
    date(2024, 8, 19),  # Raksha Bandhan
    date(2024, 9, 16),  # Ganesh Chaturthi
    date(2024, 10, 2),  # Gandhi Jayanti
    date(2024, 10, 12), # Dussehra
    date(2024, 10, 31), # Diwali Balipratipada
    date(2024, 11, 1),  # Diwali Laxmi Pujan
    date(2024, 11, 15), # Guru Nanak Jayanti
    date(2024, 12, 25), # Christmas
    # 2025
    date(2025, 1, 26),  # Republic Day
    date(2025, 2, 26),  # Mahashivratri
    date(2025, 3, 14),  # Holi
    date(2025, 3, 31),  # Id-ul-Fitr
    date(2025, 4, 10),  # Mahavir Jayanti
    date(2025, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 1),   # Maharashtra Day
    date(2025, 8, 15),  # Independence Day
    date(2025, 8, 27),  # Ganesh Chaturthi
    date(2025, 10, 2),  # Mahatma Gandhi Jayanti
    date(2025, 10, 21), # Diwali Laxmi Pujan
    date(2025, 10, 22), # Diwali Balipratipada
    date(2025, 11, 5),  # Gurunanak Jayanti
    date(2025, 12, 25), # Christmas
    # 2026
    date(2026, 1, 26),  # Republic Day
    date(2026, 3, 25),  # Ramzan Id
    date(2026, 4, 17),  # Good Friday
    date(2026, 5, 1),   # Maharashtra Day
    date(2026, 8, 15),  # Independence Day
    date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    date(2026, 10, 19), # Diwali Balipratipada
    date(2026, 10, 20), # Diwali Laxmi Pujan
    date(2026, 11, 25), # Guru Nanak Jayanti
    date(2026, 12, 25), # Christmas
}

class MarketCalendar:
    """
    Market calendar for NSE trading hours and holidays.
    This is the Single Source of Truth for all time-based market state.
    """
    
    @classmethod
    def get_current_ist_time(cls) -> datetime:
        """Get current timezone-aware time in IST."""
        return datetime.now(IST)

    @classmethod
    def is_trading_day(cls, dt: Optional[datetime] = None) -> bool:
        """Check if a given date is a valid trading day (not a holiday or weekend)."""
        if dt is None:
            dt = cls.get_current_ist_time()
        
        # Ensure we are checking the date in IST
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        else:
            dt = dt.astimezone(IST)
            
        d = dt.date()
        if d.weekday() in WEEKLY_OFF_DAYS:
            return False
        if d in NSE_HOLIDAYS:
            return False
        return True

    @classmethod
    def is_market_open(cls, dt: Optional[datetime] = None) -> bool:
        """Check if the regular trading session is currently active."""
        if dt is None:
            dt = cls.get_current_ist_time()
            
        if not cls.is_trading_day(dt):
            return False
            
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        else:
            dt = dt.astimezone(IST)
            
        return MARKET_OPEN <= dt.time() < MARKET_CLOSE

    @classmethod
    def is_pre_open(cls, dt: Optional[datetime] = None) -> bool:
        """Check if the pre-open session is currently active."""
        if dt is None:
            dt = cls.get_current_ist_time()
            
        if not cls.is_trading_day(dt):
            return False
            
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        else:
            dt = dt.astimezone(IST)
            
        return PRE_OPEN_START <= dt.time() < PRE_OPEN_END

    @classmethod
    def market_status(cls, dt: Optional[datetime] = None) -> dict:
        """Get full market status summary."""
        if dt is None:
            dt = cls.get_current_ist_time()
            
        return {
            "is_trading_day": cls.is_trading_day(dt),
            "is_open": cls.is_market_open(dt),
            "is_pre_open": cls.is_pre_open(dt),
            "timestamp_ist": dt.isoformat(),
        }

    @classmethod
    def is_trading_time(cls, dt: datetime) -> bool:
        """Check if a given timestamp is during trading hours."""
        return cls.is_market_open(dt)

# For backwards compatibility and convenience
def is_market_open() -> bool:
    return MarketCalendar.is_market_open()

def is_pre_open() -> bool:
    return MarketCalendar.is_pre_open()

def market_status() -> dict:
    return MarketCalendar.market_status()

def get_market_calendar():
    return MarketCalendar

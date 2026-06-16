from datetime import datetime, time, date
import pytz

IST = pytz.timezone("Asia/Kolkata")

from src.data.universe_tracker import NSE_HOLIDAYS

def get_holidays_for_year(year: int) -> set[date]:
    """Get holidays for a specific year."""
    return {d for d in NSE_HOLIDAYS if d.year == year}

MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN     = time(9,  0)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_market_open() -> bool:
    now  = now_ist()
    today = now.date()

    # Weekend
    if now.weekday() >= 5:
        return False

    # NSE holiday
    if today in NSE_HOLIDAYS:
        return False

    # Outside trading hours
    current_time = now.time()
    if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
        return False

    return True


def is_pre_open() -> bool:
    now = now_ist()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return PRE_OPEN <= t < MARKET_OPEN


def market_status() -> dict:
    now   = now_ist()
    open_ = is_market_open()
    return {
        "is_open":       open_,
        "is_pre_open":   is_pre_open(),
        "current_time":  now.strftime("%H:%M:%S IST"),
        "day":           now.strftime("%A"),
        "is_weekend":    now.weekday() >= 5,
        "is_holiday":    now.date() in NSE_HOLIDAYS,
        "next_open":     _next_open_str(),
    }


def _next_open_str() -> str:
    now = now_ist()
    d   = now.date()
    # If today is a trading day and we are before market open, next open is today
    if d.weekday() < 5 and d not in NSE_HOLIDAYS:
        if now.time() < MARKET_OPEN:
            return "Today at 09:15 IST"
            
    # Walk forward until we find a trading day
    for i in range(1, 10):
        import datetime as dt
        candidate = d + dt.timedelta(days=i)
        if candidate.weekday() < 5 and candidate not in NSE_HOLIDAYS:
            return f"{candidate.strftime('%A %d %b')} at 09:15 IST"
    return "Unknown"

from datetime import datetime, time, date
import pytz

IST = pytz.timezone("Asia/Kolkata")

# CRITICAL FIX: Multi-year NSE holidays support
NSE_HOLIDAYS = {
    # 2024
    date(2024, 1, 26),  # Republic Day
    date(2024, 3, 25),  # Holi
    date(2024, 3, 29),  # Good Friday
    date(2024, 4, 11),  # Eid-ul-Fitr
    date(2024, 4, 17),  # Ram Navami
    date(2024, 5, 1),   # Maharashtra Day
    date(2024, 6, 17),  # Bakri Eid
    date(2024, 8, 15),  # Independence Day
    date(2024, 10, 2),  # Gandhi Jayanti
    date(2024, 11, 1),  # Diwali Laxmi Puja
    date(2024, 11, 15), # Diwali Balipratipada
    date(2024, 12, 25), # Christmas
    # 2025
    date(2025, 1, 26),  # Republic Day
    date(2025, 3, 14),  # Holi
    date(2025, 4, 18),  # Good Friday
    date(2025, 4, 6),   # Ram Navami
    date(2025, 5, 1),   # Maharashtra Day
    date(2025, 6, 7),   # Bakri Eid
    date(2025, 8, 15),  # Independence Day
    date(2025, 10, 2),  # Gandhi Jayanti
    date(2025, 10, 21), # Diwali Laxmi Puja
    date(2025, 10, 22), # Diwali Balipratipada
    date(2025, 12, 25), # Christmas
    # 2026
    date(2026, 1, 26),  # Republic Day
    date(2026, 3, 25),  # Holi
    date(2026, 4, 14),  # Dr. Ambedkar Jayanti
    date(2026, 4, 17),  # Good Friday
    date(2026, 5,  1),  # Maharashtra Day
    date(2026, 8, 15),  # Independence Day
    date(2026, 10, 2),  # Gandhi Jayanti
    date(2026, 11, 4),  # Diwali Laxmi Puja
    date(2026, 12, 25), # Christmas
    # 2027
    date(2027, 1, 26),  # Republic Day
    date(2027, 3, 14),  # Holi
    date(2027, 4, 2),   # Good Friday
    date(2027, 5, 1),   # Maharashtra Day
    date(2027, 8, 15),  # Independence Day
    date(2027, 10, 2),  # Gandhi Jayanti
    date(2027, 10, 18), # Diwali Laxmi Puja
    date(2027, 12, 25), # Christmas
}

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
    # Walk forward until we find a trading day
    for i in range(1, 10):
        import datetime as dt
        candidate = d + dt.timedelta(days=i)
        if candidate.weekday() < 5 and candidate not in NSE_HOLIDAYS:
            return f"{candidate.strftime('%A %d %b')} at 09:15 IST"
    return "Unknown"

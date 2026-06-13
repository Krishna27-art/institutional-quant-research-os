# data/truth.py
import time
import random
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, date
import pytz
import logging

log = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

def yf_download_with_retry(tickers, max_retries=3, initial_delay=2.0, **kwargs):
    """Downloads prices from Yahoo Finance with exponential backoff and jitter."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            raw = yf.download(tickers, **kwargs)
            if not raw.empty:
                return raw
        except Exception as e:
            log.warning(f"Yahoo Finance download attempt {attempt+1} failed: {e}")
        # Add jitter
        sleep_time = delay + random.uniform(0.1, 1.0)
        time.sleep(sleep_time)
        delay *= 2
    # Last attempt
    try:
        return yf.download(tickers, **kwargs)
    except Exception as e:
        log.error(f"Yahoo Finance download failed after all retries: {e}")
        return pd.DataFrame()


def yf_download_with_fallback(tickers, max_retries=3, initial_delay=2.0, **kwargs):
    """Downloads prices with fallback tickers for unreliable symbols."""
    # First attempt with original tickers
    raw = yf_download_with_retry(tickers, max_retries, initial_delay, **kwargs)
    
    if raw.empty:
        # Try with fallback tickers
        fallback_tickers = []
        ticker_map = {}
        
        for ticker in tickers:
            if ticker in FALLBACK_TICKERS:
                fallback = FALLBACK_TICKERS[ticker][0]
                fallback_tickers.append(fallback)
                ticker_map[fallback] = ticker  # Map fallback back to original
                log.warning(f"Using fallback ticker {fallback} for {ticker}")
            else:
                fallback_tickers.append(ticker)
                ticker_map[ticker] = ticker
        
        if fallback_tickers != tickers:
            raw = yf_download_with_retry(fallback_tickers, max_retries, initial_delay, **kwargs)
            
            # If successful, rename columns back to original ticker names
            if not raw.empty:
                for fallback, original in ticker_map.items():
                    if fallback != original:
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if (col, fallback) in raw.columns:
                                raw[(col, original)] = raw[(col, fallback)]
                                del raw[(col, fallback)]
    
    return raw


def yf_actions_with_retry(ticker_symbol, max_retries=3, initial_delay=2.0):
    """Fetches actions from Yahoo Finance Ticker with exponential backoff and jitter."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(ticker_symbol)
            actions = ticker.actions
            if actions is not None:
                return actions
        except Exception as e:
            log.warning(f"Yahoo Finance actions attempt {attempt+1} failed for {ticker_symbol}: {e}")
        sleep_time = delay + random.uniform(0.1, 1.0)
        time.sleep(sleep_time)
        delay *= 2
    # Last attempt
    try:
        ticker = yf.Ticker(ticker_symbol)
        return ticker.actions
    except Exception as e:
        log.error(f"Yahoo Finance actions failed for {ticker_symbol} after all retries: {e}")
        return None

NSE_UNIVERSE = {
    "RELIANCE":    "RELIANCE.NS",
    "HDFCBANK":    "HDFCBANK.NS",
    "INFY":        "INFY.NS",
    "TCS":         "TCS.NS",
    "ICICIBANK":   "ICICIBANK.NS",
    "KOTAKBANK":   "KOTAKBANK.NS",
    "AXISBANK":    "AXISBANK.NS",
    "BAJFINANCE":  "BAJFINANCE.NS",
    "BHARTIARTL":  "BHARTIARTL.NS",
    "HINDUNILVR":  "HINDUNILVR.NS",
    "WIPRO":       "WIPRO.NS",
    "LT":          "LT.NS",
    "SBIN":        "SBIN.NS",
    "MARUTI":      "MARUTI.NS",
    "ASIANPAINT":  "ASIANPAINT.NS",
    "TITAN":       "TITAN.NS",
    "NESTLEIND":   "NESTLEIND.NS",
    "TECHM":       "TECHM.NS",
    "HCLTECH":     "HCLTECH.NS",
    "SUNPHARMA":   "SUNPHARMA.NS",
    "NIFTY":       "^NSEI",
    "BANKNIFTY":   "^NSEBANK",
    "FINNIFTY":    "NIFTY_FIN_SERVICE.NS",
    "INDIAVIX":    "^INDIAVIX",
}

# Fallback tickers for unreliable Yahoo Finance symbols
FALLBACK_TICKERS = {
    "NIFTY_FIN_SERVICE.NS": ["^NSEI"],  # Use NIFTY as proxy for FINNIFTY
    "^INDIAVIX": ["^NSEI"],  # Use NIFTY as proxy for VIX when unavailable
}

import os
from pathlib import Path

# Use absolute path to prevent DB path issues when starting from different directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(PROJECT_ROOT / "data" / "market_truth.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def _ensure_initialized():
    db_exists = os.path.exists(DB_PATH)
    is_empty = True
    if db_exists:
        try:
            con = sqlite3.connect(DB_PATH)
            res = con.execute("SELECT count(*) FROM daily_prices").fetchone()
            if res and res[0] > 0:
                is_empty = False
            con.close()
        except Exception:
            pass
    if not db_exists or is_empty:
        try:
            init_db()
            refresh_prices(period="5y")
        except Exception as e:
            log.error("Failed to auto-initialize truth DB: %s", e)

_ensure_initialized()



def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol      TEXT,
            date        TEXT,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            source      TEXT DEFAULT 'yahoo',
            fetched_at  TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS corporate_actions (
            symbol      TEXT,
            date        TEXT,
            action_type TEXT,
            value       REAL,
            PRIMARY KEY (symbol, date, action_type)
        )
    """)
    con.commit()
    con.close()


def refresh_prices(period: str = "5y") -> pd.DataFrame:
    """
    Downloads real prices from Yahoo Finance.
    Stores in SQLite. Returns clean DataFrame.
    This is the ONLY place prices are fetched.
    """
    tickers = list(NSE_UNIVERSE.values())
    log.info("Fetching %d symbols from Yahoo Finance...", len(tickers))

    raw = yf_download_with_fallback(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False,
        threads=True,
    )

    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    fetched_at = datetime.now(IST).isoformat()
    rows = []

    for sym_display, ticker in NSE_UNIVERSE.items():
        try:
            close  = raw["Close"][ticker].dropna()
            open_  = raw["Open"][ticker].dropna()
            high   = raw["High"][ticker].dropna()
            low    = raw["Low"][ticker].dropna()
            volume = raw["Volume"][ticker].dropna()

            for dt in close.index:
                rows.append((
                    sym_display,
                    str(dt.date()),
                    round(float(open_.get(dt, close[dt])), 2),
                    round(float(high.get(dt, close[dt])), 2),
                    round(float(low.get(dt, close[dt])), 2),
                    round(float(close[dt]), 2),
                    float(volume.get(dt, 0)),
                    "yahoo",
                    fetched_at,
                ))
        except Exception as e:
            log.warning("Failed %s: %s", ticker, e)

    con.executemany("""
        INSERT OR REPLACE INTO daily_prices
        (symbol, date, open, high, low, close, volume, source, fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, rows)
    con.commit()
    con.close()
    log.info("Stored %d price rows in truth DB", len(rows))

    # Also refresh corporate actions
    try:
        refresh_corporate_actions(period=period)
    except Exception as e:
        log.error("Failed to refresh corporate actions: %s", e)

    return get_latest_prices()


def refresh_corporate_actions(period: str = "5y") -> None:
    """
    Downloads historical splits and dividends from Yahoo Finance
    and stores them in the corporate_actions table.
    """
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    
    rows = []
    log.info("Fetching corporate actions for %d symbols...", len(NSE_UNIVERSE))
    for sym_display, ticker_symbol in NSE_UNIVERSE.items():
        try:
            # Retrieve historical actions with retry
            actions = yf_actions_with_retry(ticker_symbol)
            if actions is not None and not actions.empty:
                for dt, row in actions.iterrows():
                    date_str = str(dt.date())
                    div = float(row.get("Dividends", 0.0))
                    splits = float(row.get("Stock Splits", 0.0))
                    
                    if div > 0:
                        rows.append((sym_display, date_str, "dividend", div))
                    if splits > 0:
                        rows.append((sym_display, date_str, "split", splits))
        except Exception as e:
            log.warning("Failed to fetch corporate actions for %s: %s", ticker_symbol, e)
            
    if rows:
        con.executemany("""
            INSERT OR REPLACE INTO corporate_actions
            (symbol, date, action_type, value)
            VALUES (?, ?, ?, ?)
        """, rows)
        con.commit()
        log.info("Stored %d corporate actions in truth DB", len(rows))
    con.close()


def get_corporate_actions(symbol: str) -> pd.DataFrame:
    """Returns corporate actions history for one symbol."""
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    df = pd.read_sql("""
        SELECT date, action_type, value
        FROM corporate_actions
        WHERE symbol = ?
        ORDER BY date DESC
    """, con, params=(symbol,))
    con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_latest_prices() -> pd.DataFrame:
    """
    Returns latest close price for every symbol.
    This is what every other module reads.
    """
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT symbol, date, open, high, low, close, volume
        FROM daily_prices
        WHERE date = (
            SELECT MAX(date) FROM daily_prices d2
            WHERE d2.symbol = daily_prices.symbol
        )
        ORDER BY symbol
    """, con)
    con.close()
    return df


def get_price_history(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Returns price history for one symbol.
    All research and backtesting reads from here — never from yfinance directly.
    """
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT date, open, high, low, close, volume
        FROM daily_prices
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
    """, con, params=(symbol, days))
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def verify_prices() -> dict:
    """
    Sanity check. Compare our prices against dynamic median-based ranges.
    Falls back to sane absolute bounds if history is insufficient.
    Run this every morning. If it fails, do not trade.
    """
    df = get_latest_prices()
    results = {}
    
    # Modernized fallback ranges (realistic as of 2026)
    fallback_ranges = {
        "NIFTY":     (15000, 32000),
        "BANKNIFTY": (35000, 65000),
        "RELIANCE":  (1500,  4000),
        "HDFCBANK":  (1000,  2500),
        "TCS":       (2500,  5500),
        "INFY":      (1000,  2500),
        "FINNIFTY":  (15000, 28000),
        "INDIAVIX":  (5,     50),
    }
    
    all_ok = True
    con = sqlite3.connect(DB_PATH)
    
    try:
        for symbol in fallback_ranges.keys():
            row = df[df["symbol"] == symbol]
            if row.empty:
                results[symbol] = {"status": "MISSING", "price": None}
                all_ok = False
                continue
                
            price = float(row["close"].iloc[0])
            
            # Fetch last 30 daily close prices for dynamic validation
            hist = pd.read_sql("""
                SELECT close FROM daily_prices
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 30
            """, con, params=(symbol,))
            
            ok = False
            expected_range_str = ""
            
            if len(hist) >= 10:
                median_price = hist["close"].median()
                low = median_price * 0.5
                high = median_price * 1.5
                ok = low <= price <= high
                expected_range_str = f"dynamic: {low:.2f}–{high:.2f} (median: {median_price:.2f})"
            else:
                # Fallback to hardcoded sane ranges
                low, high = fallback_ranges.get(symbol, (1.0, 100000.0))
                ok = low <= price <= high
                expected_range_str = f"fallback: {low}–{high}"
                
            if not ok:
                all_ok = False
                
            results[symbol] = {
                "status": "OK" if ok else "SUSPICIOUS",
                "price": price,
                "expected": expected_range_str,
            }
    finally:
        con.close()
        
    results["all_ok"] = all_ok
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    refresh_prices()
    report = verify_prices()
    for sym, info in report.items():
        if sym == "all_ok":
            continue
        status = info["status"]
        price  = info["price"]
        exp    = info.get("expected", "")
        print(f"{sym:15s}  {status:10s}  ₹{price}  (expected {exp})")
    print()
    print("All prices OK:", report["all_ok"])

# data/truth.py
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, date
import pytz
import logging

log = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

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

DB_PATH = "data/market_truth.db"


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

    raw = yf.download(
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
            ticker = yf.Ticker(ticker_symbol)
            # Retrieve historical actions
            actions = ticker.actions
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
    Sanity check. Compare our prices against known ranges.
    Run this every morning. If it fails, do not trade.
    """
    df = get_latest_prices()
    results = {}
    known_ranges = {
        "NIFTY":     (20000, 26000),
        "RELIANCE":  (1000,  1800),
        "HDFCBANK":  (500,   1800),
        "TCS":       (1500,  4500),
        "INFY":      (800,   1800),
        "FINNIFTY":  (15000, 26000),
        "INDIAVIX":  (5,     40),
    }
    all_ok = True
    for sym, (low, high) in known_ranges.items():
        row = df[df["symbol"] == sym]
        if row.empty:
            results[sym] = {"status": "MISSING", "price": None}
            all_ok = False
            continue
        price = float(row["close"].iloc[0])
        ok    = low <= price <= high
        if not ok:
            all_ok = False
        results[sym] = {
            "status":   "OK" if ok else "SUSPICIOUS",
            "price":    price,
            "expected": f"{low}–{high}",
        }
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

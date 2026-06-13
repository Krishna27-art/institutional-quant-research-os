# research/prediction_registry.py
import sqlite3
import pandas as pd
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")
DB  = "data/predictions_cto.db"


def init():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT    NOT NULL,
            strategy      TEXT    NOT NULL,
            signal        TEXT    NOT NULL,   -- BUY / SHORT / WATCH
            signal_date   TEXT    NOT NULL,
            entry_price   REAL    NOT NULL,
            target_price  REAL    NOT NULL,
            stop_loss     REAL    NOT NULL,
            confidence    REAL,
            regime        TEXT,
            features      TEXT,              -- JSON of feature values at signal time
            exit_price    REAL,
            exit_date     TEXT,
            exit_reason   TEXT,              -- TARGET_HIT / STOP_HIT / TIME_EXIT / MANUAL
            pnl_pct       REAL,
            result        TEXT,              -- WIN / LOSS / OPEN
            notes         TEXT,
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS accuracy_daily (
            date          TEXT PRIMARY KEY,
            total         INTEGER,
            wins          INTEGER,
            losses        INTEGER,
            open_trades   INTEGER,
            win_rate      REAL,
            avg_pnl       REAL,
            sharpe_7d     REAL,
            computed_at   TEXT
        )
    """)
    con.commit()
    con.close()


def record_signal(symbol, strategy, signal, entry_price,
                  target_price, stop_loss, confidence=None,
                  regime=None, features=None):
    """Call this every time your system generates a signal."""
    import json
    con = sqlite3.connect(DB)
    con.execute("""
        INSERT INTO predictions
        (symbol, strategy, signal, signal_date, entry_price,
         target_price, stop_loss, confidence, regime, features, result)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        symbol, strategy, signal,
        datetime.now(IST).strftime("%Y-%m-%d"),
        entry_price, target_price, stop_loss,
        confidence, regime,
        json.dumps(features) if features else None,
        "OPEN",
    ))
    con.commit()
    con.close()


def update_outcome(prediction_id, exit_price, exit_reason):
    """Call this every day to mark predictions as won or lost."""
    con = sqlite3.connect(DB)
    row = con.execute(
        "SELECT signal, entry_price FROM predictions WHERE id=?",
        (prediction_id,)
    ).fetchone()

    if not row:
        con.close()
        return

    signal, entry = row
    if signal == "BUY":
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100

    result = "WIN" if pnl_pct > 0 else "LOSS"
    con.execute("""
        UPDATE predictions
        SET exit_price=?, exit_date=?, exit_reason=?,
            pnl_pct=?, result=?
        WHERE id=?
    """, (
        exit_price,
        datetime.now(IST).strftime("%Y-%m-%d"),
        exit_reason,
        round(pnl_pct, 3),
        result,
        prediction_id,
    ))
    con.commit()
    con.close()


def mark_open_predictions_daily():
    """
    Run this every day at 15:35 IST.
    Checks every open prediction against current price.
    Closes predictions that hit target or stop loss.
    """
    from data.truth import get_latest_prices
    prices = get_latest_prices().set_index("symbol")["close"].to_dict()

    con = sqlite3.connect(DB)
    open_preds = pd.read_sql(
        "SELECT * FROM predictions WHERE result='OPEN'", con
    )
    con.close()

    for _, row in open_preds.iterrows():
        sym   = row["symbol"]
        price = prices.get(sym)
        if price is None:
            continue

        if row["signal"] == "BUY":
            if price >= row["target_price"]:
                update_outcome(row["id"], price, "TARGET_HIT")
            elif price <= row["stop_loss"]:
                update_outcome(row["id"], price, "STOP_HIT")
        elif row["signal"] == "SHORT":
            if price <= row["target_price"]:
                update_outcome(row["id"], price, "TARGET_HIT")
            elif price >= row["stop_loss"]:
                update_outcome(row["id"], price, "STOP_HIT")


def get_accuracy_report() -> dict:
    """
    The real accuracy of your system.
    This replaces every hardcoded number in your dashboard.
    """
    con = sqlite3.connect(DB)
    df  = pd.read_sql(
        "SELECT * FROM predictions WHERE result != 'OPEN'", con
    )
    con.close()

    if df.empty:
        return {
            "message":       "No completed predictions yet. Start recording signals.",
            "total":         0,
            "wins":          0,
            "losses":        0,
            "win_rate":      0,
            "avg_pnl":       0,
            "avg_win":       0,
            "avg_loss":      0,
            "profit_factor": 0,
            "best_trade":    0,
            "worst_trade":   0,
        }

    wins   = df[df["result"] == "WIN"]
    losses = df[df["result"] == "LOSS"]

    avg_win  = float(wins["pnl_pct"].mean())   if len(wins)   > 0 else 0
    avg_loss = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0
    pf       = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        "total":         len(df),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(df) * 100, 1),
        "avg_pnl":       round(float(df["pnl_pct"].mean()), 2),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "profit_factor": round(pf, 2),
        "best_trade":    round(float(df["pnl_pct"].max()), 2),
        "worst_trade":   round(float(df["pnl_pct"].min()), 2),
        "by_strategy":   df.groupby("strategy")["pnl_pct"].agg(
            ["count", "mean"]
        ).round(2).to_dict(),
    }


if __name__ == "__main__":
    init()
    print("Prediction registry initialised.")
    print("Accuracy report:", get_accuracy_report())

"""Paper trading helper for Phase 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAP_EVENTS_FILE = PROJECT_ROOT / "gap_events_phase1_manual_study.csv"
JOURNAL_FILE = PROJECT_ROOT / "paper_trade_journal_phase2.csv"


def check_retail_panic_thesis(row):
    if row["gap_pct"] >= 0:
        return False, "Gap-up, not gap-down"
    if abs(row["gap_pct"]) < 0.5:
        return False, "Gap too small (< 0.5%)"
    return True, "Matches retail panic thesis"


def generate_paper_trading_signals(gap_events_df: pd.DataFrame) -> pd.DataFrame:
    signals = []
    for _, row in gap_events_df.iterrows():
        matches, _ = check_retail_panic_thesis(row)
        if not matches:
            continue
        pnl_pct = ((row["Close"] - row["Open"]) / row["Open"] * 100) if row["gap_pct"] < 0 else ((row["Open"] - row["Close"]) / row["Open"] * 100)
        signals.append(
            {
                "date": row["Date"],
                "symbol": row["symbol"],
                "gap_pct": row["gap_pct"],
                "direction": "LONG" if row["gap_pct"] < 0 else "SHORT",
                "participant_thesis_active": "retail_panic",
                "entry_price": row["Open"],
                "entry_time": "09:15:00",
                "stop_loss": row["Open"] * (1 - 0.015) if row["gap_pct"] < 0 else row["Open"] * (1 + 0.015),
                "target": row.get("prev_close", row["Close"]),
                "exit_price": row["Close"],
                "exit_time": "15:30:00",
                "exit_reason": "EOD",
                "modeled_cost": 0.001,
                "actual_cost": 0.001,
                "slippage_gap": 0.0,
                "pnl_rs": pnl_pct,
                "pnl_r_multiple": pnl_pct / 1.5,
                "was_thesis_correct": bool(row["faded"]),
                "notes": f"Gap {row['gap_pct']:.2f}%, Volume ratio: {row.get('volume_ratio', 'N/A')}, Sector: {row.get('sector', 'Unknown')}",
            }
        )
    return pd.DataFrame(signals)


def main() -> pd.DataFrame:
    if not GAP_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing gap events file: {GAP_EVENTS_FILE}")
    gap_events = pd.read_csv(GAP_EVENTS_FILE)
    signals_df = generate_paper_trading_signals(gap_events)
    signals_df.to_csv(JOURNAL_FILE, index=False)
    print(f"Generated {len(signals_df)} paper trading signals from {len(gap_events)} gap events")
    print(f"Saved {len(signals_df)} signals to {JOURNAL_FILE}")
    return signals_df


if __name__ == "__main__":
    main()

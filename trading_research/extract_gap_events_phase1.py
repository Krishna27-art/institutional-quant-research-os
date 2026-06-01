"""Extract gap events for Phase 1 manual study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "engines" / "data"
OUTPUT_FILE = PROJECT_ROOT / "gap_events_phase1_manual_study.csv"


def _load_stock_csv(symbol: str) -> pd.DataFrame:
    candidates = [
        DEFAULT_INPUT_DIR / f"{symbol}.csv",
        PROJECT_ROOT / "data" / f"{symbol}.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if "Date" not in frame.columns:
                raise ValueError(f"{path} is missing a Date column")
            return frame
    raise FileNotFoundError(f"No CSV found for {symbol} in {candidates}")


def _load_sector_map() -> pd.DataFrame:
    path = REFERENCE_DIR / "sector_map.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["symbol", "sector"])


def process_stock(df: pd.DataFrame, symbol: str, sector_map: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")
    open_col = "Open" if "Open" in frame.columns else "open"
    close_col = "Close" if "Close" in frame.columns else "close"
    volume_col = "Volume" if "Volume" in frame.columns else "volume"
    high_col = "High" if "High" in frame.columns else "high"
    low_col = "Low" if "Low" in frame.columns else "low"
    frame["prev_close"] = frame[close_col].shift(1)
    frame["gap_pct"] = ((frame[open_col] - frame["prev_close"]) / frame["prev_close"]) * 100.0
    frame["gap_type"] = np.where(frame["gap_pct"] < 0, "DOWN", np.where(frame["gap_pct"] > 0, "UP", "NONE"))
    frame["faded"] = np.where(
        (frame["gap_pct"] < 0) & (frame[close_col] > frame[open_col]),
        True,
        np.where((frame["gap_pct"] > 0) & (frame[close_col] < frame[open_col]), True, False),
    )
    frame["day_of_week"] = frame.index.day_name()
    frame["symbol"] = symbol
    symbol_key = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
    sector = sector_map[sector_map["symbol"] == symbol_key]["sector"] if not sector_map.empty else []
    frame["sector"] = sector.values[0] if len(sector) > 0 else "Unknown"
    frame["vol_avg_20"] = frame[volume_col].rolling(20, min_periods=5).median()
    frame["volume_ratio"] = frame[volume_col] / frame["vol_avg_20"]
    frame = frame.reset_index()
    frame.rename(
        columns={
            open_col: "Open",
            close_col: "Close",
            high_col: "High",
            low_col: "Low",
            volume_col: "Volume",
        },
        inplace=True,
    )
    return frame


def main() -> pd.DataFrame:
    sector_map = _load_sector_map()
    symbols = ["RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "SBIN", "WIPRO"]
    frames = [process_stock(_load_stock_csv(symbol), symbol, sector_map) for symbol in symbols]
    all_data = pd.concat(frames, ignore_index=True)
    gap_events = all_data[
        (all_data["gap_pct"].abs() >= 1.0) & (all_data["gap_pct"] != 0) & (all_data["prev_close"].notna())
    ].copy()
    gap_down = gap_events[gap_events["gap_pct"] < 0].sort_values("gap_pct").head(50)
    gap_up = gap_events[gap_events["gap_pct"] > 0].sort_values("gap_pct", ascending=False).head(50)
    manual_review = pd.concat([gap_down, gap_up], ignore_index=True)
    review_columns = [
        "Date",
        "symbol",
        "gap_pct",
        "gap_type",
        "volume_ratio",
        "faded",
        "sector",
        "day_of_week",
        "Open",
        "Close",
        "High",
        "Low",
        "prev_close",
    ]
    manual_review = manual_review[review_columns].copy()
    manual_review["Date"] = pd.to_datetime(manual_review["Date"]).dt.strftime("%Y-%m-%d")
    manual_review["volume_first_15min_vs_avg"] = "N/A (daily data only)"
    manual_review["approx_reversal_time"] = "N/A (daily data only)"
    manual_review["distance_from_expiry"] = "N/A (need expiry calendar)"
    manual_review["market_regime"] = "N/A (need NIFTY data)"
    final_columns = [
        "Date",
        "symbol",
        "gap_pct",
        "gap_type",
        "volume_ratio",
        "volume_first_15min_vs_avg",
        "faded",
        "approx_reversal_time",
        "sector",
        "day_of_week",
        "distance_from_expiry",
        "market_regime",
        "Open",
        "Close",
        "High",
        "Low",
        "prev_close",
    ]
    manual_review = manual_review[final_columns]
    manual_review.to_csv(OUTPUT_FILE, index=False)
    print(f"Extracted {len(gap_down)} gap-down events and {len(gap_up)} gap-up events")
    print(f"Total events for manual review: {len(manual_review)}")
    print(f"Saved to: {OUTPUT_FILE}")
    return manual_review


if __name__ == "__main__":
    main()

"""Data source abstraction for NIFTY research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


OHLCVColumns = ("open", "high", "low", "close", "volume")
STANDARD_COLUMNS = ("date", "open", "high", "low", "close", "volume", "adjusted_close")


def _to_naive_datetime(values: pd.Series | pd.Index) -> pd.Series | pd.DatetimeIndex:
    dt = pd.to_datetime(values, errors="coerce")
    if hasattr(dt, "dt"):
        tz = getattr(dt.dt, "tz", None)
        if tz is not None:
            dt = dt.dt.tz_convert(None)
    elif isinstance(dt, pd.DatetimeIndex) and dt.tz is not None:
        dt = dt.tz_convert(None)
    return dt


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {col: col.lower().strip().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        else:
            raise ValueError("OHLCV frame must contain a date column or DatetimeIndex")

    df["date"] = _to_naive_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]

    missing = [col for col in STANDARD_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    return df[list(STANDARD_COLUMNS)].reset_index(drop=True)


@dataclass(slots=True)
class DataSource:
    """Load clean OHLCV data from local files or a remote provider."""

    raw_data_dir: Path
    processed_data_dir: Path

    def symbol_path(self, symbol: str, processed: bool = False) -> Path:
        base = self.processed_data_dir if processed else self.raw_data_dir
        return base / f"{symbol.upper()}.csv"

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        return _normalize_frame(pd.read_csv(path))

    def load_symbol(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        prefer_processed: bool = True,
    ) -> pd.DataFrame:
        candidates = [
            self.symbol_path(symbol, processed=prefer_processed),
            self.symbol_path(symbol, processed=not prefer_processed),
        ]
        for candidate in candidates:
            if candidate.exists():
                frame = self.load_csv(candidate)
                return self._slice(frame, start=start, end=end)
        raise FileNotFoundError(f"No OHLCV file found for {symbol}")

    def load_many(self, symbols: Iterable[str], **kwargs) -> dict[str, pd.DataFrame]:
        return {symbol.upper(): self.load_symbol(symbol, **kwargs) for symbol in symbols}

    def save_symbol(self, symbol: str, frame: pd.DataFrame, processed: bool = False) -> Path:
        path = self.symbol_path(symbol, processed=processed)
        path.parent.mkdir(parents=True, exist_ok=True)
        _normalize_frame(frame).to_csv(path, index=False)
        return path

    def _slice(self, frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
        df = frame.copy()
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("yfinance is not installed in this environment") from exc

        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(start=start, end=end, auto_adjust=False, actions=False)
        if hist.empty:
            raise ValueError(f"No data returned for {symbol}")
        frame = hist.reset_index().rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adjusted_close",
                "Date": "date",
                "Datetime": "date",
            }
        )
        if "adjusted_close" not in frame.columns:
            frame["adjusted_close"] = frame["close"]
        return _normalize_frame(frame)

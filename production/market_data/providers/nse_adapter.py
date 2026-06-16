"""Optional NSE data adapter boundary inspired by nselib."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True, slots=True)
class NSERequest:
    segment: str
    symbol: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class NSELibAdapter:
    """Thin optional adapter so NSE data can be swapped without touching research code."""

    def __init__(self) -> None:
        try:
            import nselib  # type: ignore
        except Exception:
            nselib = None
        self._nselib = nselib

    @property
    def available(self) -> bool:
        return self._nselib is not None

    def require_available(self) -> None:
        if self._nselib is None:
            raise RuntimeError("nselib is not installed. Install it only when live NSE pulls are needed.")

    def normalize_bhavcopy(self, frame: pd.DataFrame) -> pd.DataFrame:
        rename = {col: col.lower().strip().replace(" ", "_") for col in frame.columns}
        df = frame.rename(columns=rename).copy()
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()
        return df

    def normalize_fii_dii(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_columns(frame)
        rename = {
            "fii_net_purchase_sales": "fii_net",
            "fii_net": "fii_net",
            "dii_net_purchase_sales": "dii_net",
            "dii_net": "dii_net",
        }
        df = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        required = {"date", "fii_net", "dii_net"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"FII/DII frame missing columns after normalization: {sorted(missing)}")
        return df[["date", "fii_net", "dii_net"]].sort_values("date").reset_index(drop=True)

    def normalize_vix(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_columns(frame)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        close_candidates = ["close", "india_vix", "vix", "current"]
        value_col = next((col for col in close_candidates if col in df.columns), None)
        if value_col is None:
            raise ValueError("VIX frame must contain one of: close, india_vix, vix, current")
        return df[["date", value_col]].rename(columns={value_col: "vix"}).sort_values("date").reset_index(drop=True)

    def normalize_delivery(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize_columns(frame)
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        rename = {
            "deliverable_qty": "deliverable_qty",
            "deliv_qty": "deliverable_qty",
            "total_traded_quantity": "traded_qty",
            "ttl_trd_qnty": "traded_qty",
            "delivery_percentage": "delivery_pct",
            "deliv_per": "delivery_pct",
        }
        df = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
        if "delivery_pct" not in df.columns and {"deliverable_qty", "traded_qty"}.issubset(df.columns):
            df["delivery_pct"] = df["deliverable_qty"] / df["traded_qty"].replace(0, pd.NA)
        required = {"date", "symbol", "delivery_pct"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Delivery frame missing columns after normalization: {sorted(missing)}")
        return df[["date", "symbol", "delivery_pct"]].sort_values(["date", "symbol"]).reset_index(drop=True)

    def fetch_with(self, fetcher: Callable[..., pd.DataFrame], *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Execute a provided nselib fetch function without coupling to its changing API."""

        self.require_available()
        result = fetcher(*args, **kwargs)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("nselib fetcher must return a pandas DataFrame")
        return result

    @staticmethod
    def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rename(columns={col: col.lower().strip().replace(" ", "_").replace("%", "pct") for col in frame.columns}).copy()


@dataclass(frozen=True, slots=True)
class NSEMarketDataset:
    """Container for market context datasets required by Nifty research."""

    fii_dii: pd.DataFrame | None = None
    vix: pd.DataFrame | None = None
    delivery: pd.DataFrame | None = None
    derivatives: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fii_dii_rows": 0 if self.fii_dii is None else len(self.fii_dii),
            "vix_rows": 0 if self.vix is None else len(self.vix),
            "delivery_rows": 0 if self.delivery is None else len(self.delivery),
            "derivatives_rows": 0 if self.derivatives is None else len(self.derivatives),
        }

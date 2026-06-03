"""Unified point-in-time feature pipeline with 25 core indicators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for the consolidated feature pipeline."""

    n_features: int = 25
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 60
    atr_period: int = 14
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    expected_session_minutes: int = 390


class FeaturePipeline:
    """Computes trailing-only features and avoids centered/global leakage."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.feature_names = [
            "relative_volume",
            "vwap_distance_pct",
            "atr_14",
            "atr_pct",
            "realized_volatility_5d",
            "realized_volatility_20d",
            "implied_volatility",
            "iv_percentile",
            "iv_rv_spread",
            "pcr",
            "fii_dii_flow",
            "order_flow_imbalance",
            "bid_ask_spread",
            "depth_imbalance",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_width",
            "momentum_5d",
            "momentum_20d",
            "gap_pct",
            "inside_outside_bar",
            "engulfing",
            "time_of_day",
        ]

    def compute_features(
        self,
        ohlcv: pd.DataFrame,
        options_data: Mapping[str, Any] | None = None,
        flow_data: Mapping[str, Any] | None = None,
        order_book: Mapping[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """Compute the current feature vector from trailing data only."""
        frame = self._normalize_frame(ohlcv)
        if frame.empty or len(frame) < max(self.config.medium_window, self.config.atr_period, self.config.rsi_period):
            return {name: 0.0 for name in self.feature_names}

        ts = timestamp or self._latest_timestamp(frame)
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        open_ = frame["open"].astype(float)
        volume = frame["volume"].astype(float)

        features: dict[str, float] = {}

        avg_volume = volume.tail(self.config.medium_window).mean()
        features["relative_volume"] = float(volume.iloc[-1] / avg_volume) if avg_volume > 0 else 0.0
        vwap = (close.tail(self.config.medium_window) * volume.tail(self.config.medium_window)).sum()
        vwap /= max(volume.tail(self.config.medium_window).sum(), 1.0)
        features["vwap_distance_pct"] = float((close.iloc[-1] - vwap) / vwap) if vwap > 0 else 0.0

        atr = self._atr(high, low, close, self.config.atr_period)
        features["atr_14"] = float(atr.iloc[-1])
        features["atr_pct"] = float(features["atr_14"] / close.iloc[-1]) if close.iloc[-1] else 0.0

        ret = close.pct_change()
        features["realized_volatility_5d"] = float(ret.tail(self.config.short_window).std() * np.sqrt(252)) if len(ret.dropna()) else 0.0
        features["realized_volatility_20d"] = float(ret.tail(self.config.medium_window).std() * np.sqrt(252)) if len(ret.dropna()) else 0.0

        iv = float((options_data or {}).get("iv", 0.0) or 0.0)
        iv_percentile = float((options_data or {}).get("iv_percentile", 0.0) or 0.0)
        features["implied_volatility"] = iv
        features["iv_percentile"] = iv_percentile
        features["iv_rv_spread"] = iv - features["realized_volatility_20d"]

        features["pcr"] = float((options_data or {}).get("pcr", 0.0) or 0.0)
        features["fii_dii_flow"] = float((flow_data or {}).get("fii_dii_flow", (flow_data or {}).get("net_flow", 0.0)) or 0.0)
        features["order_flow_imbalance"] = self._order_flow_imbalance(order_book)
        features["bid_ask_spread"] = self._bid_ask_spread(order_book, close.iloc[-1], high.iloc[-1], low.iloc[-1])
        features["depth_imbalance"] = self._depth_imbalance(order_book)

        features["rsi_14"] = self._rsi(close, self.config.rsi_period)
        macd, macd_signal, macd_hist = self._macd(close)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_hist"] = macd_hist

        bb_width = self._bollinger_width(close, self.config.bb_period, self.config.bb_std)
        features["bb_width"] = bb_width
        features["momentum_5d"] = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 and close.iloc[-6] else 0.0
        features["momentum_20d"] = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 and close.iloc[-21] else 0.0
        features["gap_pct"] = float(open_.iloc[-1] / close.iloc[-2] - 1.0) if len(close) >= 2 and close.iloc[-2] else 0.0
        features["inside_outside_bar"] = self._inside_outside_bar(open_, high, low, close)
        features["engulfing"] = self._engulfing(open_, high, low, close)
        features["time_of_day"] = self._time_of_day_fraction(ts)

        return self._clip_and_order(features)

    def validate_point_in_time(self, features: Mapping[str, float]) -> dict[str, bool]:
        """Heuristic validation for accidental leakage patterns."""
        suspicious = {"TEMP_CENTER", "NORM_GLOBAL", "GLOBAL", "CENTER"}
        return {name: not any(token in name.upper() for token in suspicious) for name in features}

    def _clip_and_order(self, features: Mapping[str, float]) -> dict[str, float]:
        ordered = {name: float(features.get(name, 0.0)) for name in self.feature_names}
        return ordered

    def _normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        renamed = frame.copy()
        rename_map = {col: str(col).strip().lower() for col in renamed.columns}
        renamed = renamed.rename(columns=rename_map)
        if not isinstance(renamed.index, pd.DatetimeIndex) and "timestamp" in renamed.columns:
            renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], errors="coerce")
            renamed = renamed.set_index("timestamp")
        return renamed.sort_index()

    def _latest_timestamp(self, frame: pd.DataFrame) -> datetime:
        return frame.index[-1].to_pydatetime() if isinstance(frame.index, pd.DatetimeIndex) else datetime.utcnow()

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean().fillna(method="bfill")

    def _rsi(self, close: pd.Series, period: int) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] not in (0, np.nan) else np.inf
        if not np.isfinite(rs):
            return 100.0 if gain.iloc[-1] > 0 else 0.0
        return float(100 - (100 / (1 + rs)))

    def _macd(self, close: pd.Series) -> tuple[float, float, float]:
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return float(macd.iloc[-1]), float(signal.iloc[-1]), float((macd - signal).iloc[-1])

    def _bollinger_width(self, close: pd.Series, period: int, std_mult: float) -> float:
        ma = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        return float((upper.iloc[-1] - lower.iloc[-1]) / ma.iloc[-1]) if ma.iloc[-1] else 0.0

    def _order_flow_imbalance(self, order_book: Mapping[str, Any] | None) -> float:
        if not order_book:
            return 0.0
        bid = float(order_book.get("bid_qty", order_book.get("bid_quantity", 0.0)) or 0.0)
        ask = float(order_book.get("ask_qty", order_book.get("ask_quantity", 0.0)) or 0.0)
        total = bid + ask
        return float((bid - ask) / total) if total else 0.0

    def _bid_ask_spread(self, order_book: Mapping[str, Any] | None, close: float, high: float, low: float) -> float:
        if order_book:
            bid = float(order_book.get("bid_price", close) or close)
            ask = float(order_book.get("ask_price", close) or close)
            if bid > 0 and ask > 0 and ask >= bid:
                return float((ask - bid) / close) if close else 0.0
        return float((high - low) / close) if close else 0.0

    def _depth_imbalance(self, order_book: Mapping[str, Any] | None) -> float:
        if not order_book:
            return 0.0
        bid_depth = float(order_book.get("bid_depth", order_book.get("bid_volume", 0.0)) or 0.0)
        ask_depth = float(order_book.get("ask_depth", order_book.get("ask_volume", 0.0)) or 0.0)
        total = bid_depth + ask_depth
        return float((bid_depth - ask_depth) / total) if total else 0.0

    def _inside_outside_bar(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        if len(close) < 2:
            return 0.0
        prev_range = high.iloc[-2] - low.iloc[-2]
        curr_range = high.iloc[-1] - low.iloc[-1]
        if high.iloc[-1] <= high.iloc[-2] and low.iloc[-1] >= low.iloc[-2]:
            return 1.0
        if high.iloc[-1] >= high.iloc[-2] and low.iloc[-1] <= low.iloc[-2]:
            return -1.0
        return float(curr_range / prev_range) if prev_range else 0.0

    def _engulfing(self, open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        if len(close) < 2:
            return 0.0
        prev_bear = close.iloc[-2] < open_.iloc[-2]
        prev_bull = close.iloc[-2] > open_.iloc[-2]
        curr_bull = close.iloc[-1] > open_.iloc[-1]
        curr_bear = close.iloc[-1] < open_.iloc[-1]
        if curr_bull and prev_bear and close.iloc[-1] >= open_.iloc[-2] and open_.iloc[-1] <= close.iloc[-2]:
            return 1.0
        if curr_bear and prev_bull and open_.iloc[-1] >= close.iloc[-2] and close.iloc[-1] <= open_.iloc[-2]:
            return -1.0
        return 0.0

    def _time_of_day_fraction(self, timestamp: datetime | None) -> float:
        if timestamp is None:
            return 0.0
        minutes = timestamp.hour * 60 + timestamp.minute
        return float(max(0.0, min(1.0, (minutes - 9 * 60 - 15) / max(self.config.expected_session_minutes, 1))))


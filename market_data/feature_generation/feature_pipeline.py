"""Unified point-in-time feature pipeline with 25 core indicators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from src.feature_store.compute import ResearchFeatures

# Import feature store
try:
    from features.feature_store import get_feature_store
    feature_store = get_feature_store()
except Exception:
    feature_store = None


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
    enable_research_features: bool = True


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
            "pct_above_20ema",
            "pct_above_50ema",
            "pct_above_200ema",
            "new_highs_20d",
            "new_lows_20d",
            "avg_correlation",
        ]
        self.research_features = ResearchFeatures()
        if self.config.enable_research_features:
            self.feature_names.extend(
                [
                    "fracdiff_close_d04",
                    "chaos_logistic_return",
                    "chaos_tent_return",
                    "hurst_60d",
                    "rough_vol_regime_60d",
                ]
            )
            
        # Cache market breadth and correlation data
        self.market_data_dir = Path(__file__).resolve().parents[2] / "market_data"
        self._breadth_df = None
        self._corr_df = None
        self._load_market_features()

    def _load_market_features(self) -> None:
        try:
            breadth_path = self.market_data_dir / "breadth_features.parquet"
            if breadth_path.exists():
                self._breadth_df = pd.read_parquet(breadth_path)
        except Exception:
            pass
            
        try:
            # Try to compute average correlation from nifty500 prices if available
            prices_path = self.market_data_dir / "nifty500.parquet"
            if prices_path.exists():
                prices = pd.read_parquet(prices_path)
                returns = prices.pct_change()
                from market_data.feature_generation.cross_sectional_features import CorrelationEngine
                engine = CorrelationEngine(window=20)
                self._corr_df = engine.compute_rolling_correlation(returns)
        except Exception:
            pass

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

        # Look up market breadth and correlation features
        pct_above_20 = 0.5
        pct_above_50 = 0.5
        pct_above_200 = 0.5
        new_highs = 0.0
        new_lows = 0.0
        avg_corr = 0.0
        
        if ts is not None:
            # Attempt to reload if they were not cached yet (e.g. download script just finished)
            if self._breadth_df is None or self._breadth_df.empty:
                self._load_market_features()
                
            if self._breadth_df is not None and not self._breadth_df.empty:
                idx = self._breadth_df.index
                valid_idx = idx[idx <= ts]
                if len(valid_idx) > 0:
                    row = self._breadth_df.loc[valid_idx[-1]]
                    pct_above_20 = float(row.get("pct_above_20ema", 0.5))
                    pct_above_50 = float(row.get("pct_above_50ema", 0.5))
                    pct_above_200 = float(row.get("pct_above_200ema", 0.5))
                    new_highs = float(row.get("new_highs_20d", 0.0))
                    new_lows = float(row.get("new_lows_20d", 0.0))
            
            if self._corr_df is not None and not self._corr_df.empty:
                idx = self._corr_df.index
                valid_idx = idx[idx <= ts]
                if len(valid_idx) > 0:
                    avg_corr = float(self._corr_df.loc[valid_idx[-1]])
                    
        features["pct_above_20ema"] = pct_above_20
        features["pct_above_50ema"] = pct_above_50
        features["pct_above_200ema"] = pct_above_200
        features["new_highs_20d"] = new_highs
        features["new_lows_20d"] = new_lows
        features["avg_correlation"] = avg_corr

        if self.config.enable_research_features:
            research_frame = frame[["close"]]
            for name in (
                "fracdiff_close_d04",
                "chaos_logistic_return",
                "chaos_tent_return",
                "hurst_60d",
                "rough_vol_regime_60d",
            ):
                series = self.research_features.compute(name, research_frame)
                features[name] = self._latest_finite(series)

        # Log features to feature store
        if feature_store:
            try:
                symbol = ohlcv.get('symbol', 'unknown') if isinstance(ohlcv, dict) else 'unknown'
                feature_store.log_features(
                    symbol=symbol,
                    features=features,
                    timestamp=ts
                )
            except Exception as e:
                pass  # Don't fail if feature store logging fails

        return self._clip_and_order(features)

    def validate_point_in_time(self, features: Mapping[str, float]) -> dict[str, bool]:
        """Heuristic validation for accidental leakage patterns."""
        suspicious = {"TEMP_CENTER", "NORM_GLOBAL", "GLOBAL", "CENTER"}
        return {name: not any(token in name.upper() for token in suspicious) for name in features}

    def _clip_and_order(self, features: Mapping[str, float]) -> dict[str, float]:
        ordered = {}
        for name in self.feature_names:
            value = float(features.get(name, 0.0) or 0.0)
            ordered[name] = value if np.isfinite(value) else 0.0
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
        return tr.rolling(period, min_periods=period).mean()

    def _rsi(self, close: pd.Series, period: int) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        latest_gain = float(gain.iloc[-1])
        latest_loss = float(loss.iloc[-1])
        if not np.isfinite(latest_gain) or not np.isfinite(latest_loss):
            return 0.0
        rs = latest_gain / latest_loss if latest_loss != 0 else np.inf
        if not np.isfinite(rs):
            return 100.0 if latest_gain > 0 else 0.0
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
        latest_ma = float(ma.iloc[-1])
        if not np.isfinite(latest_ma) or latest_ma == 0:
            return 0.0
        width = float((upper.iloc[-1] - lower.iloc[-1]) / latest_ma)
        return width if np.isfinite(width) else 0.0

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

    def _latest_finite(self, series: pd.Series) -> float:
        valid = series.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return 0.0
        value = float(valid.iloc[-1])
        return value if np.isfinite(value) else 0.0

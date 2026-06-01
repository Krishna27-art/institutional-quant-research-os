"""
VWAP Strategy for Indian Markets.
Supports both trend-following and mean-reversion modes.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VWAPSignal(Enum):
    TREND_LONG = "trend_long"
    TREND_SHORT = "trend_short"
    REVERSION_LONG = "reversion_long"
    REVERSION_SHORT = "reversion_short"
    NO_SIGNAL = "no_signal"


@dataclass
class VWAPPosition:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    vwap_at_entry: float
    vwap_distance: float
    entry_time: pd.Timestamp
    status: str = "open"


class VWAPStrategy:
    """
    VWAP-based strategy with trend and reversion modes.

    Trend Mode:
    - Enter long when price breaks above VWAP with momentum
    - Enter short when price breaks below VWAP with momentum
    - Trail stop at VWAP

    Reversion Mode:
    - Enter long when price is far below VWAP and shows reversal
    - Enter short when price is far above VWAP and shows reversal
    - Target at VWAP
    """

    def __init__(self, config: dict):
        self.config = config
        vwap_config = config.get("alpha", {}).get("vwap", {})

        self.distance_threshold = vwap_config.get("distance_threshold", 0.003)
        self.reversion_window = vwap_config.get("reversion_window", 20)
        self.trend_window = vwap_config.get("trend_window", 60)
        self.max_deviation = vwap_config.get("max_deviation", 0.015)

        self._active_positions: Dict[str, VWAPPosition] = {}

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP for a DataFrame."""
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_volume = df["Volume"].cumsum()
        cum_volume_price = (typical_price * df["Volume"]).cumsum()
        vwap = cum_volume_price / cum_volume.replace(0, np.nan)
        return vwap

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        vvg_label: str = "trending_moderate"
    ) -> Tuple[VWAPSignal, Optional[VWAPPosition]]:
        """
        Generate VWAP signal based on current market state.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame (at least 60 bars)
            vvg_label: Intraday regime label from VVG classifier

        Returns:
            (signal, position) tuple
        """
        if len(df) < self.trend_window:
            return VWAPSignal.NO_SIGNAL, None

        if symbol in self._active_positions:
            return self._manage_position(symbol, df.iloc[-1])

        # Calculate VWAP
        vwap = self.calculate_vwap(df)
        current_price = df["Close"].iloc[-1]
        current_vwap = vwap.iloc[-1]
        vwap_distance = (current_price - current_vwap) / current_vwap

        # Determine mode based on VVG label
        if vvg_label in ["trending_strong", "trending_moderate", "trending_weak"]:
            return self._generate_trend_signal(
                symbol, df, vwap, current_price, current_vwap, vwap_distance
            )
        else:
            return self._generate_reversion_signal(
                symbol, df, vwap, current_price, current_vwap, vwap_distance
            )

    def _generate_trend_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        vwap: pd.Series,
        current_price: float,
        current_vwap: float,
        vwap_distance: float
    ) -> Tuple[VWAPSignal, Optional[VWAPPosition]]:
        """Generate trend-following signal."""
        # Check for momentum confirmation
        recent_returns = df["Close"].pct_change(self.trend_window).iloc[-1]
        recent_vol = df["Volume"].tail(20).mean()
        current_vol = df["Volume"].iloc[-1]

        # Long breakout: price above VWAP + momentum + volume
        if (vwap_distance > self.distance_threshold and
            recent_returns > 0 and
            current_vol > recent_vol * 1.2):

            # Check if not already too extended
            if vwap_distance < self.max_deviation:
                position = self._create_trend_position(
                    symbol, "long", current_price, current_vwap, vwap_distance
                )
                self._active_positions[symbol] = position
                logger.info(
                    f"VWAP TREND LONG: {symbol} @ {current_price:.2f}, "
                    f"VWAP: {current_vwap:.2f}, Dist: {vwap_distance:.3f}"
                )
                return VWAPSignal.TREND_LONG, position

        # Short breakout: price below VWAP + momentum + volume
        elif (vwap_distance < -self.distance_threshold and
              recent_returns < 0 and
              current_vol > recent_vol * 1.2):

            if abs(vwap_distance) < self.max_deviation:
                position = self._create_trend_position(
                    symbol, "short", current_price, current_vwap, vwap_distance
                )
                self._active_positions[symbol] = position
                logger.info(
                    f"VWAP TREND SHORT: {symbol} @ {current_price:.2f}, "
                    f"VWAP: {current_vwap:.2f}, Dist: {vwap_distance:.3f}"
                )
                return VWAPSignal.TREND_SHORT, position

        return VWAPSignal.NO_SIGNAL, None

    def _generate_reversion_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        vwap: pd.Series,
        current_price: float,
        current_vwap: float,
        vwap_distance: float
    ) -> Tuple[VWAPSignal, Optional[VWAPPosition]]:
        """Generate mean-reversion signal."""
        # Check for extreme deviation
        if abs(vwap_distance) < self.max_deviation:
            return VWAPSignal.NO_SIGNAL, None

        # Look for reversal pattern (recent bars showing reversal)
        recent_bars = df.tail(5)
        if len(recent_bars) < 5:
            return VWAPSignal.NO_SIGNAL, None

        # Long reversion: price far below VWAP, showing reversal
        if vwap_distance < -self.max_deviation:
            # Check for bullish reversal (higher lows, higher highs)
            if (recent_bars["Close"].iloc[-1] > recent_bars["Close"].iloc[-2] and
                recent_bars["Low"].iloc[-1] > recent_bars["Low"].iloc[-2]):

                position = self._create_reversion_position(
                    symbol, "long", current_price, current_vwap, vwap_distance
                )
                self._active_positions[symbol] = position
                logger.info(
                    f"VWAP REVERSION LONG: {symbol} @ {current_price:.2f}, "
                    f"VWAP: {current_vwap:.2f}, Dist: {vwap_distance:.3f}"
                )
                return VWAPSignal.REVERSION_LONG, position

        # Short reversion: price far above VWAP, showing reversal
        elif vwap_distance > self.max_deviation:
            # Check for bearish reversal (lower highs, lower lows)
            if (recent_bars["Close"].iloc[-1] < recent_bars["Close"].iloc[-2] and
                recent_bars["High"].iloc[-1] < recent_bars["High"].iloc[-2]):

                position = self._create_reversion_position(
                    symbol, "short", current_price, current_vwap, vwap_distance
                )
                self._active_positions[symbol] = position
                logger.info(
                    f"VWAP REVERSION SHORT: {symbol} @ {current_price:.2f}, "
                    f"VWAP: {current_vwap:.2f}, Dist: {vwap_distance:.3f}"
                )
                return VWAPSignal.REVERSION_SHORT, position

        return VWAPSignal.NO_SIGNAL, None

    def _create_trend_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        vwap: float,
        vwap_distance: float
    ) -> VWAPPosition:
        """Create trend-following position."""
        if direction == "long":
            stop_loss = vwap * 0.995  # 0.5% below VWAP
            target_price = entry_price * 1.02  # 2% target
        else:  # short
            stop_loss = vwap * 1.005  # 0.5% above VWAP
            target_price = entry_price * 0.98  # 2% target

        return VWAPPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            vwap_at_entry=vwap,
            vwap_distance=vwap_distance,
            entry_time=pd.Timestamp.now(),
        )

    def _create_reversion_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        vwap: float,
        vwap_distance: float
    ) -> VWAPPosition:
        """Create mean-reversion position."""
        if direction == "long":
            stop_loss = entry_price * 0.99  # 1% below entry
            target_price = vwap  # Target at VWAP
        else:  # short
            stop_loss = entry_price * 1.01  # 1% above entry
            target_price = vwap  # Target at VWAP

        return VWAPPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            vwap_at_entry=vwap,
            vwap_distance=vwap_distance,
            entry_time=pd.Timestamp.now(),
        )

    def _manage_position(
        self,
        symbol: str,
        current_bar: pd.Series
    ) -> Tuple[VWAPSignal, Optional[VWAPPosition]]:
        """Manage existing position."""
        position = self._active_positions[symbol]
        current_price = current_bar["Close"]
        current_low = current_bar["Low"]
        current_high = current_bar["High"]

        # Check stop loss
        if position.direction == "long":
            if current_low <= position.stop_loss:
                position.status = "stopped"
                del self._active_positions[symbol]
                logger.info(f"VWAP STOP LOSS: {symbol} @ {current_price:.2f}")
                return VWAPSignal.NO_SIGNAL, position

            # Check target
            if current_high >= position.target_price:
                position.status = "target_hit"
                del self._active_positions[symbol]
                logger.info(f"VWAP TARGET HIT: {symbol} @ {current_price:.2f}")
                return VWAPSignal.NO_SIGNAL, position

        else:  # short
            if current_high >= position.stop_loss:
                position.status = "stopped"
                del self._active_positions[symbol]
                logger.info(f"VWAP STOP LOSS: {symbol} @ {current_price:.2f}")
                return VWAPSignal.NO_SIGNAL, position

            # Check target
            if current_low <= position.target_price:
                position.status = "target_hit"
                del self._active_positions[symbol]
                logger.info(f"VWAP TARGET HIT: {symbol} @ {current_price:.2f}")
                return VWAPSignal.NO_SIGNAL, position

        return VWAPSignal.NO_SIGNAL, position

    def force_close_all(self) -> None:
        """Force close all positions."""
        for symbol, position in list(self._active_positions.items()):
            position.status = "forced_close"
            logger.info(f"VWAP FORCE CLOSE: {symbol} {position.direction}")

        self._active_positions.clear()

    def get_active_positions(self) -> Dict[str, VWAPPosition]:
        """Get all active positions."""
        return self._active_positions.copy()

    def reset(self) -> None:
        """Reset strategy state."""
        self._active_positions.clear()

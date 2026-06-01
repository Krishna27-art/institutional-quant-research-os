"""
Opening Range Breakout (ORB) Strategy for Indian Markets.
Adapted from "A Profitable Day Trading Strategy" paper.

Key Insight: Stocks with abnormal opening volume (relative volume > 2.5x)
that break their 5-minute opening range have persistent trends.

Execution: 9:15-9:20 AM IST opening range, breakout entry after 9:20 AM.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ORBSignal(Enum):
    LONG_BREAKOUT = "long_breakout"
    SHORT_BREAKOUT = "short_breakout"
    NO_SIGNAL = "no_signal"
    FAILED_BREAKOUT = "failed_breakout"


@dataclass
class ORBPosition:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    target_price: float
    opening_range_high: float
    opening_range_low: float
    opening_range_mid: float
    relative_volume: float
    entry_time: pd.Timestamp
    status: str = "open"


class ORBStrategy:
    """
    Opening Range Breakout Strategy.

    Rules:
    1. Identify top N stocks by relative volume at 9:20 AM
    2. Define opening range as [9:15 - 9:20] high/low
    3. Enter long on break above range high with volume confirmation
    4. Enter short on break below range low with volume confirmation
    5. Stop loss at 1.5x ATR, target at 2x range size
    6. Exit at 3:15 PM if target not hit
    """

    def __init__(self, config: dict):
        self.config = config
        orb_config = config.get("alpha", {}).get("orb", {})

        self.range_minutes = orb_config.get("range_minutes", 5)
        self.top_n_stocks = orb_config.get("top_n_stocks", 20)
        self.rel_volume_threshold = orb_config.get("rel_volume_threshold", 2.5)
        self.atr_period = orb_config.get("atr_period", 14)
        self.stop_atr_multiplier = orb_config.get("stop_atr_multiplier", 1.5)
        self.max_positions = orb_config.get("max_positions", 5)

        self._opening_ranges: Dict[str, Dict] = {}
        self._active_positions: Dict[str, ORBPosition] = {}

    def scan_opening_range(
        self,
        intraday_data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Scan opening range (9:15-9:20) and identify candidates.

        Args:
            intraday_data: {symbol: DataFrame with 1-minute bars}

        Returns:
            List of candidate symbols that meet volume criteria
        """
        candidates = []

        for symbol, df in intraday_data.items():
            if len(df) < self.range_minutes:
                continue

            # Get opening range data
            opening_bars = df.head(self.range_minutes)

            # Calculate opening range metrics
            or_high = opening_bars["High"].max()
            or_low = opening_bars["Low"].min()
            or_mid = (or_high + or_low) / 2
            or_volume = opening_bars["Volume"].sum()

            # Calculate relative volume (vs 20-day average)
            if len(df) >= 20:
                avg_volume = df["Volume"].head(20).mean()
                rel_volume = or_volume / avg_volume if avg_volume > 0 else 0
            else:
                rel_volume = 1.0

            # Calculate ATR for stop loss
            if len(df) >= self.atr_period:
                high_low = df["High"] - df["Low"]
                high_close = np.abs(df["High"] - df["Close"].shift())
                low_close = np.abs(df["Low"] - df["Close"].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(self.atr_period).mean().iloc[-1]
            else:
                atr = (or_high - or_low)  # Fallback

            # Store opening range data
            self._opening_ranges[symbol] = {
                "high": or_high,
                "low": or_low,
                "mid": or_mid,
                "volume": or_volume,
                "relative_volume": rel_volume,
                "atr": atr,
                "range_size": or_high - or_low,
            }

            # Filter by relative volume
            if rel_volume >= self.rel_volume_threshold:
                candidates.append(symbol)

        # Sort by relative volume and take top N
        candidates.sort(
            key=lambda s: self._opening_ranges[s]["relative_volume"],
            reverse=True
        )

        return candidates[:self.top_n_stocks]

    def generate_signal(
        self,
        symbol: str,
        current_bar: pd.Series,
        position_multiplier: float = 1.0
    ) -> Tuple[ORBSignal, Optional[ORBPosition]]:
        """
        Generate ORB signal based on current price action.

        Args:
            symbol: Stock symbol
            current_bar: Current OHLCV bar
            position_multiplier: Regime-based position size multiplier

        Returns:
            (signal, position) tuple
        """
        if symbol not in self._opening_ranges:
            return ORBSignal.NO_SIGNAL, None

        if symbol in self._active_positions:
            # Manage existing position
            return self._manage_position(symbol, current_bar)

        or_data = self._opening_ranges[symbol]
        current_price = current_bar["Close"]
        current_time = current_bar.name if hasattr(current_bar, 'name') else pd.Timestamp.now()

        # Check for long breakout
        if current_price > or_data["high"]:
            # Volume confirmation (current volume > opening range average)
            if current_bar["Volume"] > or_data["volume"] / self.range_minutes:
                position = self._create_position(
                    symbol, "long", current_price, or_data, current_time, position_multiplier
                )
                self._active_positions[symbol] = position
                logger.info(
                    f"ORB LONG BREAKOUT: {symbol} @ {current_price:.2f}, "
                    f"OR: {or_data['low']:.2f}-{or_data['high']:.2f}"
                )
                return ORBSignal.LONG_BREAKOUT, position

        # Check for short breakout
        elif current_price < or_data["low"]:
            if current_bar["Volume"] > or_data["volume"] / self.range_minutes:
                position = self._create_position(
                    symbol, "short", current_price, or_data, current_time, position_multiplier
                )
                self._active_positions[symbol] = position
                logger.info(
                    f"ORB SHORT BREAKOUT: {symbol} @ {current_price:.2f}, "
                    f"OR: {or_data['low']:.2f}-{or_data['high']:.2f}"
                )
                return ORBSignal.SHORT_BREAKOUT, position

        return ORBSignal.NO_SIGNAL, None

    def _create_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        or_data: Dict,
        entry_time: pd.Timestamp,
        position_multiplier: float
    ) -> ORBPosition:
        """Create ORB position with stop loss and target."""
        atr = or_data["atr"]
        range_size = or_data["range_size"]

        if direction == "long":
            stop_loss = entry_price - (atr * self.stop_atr_multiplier)
            target_price = entry_price + (range_size * 2)
        else:  # short
            stop_loss = entry_price + (atr * self.stop_atr_multiplier)
            target_price = entry_price - (range_size * 2)

        return ORBPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            opening_range_high=or_data["high"],
            opening_range_low=or_data["low"],
            opening_range_mid=or_data["mid"],
            relative_volume=or_data["relative_volume"],
            entry_time=entry_time,
        )

    def _manage_position(
        self,
        symbol: str,
        current_bar: pd.Series
    ) -> Tuple[ORBSignal, Optional[ORBPosition]]:
        """Manage existing position (check stop loss, target, or failed breakout)."""
        position = self._active_positions[symbol]
        current_price = current_bar["Close"]
        current_low = current_bar["Low"]
        current_high = current_bar["High"]

        # Check stop loss
        if position.direction == "long":
            if current_low <= position.stop_loss:
                position.status = "stopped"
                del self._active_positions[symbol]
                logger.info(f"ORB STOP LOSS: {symbol} @ {current_price:.2f}")
                return ORBSignal.FAILED_BREAKOUT, position

            # Check target
            if current_high >= position.target_price:
                position.status = "target_hit"
                del self._active_positions[symbol]
                logger.info(f"ORB TARGET HIT: {symbol} @ {current_price:.2f}")
                return ORBSignal.FAILED_BREAKOUT, position

            # Check failed breakout (price returns below opening range mid)
            if current_price < position.opening_range_mid:
                position.status = "failed_breakout"
                del self._active_positions[symbol]
                logger.info(f"ORB FAILED: {symbol} returned to OR mid")
                return ORBSignal.FAILED_BREAKOUT, position

        else:  # short
            if current_high >= position.stop_loss:
                position.status = "stopped"
                del self._active_positions[symbol]
                logger.info(f"ORB STOP LOSS: {symbol} @ {current_price:.2f}")
                return ORBSignal.FAILED_BREAKOUT, position

            # Check target
            if current_low <= position.target_price:
                position.status = "target_hit"
                del self._active_positions[symbol]
                logger.info(f"ORB TARGET HIT: {symbol} @ {current_price:.2f}")
                return ORBSignal.FAILED_BREAKOUT, position

            # Check failed breakout (price returns above opening range mid)
            if current_price > position.opening_range_mid:
                position.status = "failed_breakout"
                del self._active_positions[symbol]
                logger.info(f"ORB FAILED: {symbol} returned to OR mid")
                return ORBSignal.FAILED_BREAKOUT, position

        return ORBSignal.NO_SIGNAL, position

    def force_close_all(self, intraday_data: Dict[str, pd.DataFrame] = None) -> None:
        """Force close all positions (EOD or emergency)."""
        for symbol, position in list(self._active_positions.items()):
            if intraday_data and symbol in intraday_data:
                current_price = intraday_data[symbol]["Close"].iloc[-1]
            else:
                current_price = position.entry_price

            position.status = "forced_close"
            logger.info(
                f"ORB FORCE CLOSE: {symbol} {position.direction} "
                f"@ {current_price:.2f}"
            )

        self._active_positions.clear()

    def get_active_positions(self) -> Dict[str, ORBPosition]:
        """Get all active positions."""
        return self._active_positions.copy()

    def reset(self) -> None:
        """Reset strategy state (new trading day)."""
        self._opening_ranges.clear()
        self._active_positions.clear()

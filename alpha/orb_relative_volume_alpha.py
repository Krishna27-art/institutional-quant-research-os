"""
Opening Range Breakout (ORB) with Relative Volume Filter Alpha Strategy

This module implements the Opening Range Breakout strategy with relative volume
filtering, which identifies strong intraday momentum based on opening range
breakouts when relative volume exceeds threshold.

Based on Zarattini et al. 2024 (in-depth 7000-stock study).
Expected Sharpe: 0.6-1.0
Expected Capacity: Medium
Decay: Years
Difficulty: Low

Priority: High (Research OS Phase 2)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ORBDirection(Enum):
    """ORB breakout direction."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class OpeningRange:
    """Opening range data."""
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    range_high: float
    range_low: float
    range_size: float
    relative_volume: float


@dataclass
class ORBSignal:
    """ORB trading signal."""
    timestamp: datetime
    symbol: str
    direction: ORBDirection
    breakout_price: float
    entry_price: float
    stop_loss: float
    target_price: float
    relative_volume: float
    signal_strength: float  # 0-1
    confidence: float


class ORBRelativeVolumeAlpha:
    """
    Opening Range Breakout with relative volume filter alpha strategy.
    
    This class implements ORB strategy that trades breakouts from the
    opening range when relative volume exceeds threshold, indicating
    strong conviction in the move.
    """
    
    def __init__(
        self,
        opening_range_minutes: int = 5,
        relative_volume_threshold: float = 1.0,  # 100% of average
        lookback_days: int = 14,
        atr_multiplier: float = 2.0,
        risk_reward_ratio: float = 2.0,
        max_position_size: float = 0.02  # 2% of portfolio
    ):
        """
        Initialize ORB alpha.
        
        Args:
            opening_range_minutes: Duration of opening range in minutes
            relative_volume_threshold: Minimum relative volume to trade
            lookback_days: Days for average volume calculation
            atr_multiplier: ATR multiplier for stop loss
            risk_reward_ratio: Risk-reward ratio for target
            max_position_size: Maximum position size as portfolio fraction
        """
        self.opening_range_minutes = opening_range_minutes
        self.relative_volume_threshold = relative_volume_threshold
        self.lookback_days = lookback_days
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.max_position_size = max_position_size
        
        self.opening_ranges: Dict[str, List[OpeningRange]] = {}
        self.signals: List[ORBSignal] = []
        self.avg_volumes: Dict[str, float] = {}
        
        logger.info(f"ORBRelativeVolumeAlpha initialized: range={opening_range_minutes}min, "
                   f"rv_threshold={relative_volume_threshold}, lookback={lookback_days}days")
    
    def calculate_average_volume(
        self,
        symbol: str,
        historical_data: pd.DataFrame
    ) -> float:
        """
        Calculate average daily volume over lookback period.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            
        Returns:
            Average volume
        """
        if len(historical_data) < self.lookback_days:
            logger.warning(f"Insufficient data for {symbol}: {len(historical_data)} < {self.lookback_days}")
            return historical_data['volume'].mean() if len(historical_data) > 0 else 0.0
        
        recent_data = historical_data.tail(self.lookback_days)
        avg_volume = recent_data['volume'].mean()
        
        self.avg_volumes[symbol] = avg_volume
        return avg_volume
    
    def calculate_opening_range(
        self,
        symbol: str,
        intraday_data: pd.DataFrame,
        date: datetime
    ) -> Optional[OpeningRange]:
        """
        Calculate opening range for a given day.
        
        Args:
            symbol: Stock symbol
            intraday_data: Intraday OHLCV data (1-minute bars)
            date: Trading date
            
        Returns:
            OpeningRange or None
        """
        # Filter data for the opening range period
        market_open = date.replace(hour=9, minute=15, second=0, microsecond=0)
        range_end = market_open + timedelta(minutes=self.opening_range_minutes)
        
        range_data = intraday_data[
            (intraday_data['timestamp'] >= market_open) &
            (intraday_data['timestamp'] < range_end)
        ]
        
        if len(range_data) == 0:
            return None
        
        # Calculate opening range
        open_price = range_data.iloc[0]['open']
        high_price = range_data['high'].max()
        low_price = range_data['low'].min()
        close_price = range_data.iloc[-1]['close']
        volume = range_data['volume'].sum()
        
        range_high = high_price
        range_low = low_price
        range_size = range_high - range_low
        
        # Calculate relative volume
        avg_volume = self.avg_volumes.get(symbol, volume)
        relative_volume = volume / avg_volume if avg_volume > 0 else 1.0
        
        opening_range = OpeningRange(
            date=date,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            range_high=range_high,
            range_low=range_low,
            range_size=range_size,
            relative_volume=relative_volume
        )
        
        # Store opening range
        if symbol not in self.opening_ranges:
            self.opening_ranges[symbol] = []
        self.opening_ranges[symbol].append(opening_range)
        
        return opening_range
    
    def calculate_atr(
        self,
        data: pd.DataFrame,
        period: int = 14
    ) -> float:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period + 1:
            return 0.0
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def detect_breakout(
        self,
        symbol: str,
        intraday_data: pd.DataFrame,
        opening_range: OpeningRange
    ) -> Optional[ORBSignal]:
        """
        Detect breakout from opening range.
        
        Args:
            symbol: Stock symbol
            intraday_data: Intraday OHLCV data
            opening_range: Opening range data
            
        Returns:
            ORBSignal or None
        """
        # Check relative volume threshold
        if opening_range.relative_volume < self.relative_volume_threshold:
            return None
        
        # Get data after opening range
        range_end = opening_range.date.replace(hour=9, minute=15, second=0, microsecond=0) + \
                    timedelta(minutes=self.opening_range_minutes)
        
        post_range_data = intraday_data[intraday_data['timestamp'] >= range_end]
        
        if len(post_range_data) == 0:
            return None
        
        # Check for breakout
        latest_bar = post_range_data.iloc[-1]
        current_price = latest_bar['close']
        
        direction = ORBDirection.NEUTRAL
        breakout_price = 0.0
        
        if current_price > opening_range.range_high:
            direction = ORBDirection.UP
            breakout_price = opening_range.range_high
        elif current_price < opening_range.range_low:
            direction = ORBDirection.DOWN
            breakout_price = opening_range.range_low
        else:
            return None  # No breakout yet
        
        # Calculate ATR for stop loss
        atr = self.calculate_atr(intraday_data)
        
        # Calculate entry, stop loss, and target
        entry_price = current_price
        
        if direction == ORBDirection.UP:
            stop_loss = entry_price - (atr * self.atr_multiplier)
            target_price = entry_price + (atr * self.atr_multiplier * self.risk_reward_ratio)
        else:
            stop_loss = entry_price + (atr * self.atr_multiplier)
            target_price = entry_price - (atr * self.atr_multiplier * self.risk_reward_ratio)
        
        # Calculate signal strength based on relative volume and range size
        signal_strength = min(opening_range.relative_volume / 2.0, 1.0)  # Cap at 1.0
        range_adjustment = min(opening_range.range_size / entry_price * 100, 0.5)  # Up to 0.5
        signal_strength = min(signal_strength + range_adjustment, 1.0)
        
        # Confidence based on relative volume
        confidence = min(opening_range.relative_volume / 1.5, 0.9)
        
        signal = ORBSignal(
            timestamp=latest_bar['timestamp'],
            symbol=symbol,
            direction=direction,
            breakout_price=breakout_price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            relative_volume=opening_range.relative_volume,
            signal_strength=signal_strength,
            confidence=confidence
        )
        
        self.signals.append(signal)
        
        return signal
    
    def process_intraday_data(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        intraday_data: pd.DataFrame,
        date: datetime
    ) -> Optional[ORBSignal]:
        """
        Process intraday data and generate ORB signal.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            intraday_data: Intraday OHLCV data (1-minute bars)
            date: Trading date
            
        Returns:
            ORBSignal or None
        """
        # Calculate average volume
        self.calculate_average_volume(symbol, historical_data)
        
        # Calculate opening range
        opening_range = self.calculate_opening_range(symbol, intraday_data, date)
        
        if opening_range is None:
            return None
        
        # Detect breakout
        signal = self.detect_breakout(symbol, intraday_data, opening_range)
        
        return signal
    
    def get_latest_signal(self, symbol: str) -> Optional[ORBSignal]:
        """
        Get the latest signal for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest ORBSignal or None
        """
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_top_signals(
        self,
        n: int = 10,
        min_relative_volume: float = 1.0
    ) -> List[ORBSignal]:
        """
        Get top N signals by signal strength.
        
        Args:
            n: Number of signals to return
            min_relative_volume: Minimum relative volume threshold
            
        Returns:
            List of top ORBSignal
        """
        filtered = [s for s in self.signals if s.relative_volume >= min_relative_volume]
        sorted_signals = sorted(filtered, key=lambda x: x.signal_strength, reverse=True)
        return sorted_signals[:n]
    
    def print_orb_report(self) -> None:
        """Print ORB analysis report."""
        print("\n" + "="*60)
        print("ORB RELATIVE VOLUME ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Opening Range: {self.opening_range_minutes} minutes")
        print(f"  Relative Volume Threshold: {self.relative_volume_threshold}x")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  ATR Multiplier: {self.atr_multiplier}x")
        print(f"  Risk-Reward Ratio: {self.risk_reward_ratio}:1")
        
        print(f"\nStatistics:")
        print(f"  Total Opening Ranges: {sum(len(ranges) for ranges in self.opening_ranges.values())}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.signals:
            up_signals = [s for s in self.signals if s.direction == ORBDirection.UP]
            down_signals = [s for s in self.signals if s.direction == ORBDirection.DOWN]
            
            print(f"\nSignal Distribution:")
            print(f"  Up Breakouts: {len(up_signals)}")
            print(f"  Down Breakouts: {len(down_signals)}")
            
            if self.signals:
                avg_rv = np.mean([s.relative_volume for s in self.signals])
                avg_strength = np.mean([s.signal_strength for s in self.signals])
                
                print(f"\nSignal Quality:")
                print(f"  Average Relative Volume: {avg_rv:.2f}x")
                print(f"  Average Signal Strength: {avg_strength:.2f}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Direction':<10} {'RV':<8} {'Strength':<10} {'Entry':<12} {'Stop':<12} {'Target':<12}")
            print("-" * 100)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.direction.value:<10} {signal.relative_volume:>7.2f}x "
                      f"{signal.signal_strength:<10.2f} {signal.entry_price:<12.2f} "
                      f"{signal.stop_loss:<12.2f} {signal.target_price:<12.2f}")
        
        print("\n" + "="*60)


def sample_orb_relative_volume_alpha():
    """Demonstrate ORB relative volume alpha."""
    print("=== ORB Relative Volume Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = ORBRelativeVolumeAlpha(
        opening_range_minutes=5,
        relative_volume_threshold=1.0,
        lookback_days=14,
        atr_multiplier=2.0,
        risk_reward_ratio=2.0
    )
    
    # Generate sample historical data
    np.random.seed(42)
    n_days = 30
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    base_price = 1000.0
    
    historical_data = pd.DataFrame({
        'date': dates,
        'open': base_price + np.random.randn(n_days) * 10,
        'high': base_price + np.random.randn(n_days) * 10 + 5,
        'low': base_price + np.random.randn(n_days) * 10 - 5,
        'close': base_price + np.random.randn(n_days) * 10,
        'volume': np.random.randint(1000000, 5000000, n_days)
    })
    
    # Generate sample intraday data for today
    n_minutes = 390
    intraday_dates = pd.date_range(
        start=datetime.now().replace(hour=9, minute=15, second=0),
        periods=n_minutes,
        freq='1min'
    )
    
    intraday_prices = base_price + np.random.randn(n_minutes).cumsum() * 0.5
    intraday_volumes = np.random.randint(10000, 50000, n_minutes)
    
    intraday_data = pd.DataFrame({
        'timestamp': intraday_dates,
        'open': intraday_prices,
        'high': intraday_prices + np.random.rand(n_minutes) * 2,
        'low': intraday_prices - np.random.rand(n_minutes) * 2,
        'close': intraday_prices,
        'volume': intraday_volumes
    })
    
    # Process data
    print("Processing intraday data...")
    signal = alpha.process_intraday_data(
        'RELIANCE',
        historical_data,
        intraday_data,
        datetime.now()
    )
    
    if signal:
        print(f"\nSignal Generated:")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Entry: {signal.entry_price:.2f}")
        print(f"  Stop Loss: {signal.stop_loss:.2f}")
        print(f"  Target: {signal.target_price:.2f}")
        print(f"  Relative Volume: {signal.relative_volume:.2f}x")
        print(f"  Signal Strength: {signal.signal_strength:.2f}")
    else:
        print("\nNo signal generated (no breakout or relative volume below threshold)")
    
    # Print report
    alpha.print_orb_report()
    
    print("\n=== ORB Relative Volume Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Opening range calculation (5-minute default)")
    print("- Relative volume filtering (100% threshold)")
    print("- Breakout detection (up/down)")
    print("- ATR-based stop loss and target calculation")
    print("- Signal strength scoring")
    print("- Expected Sharpe: 0.6-1.0")
    print("- Expected Capacity: Medium")
    print("- Decay: Years")


if __name__ == "__main__":
    sample_orb_relative_volume_alpha()

"""
VPIN Order Flow Toxicity Alpha Strategy

This module implements the Volume-Synchronized Probability of Informed Trading (VPIN)
as an alpha strategy for detecting order flow toxicity and predicting near-term
volatility and adverse selection risk.

Based on Easley et al. (2012) and recent crypto validation (2025).
Expected Sharpe: 0.6-1.0
Expected Capacity: Medium-High (index futures)
Decay: Moderate (years)
Difficulty: Medium

Priority: High (Research OS Phase 1)
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


class TradeDirection(Enum):
    """Trade direction classification."""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass
class VPINBucket:
    """Volume bucket for VPIN calculation."""
    bucket_id: int
    start_time: datetime
    end_time: datetime
    total_volume: float
    buy_volume: float
    sell_volume: float
    vpin: float
    toxicity_score: float
    is_toxic: bool


@dataclass
class VPINSignal:
    """VPIN trading signal."""
    timestamp: datetime
    symbol: str
    vpin: float
    toxicity_score: float
    signal: float  # -1 to 1, negative = reduce exposure, positive = increase
    confidence: float
    regime: str  # normal, elevated, toxic


class VPINToxicityAlpha:
    """
    VPIN-based order flow toxicity alpha strategy.
    
    This class implements the VPIN methodology to detect order flow toxicity
    and generate trading signals based on informed trading probability.
    """
    
    def __init__(
        self,
        bucket_volume: float = 1000000,  # Volume per bucket (1M shares)
        num_buckets: int = 50,  # Number of buckets for VPIN calculation
        vpin_threshold_normal: float = 0.3,
        vpin_threshold_elevated: float = 0.5,
        vpin_threshold_toxic: float = 0.7,
        lookback_periods: int = 5
    ):
        """
        Initialize VPIN toxicity alpha.
        
        Args:
            bucket_volume: Volume threshold for each bucket
            num_buckets: Number of buckets for VPIN calculation
            vpin_threshold_normal: VPIN threshold for normal regime
            vpin_threshold_elevated: VPIN threshold for elevated regime
            vpin_threshold_toxic: VPIN threshold for toxic regime
            lookback_periods: Number of lookback periods for signal smoothing
        """
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets
        self.vpin_threshold_normal = vpin_threshold_normal
        self.vpin_threshold_elevated = vpin_threshold_elevated
        self.vpin_threshold_toxic = vpin_threshold_toxic
        self.lookback_periods = lookback_periods
        
        self.buckets: List[VPINBucket] = []
        self.vpin_history: List[float] = []
        self.signals: List[VPINSignal] = []
        
        logger.info(f"VPINToxicityAlpha initialized: bucket_volume={bucket_volume}, num_buckets={num_buckets}")
    
    def classify_trade_direction(
        self,
        price: float,
        prev_price: float,
        volume: float
    ) -> TradeDirection:
        """
        Classify trade direction using tick rule.
        
        Args:
            price: Current trade price
            prev_price: Previous trade price
            volume: Trade volume
            
        Returns:
            TradeDirection
        """
        if price > prev_price:
            return TradeDirection.BUY
        elif price < prev_price:
            return TradeDirection.SELL
        else:
            return TradeDirection.NEUTRAL
    
    def create_volume_buckets(
        self,
        trades: pd.DataFrame
    ) -> List[VPINBucket]:
        """
        Create volume-synchronized buckets from trade data.
        
        Args:
            trades: DataFrame with columns: timestamp, price, volume
            
        Returns:
            List of VPINBucket
        """
        buckets = []
        current_bucket_volume = 0.0
        current_buy_volume = 0.0
        current_sell_volume = 0.0
        bucket_start = None
        bucket_id = 0
        
        prev_price = trades['price'].iloc[0] if len(trades) > 0 else 0.0
        
        for idx, row in trades.iterrows():
            timestamp = row['timestamp']
            price = row['price']
            volume = row['volume']
            
            if bucket_start is None:
                bucket_start = timestamp
            
            # Classify trade direction
            direction = self.classify_trade_direction(price, prev_price, volume)
            
            # Update bucket volumes
            current_bucket_volume += volume
            if direction == TradeDirection.BUY:
                current_buy_volume += volume
            elif direction == TradeDirection.SELL:
                current_sell_volume += volume
            
            prev_price = price
            
            # Check if bucket is full
            if current_bucket_volume >= self.bucket_volume:
                # Calculate VPIN for this bucket
                bucket_vpin = self._calculate_bucket_vpin(
                    current_buy_volume,
                    current_sell_volume,
                    current_bucket_volume
                )
                
                # Calculate toxicity score
                toxicity_score = self._calculate_toxicity_score(bucket_vpin)
                
                # Determine if toxic
                is_toxic = bucket_vpin > self.vpin_threshold_toxic
                
                bucket = VPINBucket(
                    bucket_id=bucket_id,
                    start_time=bucket_start,
                    end_time=timestamp,
                    total_volume=current_bucket_volume,
                    buy_volume=current_buy_volume,
                    sell_volume=current_sell_volume,
                    vpin=bucket_vpin,
                    toxicity_score=toxicity_score,
                    is_toxic=is_toxic
                )
                
                buckets.append(bucket)
                
                # Reset for next bucket
                current_bucket_volume = 0.0
                current_buy_volume = 0.0
                current_sell_volume = 0.0
                bucket_start = None
                bucket_id += 1
        
        return buckets
    
    def _calculate_bucket_vpin(
        self,
        buy_volume: float,
        sell_volume: float,
        total_volume: float
    ) -> float:
        """
        Calculate VPIN for a single bucket.
        
        VPIN = |Buy Volume - Sell Volume| / Total Volume
        
        Args:
            buy_volume: Buy volume in bucket
            sell_volume: Sell volume in bucket
            total_volume: Total volume in bucket
            
        Returns:
            VPIN value
        """
        if total_volume == 0:
            return 0.0
        
        vpin = abs(buy_volume - sell_volume) / total_volume
        return vpin
    
    def _calculate_toxicity_score(self, vpin: float) -> float:
        """
        Calculate toxicity score based on VPIN.
        
        Args:
            vpin: VPIN value
            
        Returns:
            Toxicity score (0-1)
        """
        # Normalize VPIN to 0-1 range
        # VPIN typically ranges from 0 to 1, but we cap at 1
        normalized_vpin = min(vpin, 1.0)
        return normalized_vpin
    
    def calculate_vpin(self, buckets: List[VPINBucket]) -> float:
        """
        Calculate overall VPIN from buckets.
        
        VPIN = Average of bucket VPINs
        
        Args:
            buckets: List of VPINBucket
            
        Returns:
            Overall VPIN
        """
        if not buckets:
            return 0.0
        
        # Use last num_buckets
        recent_buckets = buckets[-self.num_buckets:]
        
        if not recent_buckets:
            return 0.0
        
        vpin_values = [b.vpin for b in recent_buckets]
        overall_vpin = np.mean(vpin_values)
        
        return overall_vpin
    
    def determine_regime(self, vpin: float) -> str:
        """
        Determine toxicity regime based on VPIN.
        
        Args:
            vpin: VPIN value
            
        Returns:
            Regime string (normal, elevated, toxic)
        """
        if vpin < self.vpin_threshold_normal:
            return "normal"
        elif vpin < self.vpin_threshold_elevated:
            return "elevated"
        else:
            return "toxic"
    
    def generate_signal(
        self,
        symbol: str,
        current_vpin: float,
        timestamp: datetime
    ) -> VPINSignal:
        """
        Generate trading signal based on VPIN.
        
        Args:
            symbol: Stock symbol
            current_vpin: Current VPIN value
            timestamp: Signal timestamp
            
        Returns:
            VPINSignal
        """
        # Determine regime
        regime = self.determine_regime(current_vpin)
        
        # Calculate toxicity score
        toxicity_score = self._calculate_toxicity_score(current_vpin)
        
        # Generate signal
        # High VPIN = toxic flow = reduce exposure (negative signal)
        # Low VPIN = normal flow = can increase exposure (positive signal)
        if regime == "toxic":
            signal = -0.8  # Strong reduce exposure
            confidence = 0.8
        elif regime == "elevated":
            signal = -0.4  # Moderate reduce exposure
            confidence = 0.6
        else:
            signal = 0.2  # Slight increase exposure
            confidence = 0.4
        
        # Smooth signal with lookback
        if len(self.vpin_history) >= self.lookback_periods:
            recent_vpin = self.vpin_history[-self.lookback_periods:]
            avg_vpin = np.mean(recent_vpin)
            
            # Adjust signal based on trend
            if current_vpin > avg_vpin * 1.2:
                # VPIN increasing rapidly - more negative
                signal = min(signal - 0.2, -0.9)
                confidence = min(confidence + 0.1, 0.9)
            elif current_vpin < avg_vpin * 0.8:
                # VPIN decreasing rapidly - more positive
                signal = max(signal + 0.2, 0.5)
                confidence = min(confidence + 0.1, 0.9)
        
        vpin_signal = VPINSignal(
            timestamp=timestamp,
            symbol=symbol,
            vpin=current_vpin,
            toxicity_score=toxicity_score,
            signal=signal,
            confidence=confidence,
            regime=regime
        )
        
        self.signals.append(vpin_signal)
        self.vpin_history.append(current_vpin)
        
        # Keep history manageable
        if len(self.vpin_history) > 1000:
            self.vpin_history = self.vpin_history[-1000:]
        if len(self.signals) > 1000:
            self.signals = self.signals[-1000:]
        
        return vpin_signal
    
    def process_trades(
        self,
        symbol: str,
        trades: pd.DataFrame
    ) -> List[VPINSignal]:
        """
        Process trade data and generate signals.
        
        Args:
            symbol: Stock symbol
            trades: DataFrame with columns: timestamp, price, volume
            
        Returns:
            List of VPINSignal
        """
        # Create volume buckets
        buckets = self.create_volume_buckets(trades)
        self.buckets.extend(buckets)
        
        # Keep only recent buckets
        if len(self.buckets) > self.num_buckets * 2:
            self.buckets = self.buckets[-self.num_buckets * 2:]
        
        # Calculate VPIN
        current_vpin = self.calculate_vpin(buckets)
        
        # Generate signal
        signal = self.generate_signal(symbol, current_vpin, datetime.now())
        
        return [signal]
    
    def get_latest_signal(self, symbol: str) -> Optional[VPINSignal]:
        """
        Get the latest signal for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest VPINSignal or None
        """
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def print_vpin_report(self) -> None:
        """Print VPIN analysis report."""
        print("\n" + "="*60)
        print("VPIN TOXICITY ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Bucket Volume: {self.bucket_volume:,}")
        print(f"  Number of Buckets: {self.num_buckets}")
        print(f"  VPIN Thresholds: Normal={self.vpin_threshold_normal}, "
              f"Elevated={self.vpin_threshold_elevated}, Toxic={self.vpin_threshold_toxic}")
        
        print(f"\nStatistics:")
        print(f"  Total Buckets: {len(self.buckets)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.vpin_history:
            print(f"\nVPIN Statistics:")
            print(f"  Current VPIN: {self.vpin_history[-1]:.4f}")
            print(f"  Average VPIN: {np.mean(self.vpin_history):.4f}")
            print(f"  Max VPIN: {np.max(self.vpin_history):.4f}")
            print(f"  Min VPIN: {np.min(self.vpin_history):.4f}")
            print(f"  VPIN Std: {np.std(self.vpin_history):.4f}")
        
        if self.signals:
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'VPIN':<10} {'Regime':<10} {'Signal':<10} {'Confidence':<12}")
            print("-" * 85)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.vpin:<10.4f} {signal.regime:<10} {signal.signal:<10.2f} {signal.confidence:<12.2f}")
        
        print("\n" + "="*60)


def sample_vpin_toxicity_alpha():
    """Demonstrate VPIN toxicity alpha."""
    print("=== VPIN Toxicity Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = VPINToxicityAlpha(
        bucket_volume=100000,
        num_buckets=50,
        vpin_threshold_normal=0.3,
        vpin_threshold_elevated=0.5,
        vpin_threshold_toxic=0.7
    )
    
    # Generate sample trade data
    np.random.seed(42)
    n_trades = 10000
    
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), periods=n_trades, freq='1s')
    base_price = 1000.0
    prices = base_price + np.random.randn(n_trades).cumsum() * 0.1
    volumes = np.random.randint(100, 1000, n_trades)
    
    trades = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })
    
    # Process trades
    print("Processing trades...")
    signals = alpha.process_trades('RELIANCE', trades)
    
    # Print report
    alpha.print_vpin_report()
    
    print("\n=== VPIN Toxicity Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Volume-synchronized bucket creation")
    print("- VPIN calculation (Easley et al. 2012)")
    print("- Toxicity regime detection (normal, elevated, toxic)")
    print("- Trading signal generation based on toxicity")
    print("- Signal smoothing with lookback periods")
    print("- Expected Sharpe: 0.6-1.0")
    print("- Expected Capacity: Medium-High (index futures)")


if __name__ == "__main__":
    sample_vpin_toxicity_alpha()

"""
Cancel Ratio (CR) Spoofing Detection Alpha Strategy

This module implements cancel ratio spoofing detection, which identifies
manipulative order placement and cancellation patterns that indicate
fake liquidity and potential price reversals.

Based on HTX Research 2025.
Expected Sharpe: 0.5-1.0
Expected Capacity: Low (prop)
Decay: Moderate (years)
Difficulty: High

Priority: Medium (Research OS Phase 8)
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


class OrderAction(Enum):
    """Order action types."""
    PLACE = "place"
    MODIFY = "modify"
    CANCEL = "cancel"
    FILL = "fill"


class SpoofingSeverity(Enum):
    """Spoofing severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class OrderEvent:
    """Single order event."""
    order_id: str
    timestamp: datetime
    action: OrderAction
    price: float
    quantity: float
    side: str  # buy or sell


@dataclass
class CancelRatioMeasurement:
    """Cancel ratio measurement for a time window."""
    timestamp: datetime
    symbol: str
    window_start: datetime
    window_end: datetime
    total_orders: int
    placed_orders: int
    cancelled_orders: int
    modified_orders: int
    filled_orders: int
    cancel_ratio: float  # cancelled / (placed + modified)
    spoofing_score: float
    severity: SpoofingSeverity


@dataclass
class SpoofingSignal:
    """Spoofing trading signal."""
    timestamp: datetime
    symbol: str
    cancel_ratio: float
    spoofing_score: float
    severity: SpoofingSeverity
    signal: float  # -1 to 1, negative = fade spoofing
    confidence: float
    expected_reversal: float


class CancelRatioSpoofingAlpha:
    """
    Cancel ratio spoofing detection alpha strategy.
    
    This class detects manipulative order patterns by analyzing
    the ratio of cancelled orders to placed orders.
    """
    
    def __init__(
        self,
        window_seconds: int = 60,
        cancel_ratio_threshold: float = 0.7,
        spoofing_threshold: float = 0.6,
        min_orders: int = 10
    ):
        """
        Initialize cancel ratio spoofing alpha.
        
        Args:
            window_seconds: Time window for CR calculation
            cancel_ratio_threshold: Threshold for high cancel ratio
            spoofing_threshold: Threshold for spoofing detection
            min_orders: Minimum orders in window for analysis
        """
        self.window_seconds = window_seconds
        self.cancel_ratio_threshold = cancel_ratio_threshold
        self.spoofing_threshold = spoofing_threshold
        self.min_orders = min_orders
        
        self.order_events: Dict[str, List[OrderEvent]] = {}
        self.measurements: List[CancelRatioMeasurement] = []
        self.signals: List[SpoofingSignal] = []
        
        logger.info(f"CancelRatioSpoofingAlpha initialized: window={window_seconds}s, "
                   f"cr_threshold={cancel_ratio_threshold}, spoof_threshold={spoofing_threshold}")
    
    def add_order_event(
        self,
        symbol: str,
        order_id: str,
        timestamp: datetime,
        action: OrderAction,
        price: float,
        quantity: float,
        side: str
    ) -> None:
        """
        Add an order event.
        
        Args:
            symbol: Stock symbol
            order_id: Order ID
            timestamp: Event timestamp
            action: Order action
            price: Order price
            quantity: Order quantity
            side: Order side (buy/sell)
        """
        event = OrderEvent(
            order_id=order_id,
            timestamp=timestamp,
            action=action,
            price=price,
            quantity=quantity,
            side=side
        )
        
        if symbol not in self.order_events:
            self.order_events[symbol] = []
        
        self.order_events[symbol].append(event)
        
        # Keep only recent events (last hour)
        cutoff = timestamp - timedelta(hours=1)
        self.order_events[symbol] = [
            e for e in self.order_events[symbol] if e.timestamp >= cutoff
        ]
    
    def calculate_cancel_ratio(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Optional[CancelRatioMeasurement]:
        """
        Calculate cancel ratio for a symbol.
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            CancelRatioMeasurement or None
        """
        if symbol not in self.order_events:
            return None
        
        events = self.order_events[symbol]
        window_start = timestamp - timedelta(seconds=self.window_seconds)
        
        # Filter events in window
        window_events = [e for e in events if e.timestamp >= window_start]
        
        if len(window_events) < self.min_orders:
            return None
        
        # Count actions
        placed = sum(1 for e in window_events if e.action == OrderAction.PLACE)
        cancelled = sum(1 for e in window_events if e.action == OrderAction.CANCEL)
        modified = sum(1 for e in window_events if e.action == OrderAction.MODIFY)
        filled = sum(1 for e in window_events if e.action == OrderAction.FILL)
        
        total = len(window_events)
        
        # Calculate cancel ratio
        denominator = placed + modified
        if denominator == 0:
            cancel_ratio = 0.0
        else:
            cancel_ratio = cancelled / denominator
        
        # Calculate spoofing score
        spoofing_score = self._calculate_spoofing_score(
            cancel_ratio, placed, cancelled, modified, filled
        )
        
        # Determine severity
        severity = self._determine_severity(spoofing_score)
        
        measurement = CancelRatioMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            window_start=window_start,
            window_end=timestamp,
            total_orders=total,
            placed_orders=placed,
            cancelled_orders=cancelled,
            modified_orders=modified,
            filled_orders=filled,
            cancel_ratio=cancel_ratio,
            spoofing_score=spoofing_score,
            severity=severity
        )
        
        self.measurements.append(measurement)
        
        return measurement
    
    def _calculate_spoofing_score(
        self,
        cancel_ratio: float,
        placed: int,
        cancelled: int,
        modified: int,
        filled: int
    ) -> float:
        """
        Calculate spoofing score.
        
        Args:
            cancel_ratio: Cancel ratio
            placed: Number of placed orders
            cancelled: Number of cancelled orders
            modified: Number of modified orders
            filled: Number of filled orders
            
        Returns:
            Spoofing score (0-1)
        """
        # Base score from cancel ratio
        cr_score = min(cancel_ratio / self.cancel_ratio_threshold, 1.0)
        
        # Adjust for fill ratio (low fill ratio = more suspicious)
        total_actions = placed + cancelled + modified + filled
        if total_actions > 0:
            fill_ratio = filled / total_actions
            fill_penalty = 1.0 - fill_ratio
        else:
            fill_penalty = 0.5
        
        # Adjust for modification ratio (high modification = more suspicious)
        if total_actions > 0:
            mod_ratio = modified / total_actions
            mod_penalty = min(mod_ratio * 2, 1.0)
        else:
            mod_penalty = 0.0
        
        # Combined score
        spoofing_score = 0.5 * cr_score + 0.3 * fill_penalty + 0.2 * mod_penalty
        
        return min(spoofing_score, 1.0)
    
    def _determine_severity(self, spoofing_score: float) -> SpoofingSeverity:
        """
        Determine spoofing severity from score.
        
        Args:
            spoofing_score: Spoofing score
            
        Returns:
            SpoofingSeverity
        """
        if spoofing_score < 0.2:
            return SpoofingSeverity.NONE
        elif spoofing_score < 0.4:
            return SpoofingSeverity.LOW
        elif spoofing_score < 0.6:
            return SpoofingSeverity.MEDIUM
        elif spoofing_score < 0.8:
            return SpoofingSeverity.HIGH
        else:
            return SpoofingSeverity.EXTREME
    
    def generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float
    ) -> Optional[SpoofingSignal]:
        """
        Generate spoofing trading signal.
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            current_price: Current price
            
        Returns:
            SpoofingSignal or None
        """
        # Calculate cancel ratio
        measurement = self.calculate_cancel_ratio(symbol, timestamp)
        
        if measurement is None:
            return None
        
        # Check if spoofing detected
        if measurement.spoofing_score < self.spoofing_threshold:
            return None
        
        # Generate signal (fade the spoofing)
        signal = -measurement.spoofing_score  # Negative = fade
        
        # Confidence based on severity
        confidence = measurement.spoofing_score
        
        # Expected reversal (higher spoofing = larger expected reversal)
        expected_reversal = measurement.spoofing_score * 0.02  # 2% max reversal
        
        spoofing_signal = SpoofingSignal(
            timestamp=timestamp,
            symbol=symbol,
            cancel_ratio=measurement.cancel_ratio,
            spoofing_score=measurement.spoofing_score,
            severity=measurement.severity,
            signal=signal,
            confidence=confidence,
            expected_reversal=expected_reversal
        )
        
        self.signals.append(spoofing_signal)
        
        return spoofing_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[SpoofingSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_spoofing_statistics(self) -> Dict[str, any]:
        """Get spoofing statistics."""
        if not self.measurements:
            return {}
        
        cancel_ratios = [m.cancel_ratio for m in self.measurements]
        spoofing_scores = [m.spoofing_score for m in self.measurements]
        
        severity_counts = {}
        for m in self.measurements:
            severity_counts[m.severity.value] = severity_counts.get(m.severity.value, 0) + 1
        
        return {
            'total_measurements': len(self.measurements),
            'avg_cancel_ratio': np.mean(cancel_ratios),
            'avg_spoofing_score': np.mean(spoofing_scores),
            'max_spoofing_score': np.max(spoofing_scores),
            'severity_distribution': severity_counts
        }
    
    def print_spoofing_report(self) -> None:
        """Print spoofing analysis report."""
        print("\n" + "="*60)
        print("CANCEL RATIO SPOOFING DETECTION ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Window Seconds: {self.window_seconds}")
        print(f"  Cancel Ratio Threshold: {self.cancel_ratio_threshold}")
        print(f"  Spoofing Threshold: {self.spoofing_threshold}")
        print(f"  Min Orders: {self.min_orders}")
        
        print(f"\nStatistics:")
        stats = self.get_spoofing_statistics()
        print(f"  Total Measurements: {stats.get('total_measurements', 0)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if stats:
            print(f"\nSpoofing Statistics:")
            print(f"  Average Cancel Ratio: {stats.get('avg_cancel_ratio', 0):.4f}")
            print(f"  Average Spoofing Score: {stats.get('avg_spoofing_score', 0):.4f}")
            print(f"  Max Spoofing Score: {stats.get('max_spoofing_score', 0):.4f}")
            
            if stats.get('severity_distribution'):
                print(f"\nSeverity Distribution:")
                for severity, count in stats['severity_distribution'].items():
                    print(f"  {severity}: {count}")
        
        if self.signals:
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'CR':<8} {'SpoofScore':<12} {'Severity':<10} {'Signal':<10} {'Confidence':<12}")
            print("-" * 95)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.cancel_ratio:<8.4f} {signal.spoofing_score:<12.4f} "
                      f"{signal.severity.value:<10} {signal.signal:<10.3f} {signal.confidence:<12.2f}")
        
        print("\n" + "="*60)


def sample_cancel_ratio_spoofing_alpha():
    """Demonstrate cancel ratio spoofing alpha."""
    print("=== Cancel Ratio Spoofing Detection Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = CancelRatioSpoofingAlpha(
        window_seconds=60,
        cancel_ratio_threshold=0.7,
        spoofing_threshold=0.6,
        min_orders=10
    )
    
    # Generate sample order events
    np.random.seed(42)
    n_orders = 500
    
    base_time = datetime.now()
    
    # Normal order flow (low cancel ratio)
    for i in range(300):
        timestamp = base_time + timedelta(seconds=i)
        action = np.random.choice(
            [OrderAction.PLACE, OrderAction.CANCEL, OrderAction.FILL],
            p=[0.4, 0.3, 0.3]
        )
        alpha.add_order_event(
            'RELIANCE',
            f'order_{i}',
            timestamp,
            action,
            1000.0 + np.random.randn() * 10,
            100,
            'buy'
        )
    
    # Spoofing order flow (high cancel ratio)
    for i in range(300, 500):
        timestamp = base_time + timedelta(seconds=i)
        # High cancel ratio pattern
        action = np.random.choice(
            [OrderAction.PLACE, OrderAction.CANCEL, OrderAction.MODIFY, OrderAction.FILL],
            p=[0.5, 0.4, 0.08, 0.02]
        )
        alpha.add_order_event(
            'RELIANCE',
            f'order_{i}',
            timestamp,
            action,
            1000.0 + np.random.randn() * 10,
            100,
            'buy'
        )
    
    # Process and generate signals
    print("Processing order events...")
    for i in range(60, 500, 10):
        timestamp = base_time + timedelta(seconds=i)
        signal = alpha.generate_signal('RELIANCE', timestamp, 1000.0)
    
    # Print report
    alpha.print_spoofing_report()
    
    print("\n=== Cancel Ratio Spoofing Detection Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Order event tracking (place, modify, cancel, fill)")
    print("- Cancel ratio calculation")
    print("- Spoofing score calculation")
    print("- Severity classification")
    print("- Trading signal generation (fade spoofing)")
    print("- Expected Sharpe: 0.5-1.0")
    print("- Expected Capacity: Low (prop)")
    print("- Decay: Moderate (years)")


if __name__ == "__main__":
    sample_cancel_ratio_spoofing_alpha()

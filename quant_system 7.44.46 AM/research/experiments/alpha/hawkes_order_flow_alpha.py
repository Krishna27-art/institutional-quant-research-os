"""
Hawkes Process for Order Flow Clustering Alpha Strategy

This module implements the Hawkes process for modeling and detecting
clustered order flow patterns, which can predict short-term price
movements and volatility spikes.

Based on tick-hawkes literature; order flow clustering studies.
Expected Sharpe: 0.4-0.7
Expected Capacity: Medium
Decay: Persistent
Difficulty: High

Priority: Low (Research OS Phase 9)
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


class OrderFlowType(Enum):
    """Order flow event types."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    CANCEL = "cancel"


@dataclass
class OrderFlowEvent:
    """Order flow event."""
    timestamp: datetime
    event_type: OrderFlowType
    price: float
    quantity: float
    symbol: str


@dataclass
class HawkesParameters:
    """Hawkes process parameters."""
    mu: float  # Base intensity
    alpha: float  # Excitation parameter
    beta: float  # Decay parameter


@dataclass
class ClusteringMeasurement:
    """Order flow clustering measurement."""
    timestamp: datetime
    symbol: str
    event_count: int
    intensity: float
    clustering_score: float
    expected_future_events: float


@dataclass
class HawkesSignal:
    """Hawkes process trading signal."""
    timestamp: datetime
    symbol: str
    intensity: float
    clustering_score: float
    signal: float  # -1 to 1
    confidence: float
    expected_volatility: float


class HawkesOrderFlowAlpha:
    """
    Hawkes process for order flow clustering alpha strategy.
    
    This class uses Hawkes processes to model self-exciting order flow
    and generate trading signals based on clustering patterns.
    """
    
    def __init__(
        self,
        window_seconds: int = 300,
        min_events: int = 20,
        clustering_threshold: float = 0.6,
        default_mu: float = 0.1,
        default_alpha: float = 0.5,
        default_beta: float = 0.1
    ):
        """
        Initialize Hawkes order flow alpha.
        
        Args:
            window_seconds: Time window for analysis
            min_events: Minimum events for parameter estimation
            clustering_threshold: Threshold for clustering detection
            default_mu: Default base intensity
            default_alpha: Default excitation parameter
            default_beta: Default decay parameter
        """
        self.window_seconds = window_seconds
        self.min_events = min_events
        self.clustering_threshold = clustering_threshold
        self.default_mu = default_mu
        self.default_alpha = default_alpha
        self.default_beta = default_beta
        
        self.order_events: Dict[str, List[OrderFlowEvent]] = {}
        self.parameters: Dict[str, HawkesParameters] = {}
        self.measurements: List[ClusteringMeasurement] = []
        self.signals: List[HawkesSignal] = []
        
        logger.info(f"HawkesOrderFlowAlpha initialized: window={window_seconds}s, "
                   f"clustering_threshold={clustering_threshold}")
    
    def add_order_event(
        self,
        symbol: str,
        timestamp: datetime,
        event_type: OrderFlowType,
        price: float,
        quantity: float
    ) -> None:
        """
        Add an order flow event.
        
        Args:
            symbol: Stock symbol
            timestamp: Event timestamp
            event_type: Event type
            price: Event price
            quantity: Event quantity
        """
        event = OrderFlowEvent(
            timestamp=timestamp,
            event_type=event_type,
            price=price,
            quantity=quantity,
            symbol=symbol
        )
        
        if symbol not in self.order_events:
            self.order_events[symbol] = []
        
        self.order_events[symbol].append(event)
        
        # Keep only recent events (last hour)
        cutoff = timestamp - timedelta(hours=1)
        self.order_events[symbol] = [
            e for e in self.order_events[symbol] if e.timestamp >= cutoff
        ]
    
    def estimate_hawkes_parameters(
        self,
        events: List[OrderFlowEvent]
    ) -> HawkesParameters:
        """
        Estimate Hawkes process parameters from events.
        
        Args:
            events: List of order flow events
            
        Returns:
            HawkesParameters
        """
        if len(events) < self.min_events:
            return HawkesParameters(
                mu=self.default_mu,
                alpha=self.default_alpha,
                beta=self.default_beta
            )
        
        # Extract timestamps
        timestamps = [e.timestamp.timestamp() for e in events]
        timestamps = np.array(timestamps)
        
        # Simple method of moments estimation
        n = len(timestamps)
        total_time = timestamps[-1] - timestamps[0]
        
        if total_time <= 0:
            return HawkesParameters(
                mu=self.default_mu,
                alpha=self.default_alpha,
                beta=self.default_beta
            )
        
        # Estimate mu (base intensity)
        mu_hat = n / total_time
        
        # Estimate alpha and beta using inter-arrival times
        inter_arrivals = np.diff(timestamps)
        
        if len(inter_arrivals) > 0:
            # Simple estimates
            alpha_hat = min(self.default_alpha, 0.8)
            beta_hat = max(self.default_beta, 1.0 / np.mean(inter_arrivals))
        else:
            alpha_hat = self.default_alpha
            beta_hat = self.default_beta
        
        return HawkesParameters(
            mu=mu_hat,
            alpha=alpha_hat,
            beta=beta_hat
        )
    
    def calculate_conditional_intensity(
        self,
        params: HawkesParameters,
        events: List[OrderFlowEvent],
        timestamp: datetime
    ) -> float:
        """
        Calculate conditional intensity at a given timestamp.
        
        Args:
            params: Hawkes parameters
            events: Historical events
            timestamp: Target timestamp
            
        Returns:
            Conditional intensity
        """
        t = timestamp.timestamp()
        
        # Base intensity
        intensity = params.mu
        
        # Add excitation from past events
        for event in events:
            dt = t - event.timestamp.timestamp()
            if dt > 0:
                excitation = params.alpha * np.exp(-params.beta * dt)
                intensity += excitation
        
        return intensity
    
    def calculate_clustering_score(
        self,
        params: HawkesParameters,
        events: List[OrderFlowEvent],
        timestamp: datetime
    ) -> float:
        """
        Calculate clustering score.
        
        Args:
            params: Hawkes parameters
            events: Historical events
            timestamp: Target timestamp
            
        Returns:
            Clustering score (0-1)
        """
        # Current intensity
        current_intensity = self.calculate_conditional_intensity(params, events, timestamp)
        
        # Base intensity
        base_intensity = params.mu
        
        if base_intensity == 0:
            return 0.0
        
        # Clustering ratio
        clustering_ratio = current_intensity / base_intensity
        
        # Normalize to 0-1
        clustering_score = min(clustering_ratio / 5.0, 1.0)
        
        return clustering_score
    
    def measure_clustering(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Optional[ClusteringMeasurement]:
        """
        Measure order flow clustering.
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            ClusteringMeasurement or None
        """
        if symbol not in self.order_events:
            return None
        
        events = self.order_events[symbol]
        window_start = timestamp - timedelta(seconds=self.window_seconds)
        
        # Filter events in window
        window_events = [e for e in events if e.timestamp >= window_start]
        
        if len(window_events) < self.min_events:
            return None
        
        # Estimate parameters
        params = self.estimate_hawkes_parameters(window_events)
        self.parameters[symbol] = params
        
        # Calculate intensity and clustering
        intensity = self.calculate_conditional_intensity(params, window_events, timestamp)
        clustering_score = self.calculate_clustering_score(params, window_events, timestamp)
        
        # Expected future events (integral of intensity)
        expected_future = intensity * 60  # Next minute
        
        measurement = ClusteringMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            event_count=len(window_events),
            intensity=intensity,
            clustering_score=clustering_score,
            expected_future_events=expected_future
        )
        
        self.measurements.append(measurement)
        
        return measurement
    
    def generate_signal(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float
    ) -> Optional[HawkesSignal]:
        """
        Generate Hawkes trading signal.
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            current_price: Current price
            
        Returns:
            HawkesSignal or None
        """
        # Measure clustering
        measurement = self.measure_clustering(symbol, timestamp)
        
        if measurement is None:
            return None
        
        # Check if clustering detected
        if measurement.clustering_score < self.clustering_threshold:
            return None
        
        # Generate signal
        # High clustering = likely continuation or reversal depending on context
        # For simplicity, we use clustering as a volatility predictor
        signal = 0.0  # Neutral on direction, but informative on volatility
        
        # Confidence based on clustering score
        confidence = measurement.clustering_score
        
        # Expected volatility proportional to intensity
        expected_volatility = measurement.intensity * 0.01
        
        hawkes_signal = HawkesSignal(
            timestamp=timestamp,
            symbol=symbol,
            intensity=measurement.intensity,
            clustering_score=measurement.clustering_score,
            signal=signal,
            confidence=confidence,
            expected_volatility=expected_volatility
        )
        
        self.signals.append(hawkes_signal)
        
        return hawkes_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[HawkesSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_clustering_statistics(self) -> Dict[str, any]:
        """Get clustering statistics."""
        if not self.measurements:
            return {}
        
        intensities = [m.intensity for m in self.measurements]
        clustering_scores = [m.clustering_score for m in self.measurements]
        
        return {
            'total_measurements': len(self.measurements),
            'avg_intensity': np.mean(intensities),
            'avg_clustering_score': np.mean(clustering_scores),
            'max_clustering_score': np.max(clustering_scores)
        }
    
    def print_hawkes_report(self) -> None:
        """Print Hawkes analysis report."""
        print("\n" + "="*60)
        print("HAWKES ORDER FLOW CLUSTERING ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Window Seconds: {self.window_seconds}")
        print(f"  Min Events: {self.min_events}")
        print(f"  Clustering Threshold: {self.clustering_threshold}")
        
        print(f"\nStatistics:")
        stats = self.get_clustering_statistics()
        print(f"  Total Measurements: {stats.get('total_measurements', 0)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if stats:
            print(f"\nClustering Statistics:")
            print(f"  Average Intensity: {stats.get('avg_intensity', 0):.4f}")
            print(f"  Average Clustering Score: {stats.get('avg_clustering_score', 0):.4f}")
            print(f"  Max Clustering Score: {stats.get('max_clustering_score', 0):.4f}")
        
        if self.signals:
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Intensity':<12} {'Clustering':<12} {'Signal':<10} {'Confidence':<12} {'ExpVol':<10}")
            print("-" * 105)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.intensity:<12.4f} {signal.clustering_score:<12.4f} "
                      f"{signal.signal:<10.3f} {signal.confidence:<12.2f} {signal.expected_volatility:<10.4f}")
        
        print("\n" + "="*60)


def sample_hawkes_order_flow_alpha():
    """Demonstrate Hawkes order flow alpha."""
    print("=== Hawkes Order Flow Clustering Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = HawkesOrderFlowAlpha(
        window_seconds=300,
        min_events=20,
        clustering_threshold=0.6
    )
    
    # Generate sample order flow events
    np.random.seed(42)
    n_events = 1000
    
    base_time = datetime.now()
    
    # Normal order flow
    for i in range(500):
        timestamp = base_time + timedelta(seconds=i * 2)
        event_type = np.random.choice(list(OrderFlowType))
        alpha.add_order_event(
            'RELIANCE',
            timestamp,
            event_type,
            1000.0 + np.random.randn() * 10,
            100
        )
    
    # Clustered order flow (burst pattern)
    for i in range(500, 1000):
        # Create bursts
        burst_phase = (i - 500) % 50
        if burst_phase < 10:
            # High activity burst
            timestamp = base_time + timedelta(seconds=500 * 2 + burst_phase * 0.5)
        else:
            # Low activity
            timestamp = base_time + timedelta(seconds=500 * 2 + burst_phase * 2)
        
        event_type = np.random.choice(list(OrderFlowType))
        alpha.add_order_event(
            'RELIANCE',
            timestamp,
            event_type,
            1000.0 + np.random.randn() * 10,
            100
        )
    
    # Process and generate signals
    print("Processing order flow events...")
    for i in range(300, 1000, 30):
        timestamp = base_time + timedelta(seconds=i * 2)
        signal = alpha.generate_signal('RELIANCE', timestamp, 1000.0)
    
    # Print report
    alpha.print_hawkes_report()
    
    print("\n=== Hawkes Order Flow Clustering Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Order flow event tracking")
    print("- Hawkes process parameter estimation")
    print("- Conditional intensity calculation")
    print("- Clustering score calculation")
    print("- Trading signal generation")
    print("- Expected volatility prediction")
    print("- Expected Sharpe: 0.4-0.7")
    print("- Expected Capacity: Medium")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_hawkes_order_flow_alpha()

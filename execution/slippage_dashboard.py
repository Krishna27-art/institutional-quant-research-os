"""
Real-Time Slippage Measurement Dashboard

This module provides real-time slippage measurement and monitoring for execution
quality assessment, helping to identify execution inefficiencies and optimize trading.

Key Features:
- Real-time slippage tracking
- Execution quality metrics
- Slippage by order type and size
- Time-of-day slippage analysis
- Symbol-wise slippage breakdown
- Alert generation for excessive slippage
- Historical slippage trends

Based on V4 Blueprint - Institutional Architecture
Priority: Medium (Phase 3.4)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlippageType(Enum):
    """Types of slippage."""
    PRICE_SLIPPAGE = "price_slippage"
    TIMING_SLIPPAGE = "timing_slippage"
    MARKET_IMPACT = "market_impact"
    TOTAL_SLIPPAGE = "total_slippage"


@dataclass
class SlippageMeasurement:
    """Slippage measurement for an order."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    target_price: float
    execution_price: float
    order_size: int
    execution_time: datetime
    slippage_bps: float
    slippage_type: SlippageType
    market_conditions: Dict[str, float]
    
    def is_excessive(self, threshold_bps: float = 10.0) -> bool:
        """Check if slippage is excessive."""
        return abs(self.slippage_bps) > threshold_bps


@dataclass
class SlippageSummary:
    """Summary of slippage metrics."""
    symbol: str
    total_orders: int
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    std_slippage_bps: float
    excessive_slippage_count: int
    excessive_slippage_pct: float
    last_update: datetime


class SlippageDashboard:
    """
    Real-time slippage measurement dashboard.
    
    This class tracks execution slippage in real-time and provides
    comprehensive monitoring and analysis.
    """
    
    def __init__(
        self,
        excessive_threshold_bps: float = 10.0,
        window_size: int = 1000
    ):
        """
        Initialize slippage dashboard.
        
        Args:
            excessive_threshold_bps: Threshold for excessive slippage
            window_size: Size of rolling window for calculations
        """
        self.excessive_threshold_bps = excessive_threshold_bps
        self.window_size = window_size
        
        self.slippage_history: Dict[str, deque] = {}
        self.slippage_summaries: Dict[str, SlippageSummary] = {}
        self.alerts: List[Dict] = []
        
        logger.info(f"SlippageDashboard initialized: threshold={excessive_threshold_bps} bps")
    
    def measure_slippage(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        target_price: float,
        execution_price: float,
        order_size: int,
        execution_time: datetime,
        market_conditions: Optional[Dict[str, float]] = None
    ) -> SlippageMeasurement:
        """
        Measure slippage for an order.
        
        Args:
            order_id: Order identifier
            symbol: Stock symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            target_price: Target/arrival price
            execution_price: Actual execution price
            order_size: Order size
            execution_time: Execution timestamp
            market_conditions: Market conditions at execution
            
        Returns:
            SlippageMeasurement
        """
        # Calculate slippage in basis points
        if side.lower() == 'buy':
            slippage_bps = (execution_price - target_price) / target_price * 10000
        else:
            slippage_bps = (target_price - execution_price) / target_price * 10000
        
        # Determine slippage type
        if market_conditions and 'spread' in market_conditions:
            spread_bps = market_conditions['spread'] * 10000
            if abs(slippage_bps) > spread_bps * 2:
                slippage_type = SlippageType.MARKET_IMPACT
            else:
                slippage_type = SlippageType.PRICE_SLIPPAGE
        else:
            slippage_type = SlippageType.TOTAL_SLIPPAGE
        
        market_conditions = market_conditions or {}
        
        measurement = SlippageMeasurement(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            target_price=target_price,
            execution_price=execution_price,
            order_size=order_size,
            execution_time=execution_time,
            slippage_bps=slippage_bps,
            slippage_type=slippage_type,
            market_conditions=market_conditions
        )
        
        # Add to history
        if symbol not in self.slippage_history:
            self.slippage_history[symbol] = deque(maxlen=self.window_size)
        
        self.slippage_history[symbol].append(measurement)
        
        # Update summary
        self._update_summary(symbol)
        
        # Generate alert if excessive
        if measurement.is_excessive(self.excessive_threshold_bps):
            self._generate_alert(measurement)
        
        return measurement
    
    def _update_summary(self, symbol: str) -> None:
        """Update slippage summary for a symbol."""
        if symbol not in self.slippage_history or len(self.slippage_history[symbol]) == 0:
            return
        
        measurements = list(self.slippage_history[symbol])
        slippages = [m.slippage_bps for m in measurements]
        
        total_orders = len(measurements)
        avg_slippage = np.mean(slippages)
        median_slippage = np.median(slippages)
        max_slippage = max(slippages)
        min_slippage = min(slippages)
        std_slippage = np.std(slippages)
        
        excessive_count = sum(1 for m in measurements if m.is_excessive(self.excessive_threshold_bps))
        excessive_pct = (excessive_count / total_orders) * 100 if total_orders > 0 else 0
        
        summary = SlippageSummary(
            symbol=symbol,
            total_orders=total_orders,
            avg_slippage_bps=avg_slippage,
            median_slippage_bps=median_slippage,
            max_slippage_bps=max_slippage,
            min_slippage_bps=min_slippage,
            std_slippage_bps=std_slippage,
            excessive_slippage_count=excessive_count,
            excessive_slippage_pct=excessive_pct,
            last_update=datetime.now()
        )
        
        self.slippage_summaries[symbol] = summary
    
    def _generate_alert(self, measurement: SlippageMeasurement) -> None:
        """Generate alert for excessive slippage."""
        alert = {
            'timestamp': datetime.now(),
            'order_id': measurement.order_id,
            'symbol': measurement.symbol,
            'slippage_bps': measurement.slippage_bps,
            'threshold_bps': self.excessive_threshold_bps,
            'severity': 'HIGH' if abs(measurement.slippage_bps) > self.excessive_threshold_bps * 2 else 'MEDIUM',
            'message': f"Excessive slippage: {measurement.slippage_bps:.2f} bps for {measurement.symbol}"
        }
        
        self.alerts.append(alert)
        logger.warning(f"Slippage alert: {alert['message']}")
    
    def get_summary(self, symbol: str) -> Optional[SlippageSummary]:
        """Get slippage summary for a symbol."""
        return self.slippage_summaries.get(symbol)
    
    def get_all_summaries(self) -> Dict[str, SlippageSummary]:
        """Get all slippage summaries."""
        return self.slippage_summaries
    
    def get_slippage_by_time_of_day(
        self,
        symbol: str,
        bucket_minutes: int = 30
    ) -> pd.DataFrame:
        """
        Get slippage by time of day.
        
        Args:
            symbol: Stock symbol
            bucket_minutes: Time bucket size in minutes
            
        Returns:
            DataFrame with time-of-day slippage
        """
        if symbol not in self.slippage_history:
            return pd.DataFrame()
        
        measurements = list(self.slippage_history[symbol])
        
        # Create time buckets
        data = []
        for m in measurements:
            hour = m.execution_time.hour
            minute = (m.execution_time.minute // bucket_minutes) * bucket_minutes
            time_bucket = f"{hour:02d}:{minute:02d}"
            
            data.append({
                'time_bucket': time_bucket,
                'slippage_bps': m.slippage_bps,
                'order_size': m.order_size,
                'order_type': m.order_type
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Aggregate by time bucket
        result = df.groupby('time_bucket').agg({
            'slippage_bps': ['mean', 'median', 'std', 'count'],
            'order_size': 'mean'
        }).reset_index()
        
        result.columns = ['time_bucket', 'avg_slippage', 'median_slippage', 'std_slippage', 'count', 'avg_order_size']
        
        return result
    
    def get_slippage_by_order_size(
        self,
        symbol: str,
        size_buckets: List[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Get slippage by order size.
        
        Args:
            symbol: Stock symbol
            size_buckets: List of (min, max) size buckets
            
        Returns:
            DataFrame with size-based slippage
        """
        if symbol not in self.slippage_history:
            return pd.DataFrame()
        
        measurements = list(self.slippage_history[symbol])
        
        if size_buckets is None:
            # Default buckets
            size_buckets = [
                (0, 1000),
                (1000, 10000),
                (10000, 50000),
                (50000, 100000),
                (100000, float('inf'))
            ]
        
        # Categorize by size
        data = []
        for m in measurements:
            bucket_label = None
            for min_size, max_size in size_buckets:
                if min_size <= m.order_size < max_size:
                    bucket_label = f"{min_size}-{max_size if max_size != float('inf') else 'inf'}"
                    break
            
            if bucket_label:
                data.append({
                    'size_bucket': bucket_label,
                    'slippage_bps': m.slippage_bps
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Aggregate by size bucket
        result = df.groupby('size_bucket').agg({
            'slippage_bps': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        result.columns = ['size_bucket', 'avg_slippage', 'median_slippage', 'std_slippage', 'count']
        
        return result
    
    def print_dashboard(self) -> None:
        """Print slippage dashboard."""
        print("\n" + "="*60)
        print("REAL-TIME SLIPPAGE DASHBOARD")
        print("="*60)
        
        print(f"\nSymbols Monitored: {len(self.slippage_summaries)}")
        print(f"Total Alerts: {len(self.alerts)}")
        
        if self.slippage_summaries:
            print(f"\nSlippage Summary by Symbol:")
            print(f"{'Symbol':<15} {'Orders':<10} {'Avg Bps':<12} {'Median Bps':<12} {'Excessive %':<15}")
            print("-" * 70)
            
            for symbol, summary in sorted(self.slippage_summaries.items(), key=lambda x: x[1].avg_slippage_bps, reverse=True):
                print(f"{symbol:<15} {summary.total_orders:<10} {summary.avg_slippage_bps:>11.2f} "
                      f"{summary.median_slippage_bps:>11.2f} {summary.excessive_slippage_pct:>14.1f}%")
        
        if self.alerts:
            print(f"\nRecent Alerts (last 5):")
            for alert in self.alerts[-5:]:
                print(f"  [{alert['severity']}] {alert['timestamp']}: {alert['message']}")
        
        print("\n" + "="*60)


def sample_slippage_dashboard():
    """Demonstrate slippage dashboard."""
    print("=== Slippage Dashboard Demo ===\n")
    
    # Initialize dashboard
    dashboard = SlippageDashboard(
        excessive_threshold_bps=10.0,
        window_size=1000
    )
    
    # Generate sample execution data
    np.random.seed(42)
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK']
    
    for i in range(100):
        symbol = np.random.choice(symbols)
        side = np.random.choice(['buy', 'sell'])
        order_type = np.random.choice(['market', 'limit'])
        
        target_price = np.random.uniform(2000, 3000)
        slippage = np.random.normal(2, 5)  # Average 2 bps slippage
        execution_price = target_price * (1 + slippage / 10000) if side == 'buy' else target_price * (1 - slippage / 10000)
        
        order_size = int(np.random.uniform(1000, 50000))
        execution_time = datetime.now() - timedelta(minutes=np.random.randint(0, 390))
        
        market_conditions = {
            'spread': np.random.uniform(0.0001, 0.001),
            'volatility': np.random.uniform(0.01, 0.03)
        }
        
        dashboard.measure_slippage(
            order_id=f"order_{i}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            target_price=target_price,
            execution_price=execution_price,
            order_size=order_size,
            execution_time=execution_time,
            market_conditions=market_conditions
        )
    
    # Print dashboard
    dashboard.print_dashboard()
    
    # Get time-of-day analysis
    print("\nTime-of-Day Slippage Analysis (RELIANCE):")
    tod_df = dashboard.get_slippage_by_time_of_day('RELIANCE', bucket_minutes=60)
    if not tod_df.empty:
        print(tod_df.to_string(index=False))
    
    # Get size-based analysis
    print("\nSize-Based Slippage Analysis (RELIANCE):")
    size_df = dashboard.get_slippage_by_order_size('RELIANCE')
    if not size_df.empty:
        print(size_df.to_string(index=False))
    
    print("\n=== Slippage Dashboard Demo Complete ===")
    print("Key capabilities:")
    print("- Real-time slippage tracking")
    print("- Execution quality metrics")
    print("- Slippage by order type and size")
    print("- Time-of-day slippage analysis")
    print("- Symbol-wise slippage breakdown")
    print("- Alert generation for excessive slippage")


if __name__ == "__main__":
    sample_slippage_dashboard()

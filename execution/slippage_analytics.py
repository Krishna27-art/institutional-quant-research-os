"""
Real-Time Slippage Measurement
Based on Institutional Audit Recommendations

Key findings from audit:
- No tracking of execution shortfall vs. arrival price
- Slippage assumptions are wrong
- Need: Execution analytics table (broker fills)

Architecture V2 Upgrade - 90-Day Plan Item #3
Priority: P0 (Critical)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int
    limit_price: Optional[float]
    arrival_price: float  # Price at signal time
    timestamp: datetime


@dataclass
class Fill:
    """Fill representation"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    fill_price: float
    fill_time: datetime


@dataclass
class SlippageMetrics:
    """Slippage metrics for an order"""
    order_id: str
    symbol: str
    side: str
    arrival_price: float
    avg_fill_price: float
    slippage_bps: float  # Slippage in basis points
    slippage_pct: float  # Slippage in percentage
    fill_ratio: float  # Filled / ordered quantity
    fill_time_seconds: float
    execution_quality: str  # "excellent", "good", "fair", "poor"


@dataclass
class DailySlippageReport:
    """Daily slippage report"""
    date: str
    total_orders: int
    total_fills: int
    avg_slippage_bps: float
    median_slippage_bps: float
    max_slippage_bps: float
    min_slippage_bps: float
    avg_fill_ratio: float
    orders_by_quality: Dict[str, int]
    slippage_by_symbol: Dict[str, float]


class SlippageAnalyzer:
    """
    Real-time slippage analyzer.
    
    Metrics:
    - Slippage = (fill_price - arrival_price) / arrival_price * 10000 (bps)
    - For BUY: positive slippage is bad (paid more)
    - For SELL: negative slippage is bad (sold for less)
    - Fill ratio = filled_quantity / ordered_quantity
    - Execution quality: excellent (<2bps), good (2-5bps), fair (5-10bps), poor (>10bps)
    """
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, List[Fill]] = {}
        self.slippage_history: List[SlippageMetrics] = []
    
    def register_order(self, order: Order) -> None:
        """Register a new order."""
        self.orders[order.order_id] = order
        self.fills[order.order_id] = []
    
    def register_fill(self, fill: Fill) -> None:
        """Register a fill for an order."""
        if fill.order_id in self.fills:
            self.fills[fill.order_id].append(fill)
    
    def calculate_slippage(self, order_id: str) -> Optional[SlippageMetrics]:
        """
        Calculate slippage metrics for an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            SlippageMetrics or None if order not found or no fills
        """
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        fills = self.fills[order_id]
        
        if not fills:
            return None
        
        # Calculate average fill price (volume-weighted)
        total_qty = sum(f.quantity for f in fills)
        avg_fill_price = sum(f.fill_price * f.quantity for f in fills) / total_qty if total_qty > 0 else 0
        
        # Calculate slippage
        if order.side == "BUY":
            # For BUY: (fill_price - arrival_price) / arrival_price
            slippage_pct = (avg_fill_price - order.arrival_price) / order.arrival_price * 100
        else:
            # For SELL: (arrival_price - fill_price) / arrival_price
            slippage_pct = (order.arrival_price - avg_fill_price) / order.arrival_price * 100
        
        slippage_bps = slippage_pct * 100  # Convert to basis points
        
        # Calculate fill ratio
        fill_ratio = total_qty / order.quantity if order.quantity > 0 else 0
        
        # Calculate fill time
        first_fill_time = min(f.fill_time for f in fills)
        fill_time_seconds = (first_fill_time - order.timestamp).total_seconds()
        
        # Determine execution quality
        abs_slippage_bps = abs(slippage_bps)
        if abs_slippage_bps < 2:
            execution_quality = "excellent"
        elif abs_slippage_bps < 5:
            execution_quality = "good"
        elif abs_slippage_bps < 10:
            execution_quality = "fair"
        else:
            execution_quality = "poor"
        
        metrics = SlippageMetrics(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            arrival_price=order.arrival_price,
            avg_fill_price=avg_fill_price,
            slippage_bps=slippage_bps,
            slippage_pct=slippage_pct,
            fill_ratio=fill_ratio,
            fill_time_seconds=fill_time_seconds,
            execution_quality=execution_quality
        )
        
        self.slippage_history.append(metrics)
        return metrics
    
    def get_daily_report(self, date: str) -> DailySlippageReport:
        """
        Generate daily slippage report.
        
        Args:
            date: Date string
            
        Returns:
            DailySlippageReport
        """
        # Filter metrics for the date
        date_metrics = [m for m in self.slippage_history if m.order_id.startswith(date)]
        
        if not date_metrics:
            return DailySlippageReport(
                date=date,
                total_orders=0,
                total_fills=0,
                avg_slippage_bps=0.0,
                median_slippage_bps=0.0,
                max_slippage_bps=0.0,
                min_slippage_bps=0.0,
                avg_fill_ratio=0.0,
                orders_by_quality={},
                slippage_by_symbol={}
            )
        
        # Calculate statistics
        slippage_values = [m.slippage_bps for m in date_metrics]
        avg_slippage_bps = np.mean(slippage_values)
        median_slippage_bps = np.median(slippage_values)
        max_slippage_bps = max(slippage_values)
        min_slippage_bps = min(slippage_values)
        
        fill_ratios = [m.fill_ratio for m in date_metrics]
        avg_fill_ratio = np.mean(fill_ratios)
        
        # Count by quality
        orders_by_quality = {}
        for quality in ["excellent", "good", "fair", "poor"]:
            orders_by_quality[quality] = sum(1 for m in date_metrics if m.execution_quality == quality)
        
        # Slippage by symbol
        slippage_by_symbol = {}
        for symbol in set(m.symbol for m in date_metrics):
            symbol_slippage = [m.slippage_bps for m in date_metrics if m.symbol == symbol]
            slippage_by_symbol[symbol] = np.mean(symbol_slippage)
        
        return DailySlippageReport(
            date=date,
            total_orders=len(date_metrics),
            total_fills=len(date_metrics),
            avg_slippage_bps=avg_slippage_bps,
            median_slippage_bps=median_slippage_bps,
            max_slippage_bps=max_slippage_bps,
            min_slippage_bps=min_slippage_bps,
            avg_fill_ratio=avg_fill_ratio,
            orders_by_quality=orders_by_quality,
            slippage_by_symbol=slippage_by_symbol
        )
    
    def print_daily_report(self, report: DailySlippageReport) -> None:
        """Print daily slippage report."""
        print("\n" + "="*60)
        print(f"SLIPPAGE ANALYTICS REPORT: {report.date}")
        print("="*60)
        print(f"Total Orders: {report.total_orders}")
        print(f"Total Fills: {report.total_fills}")
        print(f"Average Slippage: {report.avg_slippage_bps:.2f} bps")
        print(f"Median Slippage: {report.median_slippage_bps:.2f} bps")
        print(f"Max Slippage: {report.max_slippage_bps:.2f} bps")
        print(f"Min Slippage: {report.min_slippage_bps:.2f} bps")
        print(f"Average Fill Ratio: {report.avg_fill_ratio:.2%}")
        
        print("\nExecution Quality:")
        for quality, count in report.orders_by_quality.items():
            print(f"  {quality:<10}: {count}")
        
        print("\nSlippage by Symbol:")
        for symbol, slippage in report.slippage_by_symbol.items():
            print(f"  {symbol:<10}: {slippage:.2f} bps")
        
        print("="*60)
    
    def get_adaptive_cost_model(self, position_size: float, symbol: str) -> float:
        """
        Get adaptive cost model based on historical slippage.
        
        Args:
            position_size: Position size in currency units
            symbol: Stock symbol
            
        Returns:
            Estimated slippage in bps
        """
        # Get historical slippage for symbol
        symbol_slippage = [m.slippage_bps for m in self.slippage_history if m.symbol == symbol]
        
        if not symbol_slippage:
            # Fallback to default model: 2 bps base + 0.5 bps per ₹1Cr
            return 2.0 + 0.5 * (position_size / 1e7)
        
        # Use historical average + position size adjustment
        base_slippage = np.mean(symbol_slippage)
        size_adjustment = 0.5 * (position_size / 1e7)
        
        return base_slippage + size_adjustment
    
    def to_json(self, report: DailySlippageReport) -> str:
        """Convert report to JSON."""
        report_dict = {
            "date": report.date,
            "total_orders": report.total_orders,
            "total_fills": report.total_fills,
            "avg_slippage_bps": report.avg_slippage_bps,
            "median_slippage_bps": report.median_slippage_bps,
            "max_slippage_bps": report.max_slippage_bps,
            "min_slippage_bps": report.min_slippage_bps,
            "avg_fill_ratio": report.avg_fill_ratio,
            "orders_by_quality": report.orders_by_quality,
            "slippage_by_symbol": report.slippage_by_symbol
        }
        return json.dumps(report_dict, indent=2)


def run_sample_slippage_analysis():
    """Run sample slippage analysis."""
    analyzer = SlippageAnalyzer()
    
    # Register orders
    orders = [
        Order("ORD001", "NIFTY", "BUY", 100, 20000.0, datetime(2024, 1, 1, 9, 30, 0)),
        Order("ORD002", "BANKNIFTY", "SELL", 50, 42000.0, datetime(2024, 1, 1, 9, 31, 0)),
        Order("ORD003", "RELIANCE", "BUY", 200, 2500.0, datetime(2024, 1, 1, 9, 32, 0)),
    ]
    
    for order in orders:
        analyzer.register_order(order)
    
    # Register fills
    fills = [
        Fill("FILL001", "ORD001", "NIFTY", "BUY", 100, 20000.5, datetime(2024, 1, 1, 9, 30, 2)),  # 2.5 bps slippage
        Fill("FILL002", "ORD002", "BANKNIFTY", "SELL", 50, 41999.0, datetime(2024, 1, 1, 9, 31, 1)),  # 2.38 bps slippage
        Fill("FILL003", "ORD003", "RELIANCE", "BUY", 200, 2501.5, datetime(2024, 1, 1, 9, 32, 3)),  # 6 bps slippage
    ]
    
    for fill in fills:
        analyzer.register_fill(fill)
    
    # Calculate slippage for each order
    for order in orders:
        metrics = analyzer.calculate_slippage(order.order_id)
        if metrics:
            print(f"\nOrder {order.order_id}:")
            print(f"  Arrival Price: ₹{metrics.arrival_price:.2f}")
            print(f"  Avg Fill Price: ₹{metrics.avg_fill_price:.2f}")
            print(f"  Slippage: {metrics.slippage_bps:.2f} bps")
            print(f"  Fill Ratio: {metrics.fill_ratio:.2%}")
            print(f"  Execution Quality: {metrics.execution_quality}")
    
    # Generate daily report
    report = analyzer.get_daily_report("2024-01-01")
    analyzer.print_daily_report(report)
    
    # Test adaptive cost model
    print("\nAdaptive Cost Model:")
    for size in [1e6, 1e7, 5e7]:  # ₹10L, ₹1Cr, ₹5Cr
        cost = analyzer.get_adaptive_cost_model(size, "NIFTY")
        print(f"  Position ₹{size/1e6:.0f}L: {cost:.2f} bps")
    
    return report


if __name__ == "__main__":
    run_sample_slippage_analysis()

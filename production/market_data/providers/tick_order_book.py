"""
Tick Order Book Data Integration
Based on the critique: Build data moat with tick-level order book data

Critical for institutional edge:
- Tick-by-tick trade data
- Order book depth (Level 2)
- Bid-ask spread dynamics
- Order flow imbalance
- Hidden liquidity detection

Data Sources:
- NSE/BSE direct feeds
- Third-party providers (Databento, TickData, QuantData)
- Normalized schemas for consistency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OrderBookSide(Enum):
    """Side of order book."""
    BID = "bid"
    ASK = "ask"


class TradeCondition(Enum):
    """Trade condition codes."""
    NORMAL = "normal"
    CANCELLED = "cancelled"
    OUT_OF_SEQUENCE = "out_of_sequence"
    ERROR = "error"


@dataclass
class Tick:
    """Single tick of market data."""
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: Optional[str] = None  # For trades
    condition: TradeCondition = TradeCondition.NORMAL


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    quantity: float
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book at a timestamp."""
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    spread: float = 0.0
    mid_price: float = 0.0
    
    def __post_init__(self):
        """Calculate spread and mid price."""
        if self.bids and self.asks:
            self.spread = self.asks[0].price - self.bids[0].price
            self.mid_price = (self.bids[0].price + self.asks[0].price) / 2


@dataclass
class OrderFlowMetrics:
    """Metrics derived from order flow."""
    timestamp: datetime
    symbol: str
    bid_ask_spread: float
    bid_volume: float
    ask_volume: float
    volume_imbalance: float  # (bid - ask) / (bid + ask)
    depth_imbalance: float
    mid_price: float
    vwap: float
    hidden_liquidity_ratio: float


class TickOrderBookManager:
    """
    Manager for tick-level order book data.
    
    Features:
    - Tick-by-tick trade data storage
    - Order book snapshot management
    - Order flow metrics calculation
    - Spread dynamics tracking
    - Hidden liquidity detection
    """
    
    def __init__(self):
        self.ticks: Dict[str, List[Tick]] = {}
        self.order_book_snapshots: Dict[str, List[OrderBookSnapshot]] = {}
        self.order_flow_metrics: Dict[str, List[OrderFlowMetrics]] = {}
        
        # Configuration
        self.max_ticks_per_symbol = 1000000  # Limit memory usage
        self.order_book_depth = 10  # Number of levels to track
    
    def add_tick(self, tick: Tick) -> None:
        """Add a tick to storage."""
        symbol = tick.symbol
        
        if symbol not in self.ticks:
            self.ticks[symbol] = []
        
        self.ticks[symbol].append(tick)
        
        # Limit memory usage
        if len(self.ticks[symbol]) > self.max_ticks_per_symbol:
            self.ticks[symbol] = self.ticks[symbol][-self.max_ticks_per_symbol:]
    
    def add_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add an order book snapshot."""
        symbol = snapshot.symbol
        
        if symbol not in self.order_book_snapshots:
            self.order_book_snapshots[symbol] = []
        
        self.order_book_snapshots[symbol].append(snapshot)
    
    def get_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Tick]:
        """Get ticks for a symbol in time range."""
        if symbol not in self.ticks:
            return []
        
        return [
            tick for tick in self.ticks[symbol]
            if start_time <= tick.timestamp <= end_time
        ]
    
    def get_order_book_snapshots(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[OrderBookSnapshot]:
        """Get order book snapshots for a symbol in time range."""
        if symbol not in self.order_book_snapshots:
            return []
        
        return [
            snapshot for snapshot in self.order_book_snapshots[symbol]
            if start_time <= snapshot.timestamp <= end_time
        ]
    
    def calculate_order_flow_metrics(
        self,
        symbol: str,
        snapshot: OrderBookSnapshot
    ) -> OrderFlowMetrics:
        """
        Calculate order flow metrics from order book snapshot.
        
        Args:
            symbol: Trading symbol
            snapshot: Order book snapshot
            
        Returns:
            OrderFlowMetrics with calculated metrics
        """
        # Calculate bid and ask volumes
        bid_volume = sum(level.quantity for level in snapshot.bids)
        ask_volume = sum(level.quantity for level in snapshot.asks)
        
        # Volume imbalance
        total_volume = bid_volume + ask_volume
        volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Depth imbalance (top 3 levels)
        bid_depth = sum(level.quantity for level in snapshot.bids[:3])
        ask_depth = sum(level.quantity for level in snapshot.asks[:3])
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        
        # VWAP (simplified)
        vwap = snapshot.mid_price  # Would need trade data for true VWAP
        
        # Hidden liquidity ratio (estimated from order book shape)
        # If volume drops sharply with price, suggests hidden liquidity
        if len(snapshot.bids) > 1:
            bid_ratio = snapshot.bids[1].quantity / snapshot.bids[0].quantity if snapshot.bids[0].quantity > 0 else 0
            ask_ratio = snapshot.asks[1].quantity / snapshot.asks[0].quantity if snapshot.asks[0].quantity > 0 else 0
            hidden_liquidity_ratio = (bid_ratio + ask_ratio) / 2
        else:
            hidden_liquidity_ratio = 0
        
        metrics = OrderFlowMetrics(
            timestamp=snapshot.timestamp,
            symbol=symbol,
            bid_ask_spread=snapshot.spread,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            volume_imbalance=volume_imbalance,
            depth_imbalance=depth_imbalance,
            mid_price=snapshot.mid_price,
            vwap=vwap,
            hidden_liquidity_ratio=hidden_liquidity_ratio
        )
        
        return metrics
    
    def detect_hidden_liquidity(
        self,
        symbol: str,
        window_minutes: int = 5
    ) -> Dict[str, float]:
        """
        Detect hidden liquidity in order book.
        
        Hidden liquidity manifests as:
        - Large trades without corresponding order book depth
        - Sudden price moves with small visible volume
        - Persistent order imbalance
        
        Args:
            symbol: Trading symbol
            window_minutes: Time window to analyze
            
        Returns:
            Dictionary with hidden liquidity indicators
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        # Get recent order book snapshots
        snapshots = self.get_order_book_snapshots(symbol, start_time, end_time)
        
        if not snapshots:
            return {}
        
        # Calculate metrics for each snapshot
        metrics_list = []
        for snapshot in snapshots:
            metrics = self.calculate_order_flow_metrics(symbol, snapshot)
            metrics_list.append(metrics)
        
        if not metrics_list:
            return {}
        
        # Analyze patterns
        volume_imbalances = [m.volume_imbalance for m in metrics_list]
        hidden_ratios = [m.hidden_liquidity_ratio for m in metrics_list]
        spreads = [m.bid_ask_spread for m in metrics_list]
        
        # Hidden liquidity indicators
        avg_volume_imbalance = np.mean(volume_imbalances)
        avg_hidden_ratio = np.mean(hidden_ratios)
        avg_spread = np.mean(spreads)
        
        # Persistent imbalance suggests hidden liquidity
        persistent_imbalance = abs(avg_volume_imbalance) > 0.3
        
        # High hidden ratio suggests hidden orders
        high_hidden_liquidity = avg_hidden_ratio > 0.5
        
        # Wide spread with low volume suggests hidden liquidity
        spread_volume_anomaly = avg_spread > 0.001 and avg_hidden_ratio > 0.3
        
        return {
            'avg_volume_imbalance': avg_volume_imbalance,
            'avg_hidden_ratio': avg_hidden_ratio,
            'avg_spread': avg_spread,
            'persistent_imbalance': persistent_imbalance,
            'high_hidden_liquidity': high_hidden_liquidity,
            'spread_volume_anomaly': spread_volume_anomaly,
            'hidden_liquidity_detected': persistent_imbalance or high_hidden_liquidity or spread_volume_anomaly
        }
    
    def get_tick_dataframe(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get ticks as DataFrame for analysis."""
        ticks = self.get_ticks(symbol, start_time, end_time)
        
        data = []
        for tick in ticks:
            data.append({
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'price': tick.price,
                'quantity': tick.quantity,
                'side': tick.side,
                'condition': tick.condition.value
            })
        
        return pd.DataFrame(data)
    
    def get_order_flow_dataframe(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get order flow metrics as DataFrame."""
        snapshots = self.get_order_book_snapshots(symbol, start_time, end_time)
        
        data = []
        for snapshot in snapshots:
            metrics = self.calculate_order_flow_metrics(symbol, snapshot)
            data.append({
                'timestamp': metrics.timestamp,
                'symbol': metrics.symbol,
                'bid_ask_spread': metrics.bid_ask_spread,
                'bid_volume': metrics.bid_volume,
                'ask_volume': metrics.ask_volume,
                'volume_imbalance': metrics.volume_imbalance,
                'depth_imbalance': metrics.depth_imbalance,
                'mid_price': metrics.mid_price,
                'vwap': metrics.vwap,
                'hidden_liquidity_ratio': metrics.hidden_liquidity_ratio
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the Tick Order Book Manager
    print("Testing Tick Order Book Manager...")
    
    manager = TickOrderBookManager()
    
    # Generate sample ticks
    print("\nGenerating sample ticks...")
    base_time = datetime.now()
    
    for i in range(100):
        tick = Tick(
            timestamp=base_time + timedelta(seconds=i),
            symbol="RELIANCE",
            price=2500 + np.random.uniform(-5, 5),
            quantity=np.random.uniform(100, 1000),
            side="buy" if i % 2 == 0 else "sell"
        )
        manager.add_tick(tick)
    
    print(f"Added {len(manager.ticks['RELIANCE'])} ticks")
    
    # Generate sample order book snapshots
    print("\nGenerating sample order book snapshots...")
    for i in range(50):
        mid_price = 2500 + np.random.uniform(-2, 2)
        
        bids = [
            OrderBookLevel(price=mid_price - j * 0.05, quantity=np.random.uniform(1000, 5000))
            for j in range(1, 6)
        ]
        
        asks = [
            OrderBookLevel(price=mid_price + j * 0.05, quantity=np.random.uniform(1000, 5000))
            for j in range(1, 6)
        ]
        
        snapshot = OrderBookSnapshot(
            timestamp=base_time + timedelta(seconds=i * 2),
            symbol="RELIANCE",
            bids=bids,
            asks=asks
        )
        
        manager.add_order_book_snapshot(snapshot)
    
    print(f"Added {len(manager.order_book_snapshots['RELIANCE'])} order book snapshots")
    
    # Get ticks
    print("\nRetrieving ticks...")
    ticks = manager.get_ticks("RELIANCE", base_time, base_time + timedelta(minutes=2))
    print(f"Retrieved {len(ticks)} ticks")
    
    # Get order book snapshots
    print("\nRetrieving order book snapshots...")
    snapshots = manager.get_order_book_snapshots("RELIANCE", base_time, base_time + timedelta(minutes=2))
    print(f"Retrieved {len(snapshots)} snapshots")
    
    # Calculate order flow metrics
    print("\nCalculating order flow metrics...")
    if snapshots:
        metrics = manager.calculate_order_flow_metrics("RELIANCE", snapshots[0])
        print(f"Bid-Ask Spread: {metrics.bid_ask_spread:.4f}")
        print(f"Volume Imbalance: {metrics.volume_imbalance:.4f}")
        print(f"Depth Imbalance: {metrics.depth_imbalance:.4f}")
        print(f"Hidden Liquidity Ratio: {metrics.hidden_liquidity_ratio:.4f}")
    
    # Detect hidden liquidity
    print("\nDetecting hidden liquidity...")
    hidden_liquidity = manager.detect_hidden_liquidity("RELIANCE", window_minutes=2)
    for key, value in hidden_liquidity.items():
        print(f"  {key}: {value}")
    
    # Get DataFrames
    print("\nGetting tick DataFrame...")
    tick_df = manager.get_tick_dataframe("RELIANCE", base_time, base_time + timedelta(minutes=1))
    print(f"Tick DataFrame shape: {tick_df.shape}")
    
    print("\nGetting order flow DataFrame...")
    flow_df = manager.get_order_flow_dataframe("RELIANCE", base_time, base_time + timedelta(minutes=1))
    print(f"Order Flow DataFrame shape: {flow_df.shape}")
    print(flow_df.head())

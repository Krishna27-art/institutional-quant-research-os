"""
Market Depth and Auction Imbalance Data
Based on the critique: Build data moat with market depth and auction imbalance

Critical for institutional edge:
- Pre-open auction imbalance
- Market depth at multiple price levels
- Order book shape analysis
- Large order detection
- Institutional flow detection

Auction Types (Indian Markets):
- Pre-open auction (9:00-9:08, 9:08-9:12)
- Closing auction (15:40-15:50)
- Volatility interruption auction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class AuctionType(Enum):
    """Types of auctions in Indian markets."""
    PRE_OPEN_EQUILIBRIUM = "pre_open_equilibrium"  # 9:00-9:08
    PRE_OPEN_DISCOVERY = "pre_open_discovery"  # 9:08-9:12
    CLOSING = "closing"  # 15:40-15:50
    VOLATILITY = "volatility"  # During volatility interruption


@dataclass
class AuctionData:
    """Data from an auction."""
    auction_type: AuctionType
    timestamp: datetime
    symbol: str
    buy_quantity: float
    sell_quantity: float
    buy_orders: int
    sell_orders: int
    indicative_price: float
    indicative_quantity: float
    imbalance_ratio: float  # (buy - sell) / (buy + sell)
    
    def __post_init__(self):
        """Calculate imbalance ratio."""
        total = self.buy_quantity + self.sell_quantity
        self.imbalance_ratio = (self.buy_quantity - self.sell_quantity) / total if total > 0 else 0


@dataclass
class MarketDepthLevel:
    """Single level of market depth."""
    price: float
    quantity: float
    num_orders: int
    is_hidden: bool = False


@dataclass
class MarketDepthSnapshot:
    """Snapshot of market depth at multiple levels."""
    timestamp: datetime
    symbol: str
    bid_levels: List[MarketDepthLevel]
    ask_levels: List[MarketDepthLevel]
    total_bid_quantity: float
    total_ask_quantity: float
    depth_imbalance: float
    
    def __post_init__(self):
        """Calculate depth metrics."""
        self.total_bid_quantity = sum(level.quantity for level in self.bid_levels)
        self.total_ask_quantity = sum(level.quantity for level in self.ask_levels)
        total = self.total_bid_quantity + self.total_ask_quantity
        self.depth_imbalance = (self.total_bid_quantity - self.total_ask_quantity) / total if total > 0 else 0


@dataclass
class LargeOrder:
    """Detected large order."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    significance_ratio: float  # Order size relative to average
    is_iceberg: bool = False


class MarketDepthManager:
    """
    Manager for market depth and auction data.
    
    Features:
    - Pre-open auction imbalance tracking
    - Market depth at multiple levels
    - Large order detection
    - Order book shape analysis
    - Institutional flow detection
    """
    
    def __init__(self):
        self.auction_data: Dict[str, List[AuctionData]] = {}
        self.market_depth_snapshots: Dict[str, List[MarketDepthSnapshot]] = {}
        self.large_orders: List[LargeOrder] = []
        
        # Configuration
        self.depth_levels = 20  # Number of levels to track
        self.large_order_threshold = 5.0  # 5x average order size
        self.iceberg_detection_enabled = True
    
    def add_auction_data(self, auction: AuctionData) -> None:
        """Add auction data."""
        symbol = auction.symbol
        
        if symbol not in self.auction_data:
            self.auction_data[symbol] = []
        
        self.auction_data[symbol].append(auction)
    
    def add_market_depth_snapshot(self, snapshot: MarketDepthSnapshot) -> None:
        """Add market depth snapshot."""
        symbol = snapshot.symbol
        
        if symbol not in self.market_depth_snapshots:
            self.market_depth_snapshots[symbol] = []
        
        self.market_depth_snapshots[symbol].append(snapshot)
    
    def detect_large_order(
        self,
        order_quantity: float,
        order_price: float,
        symbol: str,
        side: str,
        avg_order_size: float = 1000.0
    ) -> Optional[LargeOrder]:
        """
        Detect if an order is unusually large.
        
        Args:
            order_quantity: Order quantity
            order_price: Order price
            symbol: Trading symbol
            side: Order side
            avg_order_size: Average order size for comparison
            
        Returns:
            LargeOrder if detected, None otherwise
        """
        significance_ratio = order_quantity / avg_order_size
        
        if significance_ratio >= self.large_order_threshold:
            # Check for iceberg order (repeated similar-sized orders)
            is_iceberg = self._detect_iceberg_pattern(symbol, order_quantity, order_price)
            
            large_order = LargeOrder(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                quantity=order_quantity,
                price=order_price,
                significance_ratio=significance_ratio,
                is_iceberg=is_iceberg
            )
            
            self.large_orders.append(large_order)
            return large_order
        
        return None
    
    def _detect_iceberg_pattern(
        self,
        symbol: str,
        quantity: float,
        price: float,
        window_minutes: int = 5
    ) -> bool:
        """
        Detect iceberg order pattern.
        
        Iceberg orders manifest as:
        - Multiple orders at similar price
        - Similar quantities
        - Short time intervals
        """
        recent_orders = [
            order for order in self.large_orders
            if order.symbol == symbol
            and order.timestamp >= datetime.now() - timedelta(minutes=window_minutes)
        ]
        
        if len(recent_orders) < 3:
            return False
        
        # Check for similar quantities (within 10%)
        quantities = [order.quantity for order in recent_orders]
        avg_quantity = np.mean(quantities)
        
        similar_quantities = all(
            abs(q - avg_quantity) / avg_quantity < 0.1
            for q in quantities
        )
        
        # Check for similar prices (within 0.1%)
        prices = [order.price for order in recent_orders]
        avg_price = np.mean(prices)
        
        similar_prices = all(
            abs(p - avg_price) / avg_price < 0.001
            for p in prices
        )
        
        return similar_quantities and similar_prices
    
    def get_auction_imbalance(
        self,
        symbol: str,
        auction_type: AuctionType,
        minutes_before: int = 10
    ) -> Dict[str, float]:
        """
        Get auction imbalance metrics.
        
        Args:
            symbol: Trading symbol
            auction_type: Type of auction
            minutes_before: Minutes before current time to analyze
            
        Returns:
            Dictionary with imbalance metrics
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes_before)
        
        if symbol not in self.auction_data:
            return {}
        
        auctions = [
            auction for auction in self.auction_data[symbol]
            if auction.auction_type == auction_type
            and start_time <= auction.timestamp <= end_time
        ]
        
        if not auctions:
            return {}
        
        # Calculate metrics
        imbalances = [auction.imbalance_ratio for auction in auctions]
        buy_quantities = [auction.buy_quantity for auction in auctions]
        sell_quantities = [auction.sell_quantity for auction in auctions]
        
        return {
            'avg_imbalance': np.mean(imbalances),
            'latest_imbalance': imbalances[-1] if imbalances else 0,
            'imbalance_std': np.std(imbalances),
            'total_buy_quantity': sum(buy_quantities),
            'total_sell_quantity': sum(sell_quantities),
            'buy_sell_ratio': sum(buy_quantities) / sum(sell_quantities) if sum(sell_quantities) > 0 else 0,
            'num_auctions': len(auctions)
        }
    
    def analyze_order_book_shape(
        self,
        symbol: str,
        snapshot: MarketDepthSnapshot
    ) -> Dict[str, float]:
        """
        Analyze order book shape for institutional patterns.
        
        Order book shapes indicate:
        - V-shape: Balanced market
        - L-shape: Strong support/resistance
        - U-shape: Liquidity concentration at middle
        - Skewed: Directional bias
        """
        if not snapshot.bid_levels or not snapshot.ask_levels:
            return {}
        
        # Calculate depth profile
        bid_depths = [level.quantity for level in snapshot.bid_levels]
        ask_depths = [level.quantity for level in snapshot.ask_levels]
        
        # Depth concentration (how much at top 3 levels)
        top_3_bid = sum(bid_depths[:3]) / sum(bid_depths) if sum(bid_depths) > 0 else 0
        top_3_ask = sum(ask_depths[:3]) / sum(ask_depths) if sum(ask_depths) > 0 else 0
        
        # Depth decay rate (how quickly depth drops)
        bid_decay = bid_depths[0] / bid_depths[-1] if bid_depths[-1] > 0 else float('inf')
        ask_decay = ask_depths[0] / ask_depths[-1] if ask_depths[-1] > 0 else float('inf')
        
        # Shape classification
        if top_3_bid > 0.7 and top_3_ask > 0.7:
            shape = "U-shape"  # Liquidity concentrated near mid
        elif top_3_bid > 0.8:
            shape = "L-shape (bid support)"
        elif top_3_ask > 0.8:
            shape = "L-shape (ask resistance)"
        elif abs(snapshot.depth_imbalance) < 0.1:
            shape = "V-shape (balanced)"
        elif snapshot.depth_imbalance > 0.3:
            shape = "Skewed (bid heavy)"
        elif snapshot.depth_imbalance < -0.3:
            shape = "Skewed (ask heavy)"
        else:
            shape = "Normal"
        
        return {
            'top_3_bid_concentration': top_3_bid,
            'top_3_ask_concentration': top_3_ask,
            'bid_decay_rate': bid_decay,
            'ask_decay_rate': ask_decay,
            'depth_imbalance': snapshot.depth_imbalance,
            'order_book_shape': shape
        }
    
    def get_institutional_signals(
        self,
        symbol: str
    ) -> Dict[str, any]:
        """
        Get institutional trading signals from depth and auction data.
        
        Signals include:
        - Auction imbalance direction
        - Large order presence
        - Order book shape bias
        - Hidden liquidity indicators
        """
        signals = {}
        
        # Pre-open auction imbalance
        pre_open_imbalance = self.get_auction_imbalance(symbol, AuctionType.PRE_OPEN_DISCOVERY)
        if pre_open_imbalance:
            signals['auction_direction'] = 'bullish' if pre_open_imbalance['avg_imbalance'] > 0.2 else 'bearish' if pre_open_imbalance['avg_imbalance'] < -0.2 else 'neutral'
            signals['auction_strength'] = abs(pre_open_imbalance['avg_imbalance'])
        
        # Recent large orders
        recent_large_orders = [
            order for order in self.large_orders
            if order.symbol == symbol
            and order.timestamp >= datetime.now() - timedelta(minutes=30)
        ]
        
        if recent_large_orders:
            buy_orders = [o for o in recent_large_orders if o.side == 'buy']
            sell_orders = [o for o in recent_large_orders if o.side == 'sell']
            
            signals['large_order_direction'] = 'bullish' if len(buy_orders) > len(sell_orders) else 'bearish' if len(sell_orders) > len(buy_orders) else 'neutral'
            signals['large_order_count'] = len(recent_large_orders)
            signals['has_iceberg_orders'] = any(o.is_iceberg for o in recent_large_orders)
        
        # Latest market depth analysis
        if symbol in self.market_depth_snapshots and self.market_depth_snapshots[symbol]:
            latest_snapshot = self.market_depth_snapshots[symbol][-1]
            shape_analysis = self.analyze_order_book_shape(symbol, latest_snapshot)
            
            signals['depth_direction'] = 'bullish' if shape_analysis.get('depth_imbalance', 0) > 0.2 else 'bearish' if shape_analysis.get('depth_imbalance', 0) < -0.2 else 'neutral'
            signals['order_book_shape'] = shape_analysis.get('order_book_shape', 'unknown')
        
        return signals


if __name__ == "__main__":
    # Test the Market Depth Manager
    print("Testing Market Depth and Auction Manager...")
    
    manager = MarketDepthManager()
    
    # Generate sample auction data
    print("\nGenerating sample auction data...")
    base_time = datetime.now().replace(hour=9, minute=5, second=0)
    
    for i in range(10):
        auction = AuctionData(
            auction_type=AuctionType.PRE_OPEN_DISCOVERY,
            timestamp=base_time + timedelta(seconds=i),
            symbol="RELIANCE",
            buy_quantity=np.random.uniform(50000, 100000),
            sell_quantity=np.random.uniform(40000, 90000),
            buy_orders=np.random.randint(50, 100),
            sell_orders=np.random.randint(40, 90),
            indicative_price=2500 + np.random.uniform(-5, 5),
            indicative_quantity=np.random.uniform(100000, 200000)
        )
        manager.add_auction_data(auction)
    
    print(f"Added {len(manager.auction_data['RELIANCE'])} auction data points")
    
    # Generate sample market depth snapshots
    print("\nGenerating sample market depth snapshots...")
    for i in range(20):
        mid_price = 2500 + np.random.uniform(-2, 2)
        
        bid_levels = [
            MarketDepthLevel(price=mid_price - j * 0.05, quantity=np.random.uniform(1000, 5000))
            for j in range(1, 11)
        ]
        
        ask_levels = [
            MarketDepthLevel(price=mid_price + j * 0.05, quantity=np.random.uniform(1000, 5000))
            for j in range(1, 11)
        ]
        
        snapshot = MarketDepthSnapshot(
            timestamp=base_time + timedelta(seconds=i * 30),
            symbol="RELIANCE",
            bid_levels=bid_levels,
            ask_levels=ask_levels
        )
        
        manager.add_market_depth_snapshot(snapshot)
    
    print(f"Added {len(manager.market_depth_snapshots['RELIANCE'])} market depth snapshots")
    
    # Detect large orders
    print("\nDetecting large orders...")
    for i in range(5):
        large_order = manager.detect_large_order(
            order_quantity=np.random.uniform(5000, 10000),
            order_price=2500,
            symbol="RELIANCE",
            side="buy" if i % 2 == 0 else "sell",
            avg_order_size=1000
        )
        if large_order:
            print(f"  Large order detected: {large_order.side} {large_order.quantity:.0f} @ {large_order.price:.2f} (significance: {large_order.significance_ratio:.1f}x)")
    
    # Get auction imbalance
    print("\nGetting auction imbalance...")
    imbalance = manager.get_auction_imbalance("RELIANCE", AuctionType.PRE_OPEN_DISCOVERY)
    for key, value in imbalance.items():
        print(f"  {key}: {value}")
    
    # Analyze order book shape
    print("\nAnalyzing order book shape...")
    if manager.market_depth_snapshots["RELIANCE"]:
        shape = manager.analyze_order_book_shape("RELIANCE", manager.market_depth_snapshots["RELIANCE"][0])
        for key, value in shape.items():
            print(f"  {key}: {value}")
    
    # Get institutional signals
    print("\nGetting institutional signals...")
    signals = manager.get_institutional_signals("RELIANCE")
    for key, value in signals.items():
        print(f"  {key}: {value}")

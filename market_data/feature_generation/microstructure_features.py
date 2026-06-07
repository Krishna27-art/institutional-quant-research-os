"""
Market Microstructure Features (OFI, VPIN, Order Book Imbalance)

This module implements market microstructure features from Level 2 order book data,
including Order Flow Imbalance (OFI), Volume-Synchronized Probability of Informed Trading (VPIN),
and other order book-based features.

Key Features:
- Order Flow Imbalance (OFI) calculation
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Order book imbalance features
- Tick-direction classification (Lee-Ready)
- Spread and depth features
- Real-time feature computation

Based on V4 Blueprint - Institutional Architecture
Priority: High (Phase 1.2)
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


class TickDirection(Enum):
    """Tick direction classification (Lee-Ready algorithm)."""
    UP = "up"
    DOWN = "down"
    UNCHANGED = "unchanged"


@dataclass
class MicrostructureFeatures:
    """Microstructure features for a symbol at a timestamp."""
    symbol: str
    timestamp: datetime
    ofi: float  # Order Flow Imbalance
    vpin: float  # Volume-Synchronized Probability of Informed Trading
    order_book_imbalance: float  # Bid-ask volume imbalance
    spread: float  # Bid-ask spread
    spread_bps: float  # Spread in basis points
    depth_imbalance: float  # Bid-ask depth imbalance
    tick_direction: TickDirection
    mid_price: float
    total_volume: int
    buy_volume: int
    sell_volume: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'ofi': self.ofi,
            'vpin': self.vpin,
            'order_book_imbalance': self.order_book_imbalance,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'depth_imbalance': self.depth_imbalance,
            'tick_direction': self.tick_direction.value,
            'mid_price': self.mid_price,
            'total_volume': self.total_volume,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume
        }


class MicrostructureFeatureCalculator:
    """
    Calculator for market microstructure features.
    
    This class computes OFI, VPIN, and other order book-based features
    from Level 2 market data.
    """
    
    def __init__(self, ofi_window: int = 100, vpin_window: int = 100):
        """
        Initialize microstructure feature calculator.
        
        Args:
            ofi_window: Window size for OFI calculation
            vpin_window: Window size for VPIN calculation
        """
        self.ofi_window = ofi_window
        self.vpin_window = vpin_window
        
        # Rolling windows for calculations
        self.ofi_history: Dict[str, deque] = {}
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        
        logger.info(f"MicrostructureFeatureCalculator initialized: OFI window={ofi_window}, VPIN window={vpin_window}")
    
    def classify_tick_direction(
        self,
        current_price: float,
        previous_price: float,
        current_mid: float,
        previous_mid: float
    ) -> TickDirection:
        """
        Classify tick direction using Lee-Ready algorithm.
        
        Args:
            current_price: Current trade price
            previous_price: Previous trade price
            current_mid: Current mid price
            previous_mid: Previous mid price
            
        Returns:
            TickDirection
        """
        # Lee-Ready algorithm:
        # If trade price > previous mid: UP
        # If trade price < previous mid: DOWN
        # If trade price == previous mid: use trade price comparison
        
        if current_price > previous_mid:
            return TickDirection.UP
        elif current_price < previous_mid:
            return TickDirection.DOWN
        else:
            # Trade at mid, use price comparison
            if current_price > previous_price:
                return TickDirection.UP
            elif current_price < previous_price:
                return TickDirection.DOWN
            else:
                return TickDirection.UNCHANGED
    
    def calculate_ofi(
        self,
        bid_price: float,
        bid_qty: int,
        ask_price: float,
        ask_qty: int,
        previous_bid_price: Optional[float] = None,
        previous_bid_qty: Optional[int] = None,
        previous_ask_price: Optional[float] = None,
        previous_ask_qty: Optional[int] = None
    ) -> float:
        """
        Calculate Order Flow Imbalance (OFI).
        
        OFI = max(0, Δbid_qty) - max(0, Δask_qty)
        
        Args:
            bid_price: Current bid price
            bid_qty: Current bid quantity
            ask_price: Current ask price
            ask_qty: Current ask quantity
            previous_bid_price: Previous bid price
            previous_bid_qty: Previous bid quantity
            previous_ask_price: Previous ask price
            previous_ask_qty: Previous ask quantity
            
        Returns:
            OFI value
        """
        if previous_bid_price is None or previous_bid_qty is None:
            return 0.0
        
        # Calculate changes
        delta_bid_qty = 0
        delta_ask_qty = 0
        
        # Bid side change
        if bid_price == previous_bid_price:
            delta_bid_qty = bid_qty - previous_bid_qty
        elif bid_price > previous_bid_price:
            # Price moved up, new bid level
            delta_bid_qty = bid_qty
        else:
            # Price moved down, bid level disappeared
            delta_bid_qty = -previous_bid_qty
        
        # Ask side change
        if ask_price == previous_ask_price:
            delta_ask_qty = ask_qty - previous_ask_qty
        elif ask_price < previous_ask_price:
            # Price moved down, new ask level
            delta_ask_qty = ask_qty
        else:
            # Price moved up, ask level disappeared
            delta_ask_qty = -previous_ask_qty
        
        # OFI formula
        ofi = max(0, delta_bid_qty) - max(0, delta_ask_qty)
        
        return ofi
    
    def calculate_vpin(
        self,
        buy_volume: int,
        sell_volume: int,
        total_volume: int
    ) -> float:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
        
        VPIN = |Buy Volume - Sell Volume| / Total Volume
        
        Args:
            buy_volume: Buy volume
            sell_volume: Sell volume
            total_volume: Total volume
            
        Returns:
            VPIN value
        """
        if total_volume == 0:
            return 0.0
        
        vpin = abs(buy_volume - sell_volume) / total_volume
        return vpin
    
    def calculate_order_book_imbalance(
        self,
        bid_qty: int,
        ask_qty: int
    ) -> float:
        """
        Calculate order book imbalance.
        
        Imbalance = (Bid Qty - Ask Qty) / (Bid Qty + Ask Qty)
        
        Args:
            bid_qty: Total bid quantity
            ask_qty: Total ask quantity
            
        Returns:
            Imbalance value (-1 to 1)
        """
        total_qty = bid_qty + ask_qty
        if total_qty == 0:
            return 0.0
        
        imbalance = (bid_qty - ask_qty) / total_qty
        return imbalance
    
    def calculate_depth_imbalance(
        self,
        bid_levels: int,
        ask_levels: int
    ) -> float:
        """
        Calculate depth imbalance.
        
        Args:
            bid_levels: Number of bid levels
            ask_levels: Number of ask levels
            
        Returns:
            Depth imbalance (-1 to 1)
        """
        total_levels = bid_levels + ask_levels
        if total_levels == 0:
            return 0.0
        
        imbalance = (bid_levels - ask_levels) / total_levels
        return imbalance
    
    def compute_features_from_order_book(
        self,
        symbol: str,
        timestamp: datetime,
        bid_price: float,
        bid_qty: int,
        ask_price: float,
        ask_qty: int,
        trade_price: Optional[float] = None,
        trade_qty: Optional[int] = None,
        previous_state: Optional[Dict] = None
    ) -> MicrostructureFeatures:
        """
        Compute microstructure features from order book snapshot.
        
        Args:
            symbol: Stock symbol
            timestamp: Timestamp
            bid_price: Best bid price
            bid_qty: Best bid quantity
            ask_price: Best ask price
            ask_qty: Best ask quantity
            trade_price: Trade price (optional)
            trade_qty: Trade quantity (optional)
            previous_state: Previous order book state (optional)
            
        Returns:
            MicrostructureFeatures
        """
        # Calculate basic metrics
        mid_price = (bid_price + ask_price) / 2
        spread = ask_price - bid_price
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Calculate OFI
        ofi = 0.0
        if previous_state:
            ofi = self.calculate_ofi(
                bid_price, bid_qty, ask_price, ask_qty,
                previous_state.get('bid_price'),
                previous_state.get('bid_qty'),
                previous_state.get('ask_price'),
                previous_state.get('ask_qty')
            )
        
        # Calculate order book imbalance
        order_book_imbalance = self.calculate_order_book_imbalance(bid_qty, ask_qty)
        
        # Calculate VPIN (simplified, would need buy/sell classification in production)
        # For now, use order book imbalance as proxy
        vpin = abs(order_book_imbalance)
        
        # Calculate depth imbalance (simplified)
        depth_imbalance = 0.0  # Would need full depth levels
        
        # Classify tick direction
        tick_direction = TickDirection.UNCHANGED
        if trade_price and previous_state:
            previous_mid = (previous_state.get('bid_price', 0) + previous_state.get('ask_price', 0)) / 2
            tick_direction = self.classify_tick_direction(
                trade_price,
                previous_state.get('trade_price', trade_price),
                mid_price,
                previous_mid
            )
        
        # Volume classification (simplified)
        total_volume = trade_qty if trade_qty else 0
        buy_volume = total_volume // 2 if tick_direction == TickDirection.UP else 0
        sell_volume = total_volume // 2 if tick_direction == TickDirection.DOWN else 0
        
        features = MicrostructureFeatures(
            symbol=symbol,
            timestamp=timestamp,
            ofi=ofi,
            vpin=vpin,
            order_book_imbalance=order_book_imbalance,
            spread=spread,
            spread_bps=spread_bps,
            depth_imbalance=depth_imbalance,
            tick_direction=tick_direction,
            mid_price=mid_price,
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume
        )
        
        return features
    
    def compute_rolling_features(
        self,
        symbol: str,
        features: MicrostructureFeatures
    ) -> Dict[str, float]:
        """
        Compute rolling window features.
        
        Args:
            symbol: Stock symbol
            features: Current microstructure features
            
        Returns:
            Dict with rolling features
        """
        rolling_features = {}
        
        # Initialize history if needed
        if symbol not in self.ofi_history:
            self.ofi_history[symbol] = deque(maxlen=self.ofi_window)
        
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.vpin_window)
        
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.vpin_window)
        
        # Add current values
        self.ofi_history[symbol].append(features.ofi)
        self.price_history[symbol].append(features.mid_price)
        self.volume_history[symbol].append(features.total_volume)
        
        # Calculate rolling OFI
        if len(self.ofi_history[symbol]) > 1:
            rolling_features['ofi_mean'] = np.mean(self.ofi_history[symbol])
            rolling_features['ofi_std'] = np.std(self.ofi_history[symbol])
            rolling_features['ofi_sum'] = np.sum(self.ofi_history[symbol])
        else:
            rolling_features['ofi_mean'] = features.ofi
            rolling_features['ofi_std'] = 0.0
            rolling_features['ofi_sum'] = features.ofi
        
        # Calculate rolling VPIN
        if len(self.volume_history[symbol]) > 1:
            volumes = list(self.volume_history[symbol])
            buy_vols = [v // 2 for v in volumes]  # Simplified
            sell_vols = [v // 2 for v in volumes]
            
            rolling_vpin = 0.0
            for bv, sv in zip(buy_vols, sell_vols):
                rolling_vpin += self.calculate_vpin(bv, sv, bv + sv)
            
            rolling_features['vpin_mean'] = rolling_vpin / len(volumes)
        else:
            rolling_features['vpin_mean'] = features.vpin
        
        # Calculate price volatility
        if len(self.price_history[symbol]) > 1:
            prices = list(self.price_history[symbol])
            returns = np.diff(np.log(prices))
            rolling_features['price_volatility'] = np.std(returns) if len(returns) > 0 else 0.0
        else:
            rolling_features['price_volatility'] = 0.0
        
        return rolling_features
    
    def compute_features_dataframe(
        self,
        order_book_data: pd.DataFrame,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Compute microstructure features for a DataFrame of order book data.
        
        Args:
            order_book_data: DataFrame with order book data
            symbol_col: Symbol column name
            timestamp_col: Timestamp column name
            
        Returns:
            DataFrame with microstructure features
        """
        features_list = []
        previous_states = {}
        
        # Group by symbol
        for symbol in order_book_data[symbol_col].unique():
            symbol_data = order_book_data[order_book_data[symbol_col] == symbol].copy()
            symbol_data = symbol_data.sort_values(timestamp_col).reset_index(drop=True)
            
            for idx, row in symbol_data.iterrows():
                # Get previous state
                prev_state = previous_states.get(symbol)
                
                # Compute features
                features = self.compute_features_from_order_book(
                    symbol=symbol,
                    timestamp=row[timestamp_col],
                    bid_price=row.get('bid_price_0', 0),
                    bid_qty=row.get('bid_qty_0', 0),
                    ask_price=row.get('ask_price_0', 0),
                    ask_qty=row.get('ask_qty_0', 0),
                    trade_price=row.get('last_trade_price'),
                    trade_qty=row.get('last_trade_qty'),
                    previous_state=prev_state
                )
                
                # Compute rolling features
                rolling_features = self.compute_rolling_features(symbol, features)
                
                # Combine features
                feature_dict = features.to_dict()
                feature_dict.update(rolling_features)
                
                features_list.append(feature_dict)
                
                # Update previous state
                previous_states[symbol] = {
                    'bid_price': row.get('bid_price_0', 0),
                    'bid_qty': row.get('bid_qty_0', 0),
                    'ask_price': row.get('ask_price_0', 0),
                    'ask_qty': row.get('ask_qty_0', 0),
                    'trade_price': row.get('last_trade_price')
                }
        
        return pd.DataFrame(features_list)
    
    def print_feature_report(self, features_df: pd.DataFrame) -> None:
        """Print feature statistics report."""
        print("\n" + "="*60)
        print("MICROSTRUCTURE FEATURES REPORT")
        print("="*60)
        
        print(f"\nTotal feature rows: {len(features_df)}")
        print(f"Symbols: {features_df['symbol'].nunique()}")
        
        # OFI statistics
        if 'ofi' in features_df.columns:
            print(f"\nOFI Statistics:")
            print(f"  Mean: {features_df['ofi'].mean():.4f}")
            print(f"  Std: {features_df['ofi'].std():.4f}")
            print(f"  Min: {features_df['ofi'].min():.4f}")
            print(f"  Max: {features_df['ofi'].max():.4f}")
        
        # VPIN statistics
        if 'vpin' in features_df.columns:
            print(f"\nVPIN Statistics:")
            print(f"  Mean: {features_df['vpin'].mean():.4f}")
            print(f"  Std: {features_df['vpin'].std():.4f}")
            print(f"  Min: {features_df['vpin'].min():.4f}")
            print(f"  Max: {features_df['vpin'].max():.4f}")
        
        # Spread statistics
        if 'spread_bps' in features_df.columns:
            print(f"\nSpread Statistics (bps):")
            print(f"  Mean: {features_df['spread_bps'].mean():.2f}")
            print(f"  Std: {features_df['spread_bps'].std():.2f}")
            print(f"  Min: {features_df['spread_bps'].min():.2f}")
            print(f"  Max: {features_df['spread_bps'].max():.2f}")
        
        # Tick direction distribution
        if 'tick_direction' in features_df.columns:
            print(f"\nTick Direction Distribution:")
            for direction in features_df['tick_direction'].unique():
                count = (features_df['tick_direction'] == direction).sum()
                pct = count / len(features_df) * 100
                print(f"  {direction}: {count} ({pct:.1f}%)")
        
        print("\n" + "="*60)


def sample_microstructure_features():
    """Demonstrate microstructure feature calculation."""
    print("=== Microstructure Features Demo ===\n")
    
    # Initialize calculator
    calculator = MicrostructureFeatureCalculator(ofi_window=100, vpin_window=100)
    
    # Sample order book data
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    base_price = 2500.0
    
    for i in range(n_samples):
        timestamp = datetime(2024, 1, 1, 9, 15) + timedelta(milliseconds=i * 100)
        
        # Random walk for price
        base_price += np.random.normal(0, 0.1)
        
        # Generate bid/ask
        bid_price = base_price - np.random.uniform(0.01, 0.05)
        ask_price = base_price + np.random.uniform(0.01, 0.05)
        bid_qty = int(np.random.uniform(100, 10000))
        ask_qty = int(np.random.uniform(100, 10000))
        
        # Trade
        trade_price = base_price + np.random.normal(0, 0.02)
        trade_qty = int(np.random.uniform(10, 1000))
        
        data.append({
            'symbol': 'RELIANCE',
            'timestamp': timestamp,
            'bid_price_0': bid_price,
            'bid_qty_0': bid_qty,
            'ask_price_0': ask_price,
            'ask_qty_0': ask_qty,
            'last_trade_price': trade_price,
            'last_trade_qty': trade_qty
        })
    
    df = pd.DataFrame(data)
    
    # Compute features
    print("Computing microstructure features...")
    features_df = calculator.compute_features_dataframe(df)
    
    # Print report
    calculator.print_feature_report(features_df)
    
    # Show sample features
    print("\nSample features (first 5 rows):")
    print(features_df[['timestamp', 'ofi', 'vpin', 'spread_bps', 'tick_direction']].head())
    
    print("\n=== Microstructure Features Demo Complete ===")
    print("Key capabilities:")
    print("- Order Flow Imbalance (OFI) calculation")
    print("- VPIN (Volume-Synchronized Probability of Informed Trading)")
    print("- Order book imbalance features")
    print("- Tick-direction classification (Lee-Ready)")
    print("- Rolling window features")


if __name__ == "__main__":
    sample_microstructure_features()

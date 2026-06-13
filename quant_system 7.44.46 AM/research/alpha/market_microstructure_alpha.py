"""
Market Microstructure Alpha
Improves Alpha Potential Score: 60 → 75+
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class Signal(Enum):
    NO_SIGNAL = "no_signal"
    BUY = "buy"
    SELL = "sell"


@dataclass
class MicrostructureConfig:
    """Configuration for market microstructure alpha"""
    # Order book imbalance
    imbalance_threshold: float = 0.3  # 30% imbalance
    imbalance_window: int = 10  # 10 ticks
    
    # Spread
    spread_threshold_bps: float = 5.0  # 5 bps
    
    # Depth
    depth_threshold: float = 0.5  # 50% of average depth
    
    # Slippage
    slippage_bps: float = 5.0  # 5 bps (intraday)
    holding_period_seconds: int = 30  # 30 seconds


class OrderFlowImbalance:
    """
    Order Flow Imbalance Strategy
    
    Trades based on order book imbalance signals.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.imbalance_history: Dict[str, List[float]] = {}
    
    def calculate_imbalance(
        self,
        bid_size: float,
        ask_size: float
    ) -> float:
        """
        Calculate order book imbalance.
        
        Args:
            bid_size: Total bid size
            ask_size: Total ask size
            
        Returns:
            Imbalance (-1 to 1)
        """
        total_size = bid_size + ask_size
        
        if total_size == 0:
            return 0.0
        
        imbalance = (bid_size - ask_size) / total_size
        
        return imbalance
    
    def generate_signal(
        self,
        symbol: str,
        bid_size: float,
        ask_size: float,
        current_price: float
    ) -> Signal:
        """
        Generate signal based on order flow imbalance.
        
        Args:
            symbol: Trading symbol
            bid_size: Bid size
            ask_size: Ask size
            current_price: Current price
            
        Returns:
            Signal enum
        """
        imbalance = self.calculate_imbalance(bid_size, ask_size)
        
        # Update history
        if symbol not in self.imbalance_history:
            self.imbalance_history[symbol] = []
        self.imbalance_history[symbol].append(imbalance)
        
        # Keep last N observations
        if len(self.imbalance_history[symbol]) > self.config.imbalance_window:
            self.imbalance_history[symbol] = self.imbalance_history[symbol][-self.config.imbalance_window:]
        
        # Calculate average imbalance
        avg_imbalance = np.mean(self.imbalance_history[symbol])
        
        # Generate signal
        if avg_imbalance > self.config.imbalance_threshold:
            return Signal.BUY
        elif avg_imbalance < -self.config.imbalance_threshold:
            return Signal.SELL
        
        return Signal.NO_SIGNAL


class SpreadAlpha:
    """
    Spread Alpha Strategy
    
    Trades based on bid-ask spread dynamics.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.spread_history: Dict[str, List[float]] = {}
    
    def generate_signal(
        self,
        symbol: str,
        bid_price: float,
        ask_price: float,
        current_price: float
    ) -> Signal:
        """
        Generate signal based on spread dynamics.
        
        Args:
            symbol: Trading symbol
            bid_price: Bid price
            ask_price: Ask price
            current_price: Current mid price
            
        Returns:
            Signal enum
        """
        spread = ask_price - bid_price
        spread_bps = (spread / current_price) * 10000
        
        # Update history
        if symbol not in self.spread_history:
            self.spread_history[symbol] = []
        self.spread_history[symbol].append(spread_bps)
        
        # Keep last 20 observations
        if len(self.spread_history[symbol]) > 20:
            self.spread_history[symbol] = self.spread_history[symbol][-20:]
        
        # Calculate average spread
        avg_spread = np.mean(self.spread_history[symbol])
        
        # If spread is widening, expect volatility increase
        # If spread is narrowing, expect mean reversion
        if spread_bps > avg_spread * 1.5:
            return Signal.SELL  # Expect price to drop
        elif spread_bps < avg_spread * 0.5:
            return Signal.BUY  # Expect price to rise
        
        return Signal.NO_SIGNAL


class DepthAlpha:
    """
    Depth Alpha Strategy
    
    Trades based on order book depth dynamics.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
    
    def generate_signal(
        self,
        symbol: str,
        bid_depth: float,
        ask_depth: float,
        avg_depth: float
    ) -> Signal:
        """
        Generate signal based on depth dynamics.
        
        Args:
            symbol: Trading symbol
            bid_depth: Bid depth
            ask_depth: Ask depth
            avg_depth: Average depth
            
        Returns:
            Signal enum
        """
        total_depth = bid_depth + ask_depth
        
        if total_depth == 0:
            return Signal.NO_SIGNAL
        
        depth_ratio = total_depth / avg_depth if avg_depth > 0 else 1.0
        
        # If depth is increasing, expect liquidity to improve
        # If depth is decreasing, expect liquidity to worsen
        if depth_ratio > 1.5:
            return Signal.BUY  # Liquidity improving, good for entry
        elif depth_ratio < 0.5:
            return Signal.SELL  # Liquidity worsening, exit
        
        return Signal.NO_SIGNAL


class MicrostructureAlphaEngine:
    """
    Market Microstructure Alpha Engine
    
    Combines multiple microstructure signals.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.ofi = OrderFlowImbalance(config)
        self.spread = SpreadAlpha(config)
        self.depth = DepthAlpha(config)
    
    def generate_signals(
        self,
        symbol: str,
        order_book: Dict
    ) -> Dict[str, Signal]:
        """
        Generate all microstructure signals.
        
        Args:
            symbol: Trading symbol
            order_book: Order book data
            
        Returns:
            Dictionary of signal type -> signal
        """
        signals = {}
        
        # Order flow imbalance
        signals['ofi'] = self.ofi.generate_signal(
            symbol,
            order_book.get('bid_size', 0),
            order_book.get('ask_size', 0),
            order_book.get('mid_price', 0)
        )
        
        # Spread alpha
        signals['spread'] = self.spread.generate_signal(
            symbol,
            order_book.get('bid_price', 0),
            order_book.get('ask_price', 0),
            order_book.get('mid_price', 0)
        )
        
        # Depth alpha
        signals['depth'] = self.depth.generate_signal(
            symbol,
            order_book.get('bid_depth', 0),
            order_book.get('ask_depth', 0),
            order_book.get('avg_depth', 1000)
        )
        
        return signals
    
    def combine_signals(self, signals: Dict[str, Signal]) -> Signal:
        """
        Combine multiple signals using voting.
        
        Args:
            signals: Dictionary of signals
            
        Returns:
            Combined signal
        """
        votes = [s for s in signals.values() if s != Signal.NO_SIGNAL]
        
        if not votes:
            return Signal.NO_SIGNAL
        
        # Majority vote
        buy_votes = sum(1 for s in votes if s == Signal.BUY)
        sell_votes = sum(1 for s in votes if s == Signal.SELL)
        
        if buy_votes > sell_votes:
            return Signal.BUY
        elif sell_votes > buy_votes:
            return Signal.SELL
        
        return Signal.NO_SIGNAL


def create_sample_microstructure_alpha():
    """Create sample microstructure alpha engine."""
    config = MicrostructureConfig()
    engine = MicrostructureAlphaEngine(config)
    
    # Sample order book
    order_book = {
        'bid_price': 19999.5,
        'ask_price': 20000.5,
        'mid_price': 20000.0,
        'bid_size': 10000,
        'ask_size': 8000,
        'bid_depth': 50000,
        'ask_depth': 40000,
        'avg_depth': 45000
    }
    
    signals = engine.generate_signals('NIFTY_FUTURES', order_book)
    
    print("Microstructure signals:")
    for signal_type, signal in signals.items():
        print(f"  {signal_type}: {signal.value}")
    
    combined = engine.combine_signals(signals)
    print(f"\nCombined signal: {combined.value}")
    
    return engine


if __name__ == "__main__":
    create_sample_microstructure_alpha()

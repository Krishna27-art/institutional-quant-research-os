"""
Microstructure Strategies

Implements institutional-grade market microstructure strategies:
- Market Making with Inventory Control (Avellaneda & Stoikov 2008)
- Order Flow Imbalance
- Signal-Adaptive Quoting (Yu 2026) - already in execution/

Based on blueprint specification for multi-strategy framework
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureSignal:
    """Microstructure trading signal"""
    symbol: str
    strategy: str
    bid_price: float
    ask_price: float
    inventory: int
    signal: float
    confidence: float
    metadata: Dict = None


class MarketMakingInventory:
    """
    Market Making with Inventory Control (Avellaneda & Stoikov 2008)
    
    Formula:
    Reservation price r = s - qγσ²τ
    Quote at r ± spread/2 + k
    
    where:
    - s = mid price
    - q = inventory
    - γ = risk aversion
    - σ = volatility
    - τ = time to next quote
    - k = risk parameter for spread
    
    Expected Sharpe: 0.8
    Capacity: 100 Cr
    Turnover: 10000%/month
    Best Regime: Liquid
    Failure: Illiquid
    """
    
    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 0.2,
        tau: float = 0.01,
        k: float = 0.5,
        max_inventory: int = 100
    ):
        """
        Initialize market making with inventory control.
        
        Args:
            gamma: Risk aversion coefficient
            sigma: Volatility
            tau: Time to next quote (in years)
            k: Risk parameter for spread
            max_inventory: Maximum inventory limit
        """
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.k = k
        self.max_inventory = max_inventory
        self.inventory = 0
        
    def reservation_price(self, mid_price: float, inventory: int) -> float:
        """
        Calculate reservation price.
        
        Formula: r = s - qγσ²τ
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory (positive = long, negative = short)
            
        Returns:
            Reservation price
        """
        return mid_price - inventory * self.gamma * (self.sigma ** 2) * self.tau
    
    def optimal_spread(self, mid_price: float) -> float:
        """
        Calculate optimal spread.
        
        Formula: spread = 2k
        
        Args:
            mid_price: Current mid price
            
        Returns:
            Optimal spread
        """
        return 2 * self.k
    
    def get_quotes(
        self,
        mid_price: float,
        inventory: int
    ) -> Tuple[float, float]:
        """
        Get optimal bid and ask quotes.
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Calculate reservation price
        r = self.reservation_price(mid_price, inventory)
        
        # Calculate spread
        spread = self.optimal_spread(mid_price)
        
        # Calculate bid and ask
        bid = r - spread / 2
        ask = r + spread / 2
        
        return bid, ask
    
    def update_inventory(self, trade_side: str, quantity: int):
        """
        Update inventory after trade.
        
        Args:
            trade_side: 'buy' or 'sell'
            quantity: Trade quantity
        """
        if trade_side == 'buy':
            self.inventory += quantity
        elif trade_side == 'sell':
            self.inventory -= quantity
        
        # Clamp to max inventory
        self.inventory = max(-self.max_inventory, min(self.max_inventory, self.inventory))
    
    def should_trade(self, inventory: int) -> bool:
        """
        Check if should continue trading given inventory.
        
        Args:
            inventory: Current inventory
            
        Returns:
            True if should trade
        """
        # Stop if at inventory limits
        if abs(inventory) >= self.max_inventory:
            return False
        
        return True


class OrderFlowImbalance:
    """
    Order Flow Imbalance Strategy
    
    Trades based on order flow imbalance (OFI).
    OFI = (buy_volume - sell_volume) / total_volume
    """
    
    def __init__(
        self,
        lookback: int = 100,
        threshold: float = 0.3
    ):
        """
        Initialize order flow imbalance strategy.
        
        Args:
            lookback: Lookback period for OFI calculation
            threshold: OFI threshold for entry
        """
        self.lookback = lookback
        self.threshold = threshold
        
    def compute_ofi(
        self,
        buy_volume: pd.Series,
        sell_volume: pd.Series
    ) -> pd.Series:
        """
        Compute order flow imbalance.
        
        Args:
            buy_volume: Series of buy volumes
            sell_volume: Series of sell volumes
            
        Returns:
            OFI series
        """
        total_volume = buy_volume + sell_volume
        ofi = (buy_volume - sell_volume) / (total_volume + 1e-8)
        return ofi
    
    def get_signal(
        self,
        ofi: float,
        ofi_ma: float
    ) -> Tuple[float, str]:
        """
        Get trading signal from OFI.
        
        Args:
            ofi: Current OFI
            ofi_ma: OFI moving average
            
        Returns:
            Tuple of (signal, direction)
        """
        ofi_diff = ofi - ofi_ma
        
        if ofi_diff > self.threshold:
            # Strong buying pressure - long
            signal = 1.0
            direction = "LONG"
        elif ofi_diff < -self.threshold:
            # Strong selling pressure - short
            signal = -1.0
            direction = "SHORT"
        else:
            signal = 0.0
            direction = "HOLD"
        
        return signal, direction


class SpreadCapture:
    """
    Spread Capture Strategy
    
    Captures bid-ask spread by providing liquidity.
    """
    
    def __init__(
        self,
        min_spread: float = 0.001,
        target_capture: float = 0.5
    ):
        """
        Initialize spread capture strategy.
        
        Args:
            min_spread: Minimum spread to trade
            target_capture: Target fraction of spread to capture
        """
        self.min_spread = min_spread
        self.target_capture = target_capture
        
    def get_quotes(
        self,
        bid: float,
        ask: float,
        mid_price: float
    ) -> Tuple[float, float]:
        """
        Get quotes to capture spread.
        
        Args:
            bid: Current best bid
            ask: Current best ask
            mid_price: Mid price
            
        Returns:
            Tuple of (our_bid, our_ask)
        """
        spread = ask - bid
        spread_pct = spread / mid_price
        
        if spread_pct < self.min_spread:
            # Spread too small, don't trade
            return bid, ask
        
        # Place quotes inside the spread
        capture = spread * self.target_capture
        our_bid = bid + capture / 2
        our_ask = ask - capture / 2
        
        return our_bid, our_ask


if __name__ == "__main__":
    # Test microstructure strategies
    print("Testing Microstructure Strategies...")
    
    # Test Market Making with Inventory Control
    print("\n1. Market Making with Inventory Control:")
    mm = MarketMakingInventory(gamma=0.1, sigma=0.2, tau=0.01, k=0.5)
    
    mid_price = 100.0
    inventory = 10
    
    bid, ask = mm.get_quotes(mid_price, inventory)
    print(f"   Mid price: {mid_price:.2f}")
    print(f"   Inventory: {inventory}")
    print(f"   Bid: {bid:.2f}")
    print(f"   Ask: {ask:.2f}")
    print(f"   Spread: {ask - bid:.2f}")
    
    # Test Order Flow Imbalance
    print("\n2. Order Flow Imbalance:")
    ofi_strat = OrderFlowImbalance()
    
    buy_vol = pd.Series([100, 150, 200, 180, 220])
    sell_vol = pd.Series([80, 120, 150, 170, 160])
    
    ofi = ofi_strat.compute_ofi(buy_vol, sell_vol)
    ofi_ma = ofi.rolling(3).mean().iloc[-1]
    signal, direction = ofi_strat.get_signal(ofi.iloc[-1], ofi_ma)
    
    print(f"   OFI: {ofi.iloc[-1]:.4f}")
    print(f"   OFI MA: {ofi_ma:.4f}")
    print(f"   Signal: {signal:.2f}")
    print(f"   Direction: {direction}")
    
    # Test Spread Capture
    print("\n3. Spread Capture:")
    spread_cap = SpreadCapture()
    
    bid = 99.95
    ask = 100.05
    mid = 100.0
    
    our_bid, our_ask = spread_cap.get_quotes(bid, ask, mid)
    print(f"   Market bid: {bid:.2f}")
    print(f"   Market ask: {ask:.2f}")
    print(f"   Our bid: {our_bid:.2f}")
    print(f"   Our ask: {our_ask:.2f}")
    
    print("\n✓ All microstructure strategies tested")

"""
Market Microstructure Alpha

Generates alpha signals from order flow, bid-ask spread, and
liquidity patterns in the market microstructure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MicrostructureSignal(Enum):
    """Types of microstructure signals."""
    ORDER_FLOW_IMBALANCE = "order_flow_imbalance"
    BID_ASK_SQUEEZE = "bid_ask_squeeze"
    LIQUIDITY_DRAIN = "liquidity_drain"
    NO_SIGNAL = "no_signal"


@dataclass
class MicrostructureAlpha:
    """Alpha signal from market microstructure."""
    symbol: str
    signal_type: MicrostructureSignal
    direction: float  # -1 to 1
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    timestamp: pd.Timestamp
    metadata: Dict


class MarketMicrostructureAlpha:
    """
    Market microstructure alpha generator.
    
    Uses order book data to generate signals based on:
    - Order flow imbalance
    - Bid-ask spread dynamics
    - Liquidity patterns
    """
    
    def __init__(
        self,
        ofi_threshold: float = 0.3,  # Order flow imbalance threshold
        spread_threshold: float = 0.001,  # Spread threshold (0.1%)
        liquidity_threshold: float = 0.5  # Liquidity drain threshold
    ):
        self.ofi_threshold = ofi_threshold
        self.spread_threshold = spread_threshold
        self.liquidity_threshold = liquidity_threshold
        
    def calculate_order_flow_imbalance(
        self,
        bid_volume: float,
        ask_volume: float
    ) -> float:
        """
        Calculate order flow imbalance.
        
        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Args:
            bid_volume: Total bid volume
            ask_volume: Total ask volume
            
        Returns:
            OFI value between -1 and 1
        """
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        ofi = (bid_volume - ask_volume) / total_volume
        return ofi
    
    def calculate_spread_pressure(
        self,
        bid_price: float,
        ask_price: float,
        mid_price: float
    ) -> float:
        """
        Calculate spread pressure.
        
        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            mid_price: Mid price
            
        Returns:
            Spread pressure value
        """
        spread = (ask_price - bid_price) / mid_price
        return spread
    
    def calculate_liquidity_score(
        self,
        bid_volume: float,
        ask_volume: float,
        total_volume: float
    ) -> float:
        """
        Calculate liquidity score.
        
        Args:
            bid_volume: Bid volume
            ask_volume: Ask volume
            total_volume: Total volume
            
        Returns:
            Liquidity score between 0 and 1
        """
        if total_volume == 0:
            return 0.0
        
        # Liquidity score based on depth
        depth_score = (bid_volume + ask_volume) / total_volume
        return min(1.0, depth_score)
    
    def generate_signal(
        self,
        symbol: str,
        order_book: Dict[str, any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Optional[MicrostructureAlpha]:
        """
        Generate microstructure alpha signal.
        
        Args:
            symbol: Stock symbol
            order_book: Current order book data
            historical_data: Optional historical data for context
            
        Returns:
            MicrostructureAlpha signal or None
        """
        try:
            # Extract order book data
            bid_price = order_book.get('bid_price', 0)
            ask_price = order_book.get('ask_price', 0)
            bid_volume = order_book.get('bid_volume', 0)
            ask_volume = order_book.get('ask_volume', 0)
            total_volume = order_book.get('total_volume', 1)
            
            if bid_price == 0 or ask_price == 0:
                return None
            
            mid_price = (bid_price + ask_price) / 2
            
            # Calculate metrics
            ofi = self.calculate_order_flow_imbalance(bid_volume, ask_volume)
            spread = self.calculate_spread_pressure(bid_price, ask_price, mid_price)
            liquidity = self.calculate_liquidity_score(bid_volume, ask_volume, total_volume)
            
            # Determine signal type
            signal_type = MicrostructureSignal.NO_SIGNAL
            direction = 0.0
            strength = 0.0
            confidence = 0.0
            
            # Order flow imbalance signal
            if ofi > self.ofi_threshold:
                signal_type = MicrostructureSignal.ORDER_FLOW_IMBALANCE
                direction = 1.0  # Buy signal
                strength = min(1.0, (ofi - self.ofi_threshold) / (1 - self.ofi_threshold))
                confidence = min(0.9, ofi)
            elif ofi < -self.ofi_threshold:
                signal_type = MicrostructureSignal.ORDER_FLOW_IMBALANCE
                direction = -1.0  # Sell signal
                strength = min(1.0, (abs(ofi) - self.ofi_threshold) / (1 - self.ofi_threshold))
                confidence = min(0.9, abs(ofi))
            
            # Bid-ask squeeze signal
            elif spread < self.spread_threshold and liquidity > 0.8:
                signal_type = MicrostructureSignal.BID_ASK_SQUEEZE
                direction = 1.0  # Tight spread indicates buying pressure
                strength = min(1.0, (self.spread_threshold - spread) / self.spread_threshold)
                confidence = min(0.9, liquidity)
            
            # Liquidity drain signal
            elif liquidity < self.liquidity_threshold:
                signal_type = MicrostructureSignal.LIQUIDITY_DRAIN
                direction = -1.0  # Low liquidity is risky
                strength = min(1.0, (self.liquidity_threshold - liquidity) / self.liquidity_threshold)
                confidence = min(0.9, 1 - liquidity)
            
            if signal_type != MicrostructureSignal.NO_SIGNAL:
                return MicrostructureAlpha(
                    symbol=symbol,
                    signal_type=signal_type,
                    direction=direction,
                    strength=strength,
                    confidence=confidence,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'ofi': ofi,
                        'spread': spread,
                        'liquidity': liquidity,
                        'bid_price': bid_price,
                        'ask_price': ask_price
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate microstructure signal for {symbol}: {e}")
            return None


# Singleton instance
_microstructure_alpha = None

def get_microstructure_alpha() -> MarketMicrostructureAlpha:
    """Get the singleton microstructure alpha instance."""
    global _microstructure_alpha
    if _microstructure_alpha is None:
        _microstructure_alpha = MarketMicrostructureAlpha()
    return _microstructure_alpha


if __name__ == "__main__":
    # Test market microstructure alpha
    print("Testing Market Microstructure Alpha...")
    
    alpha = MarketMicrostructureAlpha()
    
    # Create sample order book
    order_book = {
        'bid_price': 100.0,
        'ask_price': 100.05,
        'bid_volume': 10000,
        'ask_volume': 5000,
        'total_volume': 100000
    }
    
    signal = alpha.generate_signal('TEST', order_book)
    if signal:
        print(f"Signal: {signal.signal_type}")
        print(f"Direction: {signal.direction}")
        print(f"Strength: {signal.strength:.2f}")
        print(f"Confidence: {signal.confidence:.2f}")
    else:
        print("No signal generated")

"""
Market Microstructure Alpha

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#32)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Order Flow Imbalance (OFI)
- Volume-Synchronized Probability of Informed Trading (VPIN)
- Limit Order Book (LOB) features
- Ultra-short-term alpha generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MicrostructureAlphaConfig:
    """Configuration for Microstructure Alpha"""
    # OFI parameters
    ofi_window: int = 100  # OFI calculation window
    ofi_threshold: float = 0.5  # OFI signal threshold
    
    # VPIN parameters
    vpin_bucket_size: int = 50  # Trades per VPIN bucket
    vpin_window: int = 100  # VPIN calculation window
    
    # LOB parameters
    lob_depth: int = 5  # Number of LOB levels
    lob_imbalance_threshold: float = 0.6
    
    # Signal parameters
    signal_horizon: int = 10  # Signal horizon (ticks)
    min_signal_strength: float = 0.3


class OFICalculator:
    """
    Order Flow Imbalance Calculator
    
    OFI measures the net buying/selling pressure in the market.
    """
    
    def __init__(self, config: MicrostructureAlphaConfig):
        self.config = config
        
        self.ofi_history: deque = deque(maxlen=config.ofi_window)
    
    def calculate_ofi(self, bid_prices: List[float], bid_volumes: List[int],
                      ask_prices: List[float], ask_volumes: List[int]) -> float:
        """
        Calculate Order Flow Imbalance
        
        Args:
            bid_prices: Bid prices
            bid_volumes: Bid volumes
            ask_prices: Ask prices
            ask_volumes: Ask volumes
            
        Returns:
            OFI value
        """
        if not bid_prices or not ask_prices:
            return 0.0
        
        # Calculate OFI for each level
        ofi = 0.0
        
        for i in range(min(len(bid_prices), len(ask_prices))):
            # Bid side
            if i > 0:
                delta_bid = bid_prices[i] - bid_prices[i-1]
            else:
                delta_bid = 0.0
            
            if delta_bid >= 0:
                ofi += bid_volumes[i]
            else:
                ofi -= bid_volumes[i]
            
            # Ask side
            if i > 0:
                delta_ask = ask_prices[i] - ask_prices[i-1]
            else:
                delta_ask = 0.0
            
            if delta_ask <= 0:
                ofi -= ask_volumes[i]
            else:
                ofi += ask_volumes[i]
        
        # Normalize
        total_volume = sum(bid_volumes) + sum(ask_volumes)
        if total_volume > 0:
            ofi = ofi / total_volume
        
        self.ofi_history.append(ofi)
        
        return ofi
    
    def get_ofi_signal(self) -> float:
        """Get OFI-based signal"""
        if len(self.ofi_history) < self.config.ofi_window:
            return 0.0
        
        avg_ofi = np.mean(list(self.ofi_history))
        
        # Generate signal
        if avg_ofi > self.config.ofi_threshold:
            return 1.0  # Buy signal
        elif avg_ofi < -self.config.ofi_threshold:
            return -1.0  # Sell signal
        else:
            return 0.0


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading
    
    VPIN measures the likelihood of informed trading in the market.
    """
    
    def __init__(self, config: MicrostructureAlphaConfig):
        self.config = config
        
        self.vpin_history: deque = deque(maxlen=config.vpin_window)
        self.current_bucket: List[Dict] = []
    
    def add_trade(self, price: float, volume: int, is_buy: bool) -> Optional[float]:
        """
        Add trade to current bucket
        
        Args:
            price: Trade price
            volume: Trade volume
            is_buy: Whether trade is a buy
            
        Returns:
            VPIN if bucket is complete, None otherwise
        """
        self.current_bucket.append({
            "price": price,
            "volume": volume,
            "is_buy": is_buy
        })
        
        # Check if bucket is complete
        if len(self.current_bucket) >= self.config.vpin_bucket_size:
            return self._calculate_vpin()
        
        return None
    
    def _calculate_vpin(self) -> float:
        """Calculate VPIN for current bucket"""
        # Separate buys and sells
        buy_volume = sum(t["volume"] for t in self.current_bucket if t["is_buy"])
        sell_volume = sum(t["volume"] for t in self.current_bucket if not t["is_buy"])
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        # Calculate absolute order flow
        abs_order_flow = abs(buy_volume - sell_volume)
        
        # VPIN
        vpin = abs_order_flow / total_volume
        
        self.vpin_history.append(vpin)
        self.current_bucket = []
        
        return vpin
    
    def get_vpin_signal(self) -> float:
        """Get VPIN-based signal"""
        if len(self.vpin_history) < self.config.vpin_window:
            return 0.0
        
        avg_vpin = np.mean(list(self.vpin_history))
        
        # High VPIN indicates informed trading - reduce position
        if avg_vpin > 0.3:
            return -0.5  # Reduce exposure
        else:
            return 0.0


class LOBAnalyzer:
    """
    Limit Order Book Analyzer
    
    Analyzes LOB features for alpha generation.
    """
    
    def __init__(self, config: MicrostructureAlphaConfig):
        self.config = config
    
    def calculate_lob_imbalance(self, bid_volumes: List[int], ask_volumes: List[int]) -> float:
        """
        Calculate LOB imbalance
        
        Args:
            bid_volumes: Bid volumes at each level
            ask_volumes: Ask volumes at each level
            
        Returns:
            Imbalance value (-1 to 1)
        """
        depth = min(len(bid_volumes), len(ask_volumes), self.config.lob_depth)
        
        if depth == 0:
            return 0.0
        
        total_bid = sum(bid_volumes[:depth])
        total_ask = sum(ask_volumes[:depth])
        
        total = total_bid + total_ask
        if total == 0:
            return 0.0
        
        imbalance = (total_bid - total_ask) / total
        
        return imbalance
    
    def calculate_lob_slope(self, bid_prices: List[float], ask_prices: List[float]) -> float:
        """
        Calculate LOB slope (price impact of volume)
        
        Args:
            bid_prices: Bid prices
            ask_prices: Ask prices
            
        Returns:
            Slope value
        """
        if len(bid_prices) < 2 or len(ask_prices) < 2:
            return 0.0
        
        # Bid slope
        bid_slope = (bid_prices[1] - bid_prices[0]) if len(bid_prices) > 1 else 0.0
        
        # Ask slope
        ask_slope = (ask_prices[1] - ask_prices[0]) if len(ask_prices) > 1 else 0.0
        
        return (bid_slope + ask_slope) / 2.0


class MicrostructureAlphaGenerator:
    """
    Microstructure Alpha Generator
    
    Combines OFI, VPIN, and LOB features to generate
    ultra-short-term alpha signals.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: MicrostructureAlphaConfig):
        self.config = config
        
        self.ofi_calculator = OFICalculator(config)
        self.vpin_calculator = VPINCalculator(config)
        self.lob_analyzer = LOBAnalyzer(config)
        
        # Signal history
        self.signal_history: deque = deque(maxlen=1000)
    
    def generate_signal(self, 
                       bid_prices: List[float],
                       bid_volumes: List[int],
                       ask_prices: List[float],
                       ask_volumes: List[int],
                       trade_price: float,
                       trade_volume: int,
                       is_buy: bool) -> Dict:
        """
        Generate microstructure alpha signal
        
        Args:
            bid_prices: Bid prices
            bid_volumes: Bid volumes
            ask_prices: Ask prices
            ask_volumes: Ask volumes
            trade_price: Trade price
            trade_volume: Trade volume
            is_buy: Whether trade is a buy
            
        Returns:
            Signal dictionary
        """
        # Calculate OFI
        ofi = self.ofi_calculator.calculate_ofi(bid_prices, bid_volumes, ask_prices, ask_volumes)
        ofi_signal = self.ofi_calculator.get_ofi_signal()
        
        # Calculate VPIN
        vpin = self.vpin_calculator.add_trade(trade_price, trade_volume, is_buy)
        vpin_signal = self.vpin_calculator.get_vpin_signal()
        
        # Calculate LOB features
        lob_imbalance = self.lob_analyzer.calculate_lob_imbalance(bid_volumes, ask_volumes)
        lob_slope = self.lob_analyzer.calculate_lob_slope(bid_prices, ask_prices)
        
        # Combine signals
        combined_signal = ofi_signal + vpin_signal
        
        # Add LOB contribution
        if abs(lob_imbalance) > self.config.lob_imbalance_threshold:
            combined_signal += np.sign(lob_imbalance) * 0.3
        
        # Clip signal
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        
        # Check minimum strength
        if abs(combined_signal) < self.config.min_signal_strength:
            combined_signal = 0.0
        
        signal = {
            "signal": combined_signal,
            "ofi": ofi,
            "ofi_signal": ofi_signal,
            "vpin": vpin if vpin is not None else 0.0,
            "vpin_signal": vpin_signal,
            "lob_imbalance": lob_imbalance,
            "lob_slope": lob_slope,
            "timestamp": datetime.now()
        }
        
        self.signal_history.append(signal)
        
        return signal
    
    def get_signal_summary(self) -> Dict:
        """Get signal summary statistics"""
        if not self.signal_history:
            return {}
        
        signals = [s["signal"] for s in self.signal_history]
        
        return {
            "num_signals": len(signals),
            "mean_signal": np.mean(signals),
            "std_signal": np.std(signals),
            "positive_signals": sum(1 for s in signals if s > 0),
            "negative_signals": sum(1 for s in signals if s < 0),
            "zero_signals": sum(1 for s in signals if s == 0)
        }


def simulate_lob_data(n_ticks: int = 1000) -> Tuple[List[float], List[int], List[float], List[int]]:
    """Simulate LOB data for testing"""
    np.random.seed(42)
    
    base_price = 100.0
    
    bid_prices = []
    bid_volumes = []
    ask_prices = []
    ask_volumes = []
    
    for _ in range(n_ticks):
        # Random walk price
        base_price += np.random.randn() * 0.01
        
        # Generate LOB
        current_bid_prices = [base_price - i * 0.01 for i in range(5)]
        current_ask_prices = [base_price + i * 0.01 for i in range(5)]
        
        current_bid_volumes = [int(np.random.exponential(100)) for _ in range(5)]
        current_ask_volumes = [int(np.random.exponential(100)) for _ in range(5)]
        
        bid_prices.append(current_bid_prices)
        bid_volumes.append(current_bid_volumes)
        ask_prices.append(current_ask_prices)
        ask_volumes.append(current_ask_volumes)
    
    return bid_prices, bid_volumes, ask_prices, ask_volumes


if __name__ == "__main__":
    # Example usage
    config = MicrostructureAlphaConfig(
        ofi_window=100,
        vpin_bucket_size=50,
        lob_depth=5
    )
    
    generator = MicrostructureAlphaGenerator(config)
    
    # Simulate LOB data
    print("Simulating LOB data...")
    bid_prices, bid_volumes, ask_prices, ask_volumes = simulate_lob_data(1000)
    
    # Generate signals
    print("\nGenerating microstructure signals...")
    for i in range(100):
        trade_price = (bid_prices[i][0] + ask_prices[i][0]) / 2
        trade_volume = int(np.random.exponential(100))
        is_buy = np.random.random() > 0.5
        
        signal = generator.generate_signal(
            bid_prices[i], bid_volumes[i],
            ask_prices[i], ask_volumes[i],
            trade_price, trade_volume, is_buy
        )
        
        if i % 20 == 0 and signal["signal"] != 0:
            print(f"  Tick {i}: Signal={signal['signal']:.2f}, OFI={signal['ofi']:.4f}, VPIN={signal['vpin']:.4f}")
    
    # Signal summary
    print("\nSignal Summary:")
    summary = generator.get_signal_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

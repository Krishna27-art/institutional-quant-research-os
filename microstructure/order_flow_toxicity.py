"""
Order Flow Toxicity

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#33)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Order flow toxicity detection
- Adverse selection risk
- Used to avoid toxic order flow
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
class ToxicityConfig:
    """Configuration for Order Flow Toxicity"""
    # VPIN parameters
    bucket_size: int = 50  # Trades per bucket
    n_buckets: int = 100  # Number of buckets for VPIN calculation
    
    # Toxicity thresholds
    vpin_threshold: float = 0.3  # VPIN threshold for toxicity
    toxicity_threshold: float = 0.5  # General toxicity threshold
    
    # Order imbalance parameters
    imbalance_window: int = 100  # Window for imbalance calculation
    imbalance_threshold: float = 0.7  # Imbalance threshold
    
    # Risk parameters
    max_toxicity_exposure: float = 0.2  # Maximum exposure during toxicity


class OrderFlowToxicity:
    """
    Order Flow Toxicity Detector
    
    Detects toxic order flow using VPIN and other metrics.
    Used to avoid adverse selection risk.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: ToxicityConfig):
        self.config = config
        
        # Trade buckets
        self.current_bucket: List[Dict] = []
        self.bucket_history: deque = deque(maxlen=config.n_buckets)
        
        # Toxicity history
        self.toxicity_history: deque = deque(maxlen=100)
        
        # Order imbalance history
        self.imbalance_history: deque = deque(maxlen=config.imbalance_window)
    
    def add_trade(self, price: float, volume: int, is_buy: bool, timestamp: datetime) -> Optional[float]:
        """
        Add trade to current bucket
        
        Args:
            price: Trade price
            volume: Trade volume
            is_buy: Whether trade is a buy
            timestamp: Trade timestamp
            
        Returns:
            VPIN if bucket is complete, None otherwise
        """
        self.current_bucket.append({
            "price": price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp": timestamp
        })
        
        # Update order imbalance
        self.imbalance_history.append(1 if is_buy else -1)
        
        # Check if bucket is complete
        if len(self.current_bucket) >= self.config.bucket_size:
            return self._calculate_vpin()
        
        return None
    
    def _calculate_vpin(self) -> float:
        """
        Calculate VPIN for current bucket
        
        Returns:
            VPIN value
        """
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
        
        # Store bucket
        self.bucket_history.append({
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "vpin": vpin,
            "timestamp": self.current_bucket[-1]["timestamp"]
        })
        
        # Reset bucket
        self.current_bucket = []
        
        return vpin
    
    def calculate_toxicity(self) -> float:
        """
        Calculate overall toxicity score
        
        Returns:
            Toxicity score (0-1)
        """
        if len(self.bucket_history) < self.config.n_buckets:
            return 0.0
        
        # Calculate average VPIN
        vpins = [b["vpin"] for b in self.bucket_history]
        avg_vpin = np.mean(vpins)
        
        # Calculate order imbalance
        if len(self.imbalance_history) >= self.config.imbalance_window:
            recent_imbalance = list(self.imbalance_history)[-self.config.imbalance_window:]
            avg_imbalance = abs(np.mean(recent_imbalance))
        else:
            avg_imbalance = 0.0
        
        # Combine metrics
        toxicity = 0.6 * avg_vpin + 0.4 * avg_imbalance
        
        # Normalize to 0-1
        toxicity = min(toxicity, 1.0)
        
        self.toxicity_history.append(toxicity)
        
        return toxicity
    
    def is_toxic(self) -> bool:
        """
        Check if current order flow is toxic
        
        Returns:
            True if toxic, False otherwise
        """
        toxicity = self.calculate_toxicity()
        
        return toxicity > self.config.toxicity_threshold
    
    def get_vpin(self) -> float:
        """Get current VPIN"""
        if not self.bucket_history:
            return 0.0
        
        vpins = [b["vpin"] for b in self.bucket_history]
        return np.mean(vpins)
    
    def get_order_imbalance(self) -> float:
        """Get current order imbalance"""
        if len(self.imbalance_history) < self.config.imbalance_window:
            return 0.0
        
        recent_imbalance = list(self.imbalance_history)[-self.config.imbalance_window:]
        return np.mean(recent_imbalance)
    
    def get_risk_adjustment(self) -> float:
        """
        Get position size adjustment based on toxicity
        
        Returns:
            Adjustment factor (0-1)
        """
        toxicity = self.calculate_toxicity()
        
        if toxicity > self.config.toxicity_threshold:
            # Reduce exposure
            adjustment = 1.0 - (toxicity - self.config.toxicity_threshold) / (1.0 - self.config.toxicity_threshold)
            adjustment = max(adjustment, 1.0 - self.config.max_toxicity_exposure)
        else:
            adjustment = 1.0
        
        return adjustment
    
    def get_toxicity_summary(self) -> Dict:
        """Get toxicity summary"""
        return {
            "current_toxicity": self.calculate_toxicity(),
            "is_toxic": self.is_toxic(),
            "vpin": self.get_vpin(),
            "order_imbalance": self.get_order_imbalance(),
            "risk_adjustment": self.get_risk_adjustment(),
            "num_buckets": len(self.bucket_history)
        }


def simulate_trades(n_trades: int = 5000, base_price: float = 100.0) -> List[Dict]:
    """Simulate trades for testing"""
    np.random.seed(42)
    
    trades = []
    current_price = base_price
    
    for i in range(n_trades):
        # Random walk price
        current_price += np.random.randn() * 0.01
        
        # Random volume
        volume = int(np.random.exponential(100))
        
        # Random side (with some informed trading)
        is_buy = np.random.random() > 0.5
        
        # Add some toxicity (informed trading)
        if i % 100 < 10:  # 10% of time, add informed trading
            is_buy = True if np.random.random() > 0.3 else False
            volume *= 3  # Larger volume for informed trades
        
        trade = {
            "price": current_price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp": datetime.now() + timedelta(milliseconds=i)
        }
        
        trades.append(trade)
    
    return trades


if __name__ == "__main__":
    # Example usage
    config = ToxicityConfig(
        bucket_size=50,
        n_buckets=100,
        vpin_threshold=0.3,
        toxicity_threshold=0.5
    )
    
    toxicity_detector = OrderFlowToxicity(config)
    
    # Simulate trades
    print("Simulating trades...")
    trades = simulate_trades(5000, 100.0)
    
    # Process trades
    print("\nProcessing trades...")
    for i, trade in enumerate(trades):
        vpin = toxicity_detector.add_trade(
            trade["price"],
            trade["volume"],
            trade["is_buy"],
            trade["timestamp"]
        )
        
        if i % 500 == 0:
            summary = toxicity_detector.get_toxicity_summary()
            print(f"  Trade {i}: VPIN={summary['vpin']:.4f}, Toxicity={summary['current_toxicity']:.4f}, IsToxic={summary['is_toxic']}")
    
    # Final summary
    print("\nFinal Toxicity Summary:")
    summary = toxicity_detector.get_toxicity_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Risk adjustment
    print(f"\nRecommended Position Size Adjustment: {summary['risk_adjustment']:.2%}")

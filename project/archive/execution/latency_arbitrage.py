"""
Latency Arbitrage

Based on Comprehensive Upgrade Analysis - Tier 4 Upgrade (#34)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Ultra-low latency execution
- Co-location optimization
- Market data latency arbitrage
- Cross-venue arbitrage
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
class LatencyConfig:
    """Configuration for Latency Arbitrage"""
    # Latency parameters
    target_latency_us: int = 10  # Target latency in microseconds
    max_latency_us: int = 100  # Maximum acceptable latency
    
    # Co-location parameters
    enable_colocation: bool = True
    cpu_affinity: List[int] = None
    
    # Arbitrage parameters
    min_arbitrage_bps: float = 1.0  # Minimum arbitrage in basis points
    max_position_us: int = 50  # Maximum position holding time (microseconds)
    
    # Venue parameters
    venues: List[str] = None  # List of venues to monitor


class LatencyArbitrage:
    """
    Latency Arbitrage Engine
    
    Exploits price discrepancies across venues with ultra-low latency.
    Requires co-location and optimized execution.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: LatencyConfig):
        self.config = config
        
        # Venue prices
        self.venue_prices: Dict[str, float] = {}
        self.venue_timestamps: Dict[str, datetime] = {}
        
        # Arbitrage opportunities
        self.opportunities: deque = deque(maxlen=1000)
        
        # Execution latency tracking
        self.execution_latencies: deque = deque(maxlen=1000)
    
    def update_venue_price(self, venue: str, price: float, timestamp: datetime) -> Optional[Dict]:
        """
        Update venue price and check for arbitrage
        
        Args:
            venue: Venue name
            price: Current price
            timestamp: Price timestamp
            
        Returns:
            Arbitrage opportunity if found, None otherwise
        """
        self.venue_prices[venue] = price
        self.venue_timestamps[venue] = timestamp
        
        # Check for arbitrage
        return self._check_arbitrage()
    
    def _check_arbitrage(self) -> Optional[Dict]:
        """Check for arbitrage opportunities across venues"""
        if len(self.venue_prices) < 2:
            return None
        
        # Find best bid and ask across venues
        min_price = min(self.venue_prices.values())
        max_price = max(self.venue_prices.values())
        
        # Calculate spread
        spread_bps = (max_price - min_price) / min_price * 10000
        
        # Check if arbitrage is profitable
        if spread_bps > self.config.min_arbitrage_bps:
            # Find venues
            buy_venue = min(self.venue_prices, key=self.venue_prices.get)
            sell_venue = max(self.venue_prices, key=self.venue_prices.get)
            
            opportunity = {
                "buy_venue": buy_venue,
                "sell_venue": sell_venue,
                "buy_price": self.venue_prices[buy_venue],
                "sell_price": self.venue_prices[sell_venue],
                "spread_bps": spread_bps,
                "timestamp": datetime.now()
            }
            
            self.opportunities.append(opportunity)
            
            return opportunity
        
        return None
    
    def execute_arbitrage(self, opportunity: Dict) -> Dict:
        """
        Execute arbitrage trade
        
        Args:
            opportunity: Arbitrage opportunity
            
        Returns:
            Execution result
        """
        start_time = datetime.now()
        
        # Simulate execution (in real system, this would be ultra-fast)
        # Buy at lower price, sell at higher price
        buy_price = opportunity["buy_price"]
        sell_price = opportunity["sell_price"]
        
        # Calculate profit (before costs)
        profit_bps = opportunity["spread_bps"]
        
        # Calculate execution latency
        end_time = datetime.now()
        latency_us = (end_time - start_time).total_seconds() * 1e6
        
        self.execution_latencies.append(latency_us)
        
        return {
            "success": True,
            "profit_bps": profit_bps,
            "execution_latency_us": latency_us,
            "within_target": latency_us <= self.config.target_latency_us
        }
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.execution_latencies:
            return {}
        
        latencies = list(self.execution_latencies)
        
        return {
            "mean_latency_us": np.mean(latencies),
            "median_latency_us": np.median(latencies),
            "max_latency_us": np.max(latencies),
            "min_latency_us": np.min(latencies),
            "p95_latency_us": np.percentile(latencies, 95),
            "p99_latency_us": np.percentile(latencies, 99),
            "within_target_pct": sum(1 for l in latencies if l <= self.config.target_latency_us) / len(latencies) * 100
        }
    
    def get_arbitrage_stats(self) -> Dict:
        """Get arbitrage statistics"""
        if not self.opportunities:
            return {}
        
        opportunities = list(self.opportunities)
        
        return {
            "total_opportunities": len(opportunities),
            "avg_spread_bps": np.mean([o["spread_bps"] for o in opportunities]),
            "max_spread_bps": np.max([o["spread_bps"] for o in opportunities]),
            "min_spread_bps": np.min([o["spread_bps"] for o in opportunities])
        }


def simulate_venue_data(n_venues: int = 5, n_updates: int = 1000) -> List[Dict]:
    """Simulate venue price updates for testing"""
    np.random.seed(42)
    
    base_price = 100.0
    venue_names = [f"VENUE_{i}" for i in range(n_venues)]
    
    updates = []
    
    for i in range(n_updates):
        # Each venue updates at different times
        for venue in venue_names:
            # Random delay
            if np.random.random() > 0.5:
                # Add some noise to price
                price = base_price + np.random.randn() * 0.01
                
                # Occasionally create arbitrage
                if np.random.random() < 0.05:
                    price += np.random.choice([-0.02, 0.02])
                
                updates.append({
                    "venue": venue,
                    "price": price,
                    "timestamp": datetime.now() + timedelta(microseconds=i * 100 + np.random.randint(0, 1000))
                })
    
    return updates


if __name__ == "__main__":
    # Example usage
    config = LatencyConfig(
        target_latency_us=10,
        min_arbitrage_bps=1.0,
        venues=["NSE", "BSE", "MCX", "NSE_IF", "BSE_IF"]
    )
    
    arb_engine = LatencyArbitrage(config)
    
    # Simulate venue data
    print("Simulating venue data...")
    updates = simulate_venue_data(5, 1000)
    
    # Process updates
    print("\nProcessing venue updates...")
    executed_trades = []
    
    for update in updates:
        opportunity = arb_engine.update_venue_price(
            update["venue"],
            update["price"],
            update["timestamp"]
        )
        
        if opportunity:
            # Execute arbitrage
            result = arb_engine.execute_arbitrage(opportunity)
            executed_trades.append(result)
            
            if len(executed_trades) % 10 == 0:
                print(f"  Executed {len(executed_trades)} trades, avg profit: {np.mean([t['profit_bps'] for t in executed_trades]):.2f} bps")
    
    # Statistics
    print("\nLatency Statistics:")
    latency_stats = arb_engine.get_latency_stats()
    for key, value in latency_stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nArbitrage Statistics:")
    arb_stats = arb_engine.get_arbitrage_stats()
    for key, value in arb_stats.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nTotal Executed Trades: {len(executed_trades)}")
    print(f"Total Profit: {sum(t['profit_bps'] for t in executed_trades):.2f} bps")

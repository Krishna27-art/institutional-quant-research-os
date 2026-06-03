"""
Market Impact Model

Based on Almgren-Chriss square-root impact model.
Critical for realistic backtesting of institutional-sized orders.

Model:
Impact = k * sqrt(volume / ADV)

Where:
- k: impact coefficient (typically 0.01-0.1 for Indian markets)
- volume: trade volume
- ADV: average daily volume

This model captures the fact that larger orders move prices more.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MarketImpactConfig:
    """Configuration for market impact model"""
    # Impact coefficient (higher = more impact)
    # Indian markets: 0.01-0.05 for large caps, 0.05-0.1 for mid caps
    impact_coefficient: float = 0.02  # 2% impact per sqrt(volume/ADV)
    
    # Minimum impact (even small orders have some impact)
    min_impact_bps: float = 1.0  # 1 bps minimum
    
    # Maximum impact cap (prevents unrealistic estimates)
    max_impact_bps: float = 50.0  # 50 bps maximum
    
    # Time decay (impact less for longer execution windows)
    time_decay_factor: float = 0.5  # Impact scales with sqrt(1/time_factor)
    
    # Volume participation threshold
    max_participation_pct: float = 0.10  # Max 10% of ADV per trade


class MarketImpactModel:
    """
    Market Impact Model based on Almgren-Chriss square-root model.
    
    Estimates price impact based on trade size relative to average daily volume.
    """
    
    def __init__(self, config: MarketImpactConfig):
        self.config = config
    
    def calculate_impact(
        self,
        trade_volume: float,
        adv: float,
        execution_time_minutes: float = 1.0
    ) -> float:
        """
        Calculate market impact in basis points.
        
        Args:
            trade_volume: Volume of the trade
            adv: Average daily volume
            execution_time_minutes: Time to execute the trade (minutes)
            
        Returns:
            Impact in basis points
        """
        if adv == 0:
            return self.config.min_impact_bps
        
        # Calculate participation rate
        participation = trade_volume / adv
        
        # Cap participation
        participation = min(participation, self.config.max_participation_pct)
        
        # Square-root impact model
        impact_pct = self.config.impact_coefficient * np.sqrt(participation)
        
        # Time decay (longer execution = less impact)
        if execution_time_minutes > 0:
            time_factor = np.sqrt(1.0 / execution_time_minutes)
            impact_pct *= (self.config.time_decay_factor + (1 - self.config.time_decay_factor) * time_factor)
        
        # Convert to basis points
        impact_bps = impact_pct * 10000
        
        # Apply bounds
        impact_bps = max(self.config.min_impact_bps, impact_bps)
        impact_bps = min(self.config.max_impact_bps, impact_bps)
        
        return impact_bps
    
    def calculate_price_with_impact(
        self,
        base_price: float,
        trade_volume: float,
        adv: float,
        side: str,
        execution_time_minutes: float = 1.0
    ) -> float:
        """
        Calculate execution price including market impact.
        
        Args:
            base_price: Base price without impact
            trade_volume: Volume of the trade
            adv: Average daily volume
            side: 'BUY' or 'SELL'
            execution_time_minutes: Time to execute the trade (minutes)
            
        Returns:
            Execution price including impact
        """
        impact_bps = self.calculate_impact(trade_volume, adv, execution_time_minutes)
        impact_pct = impact_bps / 10000.0
        
        if side == 'BUY':
            # Buy orders push price up
            return base_price * (1 + impact_pct)
        else:
            # Sell orders push price down
            return base_price * (1 - impact_pct)
    
    def estimate_adv(
        self,
        volume_data: pd.Series,
        lookback_days: int = 20
    ) -> float:
        """
        Estimate average daily volume.
        
        Args:
            volume_data: Series of volume data
            lookback_days: Number of days to average
            
        Returns:
            Average daily volume
        """
        if len(volume_data) < lookback_days:
            return volume_data.mean()
        
        return volume_data.iloc[-lookback_days:].mean()


def create_default_market_impact_model() -> MarketImpactModel:
    """Create a market impact model with default Indian market parameters."""
    config = MarketImpactConfig(
        impact_coefficient=0.02,  # 2% impact coefficient
        min_impact_bps=1.0,  # 1 bps minimum
        max_impact_bps=50.0,  # 50 bps maximum
        time_decay_factor=0.5,  # Time decay factor
        max_participation_pct=0.10  # Max 10% participation
    )
    return MarketImpactModel(config)


if __name__ == "__main__":
    # Example usage
    model = create_default_market_impact_model()
    
    print("Market Impact Model Examples")
    print("=" * 50)
    
    # Example 1: Small trade (low impact)
    small_trade_volume = 10000
    adv = 1000000  # 1 million shares
    impact = model.calculate_impact(small_trade_volume, adv)
    print(f"Small trade ({small_trade_volume:,} shares, ADV {adv:,}): {impact:.2f} bps")
    
    # Example 2: Medium trade (moderate impact)
    medium_trade_volume = 50000
    impact = model.calculate_impact(medium_trade_volume, adv)
    print(f"Medium trade ({medium_trade_volume:,} shares, ADV {adv:,}): {impact:.2f} bps")
    
    # Example 3: Large trade (high impact)
    large_trade_volume = 100000
    impact = model.calculate_impact(large_trade_volume, adv)
    print(f"Large trade ({large_trade_volume:,} shares, ADV {adv:,}): {impact:.2f} bps")
    
    # Example 4: Price with impact
    base_price = 100.0
    buy_price = model.calculate_price_with_impact(base_price, large_trade_volume, adv, 'BUY')
    sell_price = model.calculate_price_with_impact(base_price, large_trade_volume, adv, 'SELL')
    print(f"\nPrice impact for large trade:")
    print(f"  Base price: ₹{base_price:.2f}")
    print(f"  Buy price: ₹{buy_price:.2f} (+{((buy_price/base_price - 1) * 100):.2f}%)")
    print(f"  Sell price: ₹{sell_price:.2f} ({((sell_price/base_price - 1) * 100):.2f}%)")

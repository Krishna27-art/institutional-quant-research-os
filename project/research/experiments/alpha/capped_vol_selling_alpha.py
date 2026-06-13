"""
Capped Vol Selling Alpha Strategy

This module implements the capped volatility selling strategy that sells
volatility through put spreads to cap downside risk while still capturing
the volatility risk premium.

Based on standard options practice.
Expected Sharpe: 0.4-0.7
Expected Capacity: High
Decay: Persistent
Difficulty: Medium

Priority: Medium (Options Phase 10)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolSellingRegime(Enum):
    """Volatility selling regime."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


@dataclass
class CappedVolMeasurement:
    """Capped vol selling measurement."""
    timestamp: datetime
    symbol: str
    atm_vol: float
    otm_put_vol: float
    otm_put_delta: float
    spread_width: float  # Strike spread as percentage of spot
    vrp: float
    regime: VolSellingRegime


@dataclass
class CappedVolSignal:
    """Capped vol selling trading signal."""
    timestamp: datetime
    symbol: str
    regime: VolSellingRegime
    signal: float  # 0-1, higher = more aggressive
    short_put_strike: float
    long_put_strike: float
    short_position: float
    long_position: float
    confidence: float
    expected_return: float
    max_loss: float


class CappedVolSellingAlpha:
    """
    Capped volatility selling alpha strategy.
    
    This class sells volatility through put spreads to cap
    downside risk while capturing VRP.
    """
    
    def __init__(
        self,
        otm_delta: float = 0.25,  # OTM delta for short put
        spread_width_pct: float = 0.10,  # 10% spread width
        vrp_threshold: float = 0.02,
        max_position_size: float = 0.08
    ):
        """
        Initialize capped vol selling alpha.
        
        Args:
            otm_delta: OTM delta for short put
            spread_width_pct: Spread width as percentage of spot
            vrp_threshold: Minimum VRP for signal
            max_position_size: Maximum position size
        """
        self.otm_delta = otm_delta
        self.spread_width_pct = spread_width_pct
        self.vrp_threshold = vrp_threshold
        self.max_position_size = max_position_size
        
        self.measurements: List[CappedVolMeasurement] = []
        self.signals: List[CappedVolSignal] = []
        
        logger.info(f"CappedVolSellingAlpha initialized: otm_delta={otm_delta}, "
                   f"spread_width={spread_width_pct:.2%}, vrp_threshold={vrp_threshold}")
    
    def calculate_vrp(
        self,
        implied_vol: float,
        realized_vol: float
    ) -> float:
        """
        Calculate volatility risk premium.
        
        Args:
            implied_vol: Implied volatility
            realized_vol: Realized volatility
            
        Returns:
            VRP value
        """
        return implied_vol - realized_vol
    
    def determine_regime(
        self,
        vrp: float,
        vol_level: float
    ) -> VolSellingRegime:
        """
        Determine volatility selling regime.
        
        Args:
            vrp: Current VRP
            vol_level: Current volatility level
            
        Returns:
            VolSellingRegime
        """
        if vol_level > 0.30:  # 30% vol
            return VolSellingRegime.EXTREME
        elif vol_level > 0.25 or vrp < 0:
            return VolSellingRegime.ELEVATED
        else:
            return VolSellingRegime.NORMAL
    
    def calculate_strike_levels(
        self,
        spot: float,
        otm_vol: float
    ) -> Tuple[float, float]:
        """
        Calculate strike levels for put spread.
        
        Args:
            spot: Spot price
            otm_vol: OTM implied volatility
            
        Returns:
            (short_put_strike, long_put_strike)
        """
        # Short put at OTM delta
        short_put_strike = spot * (1 - self.otm_delta * otm_vol * np.sqrt(1/252))
        
        # Long put further OTM (capped risk)
        long_put_strike = short_put_strike * (1 - self.spread_width_pct)
        
        return short_put_strike, long_put_strike
    
    def generate_signal(
        self,
        symbol: str,
        spot: float,
        atm_vol: float,
        otm_put_vol: float,
        realized_vol: float,
        timestamp: datetime
    ) -> Optional[CappedVolSignal]:
        """
        Generate capped vol selling signal.
        
        Args:
            symbol: Underlying symbol
            spot: Spot price
            atm_vol: ATM implied volatility
            otm_put_vol: OTM put implied volatility
            realized_vol: Realized volatility
            timestamp: Signal timestamp
            
        Returns:
            CappedVolSignal or None
        """
        # Calculate VRP
        vrp = self.calculate_vrp(atm_vol, realized_vol)
        
        # Check VRP threshold
        if vrp < self.vrp_threshold:
            return None
        
        # Determine regime
        regime = self.determine_regime(vrp, atm_vol)
        
        # Calculate strike levels
        short_put_strike, long_put_strike = self.calculate_strike_levels(spot, otm_put_vol)
        
        # Generate signal based on regime
        if regime == VolSellingRegime.NORMAL:
            signal = 1.0
            short_position = self.max_position_size
            long_position = self.max_position_size * 0.8  # Hedge 80%
            confidence = min(vrp / 0.05, 0.9)
            expected_return = vrp * 0.4
            max_loss = (short_put_strike - long_put_strike) / spot
        elif regime == VolSellingRegime.ELEVATED:
            signal = 0.6
            short_position = self.max_position_size * 0.6
            long_position = self.max_position_size * 0.6  # Full hedge
            confidence = min(vrp / 0.05, 0.7)
            expected_return = vrp * 0.2
            max_loss = (short_put_strike - long_put_strike) / spot
        elif regime == VolSellingRegime.EXTREME:
            signal = 0.2
            short_position = self.max_position_size * 0.3
            long_position = self.max_position_size * 0.3  # Full hedge
            confidence = 0.5
            expected_return = vrp * 0.1
            max_loss = (short_put_strike - long_put_strike) / spot
        else:
            return None
        
        # Store measurement
        measurement = CappedVolMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            atm_vol=atm_vol,
            otm_put_vol=otm_put_vol,
            otm_put_delta=self.otm_delta,
            spread_width=self.spread_width_pct,
            vrp=vrp,
            regime=regime
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        capped_signal = CappedVolSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            signal=signal,
            short_put_strike=short_put_strike,
            long_put_strike=long_put_strike,
            short_position=short_position,
            long_position=long_position,
            confidence=confidence,
            expected_return=expected_return,
            max_loss=max_loss
        )
        
        self.signals.append(capped_signal)
        
        return capped_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[CappedVolSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_capped_vol_statistics(self) -> Dict[str, float]:
        """Get capped vol statistics."""
        if not self.measurements:
            return {}
        
        vrps = [m.vrp for m in self.measurements]
        
        return {
            'avg_vrp': np.mean(vrps),
            'std_vrp': np.std(vrps),
            'min_vrp': np.min(vrps),
            'max_vrp': np.max(vrps),
            'current_vrp': vrps[-1] if vrps else 0.0
        }
    
    def print_capped_vol_report(self) -> None:
        """Print capped vol selling analysis report."""
        print("\n" + "="*60)
        print("CAPPED VOL SELLING ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  OTM Delta: {self.otm_delta}")
        print(f"  Spread Width: {self.spread_width_pct:.2%}")
        print(f"  VRP Threshold: {self.vrp_threshold:.2%}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_capped_vol_statistics()
            print(f"\nVRP Statistics:")
            print(f"  Average VRP: {stats['avg_vrp']:.4f} ({stats['avg_vrp']*100:.2f}%)")
            print(f"  VRP Std: {stats['std_vrp']:.4f} ({stats['std_vrp']*100:.2f}%)")
            print(f"  Min VRP: {stats['min_vrp']:.4f} ({stats['min_vrp']*100:.2f}%)")
            print(f"  Max VRP: {stats['max_vrp']:.4f} ({stats['max_vrp']*100:.2f}%)")
            print(f"  Current VRP: {stats['current_vrp']:.4f} ({stats['current_vrp']*100:.2f}%)")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'VRP':<10} {'Regime':<12} {'Signal':<10} {'ShortPos':<10} {'LongPos':<10} {'MaxLoss':<10}")
            print("-" * 105)
            
            for signal in self.signals[-5]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.vrp:<10.4f} {signal.regime.value:<12} {signal.signal:<10.3f} "
                      f"{signal.short_position:<10.3f} {signal.long_position:<10.3f} {signal.max_loss:<10.3f}")
        
        print("\n" + "="*60)


def sample_capped_vol_selling_alpha():
    """Demonstrate capped vol selling alpha."""
    print("=== Capped Vol Selling Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = CappedVolSellingAlpha(
        otm_delta=0.25,
        spread_width_pct=0.10,
        vrp_threshold=0.02,
        max_position_size=0.08
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    spot = 1000.0
    
    # Generate volatility data
    base_atm_vol = 0.20
    atm_vols = base_atm_vol + np.random.randn(n_days) * 0.02
    otm_put_vols = atm_vols + 0.03 + np.random.randn(n_days) * 0.01
    
    # Generate realized vol (lower than implied)
    base_realized = 0.15
    realized_vols = base_realized + np.random.randn(n_days) * 0.01
    
    # Process data
    print("Processing capped vol selling data...")
    for i in range(30, n_days):
        signal = alpha.generate_signal(
            'NIFTY',
            spot,
            atm_vols[i],
            otm_put_vols[i],
            realized_vols[i],
            dates[i]
        )
    
    # Print report
    alpha.print_capped_vol_report()
    
    print("\n=== Capped Vol Selling Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- VRP calculation")
    print("- Regime detection (normal, elevated, extreme)")
    print("- Put spread strike calculation")
    print("- Capped risk through spread structure")
    print("- Dynamic position sizing based on regime")
    print("- Max loss calculation")
    print("- Expected Sharpe: 0.4-0.7")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_capped_vol_selling_alpha()

"""
Skew Steepener/Flattener Alpha Strategy

This module implements the skew steepener/flattener strategy that trades
the volatility skew by dynamically adjusting risk reversal positions based
on skew level and changes.

Based on Kozhan, Neuberger, Schneider 2013.
Expected Sharpe: 0.5-0.9
Expected Capacity: High
Decay: Persistent
Difficulty: High

Priority: Medium (Options Phase 3)
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


class SkewRegime(Enum):
    """Skew regime."""
    NORMAL = "normal"  # Typical skew
    STEEP = "steep"  # High skew (flattener opportunity)
    FLAT = "flat"  # Low skew (steepener opportunity)
    EXTREME = "extreme"  # Extreme skew


@dataclass
class SkewMeasurement:
    """Skew measurement."""
    timestamp: datetime
    symbol: str
    atm_vol: float
    otm_put_vol: float
    otm_call_vol: float
    skew: float  # otm_put_vol - otm_call_vol
    skew_percentile: float
    regime: SkewRegime


@dataclass
class SkewSignal:
    """Skew trading signal."""
    timestamp: datetime
    symbol: str
    skew: float
    regime: SkewRegime
    signal: float  # -1 to 1, positive = steepener, negative = flattener
    put_position: float
    call_position: float
    confidence: float
    expected_return: float


class SkewSteepenerAlpha:
    """
    Skew steepener/flattener alpha strategy.
    
    This class trades volatility skew by buying/selling risk reversals
    based on skew level and regime.
    """
    
    def __init__(
        self,
        otm_delta: float = 0.25,  # OTM delta for skew calculation
        skew_threshold: float = 0.05,  # 5% skew threshold
        lookback_days: int = 60,
        max_position_size: float = 0.08
    ):
        """
        Initialize skew steepener alpha.
        
        Args:
            otm_delta: OTM delta for skew calculation
            skew_threshold: Skew threshold for signal
            lookback_days: Lookback period for percentile
            max_position_size: Maximum position size
        """
        self.otm_delta = otm_delta
        self.skew_threshold = skew_threshold
        self.lookback_days = lookback_days
        self.max_position_size = max_position_size
        
        self.measurements: List[SkewMeasurement] = []
        self.signals: List[SkewSignal] = []
        
        logger.info(f"SkewSteepenerAlpha initialized: otm_delta={otm_delta}, "
                   f"skew_threshold={skew_threshold}")
    
    def calculate_skew(
        self,
        atm_vol: float,
        otm_put_vol: float,
        otm_call_vol: float
    ) -> float:
        """
        Calculate volatility skew.
        
        Args:
            atm_vol: ATM implied volatility
            otm_put_vol: OTM put implied volatility
            otm_call_vol: OTM call implied volatility
            
        Returns:
            Skew (otm_put_vol - otm_call_vol)
        """
        return otm_put_vol - otm_call_vol
    
    def calculate_skew_percentile(self, current_skew: float) -> float:
        """
        Calculate skew percentile based on history.
        
        Args:
            current_skew: Current skew
            
        Returns:
            Skew percentile (0-1)
        """
        if not self.measurements:
            return 0.5
        
        skews = [m.skew for m in self.measurements]
        percentile = np.sum([1 for s in skews if s <= current_skew]) / len(skews)
        return percentile
    
    def determine_regime(
        self,
        skew: float,
        skew_percentile: float
    ) -> SkewRegime:
        """
        Determine skew regime.
        
        Args:
            skew: Current skew
            skew_percentile: Skew percentile
            
        Returns:
            SkewRegime
        """
        if skew_percentile > 0.9:
            return SkewRegime.EXTREME
        elif skew_percentile > 0.75:
            return SkewRegime.STEEP
        elif skew_percentile < 0.25:
            return SkewRegime.FLAT
        else:
            return SkewRegime.NORMAL
    
    def generate_signal(
        self,
        symbol: str,
        atm_vol: float,
        otm_put_vol: float,
        otm_call_vol: float,
        timestamp: datetime
    ) -> Optional[SkewSignal]:
        """
        Generate skew trading signal.
        
        Args:
            symbol: Underlying symbol
            atm_vol: ATM implied volatility
            otm_put_vol: OTM put implied volatility
            otm_call_vol: OTM call implied volatility
            timestamp: Signal timestamp
            
        Returns:
            SkewSignal or None
        """
        # Calculate skew
        skew = self.calculate_skew(atm_vol, otm_put_vol, otm_call_vol)
        
        # Calculate skew percentile
        skew_percentile = self.calculate_skew_percentile(skew)
        
        # Determine regime
        regime = self.determine_regime(skew, skew_percentile)
        
        # Generate signal based on regime
        if regime == SkewRegime.FLAT:
            # Steepener: buy OTM puts, sell OTM calls
            signal = 1.0
            put_position = self.max_position_size
            call_position = -self.max_position_size
            confidence = 1.0 - skew_percentile
            expected_return = abs(skew) * 0.4
        elif regime == SkewRegime.STEEP:
            # Flattener: sell OTM puts, buy OTM calls
            signal = -1.0
            put_position = -self.max_position_size * 0.7
            call_position = self.max_position_size * 0.7
            confidence = skew_percentile
            expected_return = abs(skew) * 0.3
        elif regime == SkewRegime.EXTREME:
            # Extreme skew: aggressive flattener
            signal = -1.0
            put_position = -self.max_position_size * 0.5
            call_position = self.max_position_size * 0.5
            confidence = skew_percentile * 0.8
            expected_return = abs(skew) * 0.2
        else:
            return None
        
        # Store measurement
        measurement = SkewMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            atm_vol=atm_vol,
            otm_put_vol=otm_put_vol,
            otm_call_vol=otm_call_vol,
            skew=skew,
            skew_percentile=skew_percentile,
            regime=regime
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        skew_signal = SkewSignal(
            timestamp=timestamp,
            symbol=symbol,
            skew=skew,
            regime=regime,
            signal=signal,
            put_position=put_position,
            call_position=call_position,
            confidence=confidence,
            expected_return=expected_return
        )
        
        self.signals.append(skew_signal)
        
        return skew_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[SkewSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_skew_statistics(self) -> Dict[str, float]:
        """Get skew statistics."""
        if not self.measurements:
            return {}
        
        skews = [m.skew for m in self.measurements]
        
        return {
            'avg_skew': np.mean(skews),
            'std_skew': np.std(skews),
            'min_skew': np.min(skews),
            'max_skew': np.max(skews),
            'current_skew': skews[-1] if skews else 0.0
        }
    
    def print_skew_report(self) -> None:
        """Print skew analysis report."""
        print("\n" + "="*60)
        print("SKEW STEEPENER/FLATTENER ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  OTM Delta: {self.otm_delta}")
        print(f"  Skew Threshold: {self.skew_threshold:.2%}")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_skew_statistics()
            print(f"\nSkew Statistics:")
            print(f"  Average Skew: {stats['avg_skew']:.4f}")
            print(f"  Skew Std: {stats['std_skew']:.4f}")
            print(f"  Min Skew: {stats['min_skew']:.4f}")
            print(f"  Max Skew: {stats['max_skew']:.4f}")
            print(f"  Current Skew: {stats['current_skew']:.4f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Skew':<10} {'Regime':<12} {'Signal':<10} {'PutPos':<10} {'CallPos':<10}")
            print("-" * 90)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.skew:<10.4f} {signal.regime.value:<12} {signal.signal:<10.3f} "
                      f"{signal.put_position:<10.3f} {signal.call_position:<10.3f}")
        
        print("\n" + "="*60)


def sample_skew_steepener_alpha():
    """Demonstrate skew steepener alpha."""
    print("=== Skew Steepener/Flattener Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = SkewSteepenerAlpha(
        otm_delta=0.25,
        skew_threshold=0.05,
        lookback_days=60,
        max_position_size=0.08
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate volatility surface with skew
    base_vol = 0.20
    atm_vol = base_vol + np.random.randn(n_days) * 0.02
    
    # OTM puts typically higher vol (skew)
    otm_put_vol = atm_vol + 0.03 + np.random.randn(n_days) * 0.01
    otm_call_vol = atm_vol - 0.01 + np.random.randn(n_days) * 0.01
    
    # Process data
    print("Processing skew data...")
    for i in range(30, n_days):
        signal = alpha.generate_signal(
            'NIFTY',
            atm_vol[i],
            otm_put_vol[i],
            otm_call_vol[i],
            dates[i]
        )
    
    # Print report
    alpha.print_skew_report()
    
    print("\n=== Skew Steepener/Flattener Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Volatility skew calculation")
    print("- Skew percentile ranking")
    print("- Regime detection (normal, steep, flat, extreme)")
    print("- Risk reversal trading (steepener/flattener)")
    print("- Expected Sharpe: 0.5-0.9")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_skew_steepener_alpha()

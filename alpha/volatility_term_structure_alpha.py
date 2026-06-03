"""
Volatility Term Structure Trade Alpha Strategy

This module implements the volatility term structure trading strategy that
exploits the slope of the volatility term curve by trading the spread between
short-dated and long-dated volatility instruments.

Based on VIX futures basis literature.
Expected Sharpe: 0.6-1.0
Expected Capacity: Very High
Decay: Persistent
Difficulty: Medium

Priority: High (Options Phase 2)
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


class TermStructureRegime(Enum):
    """Term structure regime."""
    CONTANGO = "contango"  # Normal: short vol > long vol
    BACKWARDATION = "backwardation"  # Stress: short vol < long vol
    FLAT = "flat"  # Neutral


@dataclass
class TermStructureMeasurement:
    """Term structure measurement."""
    timestamp: datetime
    symbol: str
    vol_short: float  # Short-dated implied vol
    vol_long: float  # Long-dated implied vol
    slope: float  # vol_long - vol_short
    basis_short: float  # Short-dated futures basis
    basis_long: float  # Long-dated futures basis
    regime: TermStructureRegime
    slope_percentile: float


@dataclass
class TermStructureSignal:
    """Term structure trading signal."""
    timestamp: datetime
    symbol: str
    slope: float
    regime: TermStructureRegime
    signal: float  # -1 to 1
    position_short: float  # Short-dated position
    position_long: float  # Long-dated position
    confidence: float
    expected_return: float


class VolatilityTermStructureAlpha:
    """
    Volatility term structure trading alpha strategy.
    
    This class trades the slope of the volatility term structure,
    going long short-dated vol and short long-dated vol during contango.
    """
    
    def __init__(
        self,
        short_dte: int = 30,  # Days to expiration for short leg
        long_dte: int = 90,  # Days to expiration for long leg
        slope_threshold: float = 0.02,  # 2% slope threshold
        lookback_days: int = 60,
        max_position_size: float = 0.10
    ):
        """
        Initialize volatility term structure alpha.
        
        Args:
            short_dte: Days to expiration for short leg
            long_dte: Days to expiration for long leg
            slope_threshold: Minimum slope for signal
            lookback_days: Lookback period for percentile calculation
            max_position_size: Maximum position size
        """
        self.short_dte = short_dte
        self.long_dte = long_dte
        self.slope_threshold = slope_threshold
        self.lookback_days = lookback_days
        self.max_position_size = max_position_size
        
        self.measurements: List[TermStructureMeasurement] = []
        self.signals: List[TermStructureSignal] = []
        
        logger.info(f"VolatilityTermStructureAlpha initialized: short_dte={short_dte}, "
                   f"long_dte={long_dte}, slope_threshold={slope_threshold}")
    
    def calculate_slope(
        self,
        vol_short: float,
        vol_long: float
    ) -> float:
        """
        Calculate term structure slope.
        
        Args:
            vol_short: Short-dated implied volatility
            vol_long: Long-dated implied volatility
            
        Returns:
            Slope (vol_long - vol_short)
        """
        return vol_long - vol_short
    
    def calculate_slope_percentile(self, current_slope: float) -> float:
        """
        Calculate slope percentile based on history.
        
        Args:
            current_slope: Current slope
            
        Returns:
            Slope percentile (0-1)
        """
        if not self.measurements:
            return 0.5
        
        slopes = [m.slope for m in self.measurements]
        percentile = np.sum([1 for s in slopes if s <= current_slope]) / len(slopes)
        return percentile
    
    def determine_regime(
        self,
        slope: float,
        basis_short: float
    ) -> TermStructureRegime:
        """
        Determine term structure regime.
        
        Args:
            slope: Term structure slope
            basis_short: Short-dated futures basis
            
        Returns:
            TermStructureRegime
        """
        if slope > self.slope_threshold and basis_short > 0:
            return TermStructureRegime.CONTANGO
        elif slope < -self.slope_threshold and basis_short < 0:
            return TermStructureRegime.BACKWARDATION
        else:
            return TermStructureRegime.FLAT
    
    def generate_signal(
        self,
        symbol: str,
        vol_short: float,
        vol_long: float,
        futures_short: float,
        futures_long: float,
        spot: float,
        timestamp: datetime
    ) -> Optional[TermStructureSignal]:
        """
        Generate term structure signal.
        
        Args:
            symbol: Underlying symbol
            vol_short: Short-dated implied volatility
            vol_long: Long-dated implied volatility
            futures_short: Short-dated futures price
            futures_long: Long-dated futures price
            spot: Spot price
            timestamp: Signal timestamp
            
        Returns:
            TermStructureSignal or None
        """
        # Calculate slope
        slope = self.calculate_slope(vol_short, vol_long)
        
        # Calculate basis
        basis_short = futures_short - spot
        basis_long = futures_long - spot
        
        # Calculate slope percentile
        slope_percentile = self.calculate_slope_percentile(slope)
        
        # Determine regime
        regime = self.determine_regime(slope, basis_short)
        
        # Generate signal based on regime
        if regime == TermStructureRegime.CONTANGO:
            # Long short vol, short long vol (calendar spread)
            signal = 1.0
            position_short = self.max_position_size
            position_long = -self.max_position_size
            confidence = slope_percentile
            expected_return = abs(slope) * 0.5
        elif regime == TermStructureRegime.BACKWARDATION:
            # Short short vol, long long vol (reverse calendar spread)
            signal = -1.0
            position_short = -self.max_position_size * 0.5  # Smaller position in stress
            position_long = self.max_position_size * 0.5
            confidence = slope_percentile * 0.7
            expected_return = abs(slope) * 0.3
        else:
            return None
        
        # Store measurement
        measurement = TermStructureMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            vol_short=vol_short,
            vol_long=vol_long,
            slope=slope,
            basis_short=basis_short,
            basis_long=basis_long,
            regime=regime,
            slope_percentile=slope_percentile
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        term_signal = TermStructureSignal(
            timestamp=timestamp,
            symbol=symbol,
            slope=slope,
            regime=regime,
            signal=signal,
            position_short=position_short,
            position_long=position_long,
            confidence=confidence,
            expected_return=expected_return
        )
        
        self.signals.append(term_signal)
        
        return term_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[TermStructureSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_term_structure_statistics(self) -> Dict[str, float]:
        """Get term structure statistics."""
        if not self.measurements:
            return {}
        
        slopes = [m.slope for m in self.measurements]
        
        return {
            'avg_slope': np.mean(slopes),
            'std_slope': np.std(slopes),
            'min_slope': np.min(slopes),
            'max_slope': np.max(slopes),
            'current_slope': slopes[-1] if slopes else 0.0
        }
    
    def print_term_structure_report(self) -> None:
        """Print term structure analysis report."""
        print("\n" + "="*60)
        print("VOLATILITY TERM STRUCTURE ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Short DTE: {self.short_dte} days")
        print(f"  Long DTE: {self.long_dte} days")
        print(f"  Slope Threshold: {self.slope_threshold:.2%}")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_term_structure_statistics()
            print(f"\nTerm Structure Statistics:")
            print(f"  Average Slope: {stats['avg_slope']:.4f}")
            print(f"  Slope Std: {stats['std_slope']:.4f}")
            print(f"  Min Slope: {stats['min_slope']:.4f}")
            print(f"  Max Slope: {stats['max_slope']:.4f}")
            print(f"  Current Slope: {stats['current_slope']:.4f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Slope':<10} {'Regime':<15} {'Signal':<10} {'PosShort':<12} {'PosLong':<12}")
            print("-" * 100)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.slope:<10.4f} {signal.regime.value:<15} {signal.signal:<10.3f} "
                      f"{signal.position_short:<12.3f} {signal.position_long:<12.3f}")
        
        print("\n" + "="*60)


def sample_volatility_term_structure_alpha():
    """Demonstrate volatility term structure alpha."""
    print("=== Volatility Term Structure Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = VolatilityTermStructureAlpha(
        short_dte=30,
        long_dte=90,
        slope_threshold=0.02,
        lookback_days=60,
        max_position_size=0.10
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate volatility term structure (contango = normal)
    base_vol = 0.20
    vol_short = base_vol + np.random.randn(n_days) * 0.02
    vol_long = vol_short + 0.03 + np.random.randn(n_days) * 0.01  # Upward sloping
    
    # Generate futures
    spot = 1000.0
    futures_short = spot * (1 + vol_short / np.sqrt(252)) + np.random.randn(n_days) * 5
    futures_long = spot * (1 + vol_long / np.sqrt(252)) + np.random.randn(n_days) * 5
    
    # Process data
    print("Processing term structure data...")
    for i in range(30, n_days):
        signal = alpha.generate_signal(
            'NIFTY',
            vol_short[i],
            vol_long[i],
            futures_short[i],
            futures_long[i],
            spot,
            dates[i]
        )
    
    # Print report
    alpha.print_term_structure_report()
    
    print("\n=== Volatility Term Structure Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Term structure slope calculation")
    print("- Regime detection (contango, backwardation, flat)")
    print("- Calendar spread trading")
    print("- Slope percentile ranking")
    print("- Expected Sharpe: 0.6-1.0")
    print("- Expected Capacity: Very High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_volatility_term_structure_alpha()

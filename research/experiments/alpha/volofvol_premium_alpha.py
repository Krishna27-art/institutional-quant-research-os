"""
Vol-of-Vol Premium Harvesting Alpha Strategy

This module implements the volatility-of-volatility (VoV) premium harvesting
strategy that sells volatility when VoV is elevated, capturing the premium
for higher-order volatility risk.

Based on recent thesis 2025.
Expected Sharpe: 0.5-0.9
Expected Capacity: High
Decay: Persistent
Difficulty: High

Priority: Medium (Options Phase 7)
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


class VoVRegime(Enum):
    """Vol-of-vol regime."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


@dataclass
class VoVMeasurement:
    """Vol-of-vol measurement."""
    timestamp: datetime
    symbol: str
    current_vol: float
    vol_std: float  # Volatility of volatility
    vov: float  # Vol-of-vol (vol_std / current_vol)
    vov_percentile: float
    regime: VoVRegime


@dataclass
class VoVSignal:
    """Vol-of-vol trading signal."""
    timestamp: datetime
    symbol: str
    vov: float
    regime: VoVRegime
    signal: float  # -1 to 1, negative = short vol
    position_size: float
    confidence: float
    expected_return: float


class VolOfVolPremiumAlpha:
    """
    Vol-of-vol premium harvesting alpha strategy.
    
    This class sells volatility when VoV is elevated to capture
    the premium for higher-order volatility risk.
    """
    
    def __init__(
        self,
        vol_lookback: int = 20,
        vov_threshold: float = 0.3,  # 30% VoV threshold
        percentile_threshold: float = 0.75,
        max_position_size: float = 0.10
    ):
        """
        Initialize vol-of-vol premium alpha.
        
        Args:
            vol_lookback: Lookback period for vol calculation
            vov_threshold: VoV threshold for signal
            percentile_threshold: VoV percentile threshold
            max_position_size: Maximum position size
        """
        self.vol_lookback = vol_lookback
        self.vov_threshold = vov_threshold
        self.percentile_threshold = percentile_threshold
        self.max_position_size = max_position_size
        
        self.measurements: List[VoVMeasurement] = []
        self.signals: List[VoVSignal] = []
        self.vol_history: Dict[str, List[float]] = {}
        
        logger.info(f"VolOfVolPremiumAlpha initialized: vol_lookback={vol_lookback}, "
                   f"vov_threshold={vov_threshold}")
    
    def calculate_vov(
        self,
        current_vol: float,
        vol_history: List[float]
    ) -> float:
        """
        Calculate vol-of-volatility.
        
        Args:
            current_vol: Current volatility
            vol_history: Historical volatility values
            
        Returns:
            Vol-of-vol (vol_std / current_vol)
        """
        if len(vol_history) < self.vol_lookback:
            return 0.0
        
        recent_vols = vol_history[-self.vol_lookback:]
        vol_std = np.std(recent_vols)
        
        if current_vol == 0:
            return 0.0
        
        vov = vol_std / current_vol
        return vov
    
    def calculate_vov_percentile(self, current_vov: float) -> float:
        """
        Calculate VoV percentile based on history.
        
        Args:
            current_vov: Current VoV
            
        Returns:
            VoV percentile (0-1)
        """
        if not self.measurements:
            return 0.5
        
        vovs = [m.vov for m in self.measurements]
        percentile = np.sum([1 for v in vovs if v <= current_vov]) / len(vovs)
        return percentile
    
    def determine_regime(
        self,
        vov: float,
        vov_percentile: float
    ) -> VoVRegime:
        """
        Determine VoV regime.
        
        Args:
            vov: Current VoV
            vov_percentile: VoV percentile
            
        Returns:
            VoVRegime
        """
        if vov_percentile > 0.9:
            return VoVRegime.EXTREME
        elif vov_percentile > self.percentile_threshold or vov > self.vov_threshold:
            return VoVRegime.ELEVATED
        else:
            return VoVRegime.NORMAL
    
    def generate_signal(
        self,
        symbol: str,
        current_vol: float,
        timestamp: datetime
    ) -> Optional[VoVSignal]:
        """
        Generate VoV trading signal.
        
        Args:
            symbol: Underlying symbol
            current_vol: Current implied volatility
            timestamp: Signal timestamp
            
        Returns:
            VoVSignal or None
        """
        # Update vol history
        if symbol not in self.vol_history:
            self.vol_history[symbol] = []
        self.vol_history[symbol].append(current_vol)
        
        # Keep history manageable
        if len(self.vol_history[symbol]) > 100:
            self.vol_history[symbol] = self.vol_history[symbol][-100:]
        
        # Calculate VoV
        vov = self.calculate_vov(current_vol, self.vol_history[symbol])
        
        # Calculate VoV percentile
        vov_percentile = self.calculate_vov_percentile(vov)
        
        # Determine regime
        regime = self.determine_regime(vov, vov_percentile)
        
        # Generate signal based on regime
        if regime == VoVRegime.NORMAL:
            return None
        elif regime == VoVRegime.ELEVATED:
            signal = -0.8  # Short vol
            position_size = self.max_position_size
            confidence = vov_percentile
            expected_return = vov * 0.3
        elif regime == VoVRegime.EXTREME:
            signal = -1.0  # Aggressive short vol
            position_size = self.max_position_size * 0.7  # Smaller in extreme
            confidence = vov_percentile * 0.8
            expected_return = vov * 0.2
        else:
            return None
        
        # Calculate vol std for measurement
        if len(self.vol_history[symbol]) >= self.vol_lookback:
            vol_std = np.std(self.vol_history[symbol][-self.vol_lookback:])
        else:
            vol_std = 0.0
        
        # Store measurement
        measurement = VoVMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            current_vol=current_vol,
            vol_std=vol_std,
            vov=vov,
            vov_percentile=vov_percentile,
            regime=regime
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        vov_signal = VoVSignal(
            timestamp=timestamp,
            symbol=symbol,
            vov=vov,
            regime=regime,
            signal=signal,
            position_size=position_size,
            confidence=confidence,
            expected_return=expected_return
        )
        
        self.signals.append(vov_signal)
        
        return vov_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[VoVSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_vov_statistics(self) -> Dict[str, float]:
        """Get VoV statistics."""
        if not self.measurements:
            return {}
        
        vovs = [m.vov for m in self.measurements]
        
        return {
            'avg_vov': np.mean(vovs),
            'std_vov': np.std(vovs),
            'min_vov': np.min(vovs),
            'max_vov': np.max(vovs),
            'current_vov': vovs[-1] if vovs else 0.0
        }
    
    def print_vov_report(self) -> None:
        """Print VoV analysis report."""
        print("\n" + "="*60)
        print("VOL-OF-VOL PREMIUM HARVESTING ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Vol Lookback: {self.vol_lookback}")
        print(f"  VoV Threshold: {self.vov_threshold:.2%}")
        print(f"  Percentile Threshold: {self.percentile_threshold:.2%}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_vov_statistics()
            print(f"\nVoV Statistics:")
            print(f"  Average VoV: {stats['avg_vov']:.4f}")
            print(f"  VoV Std: {stats['std_vov']:.4f}")
            print(f"  Min VoV: {stats['min_vov']:.4f}")
            print(f"  Max VoV: {stats['max_vov']:.4f}")
            print(f"  Current VoV: {stats['current_vov']:.4f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'VoV':<10} {'Regime':<12} {'Signal':<10} {'Position':<10}")
            print("-" * 85)
            
            for signal in self.signals[-5]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.vov:<10.4f} {signal.regime.value:<12} {signal.signal:<10.3f} "
                      f"{signal.position_size:<10.3f}")
        
        print("\n" + "="*60)


def sample_volofvol_premium_alpha():
    """Demonstrate vol-of-vol premium alpha."""
    print("=== Vol-of-Vol Premium Harvesting Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = VolOfVolPremiumAlpha(
        vol_lookback=20,
        vov_threshold=0.3,
        percentile_threshold=0.75,
        max_position_size=0.10
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate volatility with VoV spikes
    base_vol = 0.20
    vols = base_vol + np.random.randn(n_days) * 0.02
    
    # Add VoV spikes (periods of high vol volatility)
    for i in range(30, 40):
        vols[i] += np.random.randn() * 0.05  # High vol volatility
    
    # Process data
    print("Processing VoV data...")
    for i in range(n_days):
        signal = alpha.generate_signal(
            'VIX',
            vols[i],
            dates[i]
        )
    
    # Print report
    alpha.print_vov_report()
    
    print("\n=== Vol-of-Vol Premium Harvesting Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Vol-of-volatility calculation")
    print("- VoV percentile ranking")
    print("- Regime detection (normal, elevated, extreme)")
    print("- Volatility selling when VoV elevated")
    print("- Position sizing based on VoV level")
    print("- Expected Sharpe: 0.5-0.9")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_volofvol_premium_alpha()

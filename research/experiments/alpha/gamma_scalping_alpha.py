"""
Gamma Scalping After Vol Spike Alpha Strategy

This module implements the gamma scalping strategy that buys options when
volatility spikes and delta-hedges to capture the gamma profit from price
movements during high volatility periods.

Based on standard options market making practice.
Expected Sharpe: 0.4-0.8
Expected Capacity: High
Decay: Persistent
Difficulty: Medium

Priority: Medium (Options Phase 4)
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


class VolSpikeRegime(Enum):
    """Volatility spike regime."""
    NORMAL = "normal"
    SPIKE = "spike"
    EXTREME = "extreme"


@dataclass
class VolSpikeMeasurement:
    """Volatility spike measurement."""
    timestamp: datetime
    symbol: str
    current_vol: float
    avg_vol: float
    vol_zscore: float
    spike_magnitude: float
    regime: VolSpikeRegime


@dataclass
class GammaScalpingSignal:
    """Gamma scalping trading signal."""
    timestamp: datetime
    symbol: str
    vol_zscore: float
    regime: VolSpikeRegime
    signal: float  # 0-1, higher = more aggressive gamma scalping
    option_position: float
    delta_hedge_ratio: float
    confidence: float
    expected_gamma_pnl: float


class GammaScalpingAlpha:
    """
    Gamma scalping after vol spike alpha strategy.
    
    This class buys options during volatility spikes and delta-hedges
    to capture gamma profits.
    """
    
    def __init__(
        self,
        vol_lookback: int = 20,
        spike_threshold: float = 2.0,  # 2 standard deviations
        hedge_frequency: int = 60,  # Hedge every 60 minutes
        max_position_size: float = 0.10
    ):
        """
        Initialize gamma scalping alpha.
        
        Args:
            vol_lookback: Lookback period for vol average
            spike_threshold: Volatility spike threshold (z-score)
            hedge_frequency: Delta hedge frequency in minutes
            max_position_size: Maximum position size
        """
        self.vol_lookback = vol_lookback
        self.spike_threshold = spike_threshold
        self.hedge_frequency = hedge_frequency
        self.max_position_size = max_position_size
        
        self.measurements: List[VolSpikeMeasurement] = []
        self.signals: List[GammaScalpingSignal] = []
        self.vol_history: Dict[str, List[float]] = {}
        
        logger.info(f"GammaScalpingAlpha initialized: vol_lookback={vol_lookback}, "
                   f"spike_threshold={spike_threshold}")
    
    def calculate_vol_zscore(
        self,
        current_vol: float,
        vol_history: List[float]
    ) -> float:
        """
        Calculate volatility z-score.
        
        Args:
            current_vol: Current volatility
            vol_history: Historical volatility values
            
        Returns:
            Z-score
        """
        if len(vol_history) < self.vol_lookback:
            return 0.0
        
        recent_vols = vol_history[-self.vol_lookback:]
        avg_vol = np.mean(recent_vols)
        std_vol = np.std(recent_vols)
        
        if std_vol == 0:
            return 0.0
        
        zscore = (current_vol - avg_vol) / std_vol
        return zscore
    
    def determine_regime(
        self,
        vol_zscore: float
    ) -> VolSpikeRegime:
        """
        Determine volatility spike regime.
        
        Args:
            vol_zscore: Volatility z-score
            
        Returns:
            VolSpikeRegime
        """
        if vol_zscore > self.spike_threshold * 2:
            return VolSpikeRegime.EXTREME
        elif vol_zscore > self.spike_threshold:
            return VolSpikeRegime.SPIKE
        else:
            return VolSpikeRegime.NORMAL
    
    def generate_signal(
        self,
        symbol: str,
        current_vol: float,
        underlying_price: float,
        timestamp: datetime
    ) -> Optional[GammaScalpingSignal]:
        """
        Generate gamma scalping signal.
        
        Args:
            symbol: Underlying symbol
            current_vol: Current implied volatility
            underlying_price: Current underlying price
            timestamp: Signal timestamp
            
        Returns:
            GammaScalpingSignal or None
        """
        # Update vol history
        if symbol not in self.vol_history:
            self.vol_history[symbol] = []
        self.vol_history[symbol].append(current_vol)
        
        # Keep history manageable
        if len(self.vol_history[symbol]) > 100:
            self.vol_history[symbol] = self.vol_history[symbol][-100:]
        
        # Calculate vol z-score
        vol_zscore = self.calculate_vol_zscore(current_vol, self.vol_history[symbol])
        
        # Calculate average vol
        if len(self.vol_history[symbol]) >= self.vol_lookback:
            avg_vol = np.mean(self.vol_history[symbol][-self.vol_lookback:])
        else:
            avg_vol = current_vol
        
        # Determine regime
        regime = self.determine_regime(vol_zscore)
        
        # Generate signal based on regime
        if regime == VolSpikeRegime.NORMAL:
            return None
        elif regime == VolSpikeRegime.SPIKE:
            signal = 0.7
            option_position = self.max_position_size
            delta_hedge_ratio = 1.0  # Full hedge
            confidence = min(vol_zscore / 3.0, 0.9)
            expected_gamma_pnl = vol_zscore * 0.01
        elif regime == VolSpikeRegime.EXTREME:
            signal = 1.0
            option_position = self.max_position_size * 1.2  # Larger position in extreme
            delta_hedge_ratio = 0.8  # Partial hedge to capture directional
            confidence = min(vol_zscore / 4.0, 0.95)
            expected_gamma_pnl = vol_zscore * 0.015
        else:
            return None
        
        # Store measurement
        measurement = VolSpikeMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            current_vol=current_vol,
            avg_vol=avg_vol,
            vol_zscore=vol_zscore,
            spike_magnitude=vol_zscore,
            regime=regime
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        gamma_signal = GammaScalpingSignal(
            timestamp=timestamp,
            symbol=symbol,
            vol_zscore=vol_zscore,
            regime=regime,
            signal=signal,
            option_position=option_position,
            delta_hedge_ratio=delta_hedge_ratio,
            confidence=confidence,
            expected_gamma_pnl=expected_gamma_pnl
        )
        
        self.signals.append(gamma_signal)
        
        return gamma_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[GammaScalpingSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_vol_spike_statistics(self) -> Dict[str, float]:
        """Get volatility spike statistics."""
        if not self.measurements:
            return {}
        
        zscores = [m.vol_zscore for m in self.measurements]
        
        return {
            'avg_zscore': np.mean(zscores),
            'std_zscore': np.std(zscores),
            'max_zscore': np.max(zscores),
            'current_zscore': zscores[-1] if zscores else 0.0
        }
    
    def print_gamma_scalping_report(self) -> None:
        """Print gamma scalping analysis report."""
        print("\n" + "="*60)
        print("GAMMA SCALPING AFTER VOL SPIKE ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Vol Lookback: {self.vol_lookback}")
        print(f"  Spike Threshold: {self.spike_threshold} std")
        print(f"  Hedge Frequency: {self.hedge_frequency} minutes")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_vol_spike_statistics()
            print(f"\nVol Spike Statistics:")
            print(f"  Average Z-Score: {stats['avg_zscore']:.4f}")
            print(f"  Z-Score Std: {stats['std_zscore']:.4f}")
            print(f"  Max Z-Score: {stats['max_zscore']:.4f}")
            print(f"  Current Z-Score: {stats['current_zscore']:.4f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Z-Score':<10} {'Regime':<12} {'Signal':<10} {'OptPos':<10} {'HedgeRatio':<12}")
            print("-" * 100)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.vol_zscore:<10.4f} {signal.regime.value:<12} {signal.signal:<10.3f} "
                      f"{signal.option_position:<10.3f} {signal.delta_hedge_ratio:<12.3f}")
        
        print("\n" + "="*60)


def sample_gamma_scalping_alpha():
    """Demonstrate gamma scalping alpha."""
    print("=== Gamma Scalping After Vol Spike Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = GammaScalpingAlpha(
        vol_lookback=20,
        spike_threshold=2.0,
        hedge_frequency=60,
        max_position_size=0.10
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate volatility with spikes
    base_vol = 0.20
    vols = base_vol + np.random.randn(n_days) * 0.02
    
    # Add volatility spikes
    spike_indices = [30, 50, 70]
    for idx in spike_indices:
        vols[idx] += 0.08  # Spike
    
    underlying_price = 1000.0
    
    # Process data
    print("Processing volatility data...")
    for i in range(n_days):
        signal = alpha.generate_signal(
            'NIFTY',
            vols[i],
            underlying_price,
            dates[i]
        )
    
    # Print report
    alpha.print_gamma_scalping_report()
    
    print("\n=== Gamma Scalping After Vol Spike Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Volatility spike detection (z-score)")
    print("- Regime classification (normal, spike, extreme)")
    print("- Option position sizing based on spike magnitude")
    print("- Delta hedge ratio adjustment")
    print("- Expected gamma PnL estimation")
    print("- Expected Sharpe: 0.4-0.8")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_gamma_scalping_alpha()

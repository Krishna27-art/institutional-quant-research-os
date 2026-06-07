"""
Volatility Risk Premium with Tail Hedge Alpha Strategy

This module implements the VRP harvesting strategy with tail hedging,
systematically selling volatility while protecting against extreme
volatility spikes using OTM options.

Based on standard VRP with tail hedge practice.
Expected Sharpe: 0.5-0.8
Expected Capacity: High
Decay: Persistent
Difficulty: Medium

Priority: Medium (Options Phase 9)
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


class VRPRegime(Enum):
    """VRP regime."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    EXTREME = "extreme"


@dataclass
class VRPTailHedgeMeasurement:
    """VRP with tail hedge measurement."""
    timestamp: datetime
    symbol: str
    implied_vol: float
    realized_vol: float
    vrp: float
    vrp_percentile: float
    regime: VRPRegime
    tail_hedge_ratio: float


@dataclass
class VRPTailHedgeSignal:
    """VRP with tail hedge trading signal."""
    timestamp: datetime
    symbol: str
    vrp: float
    regime: VRPRegime
    signal: float  # -1 to 1, negative = short vol
    short_vol_position: float
    tail_hedge_position: float
    confidence: float
    expected_return: float


class VRPTailHedgeAlpha:
    """
    VRP with tail hedge alpha strategy.
    
    This class sells volatility to harvest VRP while using
    OTM options as tail protection.
    """
    
    def __init__(
        self,
        lookback_days: int = 252,
        vrp_threshold: float = 0.02,
        tail_hedge_delta: float = 0.10,  # 10 delta for tail hedge
        max_short_position: float = 0.15,
        max_tail_hedge: float = 0.05
    ):
        """
        Initialize VRP tail hedge alpha.
        
        Args:
            lookback_days: Lookback period for VRP calculation
            vrp_threshold: Minimum VRP for signal
            tail_hedge_delta: Delta for tail hedge options
            max_short_position: Maximum short vol position
            max_tail_hedge: Maximum tail hedge position
        """
        self.lookback_days = lookback_days
        self.vrp_threshold = vrp_threshold
        self.tail_hedge_delta = tail_hedge_delta
        self.max_short_position = max_short_position
        self.max_tail_hedge = max_tail_hedge
        
        self.measurements: List[VRPTailHedgeMeasurement] = []
        self.signals: List[VRPTailHedgeSignal] = []
        
        logger.info(f"VRPTailHedgeAlpha initialized: lookback={lookback_days}days, "
                   f"vrp_threshold={vrp_threshold}, tail_hedge_delta={tail_hedge_delta}")
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate realized volatility.
        
        Args:
            returns: Return series
            annualization_factor: Annualization factor
            
        Returns:
            Realized volatility (annualized)
        """
        if len(returns) < 20:
            return 0.0
        
        realized_vol = returns.std() * np.sqrt(annualization_factor)
        return realized_vol
    
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
    
    def calculate_vrp_percentile(self, current_vrp: float) -> float:
        """
        Calculate VRP percentile based on history.
        
        Args:
            current_vrp: Current VRP
            
        Returns:
            VRP percentile (0-1)
        """
        if not self.measurements:
            return 0.5
        
        vrp_values = [m.vrp for m in self.measurements]
        percentile = np.sum([1 for v in vrp_values if v <= current_vrp]) / len(vrp_values)
        return percentile
    
    def determine_regime(
        self,
        vrp: float,
        vrp_percentile: float
    ) -> VRPRegime:
        """
        Determine VRP regime.
        
        Args:
            vrp: Current VRP
            vrp_percentile: VRP percentile
            
        Returns:
            VRPRegime
        """
        if vrp_percentile > 0.9:
            return VRPRegime.EXTREME
        elif vrp_percentile > 0.75 or vrp > self.vrp_threshold * 2:
            return VRPRegime.ELEVATED
        else:
            return VRPRegime.NORMAL
    
    def calculate_tail_hedge_ratio(self, regime: VRPRegime) -> float:
        """
        Calculate tail hedge ratio based on regime.
        
        Args:
            regime: Current regime
            
        Returns:
            Tail hedge ratio (0-1)
        """
        if regime == VRPRegime.EXTREME:
            return 1.0  # Full hedge
        elif regime == VRPRegime.ELEVATED:
            return 0.5  # Half hedge
        else:
            return 0.2  # Minimal hedge
    
    def generate_signal(
        self,
        symbol: str,
        implied_vol: float,
        returns: pd.Series,
        timestamp: datetime
    ) -> Optional[VRPTailHedgeSignal]:
        """
        Generate VRP with tail hedge signal.
        
        Args:
            symbol: Stock/index symbol
            implied_vol: Implied volatility
            returns: Return series
            timestamp: Signal timestamp
            
        Returns:
            VRPTailHedgeSignal or None
        """
        # Calculate realized volatility
        realized_vol = self.calculate_realized_volatility(returns)
        
        # Calculate VRP
        vrp = self.calculate_vrp(implied_vol, realized_vol)
        
        # Calculate VRP percentile
        vrp_percentile = self.calculate_vrp_percentile(vrp)
        
        # Determine regime
        regime = self.determine_regime(vrp, vrp_percentile)
        
        # Calculate tail hedge ratio
        tail_hedge_ratio = self.calculate_tail_hedge_ratio(regime)
        
        # Generate signal based on regime
        if regime == VRPRegime.NORMAL:
            signal = -1.0  # Short vol
            short_vol_position = self.max_short_position
            tail_hedge_position = self.max_tail_hedge * tail_hedge_ratio
            confidence = vrp_percentile
            expected_return = vrp * 0.5
        elif regime == VRPRegime.ELEVATED:
            signal = -0.7  # Reduced short vol
            short_vol_position = self.max_short_position * 0.7
            tail_hedge_position = self.max_tail_hedge * tail_hedge_ratio
            confidence = vrp_percentile * 0.8
            expected_return = vrp * 0.3
        elif regime == VRPRegime.EXTREME:
            signal = -0.3  # Minimal short vol
            short_vol_position = self.max_short_position * 0.3
            tail_hedge_position = self.max_tail_hedge * tail_hedge_ratio
            confidence = vrp_percentile * 0.6
            expected_return = vrp * 0.1
        else:
            return None
        
        # Store measurement
        measurement = VRPTailHedgeMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            implied_vol=implied_vol,
            realized_vol=realized_vol,
            vrp=vrp,
            vrp_percentile=vrp_percentile,
            regime=regime,
            tail_hedge_ratio=tail_hedge_ratio
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        vrp_signal = VRPTailHedgeSignal(
            timestamp=timestamp,
            symbol=symbol,
            vrp=vrp,
            regime=regime,
            signal=signal,
            short_vol_position=short_vol_position,
            tail_hedge_position=tail_hedge_position,
            confidence=confidence,
            expected_return=expected_return
        )
        
        self.signals.append(vrp_signal)
        
        return vrp_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[VRPTailHedgeSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_vrp_statistics(self) -> Dict[str, float]:
        """Get VRP statistics."""
        if not self.measurements:
            return {}
        
        vrp_values = [m.vrp for m in self.measurements]
        
        return {
            'avg_vrp': np.mean(vrp_values),
            'std_vrp': np.std(vrp_values),
            'min_vrp': np.min(vrp_values),
            'max_vrp': np.max(vrp_values),
            'current_vrp': vrp_values[-1] if vrp_values else 0.0
        }
    
    def print_vrp_tail_hedge_report(self) -> None:
        """Print VRP tail hedge analysis report."""
        print("\n" + "="*60)
        print("VRP WITH TAIL HEDGE ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  VRP Threshold: {self.vrp_threshold:.2%}")
        print(f"  Tail Hedge Delta: {self.tail_hedge_delta}")
        print(f"  Max Short Position: {self.max_short_position:.2%}")
        print(f"  Max Tail Hedge: {self.max_tail_hedge:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_vrp_statistics()
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
            print(f"{'Timestamp':<20} {'Symbol':<10} {'VRP':<10} {'Regime':<12} {'Signal':<10} {'ShortPos':<12} {'TailHedge':<12}")
            print("-" * 105)
            
            for signal in self.signals[-5]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.vrp:<10.4f} {signal.regime.value:<12} {signal.signal:<10.3f} "
                      f"{signal.short_vol_position:<12.3f} {signal.tail_hedge_position:<12.3f}")
        
        print("\n" + "="*60)


def sample_vrp_tail_hedge_alpha():
    """Demonstrate VRP tail hedge alpha."""
    print("=== VRP with Tail Hedge Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = VRPTailHedgeAlpha(
        lookback_days=252,
        vrp_threshold=0.02,
        tail_hedge_delta=0.10,
        max_short_position=0.15,
        max_tail_hedge=0.05
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 300
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate implied vol (systematically higher than realized)
    base_implied = 0.20
    implied_vols = base_implied + np.random.randn(n_days) * 0.03
    
    # Generate returns with lower realized vol
    base_realized = 0.15
    returns = pd.Series(np.random.randn(n_days) * base_realized / np.sqrt(252), index=dates)
    
    # Process data
    print("Processing VRP with tail hedge data...")
    for i in range(100, n_days):
        signal = alpha.generate_signal(
            'NIFTY',
            implied_vols[i],
            returns.iloc[:i],
            dates[i]
        )
    
    # Print report
    alpha.print_vrp_tail_hedge_report()
    
    print("\n=== VRP with Tail Hedge Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- VRP calculation (implied - realized volatility)")
    print("- VRP percentile ranking")
    print("- Regime detection (normal, elevated, extreme)")
    print("- Systematic volatility selling")
    print("- Tail hedging with OTM options")
    print("- Dynamic hedge ratio adjustment")
    print("- Expected Sharpe: 0.5-0.8")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_vrp_tail_hedge_alpha()

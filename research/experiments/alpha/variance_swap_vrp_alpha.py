"""
Short Variance Swap (VRP Harvesting) Alpha Strategy

This module implements the short variance swap strategy that harvests the
volatility risk premium by systematically selling volatility when implied
volatility exceeds realized volatility.

Based on Carr & Wu 2009; thesis 2010-2022.
Expected Sharpe: 0.8-1.2
Expected Capacity: Very High
Decay: Persistent
Difficulty: Medium

Priority: High (Options Phase 1)
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
    """Volatility risk premium regime."""
    NORMAL = "normal"  # Short vol
    STRESS = "stress"  # Reduce or flatten
    EXTREME = "extreme"  # Avoid short vol


@dataclass
class VRPMeasurement:
    """VRP measurement."""
    timestamp: datetime
    symbol: str
    implied_vol: float
    realized_vol: float
    vrp: float  # Implied - Realized
    vrp_percentile: float
    regime: VRPRegime


@dataclass
class VarianceSwapSignal:
    """Variance swap trading signal."""
    timestamp: datetime
    symbol: str
    vrp: float
    vrp_percentile: float
    regime: VRPRegime
    signal: float  # -1 to 1, negative = short vol
    position_size: float
    confidence: float
    expected_return: float
    tail_hedge_ratio: float


class VarianceSwapVRPAlpha:
    """
    Short variance swap VRP harvesting alpha strategy.
    
    This class implements systematic volatility selling to harvest
    the volatility risk premium.
    """
    
    def __init__(
        self,
        lookback_days: int = 252,
        vrp_threshold: float = 0.02,  # 2% VRP threshold
        percentile_threshold: float = 0.7,  # 70th percentile
        max_position_size: float = 0.15,  # 15% of portfolio
        tail_hedge: bool = True
    ):
        """
        Initialize variance swap VRP alpha.
        
        Args:
            lookback_days: Lookback period for VRP calculation
            vrp_threshold: Minimum VRP for signal
            percentile_threshold: VRP percentile threshold
            max_position_size: Maximum position size
            tail_hedge: Enable tail hedging
        """
        self.lookback_days = lookback_days
        self.vrp_threshold = vrp_threshold
        self.percentile_threshold = percentile_threshold
        self.max_position_size = max_position_size
        self.tail_hedge = tail_hedge
        
        self.vrp_history: List[VRPMeasurement] = []
        self.signals: List[VarianceSwapSignal] = []
        
        logger.info(f"VarianceSwapVRPAlpha initialized: lookback={lookback_days}days, "
                   f"vrp_threshold={vrp_threshold}")
    
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
        if not self.vrp_history:
            return 0.5
        
        vrp_values = [m.vrp for m in self.vrp_history]
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
        if vrp_percentile > 0.9 or vrp < 0:
            return VRPRegime.EXTREME
        elif vrp_percentile > 0.8:
            return VRPRegime.STRESS
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
            return 0.5  # 50% hedge
        elif regime == VRPRegime.STRESS:
            return 0.3  # 30% hedge
        else:
            return 0.1  # 10% hedge
    def generate_signal(
        self,
        symbol: str,
        implied_vol: float,
        returns: pd.Series,
        timestamp: datetime
    ) -> Optional[VarianceSwapSignal]:
        """
        Generate variance swap signal.
        
        Args:
            symbol: Stock/index symbol
            implied_vol: Implied volatility
            returns: Return series
            timestamp: Signal timestamp
            
        Returns:
            VarianceSwapSignal or None
        """
        # Calculate realized volatility
        realized_vol = self.calculate_realized_volatility(returns)
        
        # Calculate VRP
        vrp = self.calculate_vrp(implied_vol, realized_vol)
        
        # Calculate VRP percentile
        vrp_percentile = self.calculate_vrp_percentile(vrp)
        
        # Determine regime
        regime = self.determine_regime(vrp, vrp_percentile)
        
        # Check if VRP meets threshold
        if vrp < self.vrp_threshold or regime == VRPRegime.EXTREME:
            return None
        
        # Generate signal
        if regime == VRPRegime.NORMAL:
            signal = -1.0  # Short vol
            confidence = vrp_percentile
        elif regime == VRPRegime.STRESS:
            signal = -0.5  # Reduced short vol
            confidence = vrp_percentile * 0.7
        else:
            signal = 0.0
            confidence = 0.0
        
        # Position sizing based on VRP strength
        position_size = min(abs(vrp) / 0.05, 1.0) * self.max_position_size
        
        # Expected return (conservative estimate)
        expected_return = vrp * 0.5
        
        # Tail hedge ratio
        tail_hedge_ratio = self.calculate_tail_hedge_ratio(regime) if self.tail_hedge else 0.0
        
        # Store measurement
        measurement = VRPMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            implied_vol=implied_vol,
            realized_vol=realized_vol,
            vrp=vrp,
            vrp_percentile=vrp_percentile,
            regime=regime
        )
        
        self.vrp_history.append(measurement)
        
        # Keep history manageable
        if len(self.vrp_history) > 1000:
            self.vrp_history = self.vrp_history[-1000:]
        
        # Create signal
        variance_signal = VarianceSwapSignal(
            timestamp=timestamp,
            symbol=symbol,
            vrp=vrp,
            vrp_percentile=vrp_percentile,
            regime=regime,
            signal=signal,
            position_size=position_size,
            confidence=confidence,
            expected_return=expected_return,
            tail_hedge_ratio=tail_hedge_ratio
        )
        
        self.signals.append(variance_signal)
        
        return variance_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[VarianceSwapSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_vrp_statistics(self) -> Dict[str, float]:
        """Get VRP statistics."""
        if not self.vrp_history:
            return {}
        
        vrp_values = [m.vrp for m in self.vrp_history]
        
        return {
            'avg_vrp': np.mean(vrp_values),
            'std_vrp': np.std(vrp_values),
            'min_vrp': np.min(vrp_values),
            'max_vrp': np.max(vrp_values),
            'current_vrp': vrp_values[-1] if vrp_values else 0.0
        }
    
    def print_vrp_report(self) -> None:
        """Print VRP analysis report."""
        print("\n" + "="*60)
        print("VARIANCE SWAP VRP HARVESTING ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  VRP Threshold: {self.vrp_threshold:.2%}")
        print(f"  Percentile Threshold: {self.percentile_threshold:.2%}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        print(f"  Tail Hedge: {self.tail_hedge}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.vrp_history)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.vrp_history:
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
            print(f"{'Timestamp':<20} {'Symbol':<10} {'VRP':<10} {'Percentile':<12} {'Regime':<12} {'Signal':<10} {'Position':<10} {'TailHedge':<10}")
            print("-" * 105)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.vrp:<10.4f} {signal.vrp_percentile:<12.4f} {signal.regime.value:<12} "
                      f"{signal.signal:<10.3f} {signal.position_size:<10.3f} {signal.tail_hedge_ratio:<10.3f}")
        
        print("\n" + "="*60)


def sample_variance_swap_vrp_alpha():
    """Demonstrate variance swap VRP alpha."""
    print("=== Variance Swap VRP Harvesting Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = VarianceSwapVRPAlpha(
        lookback_days=252,
        vrp_threshold=0.02,
        percentile_threshold=0.7,
        max_position_size=0.15,
        tail_hedge=True
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
    print("Processing VRP data...")
    for i in range(100, n_days):
        signal = alpha.generate_signal(
            'NIFTY',
            implied_vols[i],
            returns.iloc[:i],
            dates[i]
        )
    
    # Print report
    alpha.print_vrp_report()
    
    print("\n=== Variance Swap VRP Harvesting Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- VRP calculation (implied - realized volatility)")
    print("- VRP percentile ranking")
    print("- Regime detection (normal, stress, extreme)")
    print("- Systematic volatility selling")
    print("- Tail hedging based on regime")
    print("- Expected Sharpe: 0.8-1.2")
    print("- Expected Capacity: Very High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_variance_swap_vrp_alpha()

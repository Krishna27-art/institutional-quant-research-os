"""
Volatility Risk Premium Term Structure Alpha Strategy

This module implements the volatility risk premium (VRP) term structure trading
strategy, which exploits the systematic excess of implied volatility over
realized volatility and the term structure of VIX futures.

Based on Carr & Wu 2009; Driessen, Maenhout, Vilkov 2009; CBS PhD thesis 2025.
Expected Sharpe: 0.6-1.2
Expected Capacity: Very High
Decay: Persistent
Difficulty: Medium

Priority: High (Research OS Phase 3)
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
    CONTANGO = "contango"  # Normal: short vol
    BACKWARDATION = "backwardation"  # Stress: long vol
    FLAT = "flat"  # Neutral


@dataclass
class VRPMeasurement:
    """Volatility risk premium measurement."""
    timestamp: datetime
    implied_vol: float
    realized_vol: float
    vrp: float  # Implied - Realized
    vix_spot: float
    vix_f1: float  # Front month future
    vix_f2: float  # Second month future
    basis_f1: float  # VIX_F1 - VIX_spot
    basis_f2: float  # VIX_F2 - VIX_spot
    term_structure_slope: float  # VIX_F2 - VIX_F1
    regime: VRPRegime


@dataclass
class VRPSignal:
    """VRP trading signal."""
    timestamp: datetime
    regime: VRPRegime
    signal: float  # -1 to 1, negative = short vol, positive = long vol
    position_size: float  # Fraction of capital
    confidence: float
    expected_return: float
    risk_level: str  # low, medium, high


class VolatilityRiskPremiumAlpha:
    """
    Volatility risk premium term structure alpha strategy.
    
    This class implements VRP trading based on the term structure
    of VIX futures and the implied vs realized volatility spread.
    """
    
    def __init__(
        self,
        vrp_threshold: float = 0.02,  # 2% VRP threshold
        basis_threshold: float = 1.0,  # 1 point basis threshold
        term_structure_threshold: float = 0.5,  # 0.5 point slope threshold
        lookback_days: int = 30,
        max_position_size: float = 0.10,  # 10% of portfolio
        vol_scaling: bool = True
    ):
        """
        Initialize VRP alpha.
        
        Args:
            vrp_threshold: VRP threshold for signal generation
            basis_threshold: VIX futures basis threshold
            term_structure_threshold: Term structure slope threshold
            lookback_days: Lookback period for historical analysis
            max_position_size: Maximum position size as portfolio fraction
            vol_scaling: Enable volatility-based position sizing
        """
        self.vrp_threshold = vrp_threshold
        self.basis_threshold = basis_threshold
        self.term_structure_threshold = term_structure_threshold
        self.lookback_days = lookback_days
        self.max_position_size = max_position_size
        self.vol_scaling = vol_scaling
        
        self.vrp_history: List[VRPMeasurement] = []
        self.signals: List[VRPSignal] = []
        
        logger.info(f"VolatilityRiskPremiumAlpha initialized: vrp_threshold={vrp_threshold}, "
                   f"basis_threshold={basis_threshold}, lookback={lookback_days}days")
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate realized volatility.
        
        Args:
            returns: Return series
            annualization_factor: Annualization factor (252 for daily)
            
        Returns:
            Realized volatility (annualized)
        """
        if len(returns) == 0:
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
        
        VRP = Implied Volatility - Realized Volatility
        
        Args:
            implied_vol: Implied volatility (e.g., VIX)
            realized_vol: Realized volatility
            
        Returns:
            VRP value
        """
        return implied_vol - realized_vol
    
    def determine_regime(
        self,
        basis_f1: float,
        term_structure_slope: float
    ) -> VRPRegime:
        """
        Determine VRP regime based on term structure.
        
        Args:
            basis_f1: VIX front month basis (VIX_F1 - VIX_spot)
            term_structure_slope: Term structure slope (VIX_F2 - VIX_F1)
            
        Returns:
            VRPRegime
        """
        # Contango: Futures above spot, upward sloping term structure
        if basis_f1 > self.basis_threshold and term_structure_slope > self.term_structure_threshold:
            return VRPRegime.CONTANGO
        
        # Backwardation: Futures below spot, downward sloping term structure
        elif basis_f1 < -self.basis_threshold and term_structure_slope < -self.term_structure_threshold:
            return VRPRegime.BACKWARDATION
        
        # Flat: Neither condition met
        else:
            return VRPRegime.FLAT
    
    def calculate_position_size(
        self,
        vrp: float,
        basis_f1: float,
        realized_vol: float
    ) -> float:
        """
        Calculate position size based on VRP and volatility.
        
        Args:
            vrp: Volatility risk premium
            basis_f1: VIX front month basis
            realized_vol: Realized volatility
            
        Returns:
            Position size as fraction of capital
        """
        # Base position size based on VRP magnitude
        base_size = min(abs(vrp) / self.vrp_threshold, 2.0) * 0.5
        
        # Adjust based on basis strength
        basis_adjustment = min(abs(basis_f1) / self.basis_threshold, 2.0) * 0.3
        
        # Volatility scaling
        if self.vol_scaling and realized_vol > 0:
            vol_adjustment = 0.2 / realized_vol  # Higher vol = smaller position
            vol_adjustment = min(vol_adjustment, 1.5)
        else:
            vol_adjustment = 1.0
        
        position_size = (base_size + basis_adjustment) * vol_adjustment
        
        # Cap at maximum
        position_size = min(position_size, self.max_position_size)
        
        return position_size
    
    def generate_signal(
        self,
        vix_spot: float,
        vix_f1: float,
        vix_f2: float,
        realized_vol: float,
        timestamp: datetime
    ) -> VRPSignal:
        """
        Generate VRP trading signal.
        
        Args:
            vix_spot: VIX spot price
            vix_f1: VIX front month futures price
            vix_f2: VIX second month futures price
            realized_vol: Realized volatility
            timestamp: Signal timestamp
            
        Returns:
            VRPSignal
        """
        # Calculate VRP (VIX is implied vol)
        implied_vol = vix_spot / 100.0  # Convert VIX to decimal
        vrp = self.calculate_vrp(implied_vol, realized_vol)
        
        # Calculate basis and term structure
        basis_f1 = vix_f1 - vix_spot
        basis_f2 = vix_f2 - vix_spot
        term_structure_slope = vix_f2 - vix_f1
        
        # Determine regime
        regime = self.determine_regime(basis_f1, term_structure_slope)
        
        # Generate signal
        if regime == VRPRegime.CONTANGO:
            # Short volatility (negative signal)
            signal = -1.0
            expected_return = vrp * 0.5  # Conservative estimate
            risk_level = "medium"
        elif regime == VRPRegime.BACKWARDATION:
            # Long volatility (positive signal)
            signal = 1.0
            expected_return = -vrp * 0.5  # Conservative estimate
            risk_level = "high"
        else:
            # Neutral
            signal = 0.0
            expected_return = 0.0
            risk_level = "low"
        
        # Calculate position size
        position_size = self.calculate_position_size(vrp, basis_f1, realized_vol)
        
        # Adjust signal by position size
        signal = signal * position_size
        
        # Calculate confidence based on regime strength
        if regime == VRPRegime.CONTANGO:
            confidence = min(abs(basis_f1) / (self.basis_threshold * 2), 0.9)
        elif regime == VRPRegime.BACKWARDATION:
            confidence = min(abs(basis_f1) / (self.basis_threshold * 2), 0.7)
        else:
            confidence = 0.3
        
        # Store measurement
        measurement = VRPMeasurement(
            timestamp=timestamp,
            implied_vol=implied_vol,
            realized_vol=realized_vol,
            vrp=vrp,
            vix_spot=vix_spot,
            vix_f1=vix_f1,
            vix_f2=vix_f2,
            basis_f1=basis_f1,
            basis_f2=basis_f2,
            term_structure_slope=term_structure_slope,
            regime=regime
        )
        
        self.vrp_history.append(measurement)
        
        # Create signal
        vrp_signal = VRPSignal(
            timestamp=timestamp,
            regime=regime,
            signal=signal,
            position_size=position_size,
            confidence=confidence,
            expected_return=expected_return,
            risk_level=risk_level
        )
        
        self.signals.append(vrp_signal)
        
        # Keep history manageable
        if len(self.vrp_history) > 1000:
            self.vrp_history = self.vrp_history[-1000:]
        if len(self.signals) > 1000:
            self.signals = self.signals[-1000:]
        
        return vrp_signal
    
    def get_latest_signal(self) -> Optional[VRPSignal]:
        """Get the latest signal."""
        return self.signals[-1] if self.signals else None
    
    def get_vrp_statistics(self) -> Dict[str, float]:
        """Get VRP statistics."""
        if not self.vrp_history:
            return {}
        
        vrp_values = [m.vrp for m in self.vrp_history]
        basis_values = [m.basis_f1 for m in self.vrp_history]
        
        return {
            'avg_vrp': np.mean(vrp_values),
            'std_vrp': np.std(vrp_values),
            'min_vrp': np.min(vrp_values),
            'max_vrp': np.max(vrp_values),
            'avg_basis': np.mean(basis_values),
            'std_basis': np.std(basis_values)
        }
    
    def print_vrp_report(self) -> None:
        """Print VRP analysis report."""
        print("\n" + "="*60)
        print("VOLATILITY RISK PREMIUM ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  VRP Threshold: {self.vrp_threshold:.2%}")
        print(f"  Basis Threshold: {self.basis_threshold:.2f}")
        print(f"  Term Structure Threshold: {self.term_structure_threshold:.2f}")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
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
            print(f"  Average Basis: {stats['avg_basis']:.2f}")
            print(f"  Basis Std: {stats['std_basis']:.2f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Regime':<15} {'Signal':<10} {'Position':<10} {'Confidence':<12} {'Risk':<10}")
            print("-" * 85)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.regime.value:<15} "
                      f"{signal.signal:<10.3f} {signal.position_size:<10.3f} {signal.confidence:<12.2f} "
                      f"{signal.risk_level:<10}")
        
        print("\n" + "="*60)


def sample_volatility_risk_premium_alpha():
    """Demonstrate VRP alpha."""
    print("=== Volatility Risk Premium Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = VolatilityRiskPremiumAlpha(
        vrp_threshold=0.02,
        basis_threshold=1.0,
        term_structure_threshold=0.5,
        lookback_days=30,
        max_position_size=0.10,
        vol_scaling=True
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate VIX and futures data with contango (normal regime)
    vix_spot = 20 + np.random.randn(n_days) * 3
    vix_f1 = vix_spot + 1.5 + np.random.randn(n_days) * 0.5  # Contango
    vix_f2 = vix_f1 + 0.8 + np.random.randn(n_days) * 0.3  # Upward sloping
    
    # Generate realized volatility (lower than implied)
    realized_vol = (vix_spot / 100) - 0.02 + np.random.randn(n_days) * 0.01
    
    # Process data
    print("Processing VRP data...")
    for i in range(n_days):
        signal = alpha.generate_signal(
            vix_spot[i],
            vix_f1[i],
            vix_f2[i],
            realized_vol[i],
            dates[i]
        )
    
    # Print report
    alpha.print_vrp_report()
    
    print("\n=== Volatility Risk Premium Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Volatility risk premium calculation (implied - realized)")
    print("- VIX futures term structure analysis")
    print("- Regime detection (contango, backwardation, flat)")
    print("- Trading signal generation based on regime")
    print("- Volatility-based position sizing")
    print("- Expected Sharpe: 0.6-1.2")
    print("- Expected Capacity: Very High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_volatility_risk_premium_alpha()

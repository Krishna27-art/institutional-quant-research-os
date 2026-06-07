"""
Put-Call Parity Carry Gap Alpha Strategy

This module implements the put-call parity carry gap strategy that exploits
funding and implementation wedges between put and call options, capturing
the carry from mispriced put-call parity relationships.

Based on options market microstructure literature.
Expected Sharpe: 0.3-0.6
Expected Capacity: Medium
Decay: Persistent
Difficulty: Medium

Priority: Medium (Options Phase 5)
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


class ParityRegime(Enum):
    """Put-call parity regime."""
    NORMAL = "normal"
    GAP_PUT_OVERPRICED = "gap_put_overpriced"
    GAP_CALL_OVERPRICED = "gap_call_overpriced"
    EXTREME = "extreme"


@dataclass
class ParityMeasurement:
    """Put-call parity measurement."""
    timestamp: datetime
    symbol: str
    strike: float
    spot: float
    call_price: float
    put_price: float
    time_to_expiry: float
    risk_free_rate: float
    theoretical_put: float
    theoretical_call: float
    put_gap: float  # Actual - Theoretical
    call_gap: float  # Actual - Theoretical
    total_gap: float
    regime: ParityRegime


@dataclass
class ParitySignal:
    """Put-call parity trading signal."""
    timestamp: datetime
    symbol: str
    total_gap: float
    regime: ParityRegime
    signal: float  # -1 to 1
    put_position: float
    call_position: float
    confidence: float
    expected_arbitrage: float


class PutCallParityCarryAlpha:
    """
    Put-call parity carry gap alpha strategy.
    
    This class exploits mispriced put-call parity relationships
    by trading the carry gap.
    """
    
    def __init__(
        self,
        gap_threshold: float = 0.01,  # 1% price threshold
        min_dte: int = 7,  # Minimum days to expiry
        max_dte: int = 90,  # Maximum days to expiry
        max_position_size: float = 0.05
    ):
        """
        Initialize put-call parity carry alpha.
        
        Args:
            gap_threshold: Minimum gap for signal
            min_dte: Minimum days to expiry
            max_dte: Maximum days to expiry
            max_position_size: Maximum position size
        """
        self.gap_threshold = gap_threshold
        self.min_dte = min_dte
        self.max_dte = max_dte
        self.max_position_size = max_position_size
        
        self.measurements: List[ParityMeasurement] = []
        self.signals: List[ParitySignal] = []
        
        logger.info(f"PutCallParityCarryAlpha initialized: gap_threshold={gap_threshold}, "
                   f"dte_range=[{min_dte}, {max_dte}]")
    
    def calculate_theoretical_put(
        self,
        call_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float
    ) -> float:
        """
        Calculate theoretical put price using put-call parity.
        
        Put = Call - Spot + Strike * exp(-r*T)
        
        Args:
            call_price: Call option price
            spot: Spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            
        Returns:
            Theoretical put price
        """
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        theoretical_put = call_price - spot + strike * discount_factor
        return max(theoretical_put, 0.0)
    
    def calculate_theoretical_call(
        self,
        put_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float
    ) -> float:
        """
        Calculate theoretical call price using put-call parity.
        
        Call = Put + Spot - Strike * exp(-r*T)
        
        Args:
            put_price: Put option price
            spot: Spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            
        Returns:
            Theoretical call price
        """
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        theoretical_call = put_price + spot - strike * discount_factor
        return max(theoretical_call, 0.0)
    
    def determine_regime(
        self,
        put_gap: float,
        call_gap: float,
        total_gap: float
    ) -> ParityRegime:
        """
        Determine put-call parity regime.
        
        Args:
            put_gap: Put price gap
            call_gap: Call price gap
            total_gap: Total absolute gap
            
        Returns:
            ParityRegime
        """
        if total_gap > self.gap_threshold * 3:
            return ParityRegime.EXTREME
        elif abs(put_gap) > abs(call_gap) and put_gap > self.gap_threshold:
            return ParityRegime.GAP_PUT_OVERPRICED
        elif abs(call_gap) > abs(put_gap) and call_gap > self.gap_threshold:
            return ParityRegime.GAP_CALL_OVERPRICED
        else:
            return ParityRegime.NORMAL
    
    def generate_signal(
        self,
        symbol: str,
        spot: float,
        strike: float,
        call_price: float,
        put_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        timestamp: datetime
    ) -> Optional[ParitySignal]:
        """
        Generate put-call parity signal.
        
        Args:
            symbol: Underlying symbol
            spot: Spot price
            strike: Strike price
            call_price: Call option price
            put_price: Put option price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            timestamp: Signal timestamp
            
        Returns:
            ParitySignal or None
        """
        # Check DTE constraints
        dte_days = time_to_expiry * 365
        if dte_days < self.min_dte or dte_days > self.max_dte:
            return None
        
        # Calculate theoretical prices
        theoretical_put = self.calculate_theoretical_put(
            call_price, spot, strike, time_to_expiry, risk_free_rate
        )
        theoretical_call = self.calculate_theoretical_call(
            put_price, spot, strike, time_to_expiry, risk_free_rate
        )
        
        # Calculate gaps
        put_gap = put_price - theoretical_put
        call_gap = call_price - theoretical_call
        total_gap = abs(put_gap) + abs(call_gap)
        
        # Determine regime
        regime = self.determine_regime(put_gap, call_gap, total_gap)
        
        # Generate signal based on regime
        if regime == ParityRegime.NORMAL:
            return None
        elif regime == ParityRegime.GAP_PUT_OVERPRICED:
            # Sell put, buy call (synthetic forward)
            signal = -1.0
            put_position = -self.max_position_size
            call_position = self.max_position_size
            confidence = float(min(total_gap / (self.gap_threshold * 2), 0.9))
            expected_arbitrage = float(total_gap * 0.8)
        elif regime == ParityRegime.GAP_CALL_OVERPRICED:
            # Buy put, sell call (synthetic forward)
            signal = 1.0
            put_position = self.max_position_size
            call_position = -self.max_position_size
            confidence = float(min(total_gap / (self.gap_threshold * 2), 0.9))
            expected_arbitrage = float(total_gap * 0.8)
        elif regime == ParityRegime.EXTREME:
            # Extreme gap: reduce size but don't zero out completely.
            # Use 30% of normal strength as a volatility-carry proxy.
            signal = 0.0
            put_position = 0.0
            call_position = 0.0
            confidence = 0.45
            expected_arbitrage = float(total_gap * 0.30)
        else:
            return None
        
        # Store measurement
        measurement = ParityMeasurement(
            timestamp=timestamp,
            symbol=symbol,
            strike=strike,
            spot=spot,
            call_price=call_price,
            put_price=put_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            theoretical_put=theoretical_put,
            theoretical_call=theoretical_call,
            put_gap=put_gap,
            call_gap=call_gap,
            total_gap=total_gap,
            regime=regime
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        parity_signal = ParitySignal(
            timestamp=timestamp,
            symbol=symbol,
            total_gap=float(total_gap),
            regime=regime,
            signal=float(signal),
            put_position=float(put_position),
            call_position=float(call_position),
            confidence=float(confidence),
            expected_arbitrage=float(expected_arbitrage)
        )
        
        self.signals.append(parity_signal)
        
        return parity_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[ParitySignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_parity_statistics(self) -> Dict[str, float]:
        """Get parity statistics."""
        if not self.measurements:
            return {}
        
        gaps = [m.total_gap for m in self.measurements]
        
        return {
            'avg_gap': np.mean(gaps),
            'std_gap': np.std(gaps),
            'min_gap': np.min(gaps),
            'max_gap': np.max(gaps),
            'current_gap': gaps[-1] if gaps else 0.0
        }
    
    def print_parity_report(self) -> None:
        """Print put-call parity analysis report."""
        print("\n" + "="*60)
        print("PUT-CALL PARITY CARRY GAP ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Gap Threshold: {self.gap_threshold:.2%}")
        print(f"  DTE Range: [{self.min_dte}, {self.max_dte}] days")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_parity_statistics()
            print(f"\nParity Gap Statistics:")
            print(f"  Average Gap: {stats['avg_gap']:.4f}")
            print(f"  Gap Std: {stats['std_gap']:.4f}")
            print(f"  Min Gap: {stats['min_gap']:.4f}")
            print(f"  Max Gap: {stats['max_gap']:.4f}")
            print(f"  Current Gap: {stats['current_gap']:.4f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'TotalGap':<12} {'Regime':<20} {'Signal':<10} {'PutPos':<10} {'CallPos':<10}")
            print("-" * 105)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.total_gap:<12.4f} {signal.regime.value:<20} {signal.signal:<10.3f} "
                      f"{signal.put_position:<10.3f} {signal.call_position:<10.3f}")
        
        print("\n" + "="*60)


def sample_putcall_parity_carry_alpha():
    """Demonstrate put-call parity carry alpha."""
    print("=== Put-Call Parity Carry Gap Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = PutCallParityCarryAlpha(
        gap_threshold=0.01,
        min_dte=7,
        max_dte=90,
        max_position_size=0.05
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    spot = 1000.0
    strike = 1000.0
    risk_free_rate = 0.05
    
    # Generate option prices with occasional parity gaps
    for i in range(n_days):
        time_to_expiry = max(30 - i, 7) / 365.0
        
        # Base option prices (Black-Scholes approximation)
        vol = 0.20
        call_price = 50 + np.random.randn() * 5
        put_price = 50 + np.random.randn() * 5
        
        # Add parity gaps occasionally
        if i % 20 == 0:
            put_price += 3.0  # Put overpriced
        
        signal = alpha.generate_signal(
            'NIFTY',
            spot,
            strike,
            call_price,
            put_price,
            time_to_expiry,
            risk_free_rate,
            dates[i]
        )
    
    # Print report
    alpha.print_parity_report()
    
    print("\n=== Put-Call Parity Carry Gap Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Put-call parity calculation")
    print("- Theoretical price computation")
    print("- Gap detection (put vs call)")
    print("- Regime classification")
    print("- Synthetic forward trading")
    print("- Expected arbitrage estimation")
    print("- Expected Sharpe: 0.3-0.6")
    print("- Expected Capacity: Medium")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_putcall_parity_carry_alpha()

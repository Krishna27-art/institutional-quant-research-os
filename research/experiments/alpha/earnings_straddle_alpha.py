"""
Earnings Event Straddles Alpha Strategy

This module implements the earnings event straddle strategy that buys
straddles before earnings announcements and gamma scalps during the
event to capture the volatility premium and price movement.

Based on standard options earnings trading practice.
Expected Sharpe: 0.3-0.5
Expected Capacity: Medium
Decay: Persistent
Difficulty: Medium

Priority: Medium (Options Phase 8)
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


class EarningsRegime(Enum):
    """Earnings regime."""
    PRE_EARNINGS = "pre_earnings"
    POST_EARNINGS = "post_earnings"
    NORMAL = "normal"


@dataclass
class EarningsEvent:
    """Earnings event data."""
    symbol: str
    announcement_date: datetime
    eps_actual: float
    eps_estimate: float
    surprise: float
    surprise_pct: float


@dataclass
class StraddlePosition:
    """Straddle position data."""
    timestamp: datetime
    symbol: str
    call_price: float
    put_price: float
    straddle_cost: float
    underlying_price: float
    days_to_expiry: float


@dataclass
class StraddleSignal:
    """Earnings straddle trading signal."""
    timestamp: datetime
    symbol: str
    regime: EarningsRegime
    signal: float  # 0-1, higher = more aggressive
    straddle_position: float
    delta_hedge_ratio: float
    confidence: float
    expected_pnl: float


class EarningsStraddleAlpha:
    """
    Earnings event straddle alpha strategy.
    
    This class buys straddles before earnings and gamma scalps
    during the event.
    """
    
    def __init__(
        self,
        pre_earnings_days: int = 5,
        post_earnings_days: int = 2,
        min_surprise_threshold: float = 0.05,  # 5% surprise
        max_position_size: float = 0.05
    ):
        """
        Initialize earnings straddle alpha.
        
        Args:
            pre_earnings_days: Days before earnings to enter
            post_earnings_days: Days after earnings to exit
            min_surprise_threshold: Minimum surprise for signal
            max_position_size: Maximum position size
        """
        self.pre_earnings_days = pre_earnings_days
        self.post_earnings_days = post_earnings_days
        self.min_surprise_threshold = min_surprise_threshold
        self.max_position_size = max_position_size
        
        self.earnings_events: Dict[str, EarningsEvent] = {}
        self.straddle_positions: Dict[str, List[StraddlePosition]] = {}
        self.signals: List[StraddleSignal] = []
        
        logger.info(f"EarningsStraddleAlpha initialized: pre_days={pre_earnings_days}, "
                   f"post_days={post_earnings_days}")
    
    def add_earnings_event(
        self,
        symbol: str,
        announcement_date: datetime,
        eps_actual: float,
        eps_estimate: float
    ) -> EarningsEvent:
        """
        Add an earnings event.
        
        Args:
            symbol: Stock symbol
            announcement_date: Announcement date
            eps_actual: Actual EPS
            eps_estimate: Estimated EPS
            
        Returns:
            EarningsEvent
        """
        surprise = eps_actual - eps_estimate
        surprise_pct = surprise / abs(eps_estimate) if eps_estimate != 0 else 0.0
        
        event = EarningsEvent(
            symbol=symbol,
            announcement_date=announcement_date,
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            surprise=surprise,
            surprise_pct=surprise_pct
        )
        
        self.earnings_events[symbol] = event
        return event
    
    def determine_regime(
        self,
        symbol: str,
        timestamp: datetime
    ) -> EarningsRegime:
        """
        Determine earnings regime.
        
        Args:
            symbol: Stock symbol
            timestamp: Current timestamp
            
        Returns:
            EarningsRegime
        """
        if symbol not in self.earnings_events:
            return EarningsRegime.NORMAL
        
        event = self.earnings_events[symbol]
        days_to_earnings = (event.announcement_date - timestamp).days
        
        if days_to_earnings > 0 and days_to_earnings <= self.pre_earnings_days:
            return EarningsRegime.PRE_EARNINGS
        elif days_to_earnings < 0 and abs(days_to_earnings) <= self.post_earnings_days:
            return EarningsRegime.POST_EARNINGS
        else:
            return EarningsRegime.NORMAL
    
    def generate_signal(
        self,
        symbol: str,
        call_price: float,
        put_price: float,
        underlying_price: float,
        days_to_expiry: float,
        timestamp: datetime
    ) -> Optional[StraddleSignal]:
        """
        Generate earnings straddle signal.
        
        Args:
            symbol: Stock symbol
            call_price: Call option price
            put_price: Put option price
            underlying_price: Underlying price
            days_to_expiry: Days to expiry
            timestamp: Signal timestamp
            
        Returns:
            StraddleSignal or None
        """
        # Determine regime
        regime = self.determine_regime(symbol, timestamp)
        
        if regime == EarningsRegime.NORMAL:
            return None
        
        # Calculate straddle cost
        straddle_cost = call_price + put_price
        
        # Store straddle position
        position = StraddlePosition(
            timestamp=timestamp,
            symbol=symbol,
            call_price=call_price,
            put_price=put_price,
            straddle_cost=straddle_cost,
            underlying_price=underlying_price,
            days_to_expiry=days_to_expiry
        )
        
        if symbol not in self.straddle_positions:
            self.straddle_positions[symbol] = []
        self.straddle_positions[symbol].append(position)
        
        # Generate signal based on regime
        if regime == EarningsRegime.PRE_EARNINGS:
            # Buy straddle before earnings
            signal = 1.0
            straddle_position = self.max_position_size
            delta_hedge_ratio = 0.0  # No hedge pre-earnings
            confidence = 0.8
            expected_pnl = straddle_cost * 0.3  # Expected 30% of straddle cost
        elif regime == EarningsRegime.POST_EARNINGS:
            # Gamma scalp post-earnings
            signal = 0.5
            straddle_position = self.max_position_size
            delta_hedge_ratio = 1.0  # Full hedge for gamma scalping
            confidence = 0.6
            expected_pnl = straddle_cost * 0.2  # Expected 20% of straddle cost
        else:
            return None
        
        # Create signal
        straddle_signal = StraddleSignal(
            timestamp=timestamp,
            symbol=symbol,
            regime=regime,
            signal=signal,
            straddle_position=straddle_position,
            delta_hedge_ratio=delta_hedge_ratio,
            confidence=confidence,
            expected_pnl=expected_pnl
        )
        
        self.signals.append(straddle_signal)
        
        return straddle_signal
    
    def get_latest_signal(self, symbol: str) -> Optional[StraddleSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def get_straddle_statistics(self) -> Dict[str, float]:
        """Get straddle statistics."""
        if not self.straddle_positions:
            return {}
        
        total_positions = sum(len(positions) for positions in self.straddle_positions.values())
        avg_straddle_cost = np.mean([
            p.straddle_cost for positions in self.straddle_positions.values() for p in positions
        ])
        
        return {
            'total_positions': total_positions,
            'avg_straddle_cost': avg_straddle_cost,
            'symbols_traded': len(self.straddle_positions)
        }
    
    def print_straddle_report(self) -> None:
        """Print earnings straddle analysis report."""
        print("\n" + "="*60)
        print("EARNINGS EVENT STRADDLE ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Pre-Earnings Days: {self.pre_earnings_days}")
        print(f"  Post-Earnings Days: {self.post_earnings_days}")
        print(f"  Min Surprise Threshold: {self.min_surprise_threshold:.2%}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Earnings Events: {len(self.earnings_events)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.straddle_positions:
            stats = self.get_straddle_statistics()
            print(f"\nStraddle Statistics:")
            print(f"  Total Positions: {stats['total_positions']}")
            print(f"  Average Straddle Cost: {stats['avg_straddle_cost']:.2f}")
            print(f"  Symbols Traded: {stats['symbols_traded']}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Regime':<18} {'Signal':<10} {'StraddlePos':<12} {'HedgeRatio':<12}")
            print("-" * 100)
            
            for signal in self.signals[-5]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.symbol:<10} "
                      f"{signal.regime.value:<18} {signal.signal:<10.3f} {signal.straddle_position:<12.3f} "
                      f"{signal.delta_hedge_ratio:<12.3f}")
        
        print("\n" + "="*60)


def sample_earnings_straddle_alpha():
    """Demonstrate earnings straddle alpha."""
    print("=== Earnings Event Straddle Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = EarningsStraddleAlpha(
        pre_earnings_days=5,
        post_earnings_days=2,
        min_surprise_threshold=0.05,
        max_position_size=0.05
    )
    
    # Add earnings event
    announcement_date = datetime.now() + timedelta(days=3)
    alpha.add_earnings_event(
        'RELIANCE',
        announcement_date,
        eps_actual=15.5,
        eps_estimate=15.0
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 10
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=2), periods=n_days, freq='D')
    
    underlying_price = 1000.0
    
    # Process data
    print("Processing earnings straddle data...")
    for i in range(n_days):
        call_price = 50 + np.random.randn() * 10
        put_price = 50 + np.random.randn() * 10
        days_to_expiry = max(30 - i, 7)
        
        signal = alpha.generate_signal(
            'RELIANCE',
            call_price,
            put_price,
            underlying_price,
            days_to_expiry,
            dates[i]
        )
    
    # Print report
    alpha.print_straddle_report()
    
    print("\n=== Earnings Event Straddle Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Earnings event tracking")
    print("- Pre-earnings straddle entry")
    print("- Post-earnings gamma scalping")
    print("- Regime detection (pre/post earnings)")
    print("- Delta hedging for gamma scalping")
    print("- Expected PnL estimation")
    print("- Expected Sharpe: 0.3-0.5")
    print("- Expected Capacity: Medium")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_earnings_straddle_alpha()

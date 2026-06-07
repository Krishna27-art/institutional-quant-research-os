"""
Dispersion Trading (Index vs Constituents) Alpha Strategy

This module implements the dispersion trading strategy that exploits the
difference between implied correlation (from index options) and realized
correlation (from constituent stocks) by selling index volatility and
buying constituent volatility.

Based on standard dispersion trading literature.
Expected Sharpe: 0.4-0.7
Expected Capacity: High
Decay: Persistent
Difficulty: High

Priority: Medium (Options Phase 6)
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


class DispersionRegime(Enum):
    """Dispersion trading regime."""
    NORMAL = "normal"
    HIGH_IMPLIED_CORRELATION = "high_implied_correlation"
    LOW_REALIZED_CORRELATION = "low_realized_correlation"
    OPPORTUNITY = "opportunity"


@dataclass
class DispersionMeasurement:
    """Dispersion measurement."""
    timestamp: datetime
    index_symbol: str
    index_vol: float
    constituents_vols: Dict[str, float]
    weights: Dict[str, float]
    implied_correlation: float
    realized_correlation: float
    correlation_gap: float  # Implied - Realized
    regime: DispersionRegime


@dataclass
class DispersionSignal:
    """Dispersion trading signal."""
    timestamp: datetime
    index_symbol: str
    correlation_gap: float
    regime: DispersionRegime
    signal: float  # -1 to 1, negative = short index vol, long constituent vol
    index_position: float
    constituent_positions: Dict[str, float]
    confidence: float
    expected_arbitrage: float


class DispersionTradingAlpha:
    """
    Dispersion trading alpha strategy.
    
    This class trades the difference between implied and realized
    correlation by selling index volatility and buying constituent volatility.
    """
    
    def __init__(
        self,
        correlation_gap_threshold: float = 0.15,  # 15% gap threshold
        lookback_days: int = 20,
        max_position_size: float = 0.10
    ):
        """
        Initialize dispersion trading alpha.
        
        Args:
            correlation_gap_threshold: Minimum correlation gap for signal
            lookback_days: Lookback period for realized correlation
            max_position_size: Maximum position size
        """
        self.correlation_gap_threshold = correlation_gap_threshold
        self.lookback_days = lookback_days
        self.max_position_size = max_position_size
        
        self.measurements: List[DispersionMeasurement] = []
        self.signals: List[DispersionSignal] = []
        self.constituent_returns: Dict[str, pd.Series] = {}
        
        logger.info(f"DispersionTradingAlpha initialized: gap_threshold={correlation_gap_threshold}, "
                   f"lookback={lookback_days}days")
    
    def calculate_implied_correlation(
        self,
        index_vol: float,
        constituents_vols: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate implied correlation from index and constituent vols.
        
        Implied Correlation = (Index Vol^2 - Sum(w_i^2 * Constituent Vol_i^2)) /
                            (2 * Sum_{i<j} w_i * w_j * Constituent Vol_i * Constituent Vol_j)
        
        Args:
            index_vol: Index implied volatility
            constituents_vols: Constituent implied volatilities
            weights: Constituent weights
            
        Returns:
            Implied correlation
        """
        # Calculate weighted sum of constituent vol squared
        weighted_constituent_vol_sq = sum(
            weights[symbol] ** 2 * constituents_vols[symbol] ** 2
            for symbol in constituents_vols
        )
        
        # Calculate denominator
        denominator = 0.0
        symbols = list(constituents_vols.keys())
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol_i = symbols[i]
                symbol_j = symbols[j]
                denominator += 2 * weights[symbol_i] * weights[symbol_j] * \
                              constituents_vols[symbol_i] * constituents_vols[symbol_j]
        
        if denominator == 0:
            return 0.5  # Default
        
        implied_corr = (index_vol ** 2 - weighted_constituent_vol_sq) / denominator
        return max(min(implied_corr, 1.0), -1.0)
    
    def calculate_realized_correlation(
        self,
        index_returns: pd.Series,
        constituent_returns: Dict[str, pd.Series],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate realized correlation from returns.
        
        Args:
            index_returns: Index return series
            constituent_returns: Constituent return series
            weights: Constituent weights
            
        Returns:
            Realized correlation
        """
        if len(index_returns) < self.lookback_days:
            return 0.5  # Default
        
        # Use recent returns
        recent_index = index_returns.tail(self.lookback_days)
        
        correlations = []
        for symbol, returns in constituent_returns.items():
            if len(returns) >= self.lookback_days:
                recent_constituent = returns.tail(self.lookback_days)
                if len(recent_index) == len(recent_constituent):
                    corr = recent_index.corr(recent_constituent)
                    if not pd.isna(corr):
                        correlations.append(corr)
        
        if not correlations:
            return 0.5
        
        # Weighted average correlation
        weighted_corr = sum(
            weights.get(symbol, 1.0) * corr
            for symbol, corr in zip(constituent_returns.keys(), correlations)
        ) / sum(weights.values())
        
        return weighted_corr
    
    def determine_regime(
        self,
        correlation_gap: float
    ) -> DispersionRegime:
        """
        Determine dispersion regime.
        
        Args:
            correlation_gap: Correlation gap (implied - realized)
            
        Returns:
            DispersionRegime
        """
        if correlation_gap > self.correlation_gap_threshold:
            return DispersionRegime.OPPORTUNITY
        elif correlation_gap > 0:
            return DispersionRegime.HIGH_IMPLIED_CORRELATION
        elif correlation_gap < -self.correlation_gap_threshold:
            return DispersionRegime.LOW_REALIZED_CORRELATION
        else:
            return DispersionRegime.NORMAL
    
    def generate_signal(
        self,
        index_symbol: str,
        index_vol: float,
        constituents_vols: Dict[str, float],
        weights: Dict[str, float],
        index_returns: pd.Series,
        constituent_returns: Dict[str, pd.Series],
        timestamp: datetime
    ) -> Optional[DispersionSignal]:
        """
        Generate dispersion trading signal.
        
        Args:
            index_symbol: Index symbol
            index_vol: Index implied volatility
            constituents_vols: Constituent implied volatilities
            weights: Constituent weights
            index_returns: Index return series
            constituent_returns: Constituent return series
            timestamp: Signal timestamp
            
        Returns:
            DispersionSignal or None
        """
        # Calculate implied correlation
        implied_corr = self.calculate_implied_correlation(
            index_vol, constituents_vols, weights
        )
        
        # Calculate realized correlation
        realized_corr = self.calculate_realized_correlation(
            index_returns, constituent_returns, weights
        )
        
        # Calculate correlation gap
        correlation_gap = implied_corr - realized_corr
        
        # Determine regime
        regime = self.determine_regime(correlation_gap)
        
        # Generate signal based on regime
        if regime == DispersionRegime.OPPORTUNITY:
            # Short index vol, long constituent vol
            signal = -1.0
            index_position = -self.max_position_size
            constituent_positions = {
                symbol: self.max_position_size * weights[symbol]
                for symbol in constituents_vols
            }
            confidence = min(abs(correlation_gap) / 0.3, 0.9)
            expected_arbitrage = abs(correlation_gap) * 0.5
        elif regime == DispersionRegime.HIGH_IMPLIED_CORRELATION:
            # Partial opportunity
            signal = -0.5
            index_position = -self.max_position_size * 0.5
            constituent_positions = {
                symbol: self.max_position_size * 0.5 * weights[symbol]
                for symbol in constituents_vols
            }
            confidence = min(abs(correlation_gap) / 0.2, 0.7)
            expected_arbitrage = abs(correlation_gap) * 0.3
        elif regime == DispersionRegime.LOW_REALIZED_CORRELATION:
            # Reverse opportunity
            signal = 1.0
            index_position = self.max_position_size * 0.5
            constituent_positions = {
                symbol: -self.max_position_size * 0.5 * weights[symbol]
                for symbol in constituents_vols
            }
            confidence = min(abs(correlation_gap) / 0.2, 0.6)
            expected_arbitrage = abs(correlation_gap) * 0.2
        else:
            return None
        
        # Store measurement
        measurement = DispersionMeasurement(
            timestamp=timestamp,
            index_symbol=index_symbol,
            index_vol=index_vol,
            constituents_vols=constituents_vols,
            weights=weights,
            implied_correlation=implied_corr,
            realized_correlation=realized_corr,
            correlation_gap=correlation_gap,
            regime=regime
        )
        
        self.measurements.append(measurement)
        
        # Keep history manageable
        if len(self.measurements) > 1000:
            self.measurements = self.measurements[-1000:]
        
        # Create signal
        dispersion_signal = DispersionSignal(
            timestamp=timestamp,
            index_symbol=index_symbol,
            correlation_gap=correlation_gap,
            regime=regime,
            signal=signal,
            index_position=index_position,
            constituent_positions=constituent_positions,
            confidence=confidence,
            expected_arbitrage=expected_arbitrage
        )
        
        self.signals.append(dispersion_signal)
        
        return dispersion_signal
    
    def get_latest_signal(self, index_symbol: str) -> Optional[DispersionSignal]:
        """Get the latest signal for an index."""
        for signal in reversed(self.signals):
            if signal.index_symbol == index_symbol:
                return signal
        return None
    
    def get_dispersion_statistics(self) -> Dict[str, float]:
        """Get dispersion statistics."""
        if not self.measurements:
            return {}
        
        gaps = [m.correlation_gap for m in self.measurements]
        implied_corrs = [m.implied_correlation for m in self.measurements]
        realized_corrs = [m.realized_correlation for m in self.measurements]
        
        return {
            'avg_correlation_gap': np.mean(gaps),
            'avg_implied_corr': np.mean(implied_corrs),
            'avg_realized_corr': np.mean(realized_corrs),
            'current_gap': gaps[-1] if gaps else 0.0
        }
    
    def print_dispersion_report(self) -> None:
        """Print dispersion trading analysis report."""
        print("\n" + "="*60)
        print("DISPERSION TRADING ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Correlation Gap Threshold: {self.correlation_gap_threshold:.2%}")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  Max Position Size: {self.max_position_size:.2%}")
        
        print(f"\nStatistics:")
        print(f"  Total Measurements: {len(self.measurements)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.measurements:
            stats = self.get_dispersion_statistics()
            print(f"\nCorrelation Statistics:")
            print(f"  Average Gap: {stats['avg_correlation_gap']:.4f}")
            print(f"  Average Implied Corr: {stats['avg_implied_corr']:.4f}")
            print(f"  Average Realized Corr: {stats['avg_realized_corr']:.4f}")
            print(f"  Current Gap: {stats['current_gap']:.4f}")
        
        if self.signals:
            regime_counts = {}
            for signal in self.signals:
                regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
            
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Index':<10} {'Gap':<10} {'Regime':<25} {'Signal':<10} {'IdxPos':<10}")
            print("-" * 100)
            
            for signal in self.signals[-5]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.index_symbol:<10} "
                      f"{signal.correlation_gap:<10.4f} {signal.regime.value:<25} {signal.signal:<10.3f} "
                      f"{signal.index_position:<10.3f}")
        
        print("\n" + "="*60)


def sample_dispersion_trading_alpha():
    """Demonstrate dispersion trading alpha."""
    print("=== Dispersion Trading Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = DispersionTradingAlpha(
        correlation_gap_threshold=0.15,
        lookback_days=20,
        max_position_size=0.10
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Index data
    index_vol = 0.18 + np.random.randn(n_days) * 0.02
    index_returns = pd.Series(np.random.randn(n_days) * 0.01, index=dates)
    
    # Constituents
    constituents = ['RELIANCE', 'HDFC', 'INFY', 'TCS', 'ICICI']
    weights = {c: 0.2 for c in constituents}
    constituents_vols = {c: 0.20 + np.random.randn(n_days) * 0.02 for c in constituents}
    constituent_returns = {
        c: pd.Series(np.random.randn(n_days) * 0.015, index=dates)
        for c in constituents
    }
    
    # Process data
    print("Processing dispersion data...")
    for i in range(30, n_days):
        current_constituent_vols = {c: constituents_vols[c][i] for c in constituents}
        current_constituent_returns = {c: constituent_returns[c].iloc[:i] for c in constituents}
        
        signal = alpha.generate_signal(
            'NIFTY',
            index_vol[i],
            current_constituent_vols,
            weights,
            index_returns.iloc[:i],
            current_constituent_returns,
            dates[i]
        )
    
    # Print report
    alpha.print_dispersion_report()
    
    print("\n=== Dispersion Trading Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Implied correlation calculation (from index vs constituent vols)")
    print("- Realized correlation calculation (from returns)")
    print("- Correlation gap detection")
    print("- Regime classification")
    print("- Index vs constituent volatility trading")
    print("- Expected Sharpe: 0.4-0.7")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_dispersion_trading_alpha()

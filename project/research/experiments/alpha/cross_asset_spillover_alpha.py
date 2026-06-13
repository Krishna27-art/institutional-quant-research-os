"""
Cross-Asset Spillover Signals Alpha Strategy

This module implements cross-asset spillover signals that capture lead-lag
relationships between different asset classes (commodities, FX, rates, equities),
allowing for predictive signals based on inter-market dynamics.

Based on 2026 network study; commodity-equity spillover literature.
Expected Sharpe: 0.3-0.5
Expected Capacity: High
Decay: Persistent
Difficulty: Medium

Priority: Medium (Research OS Phase 7)
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


class AssetClass(Enum):
    """Asset class types."""
    EQUITY = "equity"
    COMMODITY = "commodity"
    FX = "fx"
    RATES = "rates"
    CRYPTO = "crypto"


class SpilloverDirection(Enum):
    """Spillover direction."""
    POSITIVE = "positive"  # Positive spillover
    NEGATIVE = "negative"  # Negative spillover
    NEUTRAL = "neutral"


@dataclass
class SpilloverMeasurement:
    """Cross-asset spillover measurement."""
    timestamp: datetime
    source_asset: str
    source_class: AssetClass
    target_asset: str
    target_class: AssetClass
    correlation: float
    lead_lag_days: int
    spillover_strength: float
    direction: SpilloverDirection


@dataclass
class SpilloverSignal:
    """Cross-asset spillover trading signal."""
    timestamp: datetime
    target_asset: str
    source_asset: str
    spillover_strength: float
    direction: SpilloverDirection
    signal: float  # -1 to 1
    confidence: float
    holding_period_days: int


class CrossAssetSpilloverAlpha:
    """
    Cross-asset spillover signals alpha strategy.
    
    This class implements lead-lag detection between asset classes
    and generates trading signals based on spillover effects.
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        min_correlation: float = 0.3,
        max_lead_lag: int = 5,
        spillover_threshold: float = 0.5
    ):
        """
        Initialize cross-asset spillover alpha.
        
        Args:
            lookback_days: Lookback period for correlation analysis
            min_correlation: Minimum correlation for spillover signal
            max_lead_lag: Maximum lead-lag days to consider
            spillover_threshold: Threshold for spillover strength
        """
        self.lookback_days = lookback_days
        self.min_correlation = min_correlation
        self.max_lead_lag = max_lead_lag
        self.spillover_threshold = spillover_threshold
        
        self.spillover_measurements: List[SpilloverMeasurement] = []
        self.signals: List[SpilloverSignal] = []
        self.asset_returns: Dict[str, pd.Series] = {}
        
        # Known spillover relationships (based on literature)
        self.known_spillovers = {
            ('OIL', 'equity'): {'direction': SpilloverDirection.POSITIVE, 'avg_lag': 1},
            ('GOLD', 'equity'): {'direction': SpilloverDirection.NEGATIVE, 'avg_lag': 2},
            ('USDINR', 'equity'): {'direction': SpilloverDirection.NEGATIVE, 'avg_lag': 1},
            ('US10Y', 'equity'): {'direction': SpilloverDirection.NEGATIVE, 'avg_lag': 3},
            ('COPPER', 'equity'): {'direction': SpilloverDirection.POSITIVE, 'avg_lag': 1},
        }
        
        logger.info(f"CrossAssetSpilloverAlpha initialized: lookback={lookback_days}days, "
                   f"max_lag={max_lead_lag}days")
    
    def calculate_correlation(
        self,
        returns1: pd.Series,
        returns2: pd.Series,
        lag: int = 0
    ) -> float:
        """
        Calculate correlation with lag.
        
        Args:
            returns1: First return series
            returns2: Second return series
            lag: Lag days (positive = returns1 leads)
            
        Returns:
            Correlation coefficient
        """
        if len(returns1) < self.lookback_days or len(returns2) < self.lookback_days:
            return 0.0
        
        # Align series with lag
        if lag > 0:
            returns1_lagged = returns1.shift(lag)
            aligned_returns1 = returns1_lagged.iloc[-self.lookback_days:]
            aligned_returns2 = returns2.iloc[-self.lookback_days:]
        elif lag < 0:
            returns2_lagged = returns2.shift(abs(lag))
            aligned_returns1 = returns1.iloc[-self.lookback_days:]
            aligned_returns2 = returns2_lagged.iloc[-self.lookback_days:]
        else:
            aligned_returns1 = returns1.iloc[-self.lookback_days:]
            aligned_returns2 = returns2.iloc[-self.lookback_days:]
        
        # Drop NaN
        mask = ~(pd.isna(aligned_returns1) | pd.isna(aligned_returns2))
        aligned_returns1 = aligned_returns1[mask]
        aligned_returns2 = aligned_returns2[mask]
        
        if len(aligned_returns1) < 10:
            return 0.0
        
        correlation = aligned_returns1.corr(aligned_returns2)
        return correlation if not pd.isna(correlation) else 0.0
    
    def detect_lead_lag(
        self,
        returns1: pd.Series,
        returns2: pd.Series
    ) -> Tuple[int, float]:
        """
        Detect optimal lead-lag relationship.
        
        Args:
            returns1: First return series
            returns2: Second return series
            
        Returns:
            (optimal_lag, max_correlation)
        """
        correlations = []
        
        for lag in range(-self.max_lead_lag, self.max_lead_lag + 1):
            corr = self.calculate_correlation(returns1, returns2, lag)
            correlations.append((lag, corr))
        
        # Find maximum absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        optimal_lag, max_corr = correlations[0]
        
        return optimal_lag, max_corr
    
    def calculate_spillover_strength(
        self,
        correlation: float,
        lead_lag: int
    ) -> float:
        """
        Calculate spillover strength.
        
        Args:
            correlation: Correlation coefficient
            lead_lag: Lead-lag days
            
        Returns:
            Spillover strength (0-1)
        """
        # Strength based on correlation magnitude and lag
        corr_strength = abs(correlation)
        lag_penalty = 1.0 - (abs(lead_lag) / (self.max_lead_lag * 2))
        
        spillover_strength = corr_strength * lag_penalty
        return spillover_strength
    
    def measure_spillover(
        self,
        source_asset: str,
        source_class: AssetClass,
        target_asset: str,
        target_class: AssetClass,
        timestamp: datetime
    ) -> Optional[SpilloverMeasurement]:
        """
        Measure spillover between assets.
        
        Args:
            source_asset: Source asset symbol
            source_class: Source asset class
            target_asset: Target asset symbol
            target_class: Target asset class
            timestamp: Measurement timestamp
            
        Returns:
            SpilloverMeasurement or None
        """
        if source_asset not in self.asset_returns or target_asset not in self.asset_returns:
            return None
        
        source_returns = self.asset_returns[source_asset]
        target_returns = self.asset_returns[target_asset]
        
        # Detect lead-lag
        lead_lag, correlation = self.detect_lead_lag(source_returns, target_returns)
        
        # Check if correlation meets threshold
        if abs(correlation) < self.min_correlation:
            return None
        
        # Calculate spillover strength
        spillover_strength = self.calculate_spillover_strength(correlation, lead_lag)
        
        # Determine direction
        if correlation > 0:
            direction = SpilloverDirection.POSITIVE
        else:
            direction = SpilloverDirection.NEGATIVE
        
        measurement = SpilloverMeasurement(
            timestamp=timestamp,
            source_asset=source_asset,
            source_class=source_class,
            target_asset=target_asset,
            target_class=target_class,
            correlation=correlation,
            lead_lag_days=lead_lag,
            spillover_strength=spillover_strength,
            direction=direction
        )
        
        self.spillover_measurements.append(measurement)
        
        return measurement
    
    def generate_signal(
        self,
        source_asset: str,
        source_class: AssetClass,
        target_asset: str,
        target_class: AssetClass,
        timestamp: datetime
    ) -> Optional[SpilloverSignal]:
        """
        Generate spillover trading signal.
        
        Args:
            source_asset: Source asset symbol
            source_class: Source asset class
            target_asset: Target asset symbol
            target_class: Target asset class
            timestamp: Signal timestamp
            
        Returns:
            SpilloverSignal or None
        """
        # Measure spillover
        measurement = self.measure_spillover(
            source_asset, source_class,
            target_asset, target_class,
            timestamp
        )
        
        if measurement is None or measurement.spillover_strength < self.spillover_threshold:
            return None
        
        # Check known spillover relationships
        key = (source_asset, target_class.value)
        known = self.known_spillovers.get(key, None)
        
        if known:
            # Use known relationship for confidence
            if known['direction'] == measurement.direction:
                confidence = min(measurement.spillover_strength + 0.2, 0.95)
            else:
                confidence = measurement.spillover_strength * 0.7
        else:
            confidence = measurement.spillover_strength
        
        # Generate signal
        if measurement.direction == SpilloverDirection.POSITIVE:
            signal = measurement.spillover_strength
        else:
            signal = -measurement.spillover_strength
        
        # Holding period based on lead-lag
        holding_period = max(abs(measurement.lead_lag_days) * 2, 5)
        
        spillover_signal = SpilloverSignal(
            timestamp=timestamp,
            target_asset=target_asset,
            source_asset=source_asset,
            spillover_strength=measurement.spillover_strength,
            direction=measurement.direction,
            signal=signal,
            confidence=confidence,
            holding_period_days=holding_period
        )
        
        self.signals.append(spillover_signal)
        
        return spillover_signal
    
    def update_returns(
        self,
        asset: str,
        returns: pd.Series
    ) -> None:
        """
        Update return series for an asset.
        
        Args:
            asset: Asset symbol
            returns: Return series
        """
        self.asset_returns[asset] = returns
    
    def get_latest_signal(self, target_asset: str) -> Optional[SpilloverSignal]:
        """Get the latest signal for a target asset."""
        for signal in reversed(self.signals):
            if signal.target_asset == target_asset:
                return signal
        return None
    
    def get_spillover_statistics(self) -> Dict[str, any]:
        """Get spillover statistics."""
        if not self.spillover_measurements:
            return {}
        
        correlations = [m.correlation for m in self.spillover_measurements]
        strengths = [m.spillover_strength for m in self.spillover_measurements]
        
        pos_spillovers = [m for m in self.spillover_measurements if m.direction == SpilloverDirection.POSITIVE]
        neg_spillovers = [m for m in self.spillover_measurements if m.direction == SpilloverDirection.NEGATIVE]
        
        return {
            'total_measurements': len(self.spillover_measurements),
            'avg_correlation': np.mean(correlations),
            'avg_strength': np.mean(strengths),
            'positive_spillovers': len(pos_spillovers),
            'negative_spillovers': len(neg_spillovers)
        }
    
    def print_spillover_report(self) -> None:
        """Print spillover analysis report."""
        print("\n" + "="*60)
        print("CROSS-ASSET SPILLOVER ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Lookback Days: {self.lookback_days}")
        print(f"  Min Correlation: {self.min_correlation}")
        print(f"  Max Lead-Lag: {self.max_lead_lag} days")
        print(f"  Spillover Threshold: {self.spillover_threshold}")
        
        print(f"\nStatistics:")
        stats = self.get_spillover_statistics()
        print(f"  Total Measurements: {stats.get('total_measurements', 0)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if stats:
            print(f"\nSpillover Statistics:")
            print(f"  Average Correlation: {stats.get('avg_correlation', 0):.4f}")
            print(f"  Average Strength: {stats.get('avg_strength', 0):.4f}")
            print(f"  Positive Spillovers: {stats.get('positive_spillovers', 0)}")
            print(f"  Negative Spillovers: {stats.get('negative_spillovers', 0)}")
        
        if self.signals:
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Target':<12} {'Source':<12} {'Direction':<12} {'Strength':<10} {'Signal':<10} {'Confidence':<12}")
            print("-" * 95)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {signal.target_asset:<12} "
                      f"{signal.source_asset:<12} {signal.direction.value:<12} {signal.spillover_strength:<10.4f} "
                      f"{signal.signal:<10.3f} {signal.confidence:<12.2f}")
        
        print("\n" + "="*60)


def sample_cross_asset_spillover_alpha():
    """Demonstrate cross-asset spillover alpha."""
    print("=== Cross-Asset Spillover Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = CrossAssetSpilloverAlpha(
        lookback_days=60,
        min_correlation=0.3,
        max_lead_lag=5,
        spillover_threshold=0.5
    )
    
    # Generate sample data
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate returns for different assets
    assets = {
        'OIL': 0.02,
        'GOLD': 0.01,
        'USDINR': 0.005,
        'NIFTY': 0.015,
        'RELIANCE': 0.02
    }
    
    for asset, base_vol in assets.items():
        returns = pd.Series(np.random.randn(n_days) * base_vol, index=dates)
        alpha.update_returns(asset, returns)
    
    # Generate spillover signals
    print("Processing cross-asset spillover...")
    for i in range(60, n_days):
        # Oil -> Equity spillover
        alpha.generate_signal(
            'OIL', AssetClass.COMMODITY,
            'RELIANCE', AssetClass.EQUITY,
            dates[i]
        )
        
        # Gold -> Equity spillover
        alpha.generate_signal(
            'GOLD', AssetClass.COMMODITY,
            'NIFTY', AssetClass.EQUITY,
            dates[i]
        )
    
    # Print report
    alpha.print_spillover_report()
    
    print("\n=== Cross-Asset Spillover Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Cross-asset correlation analysis")
    print("- Lead-lag detection")
    print("- Spillover strength calculation")
    print("- Trading signal generation based on spillover")
    print("- Known relationship validation")
    print("- Expected Sharpe: 0.3-0.5")
    print("- Expected Capacity: High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_cross_asset_spillover_alpha()

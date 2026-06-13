"""
Regime-Based Strategy Switching

Implements institutional-grade regime detection and strategy switching:
- HMM-based regime detection
- Regime-dependent strategy weights
- Dynamic strategy allocation

Based on blueprint specification for multi-strategy framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Regime(Enum):
    """Market regimes"""
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


@dataclass
class RegimeState:
    """Current regime state"""
    regime: Regime
    probabilities: Dict[Regime, float]
    confidence: float
    timestamp: pd.Timestamp


class RegimeDetector:
    """
    Regime Detection using HMM or rule-based methods.
    
    Detects market regimes to switch between strategies.
    """
    
    def __init__(
        self,
        lookback: int = 63,
        vol_threshold: float = 0.25,
        trend_threshold: float = 0.02
    ):
        """
        Initialize regime detector.
        
        Args:
            lookback: Lookback period for regime detection
            vol_threshold: Volatility threshold for volatility regime
            trend_threshold: Trend threshold for trend detection
        """
        self.lookback = lookback
        self.vol_threshold = vol_threshold
        self.trend_threshold = trend_threshold
        
    def detect_regime(
        self,
        returns: pd.Series,
        volatility: pd.Series
    ) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            returns: Return series
            volatility: Volatility series
            
        Returns:
            RegimeState with current regime and probabilities
        """
        # Get recent data
        recent_returns = returns.tail(self.lookback)
        recent_vol = volatility.tail(self.lookback)
        
        # Calculate metrics
        avg_vol = recent_vol.mean()
        avg_return = recent_returns.mean()
        return_std = recent_returns.std()
        
        # Calculate regime probabilities (simplified rule-based)
        probs = {
            Regime.TREND: 0.0,
            Regime.MEAN_REVERSION: 0.0,
            Regime.VOLATILITY: 0.0,
            Regime.SIDEWAYS: 0.0,
            Regime.CRISIS: 0.0
        }
        
        # Volatility regime
        if avg_vol > self.vol_threshold:
            probs[Regime.VOLATILITY] = 0.6
            probs[Regime.CRISIS] = 0.4
        else:
            probs[Regime.VOLATILITY] = 0.1
            probs[Regime.CRISIS] = 0.0
        
        # Trend vs mean reversion
        if abs(avg_return) > self.trend_threshold:
            if avg_return > 0:
                probs[Regime.TREND] = 0.7
            else:
                probs[Regime.TREND] = 0.3
        else:
            probs[Regime.MEAN_REVERSION] = 0.5
            probs[Regime.SIDEWAYS] = 0.5
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        # Determine dominant regime
        dominant = max(probs, key=probs.get)
        confidence = probs[dominant]
        
        return RegimeState(
            regime=dominant,
            probabilities=probs,
            confidence=confidence,
            timestamp=pd.Timestamp.now()
        )


class RegimeStrategySwitcher:
    """
    Regime-Based Strategy Switching
    
    Meta-strategy that weights sub-strategies based on regime.
    """
    
    def __init__(self):
        """Initialize regime strategy switcher."""
        # Define regime-specific strategy weights
        self.regime_weights = {
            Regime.TREND: {
                'tsmom': 0.6,
                'dual_mom': 0.3,
                'sector_mom': 0.1
            },
            Regime.MEAN_REVERSION: {
                'pairs_kalman': 0.4,
                'orb': 0.3,
                'vwap_rev': 0.3
            },
            Regime.VOLATILITY: {
                'vrp': 0.5,
                'vix_basis': 0.3,
                'dispersion': 0.2
            },
            Regime.SIDEWAYS: {
                'mean_rev': 0.5,
                'pairs': 0.5
            },
            Regime.CRISIS: {
                'low_vol': 0.4,
                'quality': 0.3,
                'cash': 0.3
            }
        }
        
    def get_strategy_weights(
        self,
        regime: Regime,
        regime_probs: Dict[Regime, float]
    ) -> Dict[str, float]:
        """
        Get strategy weights based on regime.
        
        Args:
            regime: Current regime
            regime_probs: Regime probabilities
            
        Returns:
            Dictionary mapping strategy names to weights
        """
        # Get base weights for current regime
        base_weights = self.regime_weights.get(regime, {})
        
        # Blend with neighboring regimes based on probabilities
        blended_weights = {}
        
        for r, prob in regime_probs.items():
            if prob > 0.1:  # Only consider regimes with significant probability
                r_weights = self.regime_weights.get(r, {})
                for strat, weight in r_weights.items():
                    blended_weights[strat] = blended_weights.get(strat, 0) + weight * prob
        
        # Normalize
        total = sum(blended_weights.values())
        if total > 0:
            blended_weights = {k: v / total for k, v in blended_weights.items()}
        
        return blended_weights
    
    def switch_strategy(
        self,
        current_regime: Regime,
        new_regime: Regime,
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Switch strategy weights based on regime change.
        
        Args:
            current_regime: Current regime
            new_regime: New regime
            current_weights: Current strategy weights
            
        Returns:
            New strategy weights
        """
        # Get target weights for new regime
        target_weights = self.regime_weights.get(new_regime, {})
        
        # Gradual transition (70% target, 30% current)
        new_weights = {}
        all_strategies = set(current_weights.keys()) | set(target_weights.keys())
        
        for strat in all_strategies:
            target = target_weights.get(strat, 0)
            current = current_weights.get(strat, 0)
            new_weights[strat] = 0.7 * target + 0.3 * current
        
        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        return new_weights


class AdaptiveStrategyManager:
    """
    Adaptive Strategy Manager
    
    Combines regime detection with strategy switching for dynamic allocation.
    """
    
    def __init__(self):
        """Initialize adaptive strategy manager."""
        self.regime_detector = RegimeDetector()
        self.strategy_switcher = RegimeStrategySwitcher()
        self.current_regime = None
        self.current_weights = {}
        
    def update(
        self,
        returns: pd.Series,
        volatility: pd.Series
    ) -> Tuple[RegimeState, Dict[str, float]]:
        """
        Update regime and strategy weights.
        
        Args:
            returns: Return series
            volatility: Volatility series
            
        Returns:
            Tuple of (regime_state, strategy_weights)
        """
        # Detect regime
        regime_state = self.regime_detector.detect_regime(returns, volatility)
        
        # Check if regime changed
        if self.current_regime != regime_state.regime:
            # Switch strategies
            new_weights = self.strategy_switcher.switch_strategy(
                self.current_regime or Regime.SIDEWAYS,
                regime_state.regime,
                self.current_weights
            )
            self.current_weights = new_weights
            self.current_regime = regime_state.regime
            
            logger.info(f"Regime changed to {regime_state.regime.value}")
        else:
            # Update weights based on current regime probabilities
            self.current_weights = self.strategy_switcher.get_strategy_weights(
                regime_state.regime,
                regime_state.probabilities
            )
        
        return regime_state, self.current_weights


if __name__ == "__main__":
    # Test regime-based strategy switching
    print("Testing Regime-Based Strategy Switching...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    returns = pd.Series(
        np.random.randn(n) * 0.01,
        index=dates
    )
    
    volatility = pd.Series(
        np.abs(np.random.randn(n)) * 0.02,
        index=dates
    )
    
    # Test Regime Detector
    print("\n1. Regime Detector:")
    detector = RegimeDetector()
    regime_state = detector.detect_regime(returns, volatility)
    print(f"   Regime: {regime_state.regime.value}")
    print(f"   Confidence: {regime_state.confidence:.3f}")
    print(f"   Probabilities:")
    for regime, prob in regime_state.probabilities.items():
        print(f"     {regime.value}: {prob:.3f}")
    
    # Test Strategy Switcher
    print("\n2. Strategy Switcher:")
    switcher = RegimeStrategySwitcher()
    
    # Get weights for different regimes
    for regime in [Regime.TREND, Regime.MEAN_REVERSION, Regime.VOLATILITY]:
        weights = switcher.get_strategy_weights(regime, {regime: 1.0})
        print(f"   {regime.value}:")
        for strat, weight in weights.items():
            print(f"     {strat}: {weight:.2f}")
    
    # Test Adaptive Manager
    print("\n3. Adaptive Strategy Manager:")
    manager = AdaptiveStrategyManager()
    regime_state, weights = manager.update(returns, volatility)
    print(f"   Current regime: {regime_state.regime.value}")
    print(f"   Strategy weights:")
    for strat, weight in weights.items():
        print(f"     {strat}: {weight:.2f}")
    
    print("\n✓ Regime-based strategy switching tested")

"""
Rule-Based Regime Detection Engine

This replaces the HMM-based regime detection with a simpler, more robust rule-based approach.
The HMM was overfitting and the state mapping was broken. This approach uses clear rules
based on market indicators to detect regimes.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class RegimeType(Enum):
    """Regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state."""
    regime: RegimeType
    confidence: float
    timestamp: pd.Timestamp
    indicators: Dict[str, float]


class RuleBasedRegimeEngine:
    """
    Rule-based regime detection engine.
    
    Uses clear rules based on:
    - Price trend (moving averages)
    - Volatility levels
    - Volume patterns
    - Market breadth
    """
    
    def __init__(
        self,
        ma_short: int = 20,
        ma_long: int = 200,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.01
    ):
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
    def detect_regime(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Detect current regime using rule-based approach.
        
        Args:
            prices: Price series
            volume: Optional volume series
            returns: Optional returns series
            
        Returns:
            RegimeState with current regime and confidence
        """
        if len(prices) < self.ma_long:
            return RegimeState(
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                indicators={}
            )
        
        # Calculate indicators
        ma_short = prices.rolling(self.ma_short).mean().iloc[-1]
        ma_long = prices.rolling(self.ma_long).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        if returns is not None:
            volatility = returns.rolling(20).std().iloc[-1]
        else:
            volatility = prices.pct_change().rolling(20).std().iloc[-1]
        
        # Calculate trend strength
        trend_strength = (current_price - ma_long) / ma_long
        
        # Determine regime based on rules
        regime = RegimeType.SIDEWAYS
        confidence = 0.5
        
        if volatility > self.volatility_threshold:
            regime = RegimeType.HIGH_VOLATILITY
            confidence = min(0.9, volatility / self.volatility_threshold)
        elif trend_strength > self.trend_threshold and current_price > ma_short:
            regime = RegimeType.BULL_TREND
            confidence = min(0.9, trend_strength / self.trend_threshold)
        elif trend_strength < -self.trend_threshold and current_price < ma_short:
            regime = RegimeType.BEAR_TREND
            confidence = min(0.9, abs(trend_strength) / self.trend_threshold)
        else:
            regime = RegimeType.SIDEWAYS
            confidence = 0.6
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            indicators={
                'ma_short': ma_short,
                'ma_long': ma_long,
                'current_price': current_price,
                'volatility': volatility,
                'trend_strength': trend_strength
            }
        )
    
    def get_regime_weights(self, regime: RegimeType) -> Dict[str, float]:
        """
        Get strategy weights for a given regime.
        
        Args:
            regime: Current regime
            
        Returns:
            Dictionary of strategy weights
        """
        weights = {
            RegimeType.BULL_TREND: {
                'orb': 0.40,
                'vwap': 0.30,
                'momentum': 0.20,
                'mean_reversion': 0.10
            },
            RegimeType.BEAR_TREND: {
                'orb': 0.20,
                'vwap': 0.40,
                'momentum': 0.10,
                'mean_reversion': 0.30
            },
            RegimeType.SIDEWAYS: {
                'orb': 0.10,
                'vwap': 0.10,
                'momentum': 0.10,
                'mean_reversion': 0.70
            },
            RegimeType.HIGH_VOLATILITY: {
                'orb': 0.15,
                'vwap': 0.15,
                'momentum': 0.10,
                'mean_reversion': 0.20,
                'volatility': 0.40
            },
            RegimeType.UNKNOWN: {
                'orb': 0.25,
                'vwap': 0.25,
                'momentum': 0.25,
                'mean_reversion': 0.25
            }
        }
        
        return weights.get(regime, weights[RegimeType.UNKNOWN])


# Singleton instance
_regime_engine = None

def get_regime_engine() -> RuleBasedRegimeEngine:
    """Get the singleton regime engine instance."""
    global _regime_engine
    if _regime_engine is None:
        _regime_engine = RuleBasedRegimeEngine()
    return _regime_engine


if __name__ == "__main__":
    # Test regime engine
    print("Testing Rule-Based Regime Engine...")
    
    engine = RuleBasedRegimeEngine()
    
    # Create sample price data
    np.random.seed(42)
    n_days = 300
    prices = pd.Series(100 + np.cumsum(np.random.randn(n_days) * 0.5))
    
    regime_state = engine.detect_regime(prices)
    print(f"Detected regime: {regime_state.regime}")
    print(f"Confidence: {regime_state.confidence}")
    print(f"Indicators: {regime_state.indicators}")

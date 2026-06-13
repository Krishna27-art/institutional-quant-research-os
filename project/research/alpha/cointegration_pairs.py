"""
Cointegration-Based Pairs Trading Strategy

Implements statistical arbitrage using cointegration to identify
pairs of stocks that move together and trade the spread when it deviates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for pairs trading."""
    LONG_SPREAD = "long_spread"  # Long the spread (buy stock A, sell stock B)
    SHORT_SPREAD = "short_spread"  # Short the spread (sell stock A, buy stock B)
    NO_SIGNAL = "no_signal"


@dataclass
class PairSignal:
    """Signal for a trading pair."""
    stock_a: str
    stock_b: str
    signal_type: SignalType
    hedge_ratio: float
    z_score: float
    confidence: float
    timestamp: pd.Timestamp
    metadata: Dict


class CointegrationPairsStrategy:
    """
    Cointegration-based pairs trading strategy.
    
    Uses Engle-Granger two-step method to test for cointegration
    and trade the spread when it deviates from mean.
    """
    
    def __init__(
        self,
        lookback_period: int = 252,  # 1 year of daily data
        entry_threshold: float = 2.0,  # Z-score threshold for entry
        exit_threshold: float = 0.5,  # Z-score threshold for exit
        max_pairs: int = 20  # Maximum number of pairs to track
    ):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_pairs = max_pairs
        
        # Store pair information
        self.cointegrated_pairs: Dict[Tuple[str, str], Dict] = {}
        self.spread_history: Dict[Tuple[str, str], List[float]] = {}
        
    def test_cointegration(
        self,
        price_a: pd.Series,
        price_b: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Test if two price series are cointegrated using Engle-Granger method.
        
        Args:
            price_a: Price series for stock A
            price_b: Price series for stock B
            
        Returns:
            Tuple of (is_cointegrated, hedge_ratio, p_value)
        """
        try:
            # Step 1: Run regression to find hedge ratio
            # price_a = hedge_ratio * price_b + residual
            hedge_ratio = np.polyfit(price_b, price_a, 1)[0]
            
            # Step 2: Calculate spread
            spread = price_a - hedge_ratio * price_b
            
            # Step 3: Test if spread is stationary (ADF test)
            # Simplified: check if mean reversion exists
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Calculate half-life of mean reversion
            spread_lagged = spread.shift(1).dropna()
            spread_current = spread[1:]
            beta = np.polyfit(spread_lagged, spread_current - spread_mean, 1)[0]
            half_life = -np.log(2) / beta if beta < 0 else float('inf')
            
            # Simple cointegration test: half-life < lookback_period and spread is bounded
            is_cointegrated = (
                half_life < self.lookback_period and
                half_life > 0 and
                spread_std / spread_mean.abs().mean() < 0.5 if spread_mean.abs().mean() > 0 else False
            )
            
            return is_cointegrated, hedge_ratio, half_life
            
        except Exception as e:
            logger.error(f"Cointegration test failed: {e}")
            return False, 0.0, 1.0
    
    def find_cointegrated_pairs(
        self,
        prices: Dict[str, pd.Series]
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find cointegrated pairs among a universe of stocks.
        
        Args:
            prices: Dictionary of symbol to price series
            
        Returns:
            List of (stock_a, stock_b, hedge_ratio, half_life)
        """
        symbols = list(prices.keys())
        cointegrated_pairs = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol_a = symbols[i]
                symbol_b = symbols[j]
                
                price_a = prices[symbol_a]
                price_b = prices[symbol_b]
                
                is_cointegrated, hedge_ratio, half_life = self.test_cointegration(price_a, price_b)
                
                if is_cointegrated:
                    cointegrated_pairs.append((symbol_a, symbol_b, hedge_ratio, half_life))
        
        # Sort by half-life (prefer faster mean reversion)
        cointegrated_pairs.sort(key=lambda x: x[3])
        
        # Limit to max_pairs
        return cointegrated_pairs[:self.max_pairs]
    
    def generate_signals(
        self,
        prices: Dict[str, pd.Series],
        current_prices: Dict[str, float]
    ) -> List[PairSignal]:
        """
        Generate trading signals for cointegrated pairs.
        
        Args:
            prices: Historical price data for all stocks
            current_prices: Current prices for all stocks
            
        Returns:
            List of PairSignal objects
        """
        signals = []
        
        # Find cointegrated pairs
        pairs = self.find_cointegrated_pairs(prices)
        
        for stock_a, stock_b, hedge_ratio, half_life in pairs:
            # Calculate current spread
            price_a = prices[stock_a]
            price_b = prices[stock_b]
            spread = price_a - hedge_ratio * price_b
            
            # Calculate z-score of current spread
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = current_prices[stock_a] - hedge_ratio * current_prices[stock_b]
            z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Generate signal based on z-score
            if z_score > self.entry_threshold:
                # Spread is too high - short the spread
                signal_type = SignalType.SHORT_SPREAD
                confidence = min(0.9, (z_score - self.entry_threshold) / self.entry_threshold)
            elif z_score < -self.entry_threshold:
                # Spread is too low - long the spread
                signal_type = SignalType.LONG_SPREAD
                confidence = min(0.9, (abs(z_score) - self.entry_threshold) / self.entry_threshold)
            else:
                signal_type = SignalType.NO_SIGNAL
                confidence = 0.0
            
            if signal_type != SignalType.NO_SIGNAL:
                signals.append(PairSignal(
                    stock_a=stock_a,
                    stock_b=stock_b,
                    signal_type=signal_type,
                    hedge_ratio=hedge_ratio,
                    z_score=z_score,
                    confidence=confidence,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'half_life': half_life,
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                        'current_spread': current_spread
                    }
                ))
        
        return signals


# Singleton instance
_pairs_strategy = None

def get_pairs_strategy() -> CointegrationPairsStrategy:
    """Get the singleton pairs strategy instance."""
    global _pairs_strategy
    if _pairs_strategy is None:
        _pairs_strategy = CointegrationPairsStrategy()
    return _pairs_strategy


if __name__ == "__main__":
    # Test pairs trading strategy
    print("Testing Cointegration Pairs Trading Strategy...")
    
    strategy = CointegrationPairsStrategy()
    
    # Create sample price data
    np.random.seed(42)
    n_days = 300
    
    # Create two cointegrated series
    common_factor = np.cumsum(np.random.randn(n_days) * 0.5)
    price_a = pd.Series(100 + common_factor + np.random.randn(n_days) * 0.2)
    price_b = pd.Series(100 + common_factor * 0.8 + np.random.randn(n_days) * 0.2)
    
    prices = {'STOCK_A': price_a, 'STOCK_B': price_b}
    current_prices = {'STOCK_A': price_a.iloc[-1], 'STOCK_B': price_b.iloc[-1]}
    
    signals = strategy.generate_signals(prices, current_prices)
    print(f"Generated {len(signals)} signals")
    for signal in signals:
        print(f"  {signal.stock_a} - {signal.stock_b}: {signal.signal_type}, z-score: {signal.z_score:.2f}")

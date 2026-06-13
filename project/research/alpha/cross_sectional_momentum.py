"""
Cross-Sectional Momentum Strategy

Implements cross-sectional momentum by ranking stocks based on
past returns and going long the winners, short the losers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MomentumSignal(Enum):
    """Momentum signal types."""
    LONG = "long"
    SHORT = "short"
    NO_SIGNAL = "no_signal"


@dataclass
class CrossSectionalSignal:
    """Cross-sectional momentum signal."""
    symbol: str
    signal_type: MomentumSignal
    rank: int
    percentile: float
    return_momentum: float
    confidence: float
    timestamp: pd.Timestamp
    metadata: Dict


class CrossSectionalMomentum:
    """
    Cross-sectional momentum strategy.
    
    Ranks stocks by past returns and goes long top decile,
    short bottom decile.
    """
    
    def __init__(
        self,
        lookback_period: int = 252,  # 1 year
        rebalance_frequency: int = 21,  # Monthly
        top_decile_pct: float = 0.10,  # Top 10%
        bottom_decile_pct: float = 0.10,  # Bottom 10%
        min_stocks: int = 20  # Minimum stocks in universe
    ):
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.top_decile_pct = top_decile_pct
        self.bottom_decile_pct = bottom_decile_pct
        self.min_stocks = min_stocks
        
    def calculate_momentum(
        self,
        prices: pd.Series
    ) -> float:
        """
        Calculate momentum return for a stock.
        
        Args:
            prices: Price series
            
        Returns:
            Momentum return over lookback period
        """
        if len(prices) < self.lookback_period:
            return 0.0
        
        # Calculate cumulative return over lookback period
        start_price = prices.iloc[-self.lookback_period]
        end_price = prices.iloc[-1]
        
        momentum = (end_price - start_price) / start_price
        return momentum
    
    def rank_stocks(
        self,
        prices_dict: Dict[str, pd.Series]
    ) -> List[Tuple[str, float, int]]:
        """
        Rank stocks by momentum.
        
        Args:
            prices_dict: Dictionary of symbol to price series
            
        Returns:
            List of (symbol, momentum, rank)
        """
        momenta = []
        
        for symbol, prices in prices_dict.items():
            momentum = self.calculate_momentum(prices)
            momenta.append((symbol, momentum))
        
        # Sort by momentum (descending)
        momenta.sort(key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        ranked = [(symbol, momentum, rank + 1) 
                  for rank, (symbol, momentum) in enumerate(momenta)]
        
        return ranked
    
    def generate_signals(
        self,
        prices_dict: Dict[str, pd.Series]
    ) -> List[CrossSectionalSignal]:
        """
        Generate cross-sectional momentum signals.
        
        Args:
            prices_dict: Dictionary of symbol to price series
            
        Returns:
            List of CrossSectionalSignal objects
        """
        if len(prices_dict) < self.min_stocks:
            logger.warning(f"Insufficient stocks: {len(prices_dict)} < {self.min_stocks}")
            return []
        
        # Rank stocks by momentum
        ranked = self.rank_stocks(prices_dict)
        n_stocks = len(ranked)
        
        signals = []
        
        # Calculate cutoffs
        top_cutoff = int(n_stocks * self.top_decile_pct)
        bottom_cutoff = n_stocks - int(n_stocks * self.bottom_decile_pct)
        
        for rank, (symbol, momentum, rank_num) in enumerate(ranked):
            percentile = rank_num / n_stocks
            signal_type = MomentumSignal.NO_SIGNAL
            confidence = 0.0
            
            if rank < top_cutoff:
                signal_type = MomentumSignal.LONG
                confidence = 1.0 - (rank / top_cutoff)
            elif rank >= bottom_cutoff:
                signal_type = MomentumSignal.SHORT
                confidence = (rank - bottom_cutoff) / (n_stocks - bottom_cutoff)
            
            if signal_type != MomentumSignal.NO_SIGNAL:
                signals.append(CrossSectionalSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    rank=rank_num,
                    percentile=percentile,
                    return_momentum=momentum,
                    confidence=confidence,
                    timestamp=pd.Timestamp.now(),
                    metadata={
                        'lookback_period': self.lookback_period,
                        'universe_size': n_stocks
                    }
                ))
        
        return signals


# Singleton instance
_cross_sectional_momentum = None

def get_cross_sectional_momentum() -> CrossSectionalMomentum:
    """Get the singleton cross-sectional momentum instance."""
    global _cross_sectional_momentum
    if _cross_sectional_momentum is None:
        _cross_sectional_momentum = CrossSectionalMomentum()
    return _cross_sectional_momentum


if __name__ == "__main__":
    # Test cross-sectional momentum
    print("Testing Cross-Sectional Momentum Strategy...")
    
    strategy = CrossSectionalMomentum()
    
    # Create sample price data
    np.random.seed(42)
    n_days = 300
    n_stocks = 50
    
    prices_dict = {}
    for i in range(n_stocks):
        # Create random price series with different drifts
        drift = np.random.uniform(-0.001, 0.001)
        prices = pd.Series(100 + np.cumsum(np.random.randn(n_days) * 0.5 + drift))
        prices_dict[f'STOCK_{i}'] = prices
    
    signals = strategy.generate_signals(prices_dict)
    print(f"Generated {len(signals)} signals")
    for signal in signals[:5]:
        print(f"  {signal.symbol}: {signal.signal_type}, rank: {signal.rank}, momentum: {signal.return_momentum:.2%}")

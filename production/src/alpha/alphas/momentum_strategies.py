"""
Momentum Alpha Strategies

This module implements momentum-based alpha strategies including:
- Time Series Momentum (TSMOM)
- Dual Momentum
- Cross-Sectional Momentum
- Volatility-Managed Momentum

Based on Audit Report Priority 2: Alpha Generation
Research Papers: Moskowitz et al (2012), Jegadeesh & Titman (1993)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MomentumSignal:
    """Momentum trading signal."""
    symbol: str
    strategy: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    lookback_period: int
    momentum_score: float
    volatility: float
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TSMOMStrategy:
    """
    Time Series Momentum (TSMOM) Strategy.
    
    Based on Moskowitz et al (2012) - "Time Series Momentum".
    Goes long on assets with positive past returns, short on negative.
    """
    
    def __init__(self, lookback_months: int = 12):
        """
        Initialize TSMOM strategy.
        
        Args:
            lookback_months: Lookback period in months
        """
        self.lookback_months = lookback_months
        self.lookback_days = lookback_months * 21  # Approx trading days
        
        logger.info(f"TSMOMStrategy initialized with {lookback_months} month lookback")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Optional[MomentumSignal]:
        """
        Generate TSMOM signal.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            MomentumSignal
        """
        if len(data) < self.lookback_days:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} < {self.lookback_days}")
            return None
        
        # Calculate cumulative return over lookback period
        recent_data = data.tail(self.lookback_days)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return None
        
        cumulative_return = (1 + returns).prod() - 1
        
        # Calculate momentum score
        momentum_score = cumulative_return
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Generate signal based on momentum
        if momentum_score > 0:
            signal = min(1.0, momentum_score * 2)  # Scale to [-1, 1]
        else:
            signal = max(-1.0, momentum_score * 2)
        
        # Confidence based on momentum magnitude and volatility
        confidence = min(1.0, abs(momentum_score) / volatility)
        
        return MomentumSignal(
            symbol=symbol,
            strategy="TSMOM",
            signal=signal,
            confidence=confidence,
            lookback_period=self.lookback_days,
            momentum_score=momentum_score,
            volatility=volatility,
            timestamp=datetime.now(),
            metadata={
                'cumulative_return': cumulative_return,
                'lookback_months': self.lookback_months
            }
        )


class DualMomentumStrategy:
    """
    Dual Momentum Strategy.
    
    Combines absolute momentum (TSMOM) with relative momentum
    against a benchmark (e.g., NIFTY).
    """
    
    def __init__(self, lookback_months: int = 12, benchmark: str = "NIFTY"):
        """
        Initialize dual momentum strategy.
        
        Args:
            lookback_months: Lookback period in months
            benchmark: Benchmark symbol
        """
        self.lookback_months = lookback_months
        self.lookback_days = lookback_months * 21
        self.benchmark = benchmark
        
        logger.info(f"DualMomentumStrategy initialized with {lookback_months} month lookback, benchmark {benchmark}")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        symbol: str
    ) -> Optional[MomentumSignal]:
        """
        Generate dual momentum signal.
        
        Args:
            data: DataFrame with stock OHLCV data
            benchmark_data: DataFrame with benchmark OHLCV data
            symbol: Stock symbol
            
        Returns:
            MomentumSignal
        """
        if len(data) < self.lookback_days or len(benchmark_data) < self.lookback_days:
            return None
        
        # Calculate absolute momentum
        recent_data = data.tail(self.lookback_days)
        stock_returns = recent_data['close'].pct_change().dropna()
        stock_momentum = (1 + stock_returns).prod() - 1
        
        # Calculate relative momentum vs benchmark
        recent_bench = benchmark_data.tail(self.lookback_days)
        bench_returns = recent_bench['close'].pct_change().dropna()
        bench_momentum = (1 + bench_returns).prod() - 1
        
        relative_momentum = stock_momentum - bench_momentum
        
        # Calculate volatility
        volatility = stock_returns.std() * np.sqrt(252)
        
        # Generate signal based on dual momentum
        # Long if both absolute and relative momentum are positive
        if stock_momentum > 0 and relative_momentum > 0:
            signal = min(1.0, (stock_momentum + relative_momentum) / 2)
        elif stock_momentum < 0 and relative_momentum < 0:
            signal = max(-1.0, (stock_momentum + relative_momentum) / 2)
        else:
            signal = 0.0  # Neutral if signals conflict
        
        # Confidence based on agreement of signals
        abs_signal = 1 if stock_momentum > 0 else -1
        rel_signal = 1 if relative_momentum > 0 else -1
        agreement = abs_signal == rel_signal
        confidence = 0.8 if agreement else 0.4
        
        return MomentumSignal(
            symbol=symbol,
            strategy="DualMomentum",
            signal=signal,
            confidence=confidence,
            lookback_period=self.lookback_days,
            momentum_score=relative_momentum,
            volatility=volatility,
            timestamp=datetime.now(),
            metadata={
                'absolute_momentum': stock_momentum,
                'relative_momentum': relative_momentum,
                'benchmark_momentum': bench_momentum,
                'lookback_months': self.lookback_months
            }
        )


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum Strategy.
    
    Ranks stocks based on past returns and goes long on winners,
    short on losers.
    """
    
    def __init__(self, lookback_months: int = 6, top_pct: float = 0.3, bottom_pct: float = 0.3):
        """
        Initialize cross-sectional momentum strategy.
        
        Args:
            lookback_months: Lookback period in months
            top_pct: Top percentage to go long
            bottom_pct: Bottom percentage to short
        """
        self.lookback_months = lookback_months
        self.lookback_days = lookback_months * 21
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        
        logger.info(f"CrossSectionalMomentum initialized with {lookback_months} month lookback")
    
    def generate_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> List[MomentumSignal]:
        """
        Generate cross-sectional momentum signals for multiple stocks.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            
        Returns:
            List of MomentumSignals
        """
        momentum_scores = {}
        
        # Calculate momentum for each stock
        for symbol, data in data_dict.items():
            if len(data) < self.lookback_days:
                continue
            
            recent_data = data.tail(self.lookback_days)
            returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) == 0:
                continue
            
            momentum = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)
            
            momentum_scores[symbol] = {
                'momentum': momentum,
                'volatility': volatility
            }
        
        if not momentum_scores:
            return []
        
        # Rank by momentum
        sorted_symbols = sorted(momentum_scores.keys(), 
                              key=lambda s: momentum_scores[s]['momentum'])
        
        n = len(sorted_symbols)
        top_n = int(n * self.top_pct)
        bottom_n = int(n * self.bottom_pct)
        
        signals = []
        
        # Generate signals
        for i, symbol in enumerate(sorted_symbols):
            score = momentum_scores[symbol]
            
            if i < bottom_n:
                # Bottom performers - short
                signal = -1.0
                confidence = 0.7
            elif i >= n - top_n:
                # Top performers - long
                signal = 1.0
                confidence = 0.7
            else:
                # Middle - neutral
                signal = 0.0
                confidence = 0.3
            
            signals.append(MomentumSignal(
                symbol=symbol,
                strategy="CrossSectionalMomentum",
                signal=signal,
                confidence=confidence,
                lookback_period=self.lookback_days,
                momentum_score=score['momentum'],
                volatility=score['volatility'],
                timestamp=datetime.now(),
                metadata={
                    'rank': i + 1,
                    'total_stocks': n,
                    'percentile': i / n
                }
            ))
        
        return signals


class VolatilityManagedMomentum:
    """
    Volatility-Managed Momentum Strategy.
    
    Scales position sizes based on volatility to improve risk-adjusted returns.
    """
    
    def __init__(self, lookback_months: int = 12, target_vol: float = 0.15):
        """
        Initialize volatility-managed momentum strategy.
        
        Args:
            lookback_months: Lookback period in months
            target_vol: Target annualized volatility
        """
        self.lookback_months = lookback_months
        self.lookback_days = lookback_months * 21
        self.target_vol = target_vol
        
        logger.info(f"VolatilityManagedMomentum initialized with {lookback_months} month lookback")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Optional[MomentumSignal]:
        """
        Generate volatility-managed momentum signal.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            MomentumSignal
        """
        if len(data) < self.lookback_days:
            return None
        
        # Calculate momentum
        recent_data = data.tail(self.lookback_days)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return None
        
        momentum = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Scale signal by volatility
        vol_scaling = self.target_vol / volatility if volatility > 0 else 1.0
        vol_scaling = min(2.0, max(0.5, vol_scaling))  # Clamp scaling
        
        if momentum > 0:
            signal = min(1.0, momentum * vol_scaling)
        else:
            signal = max(-1.0, momentum * vol_scaling)
        
        confidence = min(1.0, abs(momentum) / volatility)
        
        return MomentumSignal(
            symbol=symbol,
            strategy="VolatilityManagedMomentum",
            signal=signal,
            confidence=confidence,
            lookback_period=self.lookback_days,
            momentum_score=momentum,
            volatility=volatility,
            timestamp=datetime.now(),
            metadata={
                'volatility_scaling': vol_scaling,
                'target_volatility': self.target_vol,
                'lookback_months': self.lookback_months
            }
        )


def get_momentum_signals(
    data_dict: Dict[str, pd.DataFrame],
    strategies: List[str] = None
) -> Dict[str, List[MomentumSignal]]:
    """
    Generate momentum signals using multiple strategies.
    
    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        strategies: List of strategy names to use
        
    Returns:
        Dictionary mapping strategy names to signal lists
    """
    if strategies is None:
        strategies = ["TSMOM", "DualMomentum", "CrossSectionalMomentum", "VolatilityManagedMomentum"]
    
    results = {}
    
    # TSMOM
    if "TSMOM" in strategies:
        tsmon = TSMOMStrategy(lookback_months=12)
        tsmon_signals = []
        for symbol, data in data_dict.items():
            signal = tsmon.generate_signal(data, symbol)
            if signal:
                tsmon_signals.append(signal)
        results["TSMOM"] = tsmon_signals
    
    # Cross-Sectional Momentum
    if "CrossSectionalMomentum" in strategies:
        csm = CrossSectionalMomentum(lookback_months=6)
        results["CrossSectionalMomentum"] = csm.generate_signals(data_dict)
    
    # Volatility-Managed Momentum
    if "VolatilityManagedMomentum" in strategies:
        vmm = VolatilityManagedMomentum(lookback_months=12)
        vmm_signals = []
        for symbol, data in data_dict.items():
            signal = vmm.generate_signal(data, symbol)
            if signal:
                vmm_signals.append(signal)
        results["VolatilityManagedMomentum"] = vmm_signals
    
    return results


if __name__ == "__main__":
    # Test momentum strategies
    print("Testing Momentum Strategies...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=300, freq='1D')
    np.random.seed(42)
    
    data_dict = {
        'RELIANCE': pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0.001, 0.02, 300)) * 1000
        }, index=dates),
        'TCS': pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0.0015, 0.018, 300)) * 3000
        }, index=dates),
        'HDFCBANK': pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0.0008, 0.022, 300)) * 1500
        }, index=dates)
    }
    
    # Generate signals
    signals = get_momentum_signals(data_dict)
    
    print(f"\nGenerated signals:")
    for strategy, signal_list in signals.items():
        print(f"  {strategy}: {len(signal_list)} signals")
        for signal in signal_list[:3]:
            print(f"    {signal.symbol}: {signal.signal:.3f} (confidence: {signal.confidence:.2f})")

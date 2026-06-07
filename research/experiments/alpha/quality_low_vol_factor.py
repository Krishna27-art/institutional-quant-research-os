"""
Quality + Low Volatility Combined Factor Alpha Strategy

This module implements a combined quality and low volatility factor strategy,
which combines the robust quality factor (profitability, low leverage) with the
low volatility anomaly to create a defensive factor portfolio with strong
risk-adjusted returns.

Based on Novy-Marx 2013; Ang et al. 2006; Research Affiliates; 2025 factor ranking.
Expected Sharpe: 0.4-0.6
Expected Capacity: Very High
Decay: Persistent
Difficulty: Low

Priority: Medium (Research OS Phase 6)
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


@dataclass
class QualityMetrics:
    """Quality factor metrics."""
    symbol: str
    roe: float  # Return on Equity
    roa: float  # Return on Assets
    profit_margin: float  # Net profit margin
    debt_to_equity: float  # Debt/Equity ratio
    accruals: float  # Accruals (lower is better)
    quality_score: float  # Combined quality score


@dataclass
class LowVolMetrics:
    """Low volatility factor metrics."""
    symbol: str
    volatility_1m: float  # 1-month volatility
    volatility_3m: float  # 3-month volatility
    volatility_12m: float  # 12-month volatility
    beta: float  # Market beta
    low_vol_score: float  # Combined low volatility score


@dataclass
class CombinedFactorSignal:
    """Combined quality + low volatility signal."""
    timestamp: datetime
    symbol: str
    quality_score: float
    low_vol_score: float
    combined_score: float
    signal: float  # -1 to 1, positive = long, negative = short
    position_size: float
    confidence: float


class QualityLowVolFactorAlpha:
    """
    Quality + Low Volatility combined factor alpha strategy.
    
    This class combines quality metrics (profitability, low leverage)
    with low volatility metrics to create a defensive factor portfolio.
    """
    
    def __init__(
        self,
        quality_weight: float = 0.5,
        low_vol_weight: float = 0.5,
        lookback_days: int = 252,
        top_decile: float = 0.1,
        bottom_decile: float = 0.1
    ):
        """
        Initialize quality + low volatility factor alpha.
        
        Args:
            quality_weight: Weight for quality factor
            low_vol_weight: Weight for low volatility factor
            lookback_days: Lookback period for calculations
            top_decile: Top decile for long positions
            bottom_decile: Bottom decile for short positions
        """
        self.quality_weight = quality_weight
        self.low_vol_weight = low_vol_weight
        self.lookback_days = lookback_days
        self.top_decile = top_decile
        self.bottom_decile = bottom_decile
        
        self.quality_metrics: Dict[str, QualityMetrics] = {}
        self.low_vol_metrics: Dict[str, LowVolMetrics] = {}
        self.signals: List[CombinedFactorSignal] = []
        
        logger.info(f"QualityLowVolFactorAlpha initialized: quality_weight={quality_weight}, "
                   f"low_vol_weight={low_vol_weight}")
    
    def calculate_quality_score(
        self,
        roe: float,
        roa: float,
        profit_margin: float,
        debt_to_equity: float,
        accruals: float
    ) -> float:
        """
        Calculate combined quality score.
        
        Args:
            roe: Return on Equity
            roa: Return on Assets
            profit_margin: Net profit margin
            debt_to_equity: Debt/Equity ratio
            accruals: Accruals
            
        Returns:
            Quality score (0-1, higher is better)
        """
        # Normalize individual metrics
        # Higher ROE, ROA, profit margin = better quality
        # Lower debt_to_equity, accruals = better quality
        
        roe_score = min(roe / 0.20, 1.0)  # 20% ROE = max score
        roa_score = min(roa / 0.10, 1.0)  # 10% ROA = max score
        margin_score = min(profit_margin / 0.15, 1.0)  # 15% margin = max score
        debt_score = max(1.0 - debt_to_equity / 2.0, 0.0)  # 2.0 D/E = min score
        accruals_score = max(1.0 - abs(accruals) / 0.10, 0.0)  # 10% accruals = min score
        
        # Weighted average
        quality_score = (
            0.3 * roe_score +
            0.2 * roa_score +
            0.2 * margin_score +
            0.2 * debt_score +
            0.1 * accruals_score
        )
        
        return quality_score
    
    def calculate_low_vol_score(
        self,
        volatility_1m: float,
        volatility_3m: float,
        volatility_12m: float,
        beta: float
    ) -> float:
        """
        Calculate combined low volatility score.
        
        Args:
            volatility_1m: 1-month volatility
            volatility_3m: 3-month volatility
            volatility_12m: 12-month volatility
            beta: Market beta
            
        Returns:
            Low volatility score (0-1, higher is better = lower vol)
        """
        # Normalize: lower volatility = higher score
        vol_1m_score = max(1.0 - volatility_1m / 0.50, 0.0)  # 50% vol = min score
        vol_3m_score = max(1.0 - volatility_3m / 0.40, 0.0)  # 40% vol = min score
        vol_12m_score = max(1.0 - volatility_12m / 0.30, 0.0)  # 30% vol = min score
        beta_score = max(1.0 - abs(beta - 1.0) / 1.0, 0.0)  # Beta near 1 = higher score
        
        # Weighted average (more weight to longer-term volatility)
        low_vol_score = (
            0.2 * vol_1m_score +
            0.3 * vol_3m_score +
            0.3 * vol_12m_score +
            0.2 * beta_score
        )
        
        return low_vol_score
    
    def calculate_quality_metrics(
        self,
        symbol: str,
        financials: Dict[str, float]
    ) -> QualityMetrics:
        """
        Calculate quality metrics for a symbol.
        
        Args:
            symbol: Stock symbol
            financials: Financial data dictionary
            
        Returns:
            QualityMetrics
        """
        roe = financials.get('roe', 0.0)
        roa = financials.get('roa', 0.0)
        profit_margin = financials.get('profit_margin', 0.0)
        debt_to_equity = financials.get('debt_to_equity', 1.0)
        accruals = financials.get('accruals', 0.0)
        
        quality_score = self.calculate_quality_score(
            roe, roa, profit_margin, debt_to_equity, accruals
        )
        
        metrics = QualityMetrics(
            symbol=symbol,
            roe=roe,
            roa=roa,
            profit_margin=profit_margin,
            debt_to_equity=debt_to_equity,
            accruals=accruals,
            quality_score=quality_score
        )
        
        self.quality_metrics[symbol] = metrics
        return metrics
    
    def calculate_low_vol_metrics(
        self,
        symbol: str,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> LowVolMetrics:
        """
        Calculate low volatility metrics for a symbol.
        
        Args:
            symbol: Stock symbol
            returns: Stock returns series
            market_returns: Market returns series
            
        Returns:
            LowVolMetrics
        """
        if len(returns) < 20:
            # Default values if insufficient data
            return LowVolMetrics(
                symbol=symbol,
                volatility_1m=0.3,
                volatility_3m=0.3,
                volatility_12m=0.3,
                beta=1.0,
                low_vol_score=0.5
            )
        
        # Calculate volatilities
        volatility_1m = returns.tail(20).std() * np.sqrt(252)
        volatility_3m = returns.tail(60).std() * np.sqrt(252)
        volatility_12m = returns.tail(252).std() * np.sqrt(252)
        
        # Calculate beta
        if len(returns) == len(market_returns) and len(returns) > 0:
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
        else:
            beta = 1.0
        
        low_vol_score = self.calculate_low_vol_score(
            volatility_1m, volatility_3m, volatility_12m, beta
        )
        
        metrics = LowVolMetrics(
            symbol=symbol,
            volatility_1m=volatility_1m,
            volatility_3m=volatility_3m,
            volatility_12m=volatility_12m,
            beta=beta,
            low_vol_score=low_vol_score
        )
        
        self.low_vol_metrics[symbol] = metrics
        return metrics
    
    def generate_signal(
        self,
        symbol: str,
        financials: Dict[str, float],
        returns: pd.Series,
        market_returns: pd.Series,
        timestamp: datetime
    ) -> CombinedFactorSignal:
        """
        Generate combined quality + low volatility signal.
        
        Args:
            symbol: Stock symbol
            financials: Financial data dictionary
            returns: Stock returns series
            market_returns: Market returns series
            timestamp: Signal timestamp
            
        Returns:
            CombinedFactorSignal
        """
        # Calculate metrics
        quality_metrics = self.calculate_quality_metrics(symbol, financials)
        low_vol_metrics = self.calculate_low_vol_metrics(symbol, returns, market_returns)
        
        # Calculate combined score
        combined_score = (
            self.quality_weight * quality_metrics.quality_score +
            self.low_vol_weight * low_vol_metrics.low_vol_score
        )
        
        # Generate signal
        # High combined score = long, low = short
        signal = (combined_score - 0.5) * 2  # Scale to -1 to 1
        
        # Position sizing based on score strength
        position_size = abs(signal) * 0.5  # Max 50% position
        
        # Confidence based on score strength
        confidence = abs(signal)
        
        combined_signal = CombinedFactorSignal(
            timestamp=timestamp,
            symbol=symbol,
            quality_score=quality_metrics.quality_score,
            low_vol_score=low_vol_metrics.low_vol_score,
            combined_score=combined_score,
            signal=signal,
            position_size=position_size,
            confidence=confidence
        )
        
        self.signals.append(combined_signal)
        
        return combined_signal
    
    def get_portfolio_signals(
        self,
        n_long: int = 20,
        n_short: int = 20
    ) -> Tuple[List[CombinedFactorSignal], List[CombinedFactorSignal]]:
        """
        Get top long and short signals.
        
        Args:
            n_long: Number of long positions
            n_short: Number of short positions
            
        Returns:
            (long_signals, short_signals)
        """
        if not self.signals:
            return [], []
        
        # Get latest signal for each symbol
        latest_signals = {}
        for signal in self.signals:
            if signal.symbol not in latest_signals or signal.timestamp > latest_signals[signal.symbol].timestamp:
                latest_signals[signal.symbol] = signal
        
        signals_list = list(latest_signals.values())
        
        # Sort by combined score
        sorted_signals = sorted(signals_list, key=lambda x: x.combined_score, reverse=True)
        
        # Top for long, bottom for short
        long_signals = sorted_signals[:n_long]
        short_signals = sorted_signals[-n_short:]
        
        return long_signals, short_signals
    
    def print_factor_report(self) -> None:
        """Print factor analysis report."""
        print("\n" + "="*60)
        print("QUALITY + LOW VOLATILITY FACTOR ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Quality Weight: {self.quality_weight}")
        print(f"  Low Vol Weight: {self.low_vol_weight}")
        print(f"  Lookback Days: {self.lookback_days}")
        
        print(f"\nStatistics:")
        print(f"  Total Quality Metrics: {len(self.quality_metrics)}")
        print(f"  Total Low Vol Metrics: {len(self.low_vol_metrics)}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.quality_metrics:
            quality_scores = [m.quality_score for m in self.quality_metrics.values()]
            print(f"\nQuality Statistics:")
            print(f"  Average Quality Score: {np.mean(quality_scores):.4f}")
            print(f"  Min Quality Score: {np.min(quality_scores):.4f}")
            print(f"  Max Quality Score: {np.max(quality_scores):.4f}")
        
        if self.low_vol_metrics:
            low_vol_scores = [m.low_vol_score for m in self.low_vol_metrics.values()]
            print(f"\nLow Vol Statistics:")
            print(f"  Average Low Vol Score: {np.mean(low_vol_scores):.4f}")
            print(f"  Min Low Vol Score: {np.min(low_vol_scores):.4f}")
            print(f"  Max Low Vol Score: {np.max(low_vol_scores):.4f}")
        
        if self.signals:
            long_signals, short_signals = self.get_portfolio_signals(10, 10)
            
            print(f"\nTop 10 Long Positions:")
            print(f"{'Symbol':<10} {'Quality':<10} {'LowVol':<10} {'Combined':<10} {'Signal':<10} {'Position':<10}")
            print("-" * 70)
            
            for signal in long_signals:
                print(f"{signal.symbol:<10} {signal.quality_score:<10.4f} {signal.low_vol_score:<10.4f} "
                      f"{signal.combined_score:<10.4f} {signal.signal:<10.3f} {signal.position_size:<10.3f}")
            
            print(f"\nTop 10 Short Positions:")
            print(f"{'Symbol':<10} {'Quality':<10} {'LowVol':<10} {'Combined':<10} {'Signal':<10} {'Position':<10}")
            print("-" * 70)
            
            for signal in short_signals:
                print(f"{signal.symbol:<10} {signal.quality_score:<10.4f} {signal.low_vol_score:<10.4f} "
                      f"{signal.combined_score:<10.4f} {signal.signal:<10.3f} {signal.position_size:<10.3f}")
        
        print("\n" + "="*60)


def sample_quality_low_vol_factor_alpha():
    """Demonstrate quality + low volatility factor alpha."""
    print("=== Quality + Low Volatility Factor Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = QualityLowVolFactorAlpha(
        quality_weight=0.5,
        low_vol_weight=0.5,
        lookback_days=252
    )
    
    # Generate sample data
    np.random.seed(42)
    n_stocks = 50
    n_days = 252
    
    symbols = [f"STOCK{i:03d}" for i in range(n_stocks)]
    dates = pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Generate market returns
    market_returns = pd.Series(np.random.randn(n_days) * 0.01, index=dates)
    
    # Generate stock data
    for symbol in symbols:
        # Generate financials
        financials = {
            'roe': np.random.uniform(0.05, 0.25),
            'roa': np.random.uniform(0.02, 0.15),
            'profit_margin': np.random.uniform(0.02, 0.20),
            'debt_to_equity': np.random.uniform(0.1, 2.0),
            'accruals': np.random.uniform(-0.05, 0.05)
        }
        
        # Generate returns
        beta = np.random.uniform(0.5, 1.5)
        stock_returns = market_returns * beta + np.random.randn(n_days) * 0.02
        returns_series = pd.Series(stock_returns, index=dates)
        
        # Generate signal
        signal = alpha.generate_signal(
            symbol,
            financials,
            returns_series,
            market_returns,
            datetime.now()
        )
    
    # Print report
    alpha.print_factor_report()
    
    print("\n=== Quality + Low Volatility Factor Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Quality metrics (ROE, ROA, profit margin, debt/equity, accruals)")
    print("- Low volatility metrics (1m, 3m, 12m vol, beta)")
    print("- Combined factor scoring")
    print("- Portfolio construction (long/short)")
    print("- Expected Sharpe: 0.4-0.6")
    print("- Expected Capacity: Very High")
    print("- Decay: Persistent")


if __name__ == "__main__":
    sample_quality_low_vol_factor_alpha()

"""
CAPM-based Alphas: Residual Momentum, Beta Neutral ORB
Based on the critique: Use CAPM for Beta, Residual Return, Idiosyncratic Alpha

Instead of:
    Buy stock

Trade:
    Alpha = stock return - beta * market return

This removes market noise and provides much stronger signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from scipy import stats


@dataclass
class CAPMMetrics:
    """CAPM metrics for a stock."""
    symbol: str
    beta: float
    alpha: float  # CAPM alpha (intercept)
    r_squared: float
    p_value: float
    is_significant: bool


@dataclass
class ResidualMomentumSignal:
    """Residual momentum signal."""
    symbol: str
    timestamp: datetime
    stock_return: float
    market_return: float
    beta: float
    residual_return: float
    signal: float  # -1 to 1
    confidence: float


@dataclass
class BetaNeutralORBSignal:
    """Beta neutral ORB signal."""
    symbol: str
    timestamp: datetime
    orb_signal: float
    beta: float
    market_signal: float
    beta_neutral_signal: float
    hedge_ratio: float


class CAPMEngine:
    """
    CAPM Engine for calculating beta, alpha, and residual returns.
    
    CAPM: E[R_i] = R_f + beta * (E[R_m] - R_f)
    
    Residual Return = R_i - beta * R_m
    This removes market exposure and isolates idiosyncratic alpha.
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.risk_free_rate = risk_free_rate
        self.capm_metrics: Dict[str, CAPMMetrics] = {}
        self.residual_signals: Dict[str, List[ResidualMomentumSignal]] = {}
        self.beta_neutral_signals: Dict[str, List[BetaNeutralORBSignal]] = {}
    
    def calculate_beta(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        window_days: int = 252
    ) -> CAPMMetrics:
        """
        Calculate beta using regression.
        
        Beta = Cov(R_i, R_m) / Var(R_m)
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns
            window_days: Window for calculation
            
        Returns:
            CAPMMetrics
        """
        # Align series
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        stock_ret = aligned.iloc[:, 0]
        market_ret = aligned.iloc[:, 1]
        
        if len(aligned) < 30:
            return CAPMMetrics(
                symbol="unknown",
                beta=1.0,
                alpha=0.0,
                r_squared=0.0,
                p_value=1.0,
                is_significant=False
            )
        
        # Calculate beta using regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market_ret, stock_ret
        )
        
        beta = slope
        alpha = intercept
        r_squared = r_value ** 2
        is_significant = p_value < 0.05
        
        metrics = CAPMMetrics(
            symbol="unknown",  # Will be set by caller
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            p_value=p_value,
            is_significant=is_significant
        )
        
        return metrics
    
    def calculate_residual_return(
        self,
        stock_return: float,
        market_return: float,
        beta: float
    ) -> float:
        """
        Calculate residual return.
        
        Residual Return = Stock Return - Beta * Market Return
        
        Args:
            stock_return: Stock return
            market_return: Market return
            beta: Beta coefficient
            
        Returns:
            Residual return
        """
        residual = stock_return - beta * market_return
        return residual
    
    def generate_residual_momentum(
        self,
        symbol: str,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        lookback_days: int = 20
    ) -> ResidualMomentumSignal:
        """
        Generate residual momentum signal.
        
        Process:
        1. Calculate beta
        2. Calculate residual returns over lookback
        3. Generate signal based on residual momentum
        
        Args:
            symbol: Trading symbol
            stock_returns: Stock returns
            market_returns: Market returns
            lookback_days: Lookback period for momentum
            
        Returns:
            ResidualMomentumSignal
        """
        # Calculate beta
        metrics = self.calculate_beta(stock_returns, market_returns)
        metrics.symbol = symbol
        self.capm_metrics[symbol] = metrics
        
        # Get recent returns
        recent_stock = stock_returns.iloc[-lookback_days:]
        recent_market = market_returns.iloc[-lookback_days:]
        
        # Calculate residual returns
        residual_returns = recent_stock - metrics.beta * recent_market
        
        # Calculate cumulative residual return
        cumulative_residual = residual_returns.sum()
        
        # Calculate stock and market returns
        stock_return = recent_stock.sum()
        market_return = recent_market.sum()
        
        # Generate signal
        # Positive residual momentum = buy signal
        signal_strength = min(abs(cumulative_residual) / 0.1, 1.0)  # Normalize to 0-1
        signal = np.sign(cumulative_residual) * signal_strength
        
        # Confidence based on R-squared
        confidence = metrics.r_squared
        
        residual_signal = ResidualMomentumSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            stock_return=stock_return,
            market_return=market_return,
            beta=metrics.beta,
            residual_return=cumulative_residual,
            signal=signal,
            confidence=confidence
        )
        
        # Store in history
        if symbol not in self.residual_signals:
            self.residual_signals[symbol] = []
        self.residual_signals[symbol].append(residual_signal)
        
        return residual_signal
    
    def generate_beta_neutral_orb(
        self,
        symbol: str,
        orb_signal: float,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        market_orb_signal: float
    ) -> BetaNeutralORBSignal:
        """
        Generate beta neutral ORB signal.
        
        Process:
        1. Calculate beta
        2. Hedge market exposure
        3. Generate beta neutral signal
        
        Args:
            symbol: Trading symbol
            orb_signal: Original ORB signal
            stock_returns: Stock returns
            market_returns: Market returns
            market_orb_signal: Market ORB signal (e.g., NIFTY ORB)
            
        Returns:
            BetaNeutralORBSignal
        """
        # Calculate beta
        metrics = self.calculate_beta(stock_returns, market_returns)
        metrics.symbol = symbol
        self.capm_metrics[symbol] = metrics
        
        # Calculate hedge ratio
        # Hedge ratio = beta * (market_orb_signal / orb_signal)
        hedge_ratio = metrics.beta
        
        # Calculate beta neutral signal
        # Beta neutral signal = ORB signal - beta * market ORB signal
        beta_neutral_signal = orb_signal - hedge_ratio * market_orb_signal
        
        # Normalize to -1 to 1
        if abs(beta_neutral_signal) > 1:
            beta_neutral_signal = beta_neutral_signal / abs(beta_neutral_signal)
        
        beta_neutral_orb = BetaNeutralORBSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            orb_signal=orb_signal,
            beta=metrics.beta,
            market_signal=market_orb_signal,
            beta_neutral_signal=beta_neutral_signal,
            hedge_ratio=hedge_ratio
        )
        
        # Store in history
        if symbol not in self.beta_neutral_signals:
            self.beta_neutral_signals[symbol] = []
        self.beta_neutral_signals[symbol].append(beta_neutral_orb)
        
        return beta_neutral_orb
    
    def get_capm_summary(self) -> pd.DataFrame:
        """Get summary of CAPM metrics for all stocks."""
        data = []
        
        for symbol, metrics in self.capm_metrics.items():
            data.append({
                'Symbol': symbol,
                'Beta': metrics.beta,
                'Alpha': metrics.alpha,
                'R-Squared': metrics.r_squared,
                'P-Value': metrics.p_value,
                'Significant': metrics.is_significant
            })
        
        return pd.DataFrame(data)
    
    def get_residual_momentum_summary(self) -> pd.DataFrame:
        """Get summary of residual momentum signals."""
        data = []
        
        for symbol, signals in self.residual_signals.items():
            latest = signals[-1]
            data.append({
                'Symbol': symbol,
                'Stock Return': latest.stock_return,
                'Market Return': latest.market_return,
                'Beta': latest.beta,
                'Residual Return': latest.residual_return,
                'Signal': latest.signal,
                'Confidence': latest.confidence
            })
        
        return pd.DataFrame(data)
    
    def get_beta_neutral_orb_summary(self) -> pd.DataFrame:
        """Get summary of beta neutral ORB signals."""
        data = []
        
        for symbol, signals in self.beta_neutral_signals.items():
            latest = signals[-1]
            data.append({
                'Symbol': symbol,
                'ORB Signal': latest.orb_signal,
                'Beta': latest.beta,
                'Market Signal': latest.market_signal,
                'Beta Neutral Signal': latest.beta_neutral_signal,
                'Hedge Ratio': latest.hedge_ratio
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the CAPM Engine
    print("Testing CAPM-based Alphas: Residual Momentum, Beta Neutral ORB...")
    
    engine = CAPMEngine()
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    # Market returns
    market_returns = pd.Series(np.random.normal(0.0005, 0.015, n), index=dates)
    
    # Stock returns with beta = 1.2
    stock_returns = pd.Series(
        0.0001 + 1.2 * market_returns + np.random.normal(0, 0.01, n),
        index=dates
    )
    
    # Calculate beta
    print("\nCalculating Beta...")
    metrics = engine.calculate_beta(stock_returns, market_returns)
    print(f"Beta: {metrics.beta:.2f}")
    print(f"Alpha: {metrics.alpha:.4f}")
    print(f"R-Squared: {metrics.r_squared:.2%}")
    print(f"Significant: {metrics.is_significant}")
    
    # Generate residual momentum signal
    print("\nGenerating Residual Momentum Signal...")
    residual_signal = engine.generate_residual_momentum(
        symbol="RELIANCE",
        stock_returns=stock_returns,
        market_returns=market_returns,
        lookback_days=20
    )
    print(f"Stock Return: {residual_signal.stock_return:.2%}")
    print(f"Market Return: {residual_signal.market_return:.2%}")
    print(f"Beta: {residual_signal.beta:.2f}")
    print(f"Residual Return: {residual_signal.residual_return:.2%}")
    print(f"Signal: {residual_signal.signal:.2f}")
    print(f"Confidence: {residual_signal.confidence:.2%}")
    
    # Generate beta neutral ORB signal
    print("\nGenerating Beta Neutral ORB Signal...")
    orb_signal = 0.8  # Strong buy ORB signal
    market_orb_signal = 0.5  # Market ORB signal
    
    beta_neutral_orb = engine.generate_beta_neutral_orb(
        symbol="RELIANCE",
        orb_signal=orb_signal,
        stock_returns=stock_returns,
        market_returns=market_returns,
        market_orb_signal=market_orb_signal
    )
    print(f"ORB Signal: {beta_neutral_orb.orb_signal:.2f}")
    print(f"Beta: {beta_neutral_orb.beta:.2f}")
    print(f"Market Signal: {beta_neutral_orb.market_signal:.2f}")
    print(f"Beta Neutral Signal: {beta_neutral_orb.beta_neutral_signal:.2f}")
    print(f"Hedge Ratio: {beta_neutral_orb.hedge_ratio:.2f}")
    
    # Get summaries
    print("\nCAPM Summary:")
    capm_summary = engine.get_capm_summary()
    print(capm_summary.to_string(index=False))
    
    print("\nResidual Momentum Summary:")
    residual_summary = engine.get_residual_momentum_summary()
    print(residual_summary.to_string(index=False))
    
    print("\nBeta Neutral ORB Summary:")
    beta_neutral_summary = engine.get_beta_neutral_orb_summary()
    print(beta_neutral_summary.to_string(index=False))

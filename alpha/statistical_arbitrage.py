"""
Statistical Arbitrage Strategies

Implements institutional-grade statistical arbitrage strategies:
- PCA-Based Pairs Trading (Avellaneda & Lee 2010)
- ETF Arbitrage (Marshall et al. 2013)
- Cross-Asset Arbitrage

Based on blueprint specification for multi-strategy framework
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatArbSignal:
    """Statistical arbitrage signal"""
    assets: List[str]
    signal: np.ndarray  # Vector of signals for each asset
    z_scores: np.ndarray
    confidence: float
    strategy: str


class PCAStatArb:
    """
    PCA-Based Statistical Arbitrage (Avellaneda & Lee 2010)
    
    Formula:
    Compute first K principal components of the return matrix.
    Residual returns ε = R - βF are mean-reverting.
    
    Expected Sharpe: 0.6
    Capacity: High
    Best Regime: Correlated markets
    Failure: Structural breaks
    """
    
    def __init__(
        self,
        n_components: int = 5,
        lookback: int = 252,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ):
        """
        Initialize PCA statistical arbitrage.
        
        Args:
            n_components: Number of principal components
            lookback: Lookback period for PCA estimation
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.n_components = n_components
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        self.pca = PCA(n_components=n_components)
        self.betas = None
        self.residuals_mean = None
        self.residuals_std = None
        
    def fit(self, returns: pd.DataFrame):
        """
        Fit PCA model to historical returns.
        
        Args:
            returns: DataFrame of asset returns
        """
        # Use lookback period
        recent_returns = returns.tail(self.lookback).dropna()
        
        if len(recent_returns) < self.lookback * 0.8:
            logger.warning("Insufficient data for PCA fitting")
            return
        
        # Fit PCA
        factors = self.pca.fit_transform(recent_returns.values)
        
        # Calculate betas (loadings)
        self.betas = np.linalg.lstsq(factors, recent_returns.values, rcond=None)[0]
        
        # Calculate residuals
        residuals = recent_returns.values - factors @ self.betas
        self.residuals_mean = residuals.mean(axis=0)
        self.residuals_std = residuals.std(axis=0)
        
        logger.info(f"PCA fitted with {self.n_components} components")
        
    def compute_signal(self, returns: pd.DataFrame) -> StatArbSignal:
        """
        Compute statistical arbitrage signal.
        
        Args:
            returns: DataFrame of current returns
            
        Returns:
            StatArbSignal
        """
        if self.betas is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get latest returns
        latest_returns = returns.iloc[-1:].values
        
        # Transform to factor space
        factors = self.pca.transform(latest_returns)
        
        # Calculate residuals
        residuals = latest_returns - factors @ self.betas
        
        # Calculate z-scores
        z_scores = (residuals - self.residuals_mean) / (self.residuals_std + 1e-8)
        z_scores = z_scores.flatten()
        
        # Generate signals (short when residual high, long when low)
        signals = -np.sign(z_scores)
        
        # Apply thresholds
        mask = np.abs(z_scores) > self.entry_threshold
        signals = signals * mask.astype(float)
        
        # Calculate confidence
        confidence = np.mean(np.abs(z_scores[mask]) / self.entry_threshold) if mask.any() else 0
        confidence = min(1.0, confidence)
        
        return StatArbSignal(
            assets=list(returns.columns),
            signal=signals,
            z_scores=z_scores,
            confidence=confidence,
            strategy="PCA_StatArb"
        )


class ETFArbitrage:
    """
    ETF Arbitrage (Marshall et al. 2013)
    
    Formula:
    Basis = (ETF_price - NAV) / NAV
    
    If basis > 2%, short ETF, buy basket
    If basis < -2%, long ETF, short basket
    
    Expected Sharpe: 0.5
    Capacity: 500 Cr
    Turnover: 100%/month
    Best Regime: Positive basis
    Failure: Negative basis
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.02,
        exit_threshold: float = 0.005
    ):
        """
        Initialize ETF arbitrage.
        
        Args:
            entry_threshold: Basis threshold for entry (2%)
            exit_threshold: Basis threshold for exit (0.5%)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def compute_basis(
        self,
        etf_price: float,
        nav: float,
        basket_prices: np.ndarray,
        basket_weights: np.ndarray
    ) -> float:
        """
        Calculate ETF basis.
        
        Args:
            etf_price: Current ETF price
            nav: Net asset value
            basket_prices: Prices of underlying basket components
            basket_weights: Weights of basket components
            
        Returns:
            Basis (ETF price - NAV) / NAV
        """
        # Calculate basket value
        basket_value = np.sum(basket_prices * basket_weights)
        
        # Calculate basis
        basis = (etf_price - basket_value) / basket_value
        
        return basis
    
    def get_signal(
        self,
        etf_price: float,
        nav: float,
        basket_prices: np.ndarray,
        basket_weights: np.ndarray
    ) -> Tuple[float, str]:
        """
        Get arbitrage signal.
        
        Args:
            etf_price: Current ETF price
            nav: Net asset value
            basket_prices: Prices of underlying basket components
            basket_weights: Weights of basket components
            
        Returns:
            Tuple of (signal, direction)
            signal: Position size (-1 to 1)
            direction: 'SHORT_ETF_LONG_BASKET' or 'LONG_ETF_SHORT_BASKET'
        """
        basis = self.compute_basis(etf_price, nav, basket_prices, basket_weights)
        
        if basis > self.entry_threshold:
            # ETF overpriced, short ETF, long basket
            signal = -1.0
            direction = "SHORT_ETF_LONG_BASKET"
        elif basis < -self.entry_threshold:
            # ETF underpriced, long ETF, short basket
            signal = 1.0
            direction = "LONG_ETF_SHORT_BASKET"
        elif abs(basis) < self.exit_threshold:
            # Close position
            signal = 0.0
            direction = "EXIT"
        else:
            signal = 0.0
            direction = "HOLD"
        
        return signal, direction


class CrossAssetArbitrage:
    """
    Cross-Asset Arbitrage
    
    Trades relationships between different asset classes (e.g., stocks vs futures,
    spot vs forward, different exchanges).
    """
    
    def __init__(self, lookback: int = 21, entry_threshold: float = 2.0):
        """
        Initialize cross-asset arbitrage.
        
        Args:
            lookback: Lookback period for relationship estimation
            entry_threshold: Z-score threshold for entry
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        
    def compute_spread(
        self,
        price1: pd.Series,
        price2: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Compute spread between two assets.
        
        Args:
            price1: Price series of asset 1
            price2: Price series of asset 2
            hedge_ratio: Optional hedge ratio (if None, estimate via OLS)
            
        Returns:
            Spread series
        """
        if hedge_ratio is None:
            # Estimate hedge ratio via OLS
            recent_data = pd.DataFrame({'p1': price1, 'p2': price2}).tail(self.lookback).dropna()
            if len(recent_data) < 10:
                hedge_ratio = 1.0
            else:
                hedge_ratio = np.cov(recent_data['p1'], recent_data['p2'])[0, 1] / np.var(recent_data['p2'])
        
        spread = price1 - hedge_ratio * price2
        return spread
    
    def get_signal(
        self,
        spread: pd.Series
    ) -> Tuple[float, float, str]:
        """
        Get arbitrage signal from spread.
        
        Args:
            spread: Spread series
            
        Returns:
            Tuple of (signal, z_score, direction)
        """
        # Calculate z-score
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-8)
        
        current_z = z_score.iloc[-1]
        
        if current_z > self.entry_threshold:
            signal = -1.0
            direction = "SHORT_SPREAD"
        elif current_z < -self.entry_threshold:
            signal = 1.0
            direction = "LONG_SPREAD"
        else:
            signal = 0.0
            direction = "HOLD"
        
        return signal, current_z, direction


if __name__ == "__main__":
    # Test statistical arbitrage strategies
    print("Testing Statistical Arbitrage Strategies...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    # Create correlated returns for PCA
    n_assets = 10
    returns = pd.DataFrame(
        np.random.randn(n, n_assets) * 0.01,
        index=dates,
        columns=[f'Asset{i}' for i in range(n_assets)]
    )
    
    # Add common factor
    common_factor = np.random.randn(n) * 0.02
    for i in range(n_assets):
        returns.iloc[:, i] += 0.5 * common_factor
    
    # Test PCA Stat Arb
    print("\n1. PCA Statistical Arbitrage:")
    pca_arb = PCAStatArb(n_components=3, lookback=252)
    pca_arb.fit(returns)
    signal = pca_arb.compute_signal(returns)
    print(f"   Signal shape: {signal.signal.shape}")
    print(f"   Non-zero signals: {(signal.signal != 0).sum()}")
    print(f"   Confidence: {signal.confidence:.3f}")
    
    # Test ETF Arbitrage
    print("\n2. ETF Arbitrage:")
    etf_arb = ETFArbitrage()
    etf_price = 100.0
    nav = 98.0
    basket_prices = np.array([20, 30, 25, 25])  # Sum = 100
    basket_weights = np.array([0.2, 0.3, 0.25, 0.25])
    
    signal, direction = etf_arb.get_signal(etf_price, nav, basket_prices, basket_weights)
    basis = etf_arb.compute_basis(etf_price, nav, basket_prices, basket_weights)
    print(f"   Basis: {basis:.4f}")
    print(f"   Signal: {signal:.2f}")
    print(f"   Direction: {direction}")
    
    # Test Cross-Asset Arbitrage
    print("\n3. Cross-Asset Arbitrage:")
    cross_arb = CrossAssetArbitrage()
    price1 = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    price2 = pd.Series(50 + np.cumsum(np.random.randn(n) * 0.3), index=dates)
    
    spread = cross_arb.compute_spread(price1, price2)
    signal, z_score, direction = cross_arb.get_signal(spread)
    print(f"   Spread z-score: {z_score:.3f}")
    print(f"   Signal: {signal:.2f}")
    print(f"   Direction: {direction}")
    
    print("\n✓ All statistical arbitrage strategies tested")

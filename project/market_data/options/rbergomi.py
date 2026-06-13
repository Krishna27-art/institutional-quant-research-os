"""
Rough Volatility Option Strategies (rBergomi)

Implements the rBergomi (Rough Bergomi) model for option pricing and
volatility trading strategies. The rBergomi model captures the roughness
of volatility observed in markets using fractional Brownian motion.

Key Features:
- rBergomi model implementation with fractional Brownian motion
- Rough volatility parameter estimation (H < 0.5)
- Option pricing under rough volatility
- Volatility trading strategies (VRP, gamma scalping)
- VIX basis trading
- Term structure arbitrage

Based on Blueprint Week 7-8: Advanced Alpha (Papers)
References:
- Gatheral et al. (2018) - Volatility is Rough
- Bayer et al. (2016) - rBergomi Model
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class FractionalBrownianMotion:
    """
    Fractional Brownian Motion (fBm).
    
    fBm is a Gaussian process with Hurst parameter H:
    - H = 0.5: Standard Brownian motion
    - H < 0.5: Rough (anti-persistent)
    - H > 0.5: Smooth (persistent)
    
    For volatility, H ≈ 0.1 (very rough).
    """
    
    def __init__(self, H: float = 0.1):
        """
        Initialize fBm.
        
        Args:
            H: Hurst parameter (0 < H < 1)
        """
        self.H = H
    
    def generate(self, n_steps: int, dt: float = 1.0) -> np.ndarray:
        """
        Generate fBm path using Cholesky decomposition.
        
        Args:
            n_steps: Number of steps
            dt: Time step
            
        Returns:
            fBm path
        """
        # Covariance matrix for fBm
        cov = np.zeros((n_steps, n_steps))
        for i in range(n_steps):
            for j in range(n_steps):
                t_i = (i + 1) * dt
                t_j = (j + 1) * dt
                cov[i, j] = 0.5 * (t_i ** (2 * self.H) + t_j ** (2 * self.H) - abs(t_i - t_j) ** (2 * self.H))
        
        # Cholesky decomposition
        L = np.linalg.cholesky(cov)
        
        # Generate normal random variables
        Z = np.random.normal(0, 1, n_steps)
        
        # fBm path
        fBm = L @ Z
        
        return fBm


class RBergomiModel:
    """
    rBergomi (Rough Bergomi) Model.
    
    The rBergomi model is defined by:
    dS_t = S_t sqrt(V_t) dW_t
    V_t = ξ * exp(η * W_t^H - 0.5 * η^2 * t^(2H))
    
    where W_t^H is fractional Brownian motion with Hurst H.
    """
    
    def __init__(
        self,
        H: float = 0.1,
        eta: float = 1.9,
        rho: float = -0.9,
        xi: float = 0.04
    ):
        """
        Initialize rBergomi model.
        
        Args:
            H: Hurst parameter (roughness, typically 0.1)
            eta: Vol of vol
            rho: Correlation between price and vol shocks
            xi: Initial volatility
        """
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi = xi
        
        self.fbm = FractionalBrownianMotion(H)
    
    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rBergomi paths.
        
        Args:
            S0: Initial price
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            
        Returns:
            Tuple of (price paths, volatility paths)
        """
        dt = T / n_steps
        
        price_paths = np.zeros((n_paths, n_steps + 1))
        vol_paths = np.zeros((n_paths, n_steps + 1))
        
        price_paths[:, 0] = S0
        vol_paths[:, 0] = self.xi
        
        for path in range(n_paths):
            # Generate fBm for volatility
            fBm = self.fbm.generate(n_steps, dt)
            
            # Generate correlated Brownian motion for price
            dW1 = np.random.normal(0, np.sqrt(dt), n_steps)
            dW2 = self.rho * dW1 + np.sqrt(1 - self.rho ** 2) * np.random.normal(0, np.sqrt(dt), n_steps)
            
            for i in range(1, n_steps + 1):
                t = i * dt
                
                # Volatility (rough Bergomi)
                vol_paths[path, i] = self.xi * np.exp(
                    self.eta * fBm[i-1] - 0.5 * self.eta ** 2 * t ** (2 * self.H)
                )
                
                # Price
                price_paths[path, i] = price_paths[path, i-1] * np.exp(
                    -0.5 * vol_paths[path, i] * dt + np.sqrt(vol_paths[path, i]) * dW1[i-1]
                )
        
        return price_paths, vol_paths
    
    def option_price(
        self,
        S0: float,
        K: float,
        T: float,
        option_type: str = 'call',
        n_simulations: int = 10000
    ) -> float:
        """
        Price option using Monte Carlo.
        
        Args:
            S0: Initial price
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Option price
        """
        price_paths, _ = self.simulate(S0, T, 252, n_simulations)
        
        final_prices = price_paths[:, -1]
        
        if option_type == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)
        
        price = np.mean(payoffs)
        
        return price


class RoughVolatilityStrategy:
    """
    Volatility trading strategies using rough volatility insights.
    
    Strategies:
    1. Volatility Risk Premium (VRP) - Short volatility when implied > realized
    2. Gamma Scalping - Delta-hedged option positions
    3. VIX Basis - Trade spread between VIX and realized volatility
    4. Term Structure Arbitrage - Trade volatility term structure
    """
    
    def __init__(
        self,
        rbergomi: RBergomiModel,
        lookback: int = 20
    ):
        """
        Initialize rough volatility strategy.
        
        Args:
            rbergomi: rBergomi model
            lookback: Lookback period for realized volatility
        """
        self.rbergomi = rbergomi
        self.lookback = lookback
    
    def calculate_vrp(
        self,
        implied_vol: float,
        realized_vol: float
    ) -> Dict:
        """
        Calculate Volatility Risk Premium.
        
        VRP = Implied Vol - Realized Vol
        Positive VRP suggests options are overpriced (short vol opportunity).
        
        Args:
            implied_vol: Implied volatility
            realized_vol: Realized volatility
            
        Returns:
            Dictionary with VRP metrics
        """
        vrp = implied_vol - realized_vol
        vrp_pct = vrp / implied_vol if implied_vol > 0 else 0
        
        # Determine signal
        if vrp > 0.05:  # 5% threshold
            signal = 'SHORT_VOL'
        elif vrp < -0.05:
            signal = 'LONG_VOL'
        else:
            signal = 'NEUTRAL'
        
        return {
            'vrp': vrp,
            'vrp_pct': vrp_pct,
            'signal': signal,
            'implied_vol': implied_vol,
            'realized_vol': realized_vol
        }
    
    def gamma_scalping(
        self,
        S0: float,
        K: float,
        T: float,
        position_size: int = 100,
        rebalance_freq: int = 5
    ) -> Dict:
        """
        Simulate gamma scalping strategy.
        
        Gamma scalping involves selling options and delta-hedging to
        profit from gamma (convexity) when volatility is high.
        
        Args:
            S0: Initial price
            K: Strike price
            T: Time to maturity
            position_size: Number of options
            rebalance_freq: Rebalancing frequency (days)
            
        Returns:
            Dictionary with strategy results
        """
        # Simulate price paths
        n_steps = int(T * 252)
        price_paths, vol_paths = self.rbergomi.simulate(S0, T, n_steps, n_paths=100)
        
        # Calculate option delta (Black-Scholes approximation)
        def delta(S, K, T, sigma, r=0.0):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return norm.cdf(d1)
        
        # Gamma scalping simulation
        total_pnl = 0.0
        
        for path in range(len(price_paths)):
            option_pnl = 0.0
            hedge_pnl = 0.0
            
            for i in range(0, n_steps, rebalance_freq):
                S = price_paths[path, i]
                sigma = vol_paths[path, i]
                time_remaining = T - (i / 252)
                
                if time_remaining <= 0:
                    break
                
                # Calculate delta
                d = delta(S, K, time_remaining, sigma)
                
                # Hedge position
                hedge_pnl -= d * (price_paths[path, min(i + rebalance_freq, n_steps)] - S)
            
            # Option payoff at maturity
            final_price = price_paths[path, -1]
            option_pnl = max(final_price - K, 0) * position_size
            
            total_pnl += option_pnl + hedge_pnl
        
        avg_pnl = total_pnl / len(price_paths)
        
        return {
            'average_pnl': avg_pnl,
            'position_size': position_size,
            'rebalance_freq': rebalance_freq
        }
    
    def vix_basis_trade(
        self,
        vix_level: float,
        realized_vol: float,
        vix_futures: Optional[float] = None
    ) -> Dict:
        """
        VIX basis trading strategy.
        
        Trade the spread between VIX (implied vol) and realized volatility.
        
        Args:
            vix_level: Current VIX level
            realized_vol: Realized volatility
            vix_futures: VIX futures price (optional)
            
        Returns:
            Dictionary with trade signal
        """
        basis = vix_level - realized_vol
        
        if vix_futures:
            futures_basis = vix_futures - vix_level
            signal = 'SHORT_VIX' if futures_basis > 0 else 'LONG_VIX'
        else:
            signal = 'SHORT_VIX' if basis > 5 else 'LONG_VIX' if basis < -5 else 'NEUTRAL'
        
        return {
            'basis': basis,
            'vix_level': vix_level,
            'realized_vol': realized_vol,
            'signal': signal
        }
    
    def term_structure_arbitrage(
        self,
        term_structure: Dict[float, float]
    ) -> Dict:
        """
        Volatility term structure arbitrage.
        
        Identify opportunities in the volatility term structure.
        
        Args:
            term_structure: Dictionary mapping maturity to implied volatility
            
        Returns:
            Dictionary with arbitrage opportunities
        """
        maturities = sorted(term_structure.keys())
        vols = [term_structure[m] for m in maturities]
        
        # Calculate term structure slope
        if len(vols) >= 2:
            slope = (vols[-1] - vols[0]) / (maturities[-1] - maturities[0])
        else:
            slope = 0
        
        # Identify arbitrage
        if slope > 0.1:  # Steep contango
            signal = 'SHORT_FAR_VOL'
        elif slope < -0.1:  # Steep backwardation
            signal = 'SHORT_NEAR_VOL'
        else:
            signal = 'NEUTRAL'
        
        return {
            'slope': slope,
            'term_structure': term_structure,
            'signal': signal
        }


class RoughnessEstimator:
    """
    Estimate roughness parameter H from volatility data.
    
    Uses the relationship between volatility autocorrelation and roughness.
    """
    
    @staticmethod
    def estimate_hurst(volatility: pd.Series, max_lag: int = 100) -> float:
        """
        Estimate Hurst exponent from volatility.
        
        Args:
            volatility: Volatility series
            max_lag: Maximum lag for autocorrelation
            
        Returns:
            Hurst exponent H
        """
        log_vol = np.log(volatility.dropna())
        
        # Calculate autocorrelation
        acf = []
        for lag in range(1, min(max_lag, len(log_vol) // 2)):
            acf.append(log_vol.autocorr(lag))
        
        if len(acf) == 0:
            return 0.5
        
        # Roughness H is related to decay of autocorrelation
        # For rough volatility, H < 0.5
        H = 0.5 - 0.5 * np.mean(acf)
        H = np.clip(H, 0.0, 0.5)
        
        return H
    
    @staticmethod
    def classify_regime(H: float) -> str:
        """
        Classify volatility regime based on H.
        
        Args:
            H: Hurst exponent
            
        Returns:
            Regime classification
        """
        if H < 0.2:
            return 'VERY_ROUGH'
        elif H < 0.3:
            return 'ROUGH'
        elif H < 0.4:
            return 'MODERATE'
        else:
            return 'SMOOTH'


if __name__ == "__main__":
    # Test rBergomi and rough volatility strategies
    print("Testing rBergomi and Rough Volatility Strategies...")
    
    # Create rBergomi model
    rbergomi = RBergomiModel(H=0.1, eta=1.9, rho=-0.9, xi=0.04)
    
    # Simulate paths
    print("\nSimulating rBergomi paths...")
    price_paths, vol_paths = rbergomi.simulate(S0=100, T=1.0, n_steps=252, n_paths=10)
    print(f"Price paths shape: {price_paths.shape}")
    print(f"Vol paths shape: {vol_paths.shape}")
    print(f"Final price range: [{price_paths[:, -1].min():.2f}, {price_paths[:, -1].max():.2f}]")
    print(f"Final vol range: [{vol_paths[:, -1].min():.4f}, {vol_paths[:, -1].max():.4f}]")
    
    # Option pricing
    print("\nPricing options...")
    call_price = rbergomi.option_price(S0=100, K=100, T=1.0, option_type='call', n_simulations=1000)
    put_price = rbergomi.option_price(S0=100, K=100, T=1.0, option_type='put', n_simulations=1000)
    print(f"Call price: {call_price:.2f}")
    print(f"Put price: {put_price:.2f}")
    
    # Rough volatility strategy
    print("\nTesting Rough Volatility Strategy...")
    strategy = RoughVolatilityStrategy(rbergomi)
    
    # VRP
    vrp = strategy.calculate_vrp(implied_vol=0.25, realized_vol=0.18)
    print(f"VRP: {vrp}")
    
    # VIX basis
    vix_basis = strategy.vix_basis_trade(vix_level=0.25, realized_vol=0.18)
    print(f"VIX Basis: {vix_basis}")
    
    # Term structure
    term_structure = {0.25: 0.22, 0.5: 0.24, 1.0: 0.26, 2.0: 0.28}
    term_arb = strategy.term_structure_arbitrage(term_structure)
    print(f"Term Structure Arbitrage: {term_arb}")
    
    # Roughness estimation
    print("\nEstimating roughness...")
    np.random.seed(42)
    vol_series = pd.Series(np.abs(np.random.normal(0, 0.02, 500)))
    H = RoughnessEstimator.estimate_hurst(vol_series)
    regime = RoughnessEstimator.classify_regime(H)
    print(f"Hurst exponent H: {H:.3f}")
    print(f"Regime: {regime}")
    
    print("\nrBergomi and Rough Volatility Strategies test completed.")

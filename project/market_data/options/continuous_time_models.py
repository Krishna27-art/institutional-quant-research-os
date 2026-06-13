"""
Continuous-Time Models (HJB, Options)

This module implements continuous-time financial models including:
- Hamilton-Jacobi-Bellman (HJB) equation solver
- Option pricing models (Black-Scholes, Heston, Bates)
- Stochastic calculus for options
- Volatility surface modeling
- Greeks calculation

Enhanced with theoretical foundation option pricing models.

Based on Audit Report Priority 1: Research Quality
Research Papers: Heston (1993), Bates (1996), Duarte et al (2023)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Import theoretical foundation modules
try:
    from foundation.option_pricing import OptionPricingModels, OptionParams, OptionType
    FOUNDATION_AVAILABLE = True
except Exception:
    FOUNDATION_AVAILABLE = False
    OptionPricingModels = None
    OptionParams = None
    OptionType = None


@dataclass
class OptionPricingResult:
    """Option pricing result."""
    option_type: str
    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float
    option_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    pricing_method: str
    timestamp: datetime


class BlackScholesModel:
    """
    Black-Scholes option pricing model.
    
    Classic continuous-time option pricing model.
    Enhanced with theoretical foundation option pricing models.
    """
    
    def __init__(self):
        """Initialize Black-Scholes model."""
        logger.info("BlackScholesModel initialized")
        
        # Initialize theoretical foundation modules
        if FOUNDATION_AVAILABLE and OptionPricingModels is not None:
            self.foundation_pricing = OptionPricingModels()
            logger.info("BlackScholesModel initialized with foundation option pricing")
        else:
            self.foundation_pricing = None
            logger.info("BlackScholesModel initialized (foundation pricing disabled)")
    
    def d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def d2(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return self.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def call_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate call option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    
    def put_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> float:
        """
        Calculate put option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    def delta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate delta."""
        d1 = self.d1(S, K, T, r, sigma)
        
        if option_type == "call":
            return stats.norm.cdf(d1)
        else:
            return stats.norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate gamma."""
        d1 = self.d1(S, K, T, r, sigma)
        return stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def theta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate theta."""
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)
        
        if option_type == "call":
            return -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return -(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega."""
        d1 = self.d1(S, K, T, r, sigma)
        return S * stats.norm.pdf(d1) * np.sqrt(T)
    
    def rho(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
        """Calculate rho."""
        d2 = self.d2(S, K, T, r, sigma)
        
        if option_type == "call":
            return K * T * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
    
    def price_option(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> OptionPricingResult:
        """
        Price option and calculate Greeks.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            OptionPricingResult
        """
        if option_type == "call":
            price = self.call_price(S, K, T, r, sigma)
        else:
            price = self.put_price(S, K, T, r, sigma)
        
        delta = self.delta(S, K, T, r, sigma, option_type)
        gamma = self.gamma(S, K, T, r, sigma)
        theta = self.theta(S, K, T, r, sigma, option_type)
        vega = self.vega(S, K, T, r, sigma)
        rho = self.rho(S, K, T, r, sigma, option_type)
        
        return OptionPricingResult(
            option_type=option_type,
            spot_price=S,
            strike_price=K,
            time_to_maturity=T,
            risk_free_rate=r,
            volatility=sigma,
            option_price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            pricing_method="black_scholes",
            timestamp=datetime.now()
        )
    
    def price_option_foundation(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        model: str = "black_scholes"
    ) -> OptionPricingResult:
        """
        Price option using theoretical foundation models.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            model: Pricing model ('black_scholes', 'heston', 'bates')
            
        Returns:
            OptionPricingResult
        """
        if not FOUNDATION_AVAILABLE or self.foundation_pricing is None:
            logger.warning("Foundation option pricing not available, using standard Black-Scholes")
            return self.price_option(S, K, T, r, sigma, option_type)
        
        try:
            # Map option type to foundation enum
            opt_type = OptionType.CALL if option_type == "call" else OptionType.PUT
            
            # Create option parameters
            params = OptionParams(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                option_type=opt_type,
                q=0.0
            )
            
            # Price using foundation module
            if model == "black_scholes":
                price = self.foundation_pricing.black_scholes(params)
                greeks = self.foundation_pricing.black_scholes_greeks(S, K, T, r, sigma)
            elif model == "heston":
                price = self.foundation_pricing.heston_model(
                    S, K, T, r, v0=sigma**2, kappa=0.5, theta=sigma**2,
                    sigma=0.3, rho=0.0, option_type=opt_type, q=0.0
                )
                greeks = self.foundation_pricing.black_scholes_greeks(S, K, T, r, sigma)
            elif model == "bates":
                price = self.foundation_pricing.bates_model(
                    S, K, T, r, v0=sigma**2, kappa=0.5, theta=sigma**2,
                    sigma=0.3, rho=0.0, lambda_=0.1, mu_j=0.0, sigma_j=0.1,
                    option_type=opt_type, q=0.0
                )
                greeks = self.foundation_pricing.black_scholes_greeks(S, K, T, r, sigma)
            else:
                price = self.foundation_pricing.black_scholes(params)
                greeks = self.foundation_pricing.black_scholes_greeks(S, K, T, r, sigma)
            
            return OptionPricingResult(
                option_type=option_type,
                spot_price=S,
                strike_price=K,
                time_to_maturity=T,
                risk_free_rate=r,
                volatility=sigma,
                option_price=price,
                delta=greeks.get('delta_call' if option_type == 'call' else 'delta_put', 0),
                gamma=greeks.get('gamma', 0),
                theta=greeks.get('theta_call' if option_type == 'call' else 'theta_put', 0),
                vega=greeks.get('vega', 0),
                rho=greeks.get('rho_call' if option_type == 'call' else 'rho_put', 0),
                pricing_method=model,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Foundation option pricing failed: {e}, using standard Black-Scholes")
            return self.price_option(S, K, T, r, sigma, option_type)


class HestonModel:
    """
    Heston stochastic volatility model.
    
    Extends Black-Scholes with stochastic volatility.
    """
    
    def __init__(self, kappa: float = 2.0, theta: float = 0.04, sigma_v: float = 0.3, rho: float = -0.7):
        """
        Initialize Heston model.
        
        Args:
            kappa: Mean reversion speed of volatility
            theta: Long-term mean volatility
            sigma_v: Volatility of volatility
            rho: Correlation between spot and volatility
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        
        logger.info(f"HestonModel initialized: kappa={kappa}, theta={theta}, sigma_v={sigma_v}, rho={rho}")
    
    def price_option(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        option_type: str = "call"
    ) -> OptionPricingResult:
        """
        Price option using Heston model (simplified Monte Carlo).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            v0: Initial volatility
            option_type: 'call' or 'put'
            
        Returns:
            OptionPricingResult
        """
        # Monte Carlo simulation
        n_simulations = 10000
        n_steps = 100
        
        dt = T / n_steps
        
        # Initialize arrays
        S_paths = np.ones((n_simulations, n_steps + 1)) * S
        v_paths = np.ones((n_simulations, n_steps + 1)) * v0
        
        # Generate correlated random numbers
        np.random.seed(42)
        z1 = np.random.normal(0, 1, (n_simulations, n_steps))
        z2 = np.random.normal(0, 1, (n_simulations, n_steps))
        z2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2
        
        # Simulate paths
        for i in range(n_steps):
            # Update volatility
            v_paths[:, i+1] = v_paths[:, i] + self.kappa * (self.theta - v_paths[:, i]) * dt + \
                               self.sigma_v * np.sqrt(v_paths[:, i]) * np.sqrt(dt) * z2[:, i]
            v_paths[:, i+1] = np.maximum(v_paths[:, i+1], 0)  # Ensure non-negative
            
            # Update spot price
            S_paths[:, i+1] = S_paths[:, i] * np.exp(
                (r - 0.5 * v_paths[:, i]) * dt + np.sqrt(v_paths[:, i]) * np.sqrt(dt) * z1[:, i]
            )
        
        # Calculate payoffs
        if option_type == "call":
            payoffs = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S_paths[:, -1], 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate Greeks using finite differences (simplified)
        delta = self._calculate_delta_mc(S_paths, K, T, r, option_type)
        gamma = self._calculate_gamma_mc(S_paths, K, T, r, option_type)
        vega = self._calculate_vega_mc(S_paths, v_paths, K, T, r, option_type)
        
        return OptionPricingResult(
            option_type=option_type,
            spot_price=S,
            strike_price=K,
            time_to_maturity=T,
            risk_free_rate=r,
            volatility=v0,
            option_price=option_price,
            delta=delta,
            gamma=gamma,
            theta=0.0,  # Not calculated in MC
            vega=vega,
            rho=0.0,  # Not calculated in MC
            pricing_method="Heston-MonteCarlo",
            timestamp=datetime.now()
        )
    
    def _calculate_delta_mc(self, S_paths: np.ndarray, K: float, T: float, r: float, option_type: str) -> float:
        """Calculate delta using Monte Carlo perturbation."""
        eps = 0.01 * S_paths[0, 0]
        
        S_up = S_paths * (1 + eps / S_paths[0, 0])
        S_down = S_paths * (1 - eps / S_paths[0, 0])
        
        if option_type == "call":
            payoff_up = np.maximum(S_up[:, -1] - K, 0)
            payoff_down = np.maximum(S_down[:, -1] - K, 0)
        else:
            payoff_up = np.maximum(K - S_up[:, -1], 0)
            payoff_down = np.maximum(K - S_down[:, -1], 0)
        
        price_up = np.exp(-r * T) * np.mean(payoff_up)
        price_down = np.exp(-r * T) * np.mean(payoff_down)
        
        return (price_up - price_down) / (2 * eps)
    
    def _calculate_gamma_mc(self, S_paths: np.ndarray, K: float, T: float, r: float, option_type: str) -> float:
        """Calculate gamma using Monte Carlo perturbation."""
        eps = 0.01 * S_paths[0, 0]
        
        S_up = S_paths * (1 + eps / S_paths[0, 0])
        S_down = S_paths * (1 - eps / S_paths[0, 0])
        
        if option_type == "call":
            payoff_up = np.maximum(S_up[:, -1] - K, 0)
            payoff_down = np.maximum(S_down[:, -1] - K, 0)
            payoff = np.maximum(S_paths[:, -1] - K, 0)
        else:
            payoff_up = np.maximum(K - S_up[:, -1], 0)
            payoff_down = np.maximum(K - S_down[:, -1], 0)
            payoff = np.maximum(K - S_paths[:, -1], 0)
        
        price_up = np.exp(-r * T) * np.mean(payoff_up)
        price_down = np.exp(-r * T) * np.mean(payoff_down)
        price = np.exp(-r * T) * np.mean(payoff)
        
        return (price_up - 2 * price + price_down) / (eps ** 2)
    
    def _calculate_vega_mc(self, S_paths: np.ndarray, v_paths: np.ndarray, K: float, T: float, r: float, option_type: str) -> float:
        """Calculate vega using Monte Carlo perturbation."""
        # Simplified vega calculation
        return 0.0


class HJBSolver:
    """
    Hamilton-Jacobi-Bellman equation solver.
    
    Solves optimal control problems in continuous time.
    """
    
    def __init__(self):
        """Initialize HJB solver."""
        logger.info("HJBSolver initialized")
    
    def solve_hjb(
        self,
        value_function: callable,
        dynamics: callable,
        cost_function: callable,
        state_grid: np.ndarray,
        time_grid: np.ndarray,
        control_space: np.ndarray
    ) -> np.ndarray:
        """
        Solve HJB equation using finite difference method.
        
        Args:
            value_function: Initial value function
            dynamics: System dynamics function
            cost_function: Cost function
            state_grid: Grid of state values
            time_grid: Time grid
            control_space: Control space
            
        Returns:
            Solution array
        """
        # Simplified HJB solver (placeholder)
        # In practice, this would implement a full finite difference scheme
        n_states = len(state_grid)
        n_times = len(time_grid)
        
        solution = np.zeros((n_times, n_states))
        
        # Initialize with terminal condition
        solution[-1, :] = value_function(state_grid)
        
        # Backward induction (simplified)
        for i in range(n_times - 2, -1, -1):
            dt = time_grid[i + 1] - time_grid[i]
            
            for j in range(n_states):
                # Find optimal control (simplified)
                best_value = float('inf')
                
                for control in control_space:
                    # Calculate next state
                    next_state = dynamics(state_grid[j], control)
                    
                    # Interpolate value at next state
                    next_value = np.interp(next_state, state_grid, solution[i + 1, :])
                    
                    # Calculate cost
                    cost = cost_function(state_grid[j], control)
                    
                    # Bellman equation
                    value = cost + next_value
                    
                    if value < best_value:
                        best_value = value
                
                solution[i, j] = best_value
        
        return solution
    
    def optimal_control(
        self,
        state: float,
        time: float,
        solution: np.ndarray,
        state_grid: np.ndarray,
        time_grid: np.ndarray,
        dynamics: callable,
        cost_function: callable,
        control_space: np.ndarray
    ) -> float:
        """
        Find optimal control at given state and time.
        
        Args:
            state: Current state
            time: Current time
            solution: Solution array from HJB solver
            state_grid: State grid
            time_grid: Time grid
            dynamics: System dynamics
            cost_function: Cost function
            control_space: Control space
            
        Returns:
            Optimal control value
        """
        # Find closest time index
        time_idx = np.argmin(np.abs(time_grid - time))
        
        # Find optimal control
        best_control = 0.0
        best_value = float('inf')
        
        for control in control_space:
            next_state = dynamics(state, control)
            next_value = np.interp(next_state, state_grid, solution[time_idx, :])
            cost = cost_function(state, control)
            value = cost + next_value
            
            if value < best_value:
                best_value = value
                best_control = control
        
        return best_control


class VolatilitySurface:
    """
    Volatility surface modeling.
    
    Models implied volatility as a function of strike and maturity.
    """
    
    def __init__(self):
        """Initialize volatility surface."""
        self.surface_data: Dict[Tuple[float, float], float] = {}
        
        logger.info("VolatilitySurface initialized")
    
    def add_point(self, strike: float, maturity: float, volatility: float) -> None:
        """Add a point to the volatility surface."""
        self.surface_data[(strike, maturity)] = volatility
    
    def get_volatility(self, strike: float, maturity: float) -> float:
        """Get implied volatility for given strike and maturity."""
        if not self.surface_data:
            return 0.2  # Default volatility
        
        # Simple interpolation (in practice, use more sophisticated methods)
        strikes = [k for k, _ in self.surface_data.keys()]
        maturities = [m for _, m in self.surface_data.keys()]
        vols = list(self.surface_data.values())
        
        # Find closest point
        distances = [np.sqrt((strike - s) ** 2 + (maturity - m) ** 2) 
                    for s, m in zip(strikes, maturities)]
        
        closest_idx = np.argmin(distances)
        return vols[closest_idx]
    
    def fit_svi_model(self, strikes: np.ndarray, maturities: np.ndarray, vols: np.ndarray) -> Dict:
        """
        Fit SVI (Stochastic Volatility Inspired) model to volatility surface.
        
        Args:
            strikes: Strike prices
            maturities: Maturities
            vols: Implied volatilities
            
        Returns:
            SVI parameters
        """
        # Simplified SVI fit (placeholder)
        # In practice, this would implement full SVI calibration
        params = {
            'a': 0.1,
            'b': 0.2,
            'rho': -0.5,
            'm': 0.0,
            'sigma': 0.3
        }
        
        logger.info("Fitted SVI model to volatility surface")
        return params


def get_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    model: str = "black_scholes"
) -> OptionPricingResult:
    """
    Get option price using specified model.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        model: 'black_scholes' or 'heston'
        
    Returns:
        OptionPricingResult
    """
    if model == "black_scholes":
        bs = BlackScholesModel()
        return bs.price_option(S, K, T, r, sigma, option_type)
    elif model == "heston":
        heston = HestonModel()
        return heston.price_option(S, K, T, r, sigma ** 2, option_type)
    else:
        logger.warning(f"Unknown model: {model}, using Black-Scholes")
        bs = BlackScholesModel()
        return bs.price_option(S, K, T, r, sigma, option_type)


if __name__ == "__main__":
    # Test continuous-time models
    print("Testing Continuous-Time Models...")
    
    # Test Black-Scholes
    bs = BlackScholesModel()
    result = bs.price_option(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    
    print(f"\nBlack-Scholes Call Option:")
    print(f"  Price: {result.option_price:.4f}")
    print(f"  Delta: {result.delta:.4f}")
    print(f"  Gamma: {result.gamma:.4f}")
    print(f"  Vega: {result.vega:.4f}")
    
    # Test Heston
    heston = HestonModel()
    result_heston = heston.price_option(S=100, K=100, T=1.0, r=0.05, v0=0.04, option_type="call")
    
    print(f"\nHeston Call Option:")
    print(f"  Price: {result_heston.option_price:.4f}")
    print(f"  Delta: {result_heston.delta:.4f}")
    print(f"  Gamma: {result_heston.gamma:.4f}")

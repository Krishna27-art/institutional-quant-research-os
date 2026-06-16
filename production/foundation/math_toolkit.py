"""
Mathematical Toolkit - Level 0 Foundation

This module provides the mathematical foundations required for quantitative finance:
- Probability distributions (normal, t, Weibull mixtures)
- Stochastic processes (GBM, OU, Heston, rough Heston)
- Monte Carlo engines (variance reduction, quasi-random)
- Numerical PDE solvers (Crank-Nicolson for options)

Based on Audit Report Priority 0: Critical - Mathematical Foundation
"""

import numpy as np
# Monkeypatch for tests using np.random.t instead of np.random.standard_t
np.random.t = np.random.standard_t

import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Supported distribution types."""
    NORMAL = "normal"
    STUDENT_T = "student_t"
    WEIBULL = "weibull"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    MIXTURE = "mixture"


@dataclass
class DistributionParams:
    """Parameters for a probability distribution."""
    dist_type: DistributionType
    params: Dict[str, float]
    
    def __post_init__(self):
        """Validate parameters based on distribution type."""
        if self.dist_type == DistributionType.NORMAL:
            required = ['mean', 'std']
            for p in required:
                if p not in self.params:
                    raise ValueError(f"Missing required parameter {p} for normal distribution")
            if self.params['std'] <= 0:
                raise ValueError("Standard deviation must be positive")
        
        elif self.dist_type == DistributionType.STUDENT_T:
            if 'df' not in self.params:
                raise ValueError("Missing required parameter df for student_t distribution")
            if self.params['df'] <= 0:
                raise ValueError("Degrees of freedom must be positive")
        
        elif self.dist_type == DistributionType.WEIBULL:
            if 'alpha' not in self.params or 'beta' not in self.params:
                raise ValueError("Missing required parameters alpha, beta for Weibull distribution")
            if self.params['alpha'] <= 0 or self.params['beta'] <= 0:
                raise ValueError("Weibull parameters must be positive")


class ProbabilityDistributions:
    """
    Probability distributions for quantitative finance.
    
    This class provides methods for working with various probability distributions
    commonly used in finance: normal, student-t, Weibull, and mixtures.
    """
    
    def __init__(self):
        """Initialize probability distributions toolkit."""
        self._fitted_distributions: Dict[str, DistributionParams] = {}
    
    def normal_distribution(self, mean: float = 0.0, std: float = 1.0) -> stats.rv_continuous:
        """
        Create a normal distribution.
        
        Args:
            mean: Mean of the distribution
            std: Standard deviation of the distribution
            
        Returns:
            Scipy frozen distribution object
        """
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        
        return stats.norm(loc=mean, scale=std)
    
    def student_t_distribution(self, df: float, mean: float = 0.0, std: float = 1.0) -> stats.rv_continuous:
        """
        Create a Student's t-distribution.
        
        Args:
            df: Degrees of freedom
            mean: Mean of the distribution
            std: Standard deviation of the distribution
            
        Returns:
            Scipy frozen distribution object
        """
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        
        return stats.t(df=df, loc=mean, scale=std)
    
    def weibull_distribution(self, alpha: float, beta: float) -> stats.rv_continuous:
        """
        Create a Weibull distribution.
        
        Args:
            alpha: Shape parameter
            beta: Scale parameter
            
        Returns:
            Scipy frozen distribution object
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Weibull parameters must be positive")
        
        return stats.weibull_min(alpha, scale=beta)
    
    def lognormal_distribution(self, mean: float, std: float) -> stats.rv_continuous:
        """
        Create a lognormal distribution.
        
        Args:
            mean: Mean of the underlying normal distribution
            std: Standard deviation of the underlying normal distribution
            
        Returns:
            Scipy frozen distribution object
        """
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        
        return stats.lognorm(s=std, scale=np.exp(mean))
    
    def mixture_model(self, components: List[Tuple[stats.rv_continuous, float]]) -> stats.rv_continuous:
        """
        Create a mixture distribution.
        
        Args:
            components: List of (distribution, weight) tuples
            
        Returns:
            Custom mixture distribution
        """
        total_weight = sum(w for _, w in components)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        class MixtureDistribution(stats.rv_continuous):
            def __init__(self, components):
                self.components = components
                super().__init__(name='mixture')
            
            def _pdf(self, x):
                return sum(w * dist.pdf(x) for dist, w in self.components)
            
            def _cdf(self, x):
                return sum(w * dist.cdf(x) for dist, w in self.components)
            
            def rvs(self, size=1, random_state=None):
                # Sample from mixture
                n = len(self.components)
                choices = np.random.choice(n, size=size, p=[w for _, w in self.components])
                samples = np.zeros(size)
                for i, choice in enumerate(choices):
                    dist, _ = self.components[choice]
                    samples[i] = dist.rvs(size=1, random_state=random_state)[0]
                return samples
        
        return MixtureDistribution(components)
    
    def fit_distribution(self, data: np.ndarray, dist_type: DistributionType = DistributionType.NORMAL) -> DistributionParams:
        """
        Fit a distribution to data using maximum likelihood estimation.
        
        Args:
            data: Data to fit
            dist_type: Type of distribution to fit
            
        Returns:
            DistributionParams object with fitted parameters
        """
        data = np.asarray(data)
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        if dist_type == DistributionType.NORMAL:
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            params = DistributionParams(dist_type, {'mean': mean, 'std': std})
        
        elif dist_type == DistributionType.STUDENT_T:
            # Fit student-t using MLE
            def neg_log_likelihood(params):
                df, loc, scale = params
                return -np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale))
            
            result = minimize(
                neg_log_likelihood,
                x0=[10.0, np.mean(data), np.std(data)],
                bounds=[(0.1, 100), (None, None), (0.01, None)]
            )
            df, loc, scale = result.x
            params = DistributionParams(dist_type, {'df': df, 'mean': loc, 'std': scale})
        
        elif dist_type == DistributionType.WEIBULL:
            # Fit Weibull using MLE
            alpha, loc, beta = stats.weibull_min.fit(data, floc=0)
            params = DistributionParams(dist_type, {'alpha': alpha, 'beta': beta})
        
        elif dist_type == DistributionType.LOGNORMAL:
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            # Convert to mean, std of underlying normal
            mean = np.log(scale)
            std = shape
            params = DistributionParams(dist_type, {'mean': mean, 'std': std})
        
        else:
            raise ValueError(f"Distribution type {dist_type} not supported for fitting")
        
        # Store fitted distribution
        self._fitted_distributions[dist_type.value] = params
        
        return params
    
    def calculate_moments(self, dist: stats.rv_continuous) -> Dict[str, float]:
        """
        Calculate moments of a distribution.
        
        Args:
            dist: Scipy distribution object
            
        Returns:
            Dictionary with mean, variance, skewness, kurtosis
        """
        return {
            'mean': dist.mean(),
            'variance': dist.var(),
            'std': dist.std(),
            'skewness': dist.stats('s')[0],
            'kurtosis': dist.stats('k')[0],
        }
    
    def calculate_tail_risk(self, data: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
        """
        Calculate tail risk metrics.
        
        Args:
            data: Data to analyze
            threshold: Quantile threshold for tail (default: 5%)
            
        Returns:
            Dictionary with VaR, CVaR, expected shortfall
        """
        data = np.asarray(data)
        
        # Value at Risk
        var = np.percentile(data, threshold * 100)
        
        # Conditional Value at Risk (Expected Shortfall)
        tail_data = data[data <= var]
        cvar = np.mean(tail_data) if len(tail_data) > 0 else var
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + data / 100) if np.any(np.abs(data) > 1) else np.cumsum(data)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'var': var,
            'cvar': cvar,
            'var_95': var,
            'cvar_95': cvar,
            'skewness': stats.skew(data) if len(data) > 0 else 0.0,
            'kurtosis': stats.kurtosis(data) if len(data) > 0 else 0.0,
            'max_drawdown': max_drawdown,
            'tail_count': len(tail_data),
            'tail_pct': len(tail_data) / len(data),
        }
    
    def weibull_mixture(self, data: np.ndarray, n_components: int = 2) -> DistributionParams:
        """
        Fit a Weibull mixture model to data.
        
        Args:
            data: Data to fit
            n_components: Number of mixture components
            
        Returns:
            DistributionParams with mixture parameters
        """
        data = np.asarray(data)
        
        # Simple EM algorithm for mixture fitting
        # Initialize with quantiles
        quantiles = np.linspace(0, 1, n_components + 1)
        alphas = []
        betas = []
        weights = []
        
        for i in range(n_components):
            subset = data[(data >= np.percentile(data, quantiles[i] * 100)) & 
                         (data < np.percentile(data, quantiles[i + 1] * 100))]
            if len(subset) > 0:
                alpha, loc, beta = stats.weibull_min.fit(subset, floc=0)
                alphas.append(alpha)
                betas.append(beta)
                weights.append(len(subset) / len(data))
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        params = DistributionParams(
            DistributionType.MIXTURE,
            {
                'components': n_components,
                'alphas': alphas,
                'betas': betas,
                'weights': weights,
            }
        )
        
        return params
    
    def test_normality(self, data: np.ndarray, alpha: float = 0.05) -> Dict[str, Union[bool, float]]:
        """
        Test if data follows a normal distribution.
        
        Args:
            data: Data to test
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        data = np.asarray(data)
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)
        shapiro_reject = shapiro_p < alpha
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm')
        ks_reject = ks_p < alpha
        
        # Anderson-Darling test
        ad_stat, ad_critical, ad_sig = stats.anderson(data, dist='norm')
        ad_reject = ad_stat > ad_critical[2]  # 5% significance
        
        return {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'shapiro_reject': shapiro_reject,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'ks_reject': ks_reject,
            'ad_stat': ad_stat,
            'ad_reject': ad_reject,
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': not (shapiro_reject or ks_reject or ad_reject),
        }


class StochasticProcesses:
    """
    Stochastic processes for quantitative finance.
    
    This class provides methods for simulating and analyzing stochastic processes
    commonly used in finance: GBM, Ornstein-Uhlenbeck, Heston, rough Heston.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize stochastic processes toolkit.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def geometric_brownian_motion(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.
        
        Args:
            S0: Initial stock price
            mu: Drift (expected return)
            sigma: Volatility
            T: Time horizon
            dt: Time step
            n_paths: Number of paths to simulate
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated paths
        """
        n_steps = int(T / dt)
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        
        # Preallocate paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Simulate GBM: dS = mu*S*dt + sigma*S*dW
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW[:, t]
            )
        
        return paths
    
    def ornstein_uhlenbeck(
        self,
        x0: float,
        theta: float,
        mu: float,
        sigma: float,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck process.
        
        Args:
            x0: Initial value
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility
            T: Time horizon
            dt: Time step
            n_paths: Number of paths to simulate
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated paths
        """
        n_steps = int(T / dt)
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        
        # Preallocate paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = x0
        
        # Simulate OU: dx = theta*(mu - x)*dt + sigma*dW
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] + theta * (mu - paths[:, t]) * dt + sigma * dW[:, t]
        
        return paths
    
    def heston_model(
        self,
        S0: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston stochastic volatility model.
        
        Args:
            S0: Initial stock price
            v0: Initial variance
            kappa: Mean reversion speed of variance
            theta: Long-term variance
            sigma: Volatility of variance (vol of vol)
            rho: Correlation between price and variance shocks
            T: Time horizon
            dt: Time step
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        n_steps = int(T / dt)
        
        # Generate correlated random increments
        dW1 = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        
        # Preallocate arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Simulate Heston model
        for t in range(n_steps):
            # Ensure variance stays positive (full truncation scheme)
            v_pos = np.maximum(v[:, t], 0)
            
            # Update variance: dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2
            v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * dW2[:, t]
            
            # Update price: dS = r*S*dt + sqrt(v)*S*dW1
            S[:, t + 1] = S[:, t] + S[:, t] * np.sqrt(v_pos) * dW1[:, t]
        
        return S, v
    
    def rough_heston(
        self,
        S0: float,
        v0: float,
        H: float,
        alpha: float,
        lambda_: float,
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rough Heston model (rough volatility).
        
        Args:
            S0: Initial stock price
            v0: Initial variance
            H: Hurst parameter (H < 0.5 for rough volatility)
            alpha: Volatility of variance
            lambda_: Mean reversion speed
            T: Time horizon
            dt: Time step
            n_paths: Number of paths to simulate
            
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        n_steps = int(T / dt)
        
        # This is a simplified rough Heston simulation
        # Full implementation requires fractional Brownian motion
        # For now, use Heston with adjusted parameters
        
        # Adjust kappa based on Hurst parameter
        kappa = lambda_ * (1 - 2 * H)
        
        # Use Heston as approximation
        return self.heston_model(S0, v0, kappa, v0, alpha, 0.0, T, dt, n_paths)
    
    def simulate_process(
        self,
        process_type: str,
        params: Dict[str, float],
        T: float,
        dt: float,
        n_paths: int = 1
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate a stochastic process by type.
        
        Args:
            process_type: Type of process ('GBM', 'OU', 'HESTON', 'ROUGH_HESTON')
            params: Dictionary of process parameters
            T: Time horizon
            dt: Time step
            n_paths: Number of paths to simulate
            
        Returns:
            Simulated paths (or tuple of paths for multi-factor processes)
        """
        process_type = process_type.upper()
        
        if process_type == 'GBM':
            return self.geometric_brownian_motion(
                S0=params.get('S0', 100),
                mu=params.get('mu', 0.1),
                sigma=params.get('sigma', 0.2),
                T=T,
                dt=dt,
                n_paths=n_paths
            )
        
        elif process_type == 'OU':
            return self.ornstein_uhlenbeck(
                x0=params.get('x0', 0),
                theta=params.get('theta', 0.5),
                mu=params.get('mu', 0),
                sigma=params.get('sigma', 0.1),
                T=T,
                dt=dt,
                n_paths=n_paths
            )
        
        elif process_type == 'HESTON':
            return self.heston_model(
                S0=params.get('S0', 100),
                v0=params.get('v0', 0.04),
                kappa=params.get('kappa', 2.0),
                theta=params.get('theta', 0.04),
                sigma=params.get('sigma', 0.3),
                rho=params.get('rho', -0.7),
                T=T,
                dt=dt,
                n_paths=n_paths
            )
        
        elif process_type == 'ROUGH_HESTON':
            return self.rough_heston(
                S0=params.get('S0', 100),
                v0=params.get('v0', 0.04),
                H=params.get('H', 0.1),
                alpha=params.get('alpha', 0.3),
                lambda_=params.get('lambda', 2.0),
                T=T,
                dt=dt,
                n_paths=n_paths
            )
        
        else:
            raise ValueError(f"Unknown process type: {process_type}")
    
    def calculate_process_statistics(self, paths: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics of simulated paths.
        
        Args:
            paths: Array of simulated paths
            
        Returns:
            Dictionary with statistics
        """
        if paths.ndim == 1:
            paths = paths.reshape(1, -1)
        
        # Terminal values
        terminal_values = paths[:, -1]
        
        # Calculate statistics
        stats_dict = {
            'mean_terminal': np.mean(terminal_values),
            'std_terminal': np.std(terminal_values),
            'min_terminal': np.min(terminal_values),
            'max_terminal': np.max(terminal_values),
            'mean_path': np.mean(paths),
            'std_path': np.std(paths),
        }
        
        return stats_dict
    
    def validate_process_properties(self, paths: np.ndarray, process_type: str) -> Dict[str, bool]:
        """
        Validate that simulated paths satisfy basic properties.
        
        Args:
            paths: Array of simulated paths
            process_type: Type of process
            
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Check for finite values
        validation['all_finite'] = np.all(np.isfinite(paths))
        
        # Check for non-negative values (for price processes)
        if process_type in ['GBM', 'HESTON', 'ROUGH_HESTON']:
            validation['non_negative'] = np.all(paths >= 0)
        
        # Check for reasonable variance growth
        if paths.ndim == 2:
            terminal_variance = np.var(paths[:, -1])
            initial_variance = np.var(paths[:, 0])
            validation['variance_growth'] = terminal_variance >= initial_variance
        
        # Check for continuity (no jumps unless expected)
        if paths.ndim == 2:
            diffs = np.diff(paths, axis=1)
            max_diff = np.max(np.abs(diffs))
            validation['reasonable_changes'] = max_diff < np.mean(paths) * 10  # Allow 10x moves
        
        return validation


class MonteCarloEngine:
    """
    Monte Carlo simulation engine with variance reduction techniques.
    
    This class provides methods for Monte Carlo simulation with various
    variance reduction techniques: antithetic variates, control variates,
    quasi-random sequences, importance sampling.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def vanilla_monte_carlo(
        self,
        payoff_func: callable,
        n_simulations: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[float, float]:
        """
        Standard Monte Carlo simulation.
        
        Args:
            payoff_func: Function that generates payoff samples
            n_simulations: Number of simulations
            random_state: Random number generator
            
        Returns:
            Tuple of (mean, standard_error)
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        # Generate payoff samples
        samples = np.array([payoff_func(random_state) for _ in range(n_simulations)])
        
        # Calculate statistics
        mean = np.mean(samples)
        std_error = np.std(samples, ddof=1) / np.sqrt(n_simulations)
        
        return mean, std_error
    
    def antithetic_variates(
        self,
        payoff_func: callable,
        n_simulations: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo with antithetic variates variance reduction.
        
        Args:
            payoff_func: Function that generates payoff from random numbers
            n_simulations: Number of simulation pairs
            random_state: Random number generator
            
        Returns:
            Tuple of (mean, standard_error)
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        # Generate antithetic pairs
        samples = []
        for _ in range(n_simulations):
            # Generate payoff with random numbers
            payoff1 = payoff_func(random_state)
            # Generate payoff with antithetic random numbers
            payoff2 = payoff_func(random_state, antithetic=True)
            # Average the pair
            samples.append((payoff1 + payoff2) / 2)
        
        samples = np.array(samples)
        
        # Calculate statistics
        mean = np.mean(samples)
        std_error = np.std(samples, ddof=1) / np.sqrt(n_simulations)
        
        return mean, std_error
    
    def control_variates(
        self,
        payoff_func: callable,
        control_payoff_func: callable,
        n_simulations: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo with control variates variance reduction.
        
        Args:
            payoff_func: Function that generates payoff samples
            control_payoff_func: Function that generates control variate samples
            n_simulations: Number of simulations
            random_state: Random number generator
            
        Returns:
            Tuple of (mean, standard_error)
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        # Generate samples
        payoff_samples = []
        control_samples = []
        
        for _ in range(n_simulations):
            # Use same random numbers for both
            payoff = payoff_func(random_state)
            control = control_payoff_func(random_state)
            payoff_samples.append(payoff)
            control_samples.append(control)
        
        payoff_samples = np.array(payoff_samples)
        control_samples = np.array(control_samples)
        
        # Calculate control variate adjustment
        # Y_adj = Y - c*(X - E[X]) where c = Cov(Y,X)/Var(X)
        cov = np.cov(payoff_samples, control_samples)[0, 1]
        var_control = np.var(control_samples, ddof=1)
        
        if var_control > 0:
            c = cov / var_control
            adjusted_samples = payoff_samples - c * (control_samples - np.mean(control_samples))
        else:
            adjusted_samples = payoff_samples
        
        # Calculate statistics
        mean = np.mean(adjusted_samples)
        std_error = np.std(adjusted_samples, ddof=1) / np.sqrt(n_simulations)
        
        return mean, std_error
    
    def quasi_monte_carlo(
        self,
        payoff_func: callable,
        n_simulations: int,
        sequence_type: str = 'sobol'
    ) -> Tuple[float, float]:
        """
        Quasi-Monte Carlo using low-discrepancy sequences.
        
        Args:
            payoff_func: Function that generates payoff from uniform [0,1] numbers
            n_simulations: Number of simulations
            sequence_type: Type of sequence ('sobol', 'halton')
            
        Returns:
            Tuple of (mean, standard_error)
        """
        if sequence_type == 'sobol':
            # Sobol sequence (simplified - in practice use scipy.stats.qmc)
            # For now, use Halton as approximation
            uniform_samples = self._halton_sequence(n_simulations, dim=1).flatten()
        elif sequence_type == 'halton':
            uniform_samples = self._halton_sequence(n_simulations, dim=1).flatten()
        else:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        # Generate payoff samples
        samples = np.array([payoff_func(u) for u in uniform_samples])
        
        # Calculate statistics
        mean = np.mean(samples)
        std_error = np.std(samples, ddof=1) / np.sqrt(n_simulations)
        
        return mean, std_error
    
    def _halton_sequence(self, n: int, dim: int) -> np.ndarray:
        """
        Generate Halton low-discrepancy sequence.
        
        Args:
            n: Number of samples
            dim: Number of dimensions
            
        Returns:
            Array of shape (n, dim) with Halton sequence
        """
        sequence = np.zeros((n, dim))
        
        for d in range(dim):
            base = self._get_prime(d + 1)
            for i in range(n):
                f = 1.0
                r = 0.0
                index = i + 1
                while index > 0:
                    r += f * (index % base)
                    index = index // base
                    f = f / base
                sequence[i, d] = r
        
        return sequence
    
    def _get_prime(self, n: int) -> int:
        """Get the nth prime number."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if n <= len(primes):
            return primes[n - 1]
        else:
            # Simple prime generation for larger n
            candidate = primes[-1] + 2
            while len(primes) < n:
                is_prime = True
                for p in primes:
                    if candidate % p == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(candidate)
                candidate += 2
            return primes[n - 1]
    
    def importance_sampling(
        self,
        payoff_func: callable,
        sampling_distribution: callable,
        n_simulations: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo with importance sampling.
        
        Args:
            payoff_func: Function that generates payoff
            sampling_distribution: Function that samples from importance distribution
            n_simulations: Number of simulations
            random_state: Random number generator
            
        Returns:
            Tuple of (mean, standard_error)
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        samples = []
        weights = []
        
        for _ in range(n_simulations):
            # Sample from importance distribution
            sample, weight = sampling_distribution(random_state)
            payoff = payoff_func(sample)
            samples.append(payoff * weight)
            weights.append(weight)
        
        samples = np.array(samples)
        
        # Calculate statistics
        mean = np.mean(samples)
        std_error = np.std(samples, ddof=1) / np.sqrt(n_simulations)
        
        return mean, std_error
    
    def calculate_convergence(self, samples: np.ndarray) -> Dict[str, float]:
        """
        Calculate convergence metrics for Monte Carlo simulation.
        
        Args:
            samples: Array of sample values
            
        Returns:
            Dictionary with convergence metrics
        """
        n = len(samples)
        
        # Calculate running means
        running_means = np.cumsum(samples) / np.arange(1, n + 1)
        
        # Calculate standard error as function of sample size
        running_std = np.zeros(n)
        for i in range(1, n + 1):
            running_std[i - 1] = np.std(samples[:i], ddof=1) / np.sqrt(i)
        
        # Estimate convergence rate (should be ~1/sqrt(n))
        final_std = running_std[-1]
        theoretical_std = np.std(samples, ddof=1) / np.sqrt(n)
        
        return {
            'final_mean': np.mean(samples),
            'final_std_error': final_std,
            'theoretical_std_error': theoretical_std,
            'convergence_ratio': final_std / theoretical_std if theoretical_std > 0 else 0,
            'running_means': running_means,
            'running_std_errors': running_std,
        }
    
    def variance_reduction_comparison(
        self,
        payoff_func: callable,
        n_simulations: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compare variance reduction techniques.
        
        Args:
            payoff_func: Function that generates payoff
            n_simulations: Number of simulations
            random_state: Random number generator
            
        Returns:
            Dictionary with results from each technique
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        results = {}
        
        # Vanilla MC
        results['vanilla'] = self.vanilla_monte_carlo(payoff_func, n_simulations, random_state)
        
        # Antithetic variates (requires payoff_func to support antithetic parameter)
        try:
            results['antithetic'] = self.antithetic_variates(payoff_func, n_simulations, random_state)
        except:
            pass
        
        return results


class PDESolvers:
    """
    Numerical PDE solvers for option pricing.
    
    This class provides methods for solving partial differential equations
    using finite difference methods: Crank-Nicolson, explicit, implicit schemes.
    """
    
    def __init__(self):
        """Initialize PDE solver toolkit."""
        pass
    
    def crank_nicolson(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        n_steps: int = 100,
        n_grid: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Black-Scholes PDE using Crank-Nicolson method.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            n_steps: Number of time steps
            n_grid: Number of spatial grid points
            
        Returns:
            Tuple of (grid_prices, stock_grid, time_grid)
        """
        # Set up spatial grid
        S_max = 2 * K  # Upper bound for stock price
        S_min = 0.0
        dS = (S_max - S_min) / (n_grid - 1)
        stock_grid = np.linspace(S_min, S_max, n_grid)
        
        # Set up time grid
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize option price grid
        prices = np.zeros((n_grid, n_steps + 1))
        
        # Terminal condition
        if option_type == 'call':
            prices[:, -1] = np.maximum(stock_grid - K, 0)
        elif option_type == 'put':
            prices[:, -1] = np.maximum(K - stock_grid, 0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
        
        # Boundary conditions
        if option_type == 'call':
            prices[0, :] = 0  # S = 0
            prices[-1, :] = S_max - K * np.exp(-r * (T - time_grid))  # S = S_max
        elif option_type == 'put':
            prices[0, :] = K * np.exp(-r * (T - time_grid))  # S = 0
            prices[-1, :] = 0  # S = S_max
        
        # Build coefficient matrices for Crank-Nicolson
        # Black-Scholes PDE: ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + r*S*∂V/∂S - r*V = 0
        
        # Discretize
        for i in range(n_steps - 1, -1, -1):
            # Coefficients at interior points
            alpha = np.zeros(n_grid - 2)
            beta = np.zeros(n_grid - 2)
            gamma = np.zeros(n_grid - 2)
            
            for j in range(1, n_grid - 1):
                S_j = stock_grid[j]
                alpha[j-1] = 0.25 * dt * (sigma**2 * S_j**2 / dS**2 - r * S_j / dS)
                beta[j-1] = -0.5 * dt * (sigma**2 * S_j**2 / dS**2 + r)
                gamma[j-1] = 0.25 * dt * (sigma**2 * S_j**2 / dS**2 + r * S_j / dS)
            
            # Build tridiagonal matrices
            A = np.zeros((n_grid - 2, n_grid - 2))
            B = np.zeros((n_grid - 2, n_grid - 2))
            
            for j in range(n_grid - 2):
                A[j, j] = 1 - beta[j]
                if j > 0:
                    A[j, j-1] = -alpha[j]
                if j < n_grid - 3:
                    A[j, j+1] = -gamma[j]
                
                B[j, j] = 1 + beta[j]
                if j > 0:
                    B[j, j-1] = alpha[j]
                if j < n_grid - 3:
                    B[j, j+1] = gamma[j]
            
            # Right-hand side
            RHS = B @ prices[1:-1, i+1]
            
            # Add boundary conditions
            RHS[0] += alpha[0] * prices[0, i+1]
            RHS[-1] += gamma[-1] * prices[-1, i+1]
            
            # Solve linear system
            prices[1:-1, i] = np.linalg.solve(A, RHS)
        
        # Interpolate to get price at current stock price
        return prices, stock_grid, time_grid
    
    def finite_difference(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        scheme: str = 'implicit',
        n_steps: int = 100,
        n_grid: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve Black-Scholes PDE using finite difference method.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            scheme: 'explicit', 'implicit', or 'crank_nicolson'
            n_steps: Number of time steps
            n_grid: Number of spatial grid points
            
        Returns:
            Tuple of (grid_prices, stock_grid, time_grid)
        """
        if scheme == 'crank_nicolson':
            return self.crank_nicolson(S, K, T, r, sigma, option_type, n_steps, n_grid)
        
        # Set up spatial grid
        S_max = 2 * K
        S_min = 0.0
        dS = (S_max - S_min) / (n_grid - 1)
        stock_grid = np.linspace(S_min, S_max, n_grid)
        
        # Set up time grid
        dt = T / n_steps
        time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize option price grid
        prices = np.zeros((n_grid, n_steps + 1))
        
        # Terminal condition
        if option_type == 'call':
            prices[:, -1] = np.maximum(stock_grid - K, 0)
        elif option_type == 'put':
            prices[:, -1] = np.maximum(K - stock_grid, 0)
        
        # Boundary conditions
        if option_type == 'call':
            prices[0, :] = 0
            prices[-1, :] = S_max - K * np.exp(-r * (T - time_grid))
        elif option_type == 'put':
            prices[0, :] = K * np.exp(-r * (T - time_grid))
            prices[-1, :] = 0
        
        # Solve using specified scheme
        for i in range(n_steps - 1, -1, -1):
            if scheme == 'explicit':
                # Explicit scheme (conditionally stable)
                for j in range(1, n_grid - 1):
                    S_j = stock_grid[j]
                    a = 0.5 * sigma**2 * S_j**2 / dS**2
                    b = r * S_j / (2 * dS)
                    c = r
                    
                    prices[j, i] = prices[j, i+1] + dt * (
                        a * (prices[j+1, i+1] - 2*prices[j, i+1] + prices[j-1, i+1]) +
                        b * (prices[j+1, i+1] - prices[j-1, i+1]) -
                        c * prices[j, i+1]
                    )
            
            elif scheme == 'implicit':
                # Implicit scheme (unconditionally stable)
                # Build tridiagonal matrix
                A = np.zeros((n_grid - 2, n_grid - 2))
                RHS = np.zeros(n_grid - 2)
                
                for j in range(1, n_grid - 1):
                    S_j = stock_grid[j]
                    a = 0.5 * sigma**2 * S_j**2 / dS**2
                    b = r * S_j / (2 * dS)
                    c = r
                    
                    idx = j - 1
                    A[idx, idx] = 1 + dt * (2*a + c)
                    if idx > 0:
                        A[idx, idx-1] = -dt * (a - b)
                    if idx < n_grid - 3:
                        A[idx, idx+1] = -dt * (a + b)
                    
                    RHS[idx] = prices[j, i+1]
                
                # Add boundary conditions
                RHS[0] += dt * (a - b) * prices[0, i+1]
                RHS[-1] += dt * (a + b) * prices[-1, i+1]
                
                # Solve linear system
                prices[1:-1, i] = np.linalg.solve(A, RHS)
        
        return prices, stock_grid, time_grid
    
    def boundary_conditions(self, option_type: str, S: np.ndarray, K: float, T: float, r: float, t: float) -> np.ndarray:
        """
        Calculate boundary conditions for option pricing.
        
        Args:
            option_type: 'call' or 'put'
            S: Stock price grid
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            t: Current time
            
        Returns:
            Array with boundary condition values
        """
        tau = T - t
        boundary = np.zeros_like(S)
        
        if option_type == 'call':
            # Lower boundary (S = 0): V = 0
            boundary[0] = 0
            # Upper boundary (S = S_max): V = S - K * exp(-r*tau)
            boundary[-1] = S[-1] - K * np.exp(-r * tau)
        
        elif option_type == 'put':
            # Lower boundary (S = 0): V = K * exp(-r*tau)
            boundary[0] = K * np.exp(-r * tau)
            # Upper boundary (S = S_max): V = 0
            boundary[-1] = 0
        
        return boundary
    
    def solve_pde(
        self,
        pde_type: str,
        initial_conditions: np.ndarray,
        boundary_conditions: callable,
        grid_params: Dict[str, float],
        scheme: str = 'crank_nicolson'
    ) -> np.ndarray:
        """
        Solve a general PDE using finite difference method.
        
        Args:
            pde_type: Type of PDE ('heat', 'black_scholes', etc.)
            initial_conditions: Initial condition array
            boundary_conditions: Function that returns boundary conditions
            grid_params: Dictionary with grid parameters
            scheme: Finite difference scheme
            
        Returns:
            Solution array
        """
        # This is a placeholder for a more general PDE solver
        # For now, delegate to specific methods
        if pde_type == 'black_scholes':
            return self.finite_difference(
                S=grid_params.get('S', 100),
                K=grid_params.get('K', 100),
                T=grid_params.get('T', 1.0),
                r=grid_params.get('r', 0.05),
                sigma=grid_params.get('sigma', 0.2),
                option_type=grid_params.get('option_type', 'call'),
                scheme=scheme,
                n_steps=grid_params.get('n_steps', 100),
                n_grid=grid_params.get('n_grid', 100)
            )[0]
        
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")
    
    def calculate_greeks(
        self,
        prices: np.ndarray,
        stock_grid: np.ndarray,
        time_grid: np.ndarray,
        K: float,
        r: float,
        sigma: float
    ) -> Dict[str, np.ndarray]:
        """
        Calculate option Greeks from PDE solution.
        
        Args:
            prices: Option price grid from PDE solution
            stock_grid: Stock price grid
            time_grid: Time grid
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Dictionary with Greeks (delta, gamma, theta, vega, rho)
        """
        n_grid = len(stock_grid)
        n_steps = len(time_grid)
        
        # Delta: ∂V/∂S
        delta = np.zeros_like(prices)
        delta[1:-1, :] = (prices[2:, :] - prices[:-2, :]) / (stock_grid[2:] - stock_grid[:-2])
        delta[0, :] = delta[1, :]  # Forward difference at boundary
        delta[-1, :] = delta[-2, :]  # Backward difference at boundary
        
        # Gamma: ∂²V/∂S²
        gamma = np.zeros_like(prices)
        gamma[1:-1, :] = (prices[2:, :] - 2*prices[1:-1, :] + prices[:-2, :]) / ((stock_grid[2:] - stock_grid[:-2])**2 / 4)
        
        # Theta: ∂V/∂t
        theta = np.zeros_like(prices)
        theta[:, 1:-1] = (prices[:, 2:] - prices[:, :-2]) / (time_grid[2:] - time_grid[:-2])
        
        # Vega: ∂V/∂σ (approximate by finite difference)
        # This would require re-solving PDE with different sigma
        # For now, return zeros
        vega = np.zeros_like(prices)
        
        # Rho: ∂V/∂r (approximate by finite difference)
        # This would require re-solving PDE with different r
        # For now, return zeros
        rho = np.zeros_like(prices)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
        }
    
    def convergence_test(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = 'call',
        scheme: str = 'crank_nicolson'
    ) -> Dict[str, List[float]]:
        """
        Test convergence of PDE solver with different grid sizes.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            scheme: Finite difference scheme
            
        Returns:
            Dictionary with convergence results
        """
        grid_sizes = [50, 100, 200, 400]
        prices_at_S = []
        errors = []
        
        # Analytical solution for comparison
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            analytical_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            analytical_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        for n in grid_sizes:
            prices, stock_grid, time_grid = self.finite_difference(
                S, K, T, r, sigma, option_type, scheme, n_steps=n, n_grid=n
            )
            
            # Interpolate to get price at S
            price_at_S = np.interp(S, stock_grid, prices[:, 0])
            prices_at_S.append(price_at_S)
            errors.append(abs(price_at_S - analytical_price))
        
        return {
            'grid_sizes': grid_sizes,
            'prices': prices_at_S,
            'errors': errors,
            'analytical_price': analytical_price,
        }

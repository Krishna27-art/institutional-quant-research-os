"""
Cornish-Fisher expansion for VaR with skewness and kurtosis correction.

The Cornish-Fisher expansion adjusts the normal distribution quantile
to account for non-normality in the return distribution (skewness and kurtosis).

This is critical for accurate VaR estimation because:
- Financial returns are not normally distributed
- They exhibit fat tails (high kurtosis)
- They are often skewed (asymmetric)
- Parametric VaR assuming normality underestimates tail risk by 2-5x

Reference:
Cornish, E. A., & Fisher, R. A. (1937). Moments and cumulants in the
specification of distributions. Revue de l'Institut International de Statistique.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple


def cornish_fisher_quantile(
    z: float,
    skew: float,
    kurt: float,
    order: int = 3
) -> float:
    """
    Compute Cornish-Fisher adjusted quantile.
    
    The expansion adjusts the normal quantile z to account for skewness
    and kurtosis of the distribution.
    
    Args:
        z: Normal distribution quantile (e.g., norm.ppf(0.95))
        skew: Skewness of returns
        kurt: Excess kurtosis of returns (kurtosis - 3)
        order: Order of expansion (typically 3 or 4)
        
    Returns:
        Adjusted quantile z_cf
    """
    if order == 3:
        # Third-order Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
    elif order == 4:
        # Fourth-order Cornish-Fisher expansion (more accurate)
        z_cf = (z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
                - (2*z**3 - 5*z) * skew**2 / 36)
    else:
        raise ValueError(f"Order must be 3 or 4, got {order}")
    
    return z_cf


def cornish_fisher_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    order: int = 3
) -> Tuple[float, float, float, float]:
    """
    Compute Value at Risk using Cornish-Fisher expansion.
    
    Args:
        returns: Array of returns (simple returns, decimal)
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        order: Order of Cornish-Fisher expansion (3 or 4)
        
    Returns:
        Tuple of (var_pct, var_absolute, skew, kurt)
        - var_pct: VaR as percentage of portfolio
        - var_absolute: VaR as absolute loss (requires portfolio value)
        - skew: Skewness of returns
        - kurt: Excess kurtosis of returns
    """
    returns = np.asarray(returns)
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 10:
        raise ValueError("Need at least 10 returns for Cornish-Fisher VaR")
    
    # Compute moments
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    if sigma <= 0:
        raise ValueError(f"Standard deviation must be positive, got {sigma}")
    
    # Compute skewness and kurtosis
    skew = np.mean(((returns - mu) / sigma) ** 3)
    kurt = np.mean(((returns - mu) / sigma) ** 4) - 3  # excess kurtosis
    
    # Normal quantile
    z = norm.ppf(confidence)
    
    # Cornish-Fisher adjusted quantile
    z_cf = cornish_fisher_quantile(z, skew, kurt, order)
    
    # VaR at confidence level (as percentage)
    var_pct = mu + z_cf * sigma
    
    # VaR is the loss (negative return), so take absolute if negative
    var_pct = abs(var_pct) if var_pct < 0 else 0.0
    
    return var_pct, abs(var_pct), skew, kurt


def cornish_fisher_var_absolute(
    returns: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.95,
    order: int = 3
) -> float:
    """
    Compute absolute VaR using Cornish-Fisher expansion.
    
    Args:
        returns: Array of returns (simple returns, decimal)
        portfolio_value: Current portfolio value in INR
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        order: Order of Cornish-Fisher expansion (3 or 4)
        
    Returns:
        Absolute VaR in INR
    """
    var_pct, _, _, _ = cornish_fisher_var(returns, confidence, order)
    return portfolio_value * var_pct


def compare_var_methods(
    returns: np.ndarray,
    confidence: float = 0.95,
    portfolio_value: float = 1_000_000
) -> dict:
    """
    Compare VaR from different methods to show the impact of skew/kurtosis.
    
    Args:
        returns: Array of returns (simple returns, decimal)
        confidence: Confidence level
        portfolio_value: Portfolio value for absolute VaR
        
    Returns:
        Dictionary with comparison results
    """
    returns = np.asarray(returns)
    returns = returns[np.isfinite(returns)]
    
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # Normal VaR (parametric, assumes normality)
    z = norm.ppf(confidence)
    var_normal_pct = abs(mu + z * sigma)
    var_normal_abs = portfolio_value * var_normal_pct
    
    # Cornish-Fisher VaR (accounts for skew/kurtosis)
    var_cf_pct, _, skew, kurt = cornish_fisher_var(returns, confidence, order=3)
    var_cf_abs = portfolio_value * var_cf_pct
    
    # Ratio
    ratio = var_cf_abs / var_normal_abs if var_normal_abs > 0 else 1.0
    
    return {
        'normal_var_pct': var_normal_pct,
        'normal_var_abs': var_normal_abs,
        'cornish_fisher_var_pct': var_cf_pct,
        'cornish_fisher_var_abs': var_cf_abs,
        'ratio': ratio,
        'skewness': skew,
        'excess_kurtosis': kurt,
        'interpretation': (
            f"VaR is {ratio:.2f}x higher than normal due to "
            f"skew={skew:.2f} and kurt={kurt:.2f}"
        )
    }


if __name__ == "__main__":
    # Example usage
    print("Cornish-Fisher VaR Example")
    print("=" * 60)
    
    # Simulate returns with fat tails (t-distribution)
    np.random.seed(42)
    n = 1000
    returns = np.random.standard_t(3, n) * 0.02  # 2% daily vol, fat tails
    
    # Compare VaR methods
    comparison = compare_var_methods(returns, confidence=0.95, portfolio_value=10_000_000)
    
    print(f"Normal VaR (95%): ₹{comparison['normal_var_abs']:,.2f} ({comparison['normal_var_pct']:.2%})")
    print(f"Cornish-Fisher VaR (95%): ₹{comparison['cornish_fisher_var_abs']:,.2f} ({comparison['cornish_fisher_var_pct']:.2%})")
    print(f"Ratio: {comparison['ratio']:.2f}x")
    print(f"Skewness: {comparison['skewness']:.2f}")
    print(f"Excess Kurtosis: {comparison['excess_kurtosis']:.2f}")
    print()
    print(comparison['interpretation'])

"""
Bayesian shrinkage estimators for statistical moments.

Point estimates of moments (mean, variance, covariance) from limited data
have high uncertainty and can lead to overfitting. Bayesian shrinkage
combines the sample estimate with a prior (typically a structured estimator)
to produce a more robust estimate.

This is critical for:
- Kelly criterion (win rate, win/loss ratio have high uncertainty)
- Covariance matrix estimation (singular with more assets than observations)
- Risk calculations (overfitting leads to underestimated risk)

Methods:
1. James-Stein estimator for mean shrinkage
2. Ledoit-Wolf shrinkage for covariance matrices
3. Shrinkage for win rate and win/loss ratio in Kelly
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
from sklearn.covariance import LedoitWolf


def james_stein_shrinkage(
    sample_mean: float,
    sample_var: float,
    n_obs: int,
    prior_mean: float = 0.0,
    prior_var: float = 1.0
) -> Tuple[float, float]:
    """
    James-Stein estimator for shrinking the mean.
    
    The James-Stein estimator shrinks the sample mean towards a prior mean,
    reducing estimation error when the true mean is close to the prior.
    
    Args:
        sample_mean: Sample mean estimate
        sample_var: Sample variance
        n_obs: Number of observations
        prior_mean: Prior mean (typically 0 for returns)
        prior_var: Prior variance
        
    Returns:
        Tuple of (shrunken_mean, shrinkage_factor)
    """
    # Shrinkage factor
    if n_obs <= 1:
        return prior_mean, 1.0
    
    # James-Stein shrinkage factor
    shrinkage = max(0, 1 - (n_obs - 3) * sample_var / ((n_obs - 1) * prior_var))
    
    # Shrunken estimate
    shrunken_mean = shrinkage * prior_mean + (1 - shrinkage) * sample_mean
    
    return shrunken_mean, shrinkage


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage for covariance matrix estimation.
    
    Shrinks the sample covariance towards a structured estimator
(diagonal matrix of variances), which is well-conditioned and
    more robust when n_assets > n_observations.
    
    Args:
        returns: T x N matrix of returns (T observations, N assets)
        
    Returns:
        Shrunken covariance matrix (N x N)
    """
    lw = LedoitWolf().fit(returns)
    return lw.covariance_


def shrinkage_factor_for_win_rate(
    wins: int,
    losses: int,
    prior_win_rate: float = 0.5,
    prior_strength: float = 5.0
) -> Tuple[float, float]:
    """
    Bayesian shrinkage for win rate estimation.
    
    Uses a Beta prior to shrink the observed win rate towards a prior.
    This is critical for Kelly criterion where win rate uncertainty
    can lead to over-aggressive position sizing.
    
    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        prior_win_rate: Prior belief about win rate (default 0.5)
        prior_strength: Strength of prior (equivalent to prior_strength observations)
        
    Returns:
        Tuple of (shrunken_win_rate, shrinkage_factor)
    """
    total_trades = wins + losses
    
    if total_trades == 0:
        return prior_win_rate, 1.0
    
    # Beta prior parameters
    alpha_prior = prior_win_rate * prior_strength
    beta_prior = (1 - prior_win_rate) * prior_strength
    
    # Posterior parameters
    alpha_post = alpha_prior + wins
    beta_post = beta_prior + losses
    
    # Posterior mean (shrunken win rate)
    shrunken_win_rate = alpha_post / (alpha_post + beta_post)
    
    # Sample win rate
    sample_win_rate = wins / total_trades if total_trades > 0 else prior_win_rate
    
    # Shrinkage factor (how much we moved from sample to prior)
    shrinkage_factor = abs(shrunken_win_rate - sample_win_rate) / abs(sample_win_rate - prior_win_rate) if sample_win_rate != prior_win_rate else 0.0
    
    return shrunken_win_rate, shrinkage_factor


def shrinkage_for_win_loss_ratio(
    avg_win: float,
    avg_loss: float,
    n_wins: int,
    n_losses: int,
    prior_ratio: float = 1.0,
    prior_strength: float = 5.0
) -> Tuple[float, float]:
    """
    Bayesian shrinkage for win/loss ratio estimation.
    
    Shrinks the observed win/loss ratio towards a prior to reduce
    estimation error. Critical for Kelly criterion.
    
    Args:
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        n_wins: Number of winning trades
        n_losses: Number of losing trades
        prior_ratio: Prior belief about win/loss ratio (default 1.0)
        prior_strength: Strength of prior
        
    Returns:
        Tuple of (shrunken_ratio, shrinkage_factor)
    """
    if n_wins == 0 or n_losses == 0:
        return prior_ratio, 1.0
    
    sample_ratio = avg_win / avg_loss if avg_loss > 0 else prior_ratio
    
    # Shrink towards prior
    effective_n = n_wins + n_losses
    shrinkage = prior_strength / (prior_strength + effective_n)
    
    shrunken_ratio = shrinkage * prior_ratio + (1 - shrinkage) * sample_ratio
    
    return shrunken_ratio, shrinkage


def shrunk_kelly_fraction(
    wins: int,
    losses: int,
    avg_win: float,
    avg_loss: float,
    prior_win_rate: float = 0.5,
    prior_ratio: float = 1.0,
    prior_strength: float = 5.0,
    kelly_fraction: float = 0.25
) -> Tuple[float, dict]:
    """
    Compute Kelly fraction with Bayesian shrinkage on inputs.
    
    This prevents over-aggressive position sizing by shrinking
    uncertain estimates towards conservative priors.
    
    Args:
        wins: Number of winning trades
        losses: Number of losing trades
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive)
        prior_win_rate: Prior win rate
        prior_ratio: Prior win/loss ratio
        prior_strength: Strength of priors
        kelly_fraction: Fraction of optimal Kelly to use (e.g., 0.25 for quarter-Kelly)
        
    Returns:
        Tuple of (shrunken_kelly, diagnostics)
    """
    # Shrink win rate
    shrunken_win_rate, win_rate_shrinkage = shrinkage_factor_for_win_rate(
        wins, losses, prior_win_rate, prior_strength
    )
    
    # Shrink win/loss ratio
    shrunken_ratio, ratio_shrinkage = shrinkage_for_win_loss_ratio(
        avg_win, avg_loss, wins, losses, prior_ratio, prior_strength
    )
    
    # Compute Kelly with shrunken estimates
    p = shrunken_win_rate
    b = shrunken_ratio
    
    if b <= 0:
        return 0.0, {'error': 'Invalid win/loss ratio'}
    
    kelly = (p * b - (1 - p)) / b
    kelly = max(0, kelly)  # No negative Kelly
    
    # Apply fractional Kelly
    shrunken_kelly = kelly * kelly_fraction
    
    # Cap at reasonable maximum
    shrunken_kelly = min(shrunken_kelly, 0.25)
    
    diagnostics = {
        'sample_win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'shrunken_win_rate': shrunken_win_rate,
        'win_rate_shrinkage': win_rate_shrinkage,
        'sample_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
        'shrunken_ratio': shrunken_ratio,
        'ratio_shrinkage': ratio_shrinkage,
        'raw_kelly': kelly,
        'fractional_kelly': shrunken_kelly
    }
    
    return shrunken_kelly, diagnostics


def constant_correlation_shrinkage(
    returns: np.ndarray,
    shrinkage_intensity: float = 0.1
) -> np.ndarray:
    """
    Constant correlation shrinkage for covariance matrix.
    
    Shrinks the sample covariance towards a constant correlation model
    (all correlations equal to the average correlation). This is more
    stable than the sample covariance when n_assets is large.
    
    Args:
        returns: T x N matrix of returns
        shrinkage_intensity: Shrinkage intensity (0 = no shrinkage, 1 = full shrinkage)
        
    Returns:
        Shrunken covariance matrix
    """
    n_assets = returns.shape[1]
    
    # Sample covariance
    sample_cov = np.cov(returns, rowvar=False)
    
    # Extract variances (diagonal)
    variances = np.diag(sample_cov)
    stds = np.sqrt(variances)
    
    # Compute correlation matrix
    corr_matrix = sample_cov / np.outer(stds, stds)
    
    # Average correlation (excluding diagonal)
    avg_corr = (np.sum(corr_matrix) - n_assets) / (n_assets * (n_assets - 1))
    
    # Constant correlation matrix
    constant_corr = np.ones((n_assets, n_assets)) * avg_corr
    np.fill_diagonal(constant_corr, 1.0)
    
    # Convert back to covariance
    constant_cov = constant_corr * np.outer(stds, stds)
    
    # Shrink
    shrunk_cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * constant_cov
    
    return shrunk_cov


if __name__ == "__main__":
    # Example usage
    print("Bayesian Shrinkage Example")
    print("=" * 60)
    
    # Example: Kelly with limited data
    wins = 15
    losses = 10
    avg_win = 0.02
    avg_loss = 0.015
    
    print(f"Sample data: {wins} wins, {losses} losses")
    print(f"Sample win rate: {wins/(wins+losses):.2%}")
    print(f"Sample win/loss ratio: {avg_win/avg_loss:.2f}")
    print()
    
    # Compute Kelly with shrinkage
    shrunken_kelly, diagnostics = shrunk_kelly_fraction(
        wins, losses, avg_win, avg_loss,
        prior_win_rate=0.5, prior_ratio=1.0, prior_strength=5.0
    )
    
    print("Shrunken Kelly:")
    print(f"  Shrunken win rate: {diagnostics['shrunken_win_rate']:.2%}")
    print(f"  Shrunken ratio: {diagnostics['shrunken_ratio']:.2f}")
    print(f"  Raw Kelly: {diagnostics['raw_kelly']:.2%}")
    print(f"  Fractional Kelly (25%): {shrunken_kelly:.2%}")
    print()
    
    # Example: Covariance shrinkage
    np.random.seed(42)
    n_assets = 50
    n_obs = 100
    returns = np.random.multivariate_normal(
        np.zeros(n_assets),
        np.eye(n_assets) * 0.02**2,
        n_obs
    )
    
    print(f"Covariance matrix: {n_assets} assets, {n_obs} observations")
    print("Sample covariance condition number:", np.linalg.cond(np.cov(returns, rowvar=False)))
    
    shrunk_cov = ledoit_wolf_covariance(returns)
    print("Shrunken covariance condition number:", np.linalg.cond(shrunk_cov))
    print()
    print("Shrinkage improves conditioning significantly!")

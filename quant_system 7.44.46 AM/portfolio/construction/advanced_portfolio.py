"""
Advanced Portfolio Construction - HRP, Kelly, and Volatility Targeting

Implements institutional-grade portfolio construction methods:
- Hierarchical Risk Parity (HRP) - robust to correlation regime changes
- Kelly Criterion - optimal growth with bounded risk
- Black-Litterman - combine views with equilibrium
- Risk Parity - equal risk contribution
- Volatility Targeting - dynamic leverage adjustment
- Turnover Control - penalty to reduce transaction costs

These methods are used by top quant funds (Renaissance, Citadel, Two Sigma)
for portfolio construction and risk management.

Key Features:
- HRP with hierarchical clustering
- Kelly Criterion with fractional Kelly
- Black-Litterman with confidence levels
- Risk Parity with risk budgeting
- Volatility targeting with dynamic leverage
- Turnover control with transaction costs

Based on Blueprint Week 9-10: Portfolio & Risk
References:
- De Prado (2016) - Building Diversified Portfolios
- Kelly (1956) - A New Interpretation of Information Rate
- Black-Litterman (1992) - Global Portfolio Optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP).
    
    HRP constructs portfolios by clustering assets based on correlation
    and allocating weights inversely proportional to risk within clusters.
    This makes HRP robust to correlation regime changes.
    """
    
    def __init__(self):
        """Initialize HRP."""
        self.linkage_matrix = None
        self.clusters = None
    
    def get_cluster_var(self, cov: np.ndarray, cluster_items: List[int]) -> float:
        """
        Calculate intra-cluster variance.
        
        Args:
            cov: Covariance matrix
            cluster_items: Indices of items in cluster
            
        Returns:
            Intra-cluster variance
        """
        cov_cluster = cov[np.ix_(cluster_items, cluster_items)]
        w = self._get_ivp(cov_cluster)
        return np.dot(w, np.dot(cov_cluster, w))
    
    def _get_ivp(self, cov: np.ndarray) -> np.ndarray:
        """
        Get inverse variance portfolio weights.
        
        Args:
            cov: Covariance matrix
            
        Returns:
            IVP weights
        """
        ivp = 1.0 / np.diag(cov)
        ivp /= ivp.sum()
        return ivp
    
    def get_rec_bifurcation(
        self,
        cov: np.ndarray,
        items: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Perform recursive bisection for clustering.
        
        Args:
            cov: Covariance matrix
            items: Indices of items to cluster
            
        Returns:
            Tuple of (cluster1, cluster2)
        """
        # Calculate covariance matrix for items
        cov_items = cov[np.ix_(items, items)]
        
        # Calculate distance matrix
        dist_matrix = np.sqrt(np.diag(cov_items))
        dist_matrix = dist_matrix[:, None] + dist_matrix[None, :] - 2 * cov_items
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(dist_matrix), method='ward')
        
        # Get two clusters
        clusters = fcluster(linkage_matrix, 2, criterion='maxclust')
        
        cluster1 = [items[i] for i in range(len(items)) if clusters[i] == 1]
        cluster2 = [items[i] for i in range(len(items)) if clusters[i] == 2]
        
        return cluster1, cluster2
    
    def allocate(
        self,
        cov: np.ndarray,
        items: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Allocate weights using HRP.
        
        Args:
            cov: Covariance matrix
            items: Indices of items (if None, use all)
            
        Returns:
            Weight array
        """
        if items is None:
            items = list(range(cov.shape[0]))
        
        # Single item
        if len(items) == 1:
            weights = np.zeros(cov.shape[0])
            weights[items[0]] = 1.0
            return weights
        
        # Recursive bisection
        cluster1, cluster2 = self.get_rec_bifurcation(cov, items)
        
        # Allocate within clusters
        w1 = self.allocate(cov, cluster1)
        w2 = self.allocate(cov, cluster2)
        
        # Combine clusters
        var1 = self.get_cluster_var(cov, cluster1)
        var2 = self.get_cluster_var(cov, cluster2)
        
        alpha = 1 - var1 / (var1 + var2)
        
        weights = alpha * w1 + (1 - alpha) * w2
        
        return weights


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    
    The Kelly Criterion maximizes the expected logarithmic utility,
    leading to optimal growth with bounded risk.
    """
    
    def __init__(self, max_leverage: float = 2.0, fractional_kelly: float = 0.5):
        """
        Initialize Kelly Criterion.
        
        Args:
            max_leverage: Maximum leverage allowed
            fractional_kelly: Fraction of full Kelly to use (for safety)
        """
        self.max_leverage = max_leverage
        self.fractional_kelly = fractional_kelly
    
    def calculate_kelly(
        self,
        expected_return: float,
        variance: float,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Formula: f* = (μ - r) / σ²
        
        Args:
            expected_return: Expected return
            variance: Variance of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Kelly fraction
        """
        if variance <= 0:
            return 0.0
        
        kelly = (expected_return - risk_free_rate) / variance
        
        # Apply fractional Kelly for safety
        kelly *= self.fractional_kelly
        
        # Cap at max leverage
        kelly = np.clip(kelly, -self.max_leverage, self.max_leverage)
        
        return kelly
    
    def calculate_portfolio_kelly(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> np.ndarray:
        """
        Calculate Kelly weights for portfolio.
        
        Formula: f* = Σ⁻¹ (μ - r)
        
        Args:
            expected_returns: Expected returns vector
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            
        Returns:
            Kelly weight vector
        """
        try:
            # Invert covariance matrix
            cov_inv = np.linalg.inv(cov_matrix)
            
            # Calculate excess returns
            excess_returns = expected_returns - risk_free_rate
            
            # Calculate Kelly weights
            kelly_weights = np.dot(cov_inv, excess_returns)
            
            # Apply fractional Kelly
            kelly_weights *= self.fractional_kelly
            
            # Cap weights
            kelly_weights = np.clip(kelly_weights, -self.max_leverage, self.max_leverage)
            
            return kelly_weights
            
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix not invertible, using diagonal")
            # Fallback to diagonal
            variances = np.diag(cov_matrix)
            kelly_weights = (expected_returns - risk_free_rate) / variances
            kelly_weights *= self.fractional_kelly
            kelly_weights = np.clip(kelly_weights, -self.max_leverage, self.max_leverage)
            return kelly_weights


class BlackLitterman:
    """
    Black-Litterman model for combining views with equilibrium.
    
    The Black-Litterman model combines investor views with market
    equilibrium to produce more stable and intuitive portfolio weights.
    """
    
    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 1.0
    ):
        """
        Initialize Black-Litterman model.
        
        Args:
            tau: Uncertainty parameter (typically 0.05)
            risk_aversion: Risk aversion coefficient
        """
        self.tau = tau
        self.risk_aversion = risk_aversion
    
    def combine_views(
        self,
        equilibrium_returns: np.ndarray,
        cov_matrix: np.ndarray,
        view_matrix: np.ndarray,
        view_returns: np.ndarray,
        view_confidences: np.ndarray
    ) -> np.ndarray:
        """
        Combine views with equilibrium returns.
        
        Formula:
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
        
        Args:
            equilibrium_returns: Equilibrium returns (π)
            cov_matrix: Covariance matrix (Σ)
            view_matrix: View matrix (P)
            view_returns: View returns (Q)
            view_confidences: View confidence matrix (Ω)
            
        Returns:
            Black-Litterman expected returns
        """
        try:
            # Calculate components
            tau_sigma = self.tau * cov_matrix
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            
            omega_inv = np.linalg.inv(view_confidences)
            
            # Calculate Black-Litterman returns
            term1 = np.linalg.inv(tau_sigma_inv + view_matrix.T @ omega_inv @ view_matrix)
            term2 = tau_sigma_inv @ equilibrium_returns + view_matrix.T @ omega_inv @ view_returns
            
            bl_returns = term1 @ term2
            
            return bl_returns
            
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion error in Black-Litterman")
            return equilibrium_returns


class RiskParity:
    """
    Risk Parity portfolio construction.
    
    Risk parity allocates capital such that each asset contributes
    equal risk to the portfolio. This is achieved by solving for
    weights that equalize marginal risk contributions.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize Risk Parity.
        
        Args:
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def allocate(
        self,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Allocate weights using risk parity.
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Risk parity weights
        """
        n = cov_matrix.shape[0]
        
        # Initialize equal weights
        weights = np.ones(n) / n
        
        for iteration in range(self.max_iterations):
            # Calculate marginal risk contributions
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_risk = cov_matrix @ weights / portfolio_vol
            
            # Calculate risk contributions
            risk_contributions = weights * marginal_risk
            
            # Target risk contribution (equal)
            target_risk = portfolio_vol / n
            
            # Update weights
            new_weights = weights * (target_risk / risk_contributions)
            new_weights = new_weights / new_weights.sum()
            
            # Check convergence
            if np.max(np.abs(new_weights - weights)) < self.tolerance:
                break
            
            weights = new_weights
        
        return weights


class VolatilityTargeting:
    """
    Volatility Targeting for dynamic leverage adjustment.
    
    Dynamically adjusts portfolio leverage to target a specific
    volatility level, reducing exposure during high volatility periods.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
        min_leverage: float = 0.5,
        window: int = 20
    ):
        """
        Initialize volatility targeting.
        
        Args:
            target_volatility: Target annualized volatility
            max_leverage: Maximum leverage
            min_leverage: Minimum leverage
            window: Window for volatility calculation
        """
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.window = window
    
    def calculate_leverage(
        self,
        returns: pd.Series
    ) -> float:
        """
        Calculate optimal leverage based on current volatility.
        
        Formula: leverage = target_vol / current_vol
        
        Args:
            returns: Return series
            
        Returns:
            Optimal leverage
        """
        # Calculate current volatility
        current_vol = returns.rolling(window=self.window).std().iloc[-1] * np.sqrt(252)
        
        if current_vol == 0:
            return self.min_leverage
        
        # Calculate leverage
        leverage = self.target_volatility / current_vol
        
        # Clip to bounds
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        return leverage
    
    def adjust_weights(
        self,
        weights: np.ndarray,
        returns: pd.Series
    ) -> np.ndarray:
        """
        Adjust weights based on volatility targeting.
        
        Args:
            weights: Original weights
            returns: Return series
            
        Returns:
            Adjusted weights
        """
        leverage = self.calculate_leverage(returns)
        adjusted_weights = weights * leverage
        
        return adjusted_weights


class AdvancedPortfolioConstructor:
    """
    Advanced Portfolio Constructor combining all methods.
    
    This class provides a unified interface for portfolio construction
    using HRP, Kelly, Black-Litterman, Risk Parity, and Volatility Targeting.
    """
    
    def __init__(
        self,
        method: str = 'hrp',
        target_volatility: float = 0.15,
        max_leverage: float = 2.0,
        turnover_penalty: float = 0.001
    ):
        """
        Initialize advanced portfolio constructor.
        
        Args:
            method: Portfolio construction method ('hrp', 'kelly', 'risk_parity', 'black_litterman')
            target_volatility: Target volatility for volatility targeting
            max_leverage: Maximum leverage
            turnover_penalty: Penalty for turnover
        """
        self.method = method
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.turnover_penalty = turnover_penalty
        
        # Initialize methods
        self.hrp = HierarchicalRiskParity()
        self.kelly = KellyCriterion(max_leverage=max_leverage)
        self.risk_parity = RiskParity()
        self.black_litterman = BlackLitterman()
        self.vol_targeting = VolatilityTargeting(
            target_volatility=target_volatility,
            max_leverage=max_leverage
        )
        
        # Track previous weights for turnover calculation
        self.previous_weights = None
    
    def construct_portfolio(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        views: Optional[Dict] = None
    ) -> Dict:
        """
        Construct portfolio using specified method.
        
        Args:
            returns: Historical returns DataFrame
            expected_returns: Expected returns (for Kelly, Black-Litterman)
            views: Views for Black-Litterman
            
        Returns:
            Dictionary with portfolio information
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov().values
        
        # Calculate weights based on method
        if self.method == 'hrp':
            weights = self.hrp.allocate(cov_matrix)
        elif self.method == 'kelly':
            if expected_returns is None:
                expected_returns = returns.mean().values
            weights = self.kelly.calculate_portfolio_kelly(expected_returns, cov_matrix)
        elif self.method == 'risk_parity':
            weights = self.risk_parity.allocate(cov_matrix)
        elif self.method == 'black_litterman':
            if expected_returns is None:
                expected_returns = returns.mean().values
            # Use equilibrium returns
            equilibrium_returns = expected_returns
            weights = self._black_litterman_weights(
                equilibrium_returns, cov_matrix, views
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply volatility targeting
        portfolio_returns = returns.mean(axis=1)
        weights = self.vol_targeting.adjust_weights(weights, portfolio_returns)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Calculate turnover
        turnover = 0.0
        if self.previous_weights is not None:
            turnover = np.sum(np.abs(weights - self.previous_weights)) / 2
        
        # Apply turnover penalty
        if turnover > 0:
            weights = self._apply_turnover_penalty(weights, self.previous_weights)
        
        # Update previous weights
        self.previous_weights = weights.copy()
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(weights, returns, cov_matrix)
        
        return {
            'weights': weights,
            'method': self.method,
            'turnover': turnover,
            'leverage': np.sum(np.abs(weights)),
            **portfolio_metrics
        }
    
    def _black_litterman_weights(
        self,
        equilibrium_returns: np.ndarray,
        cov_matrix: np.ndarray,
        views: Optional[Dict]
    ) -> np.ndarray:
        """Calculate Black-Litterman weights."""
        if views is None:
            # Fallback to Kelly
            return self.kelly.calculate_portfolio_kelly(equilibrium_returns, cov_matrix)
        
        # Extract view components
        view_matrix = views.get('P', np.eye(len(equilibrium_returns)))
        view_returns = views.get('Q', equilibrium_returns)
        view_confidences = views.get('Omega', np.eye(len(equilibrium_returns)))
        
        # Combine views
        bl_returns = self.black_litterman.combine_views(
            equilibrium_returns, cov_matrix,
            view_matrix, view_returns, view_confidences
        )
        
        # Calculate weights from BL returns
        weights = self.kelly.calculate_portfolio_kelly(bl_returns, cov_matrix)
        
        return weights
    
    def _apply_turnover_penalty(
        self,
        weights: np.ndarray,
        previous_weights: np.ndarray
    ) -> np.ndarray:
        """Apply turnover penalty to weights."""
        if previous_weights is None:
            return weights
        
        # Calculate penalty
        turnover = np.sum(np.abs(weights - previous_weights)) / 2
        penalty = 1.0 - self.turnover_penalty * turnover
        
        # Apply penalty (move towards previous weights)
        adjusted_weights = penalty * weights + (1 - penalty) * previous_weights
        
        return adjusted_weights
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        cov_matrix: np.ndarray
    ) -> Dict:
        """Calculate portfolio metrics."""
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Metrics
        mean_return = portfolio_returns.mean() * 252
        std_return = portfolio_returns.std() * np.sqrt(252)
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        
        # Portfolio variance
        portfolio_var = weights @ cov_matrix @ weights
        
        # Risk contribution
        marginal_risk = cov_matrix @ weights
        risk_contributions = weights * marginal_risk
        risk_contributions_pct = risk_contributions / risk_contributions.sum()
        
        return {
            'expected_return': mean_return,
            'volatility': std_return,
            'sharpe': sharpe,
            'portfolio_variance': portfolio_var,
            'risk_contributions': risk_contributions_pct
        }


if __name__ == "__main__":
    # Test advanced portfolio construction
    print("Testing Advanced Portfolio Construction...")
    
    # Create sample returns
    np.random.seed(42)
    n_assets = 10
    n_samples = 252
    
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            np.zeros(n_assets),
            np.eye(n_assets) * 0.02,
            n_samples
        ),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Test HRP
    print("\nTesting HRP...")
    constructor = AdvancedPortfolioConstructor(method='hrp')
    result = constructor.construct_portfolio(returns)
    print(f"Method: {result['method']}")
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Leverage: {result['leverage']:.2f}")
    print(f"Top 5 weights: {np.sort(result['weights'])[-5:][::-1]}")
    
    # Test Kelly
    print("\nTesting Kelly...")
    constructor = AdvancedPortfolioConstructor(method='kelly')
    result = constructor.construct_portfolio(returns)
    print(f"Method: {result['method']}")
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Leverage: {result['leverage']:.2f}")
    
    # Test Risk Parity
    print("\nTesting Risk Parity...")
    constructor = AdvancedPortfolioConstructor(method='risk_parity')
    result = constructor.construct_portfolio(returns)
    print(f"Method: {result['method']}")
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Leverage: {result['leverage']:.2f}")
    
    # Test Volatility Targeting
    print("\nTesting Volatility Targeting...")
    vol_targeting = VolatilityTargeting(target_volatility=0.15)
    portfolio_returns = returns.mean(axis=1)
    leverage = vol_targeting.calculate_leverage(portfolio_returns)
    print(f"Calculated leverage: {leverage:.2f}")
    
    print("\nAdvanced Portfolio Construction test completed.")

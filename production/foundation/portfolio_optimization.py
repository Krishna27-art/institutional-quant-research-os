"""
Portfolio Optimization - Level 2 Foundation

This module provides portfolio optimization methods:
- Mean-variance optimization (Markowitz 1952)
- Black-Litterman model
- Hierarchical Risk Parity (HRP)
- Risk parity
- Minimum variance
- Maximum diversification
- Regime-based optimization

Based on Audit Report Priority 2: Asset Pricing Theories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import optimize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Types of portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hierarchical_risk_parity"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class PortfolioWeights:
    """Portfolio weights."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    
    def __post_init__(self):
        """Validate portfolio weights."""
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")


class PortfolioOptimization:
    """
    Portfolio optimization methods.
    
    This class implements various portfolio optimization techniques
    based on Modern Portfolio Theory (Markowitz 1952) and extensions.
    """
    
    def __init__(self):
        """Initialize portfolio optimization toolkit."""
        pass
    
    def mean_variance_optimization(
        self,
        returns: pd.DataFrame,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> PortfolioWeights:
        """
        Mean-variance optimization (Markowitz 1952).
        
        Maximize: w'μ - (λ/2) * w'Σw
        Subject to: sum(w) = 1, w >= 0
        
        Args:
            returns: DataFrame of asset returns
            risk_aversion: Risk aversion parameter (λ)
            constraints: Additional constraints
            
        Returns:
            PortfolioWeights object
        """
        n_assets = returns.shape[1]
        assets = returns.columns.tolist()
        
        # Calculate mean returns and covariance matrix
        mu = returns.mean().values
        sigma = returns.cov().values
        
        # Objective function: minimize -w'μ + (λ/2) * w'Σw
        def objective(w):
            return -np.dot(w, mu) + 0.5 * risk_aversion * np.dot(w, np.dot(sigma, w))
        
        # Constraints
        constraints_list = []
        
        # Sum of weights = 1
        constraints_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Additional constraints
        if constraints:
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                bounds = [(0, max_weight) for _ in range(n_assets)]
            else:
                bounds = [(0, 1) for _ in range(n_assets)]
            
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                bounds = [(min_weight, 1) for _ in range(n_assets)]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            w = w0
        else:
            w = result.x
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        weights_dict = {assets[i]: w[i] for i in range(n_assets)}
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )
    
    def black_litterman(
        self,
        returns: pd.DataFrame,
        views: Dict[str, float],
        uncertainty_matrix: Optional[np.ndarray] = None,
        tau: float = 0.05,
        risk_aversion: float = 1.0
    ) -> PortfolioWeights:
        """
        Black-Litterman portfolio optimization.
        
        Combines market equilibrium with investor views.
        
        Args:
            returns: DataFrame of asset returns
            views: Dictionary of investor views {asset: expected_return}
            uncertainty_matrix: Uncertainty matrix for views
            tau: Scaling parameter for uncertainty
            risk_aversion: Risk aversion parameter
            
        Returns:
            PortfolioWeights object
        """
        n_assets = returns.shape[1]
        assets = returns.columns.tolist()
        
        # Calculate market equilibrium
        mu = returns.mean().values
        sigma = returns.cov().values
        
        # Market equilibrium excess returns (simplified)
        pi = mu  # In practice, use CAPM to get equilibrium returns
        
        # Views matrix
        P = np.zeros((len(views), n_assets))
        q = np.zeros(len(views))
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in assets:
                P[i, assets.index(asset)] = 1
                q[i] = view_return
        
        # Uncertainty matrix
        if uncertainty_matrix is None:
            # Default: diagonal with proportional uncertainty
            omega = np.diag(np.diag(sigma)) * tau
        else:
            omega = uncertainty_matrix
        
        # Black-Litterman formula
        # μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)q]
        sigma_inv = np.linalg.inv(sigma)
        omega_inv = np.linalg.inv(omega)
        
        M1 = tau * sigma_inv
        M2 = P.T @ omega_inv @ P
        M = np.linalg.inv(M1 + M2)
        
        M3 = M1 @ pi
        M4 = P.T @ omega_inv @ q
        mu_bl = M @ (M3 + M4)
        
        # Updated covariance
        sigma_bl = (1 + tau) * sigma
        
        # Optimize with Black-Litterman returns
        def objective(w):
            return -np.dot(w, mu_bl) + 0.5 * risk_aversion * np.dot(w, np.dot(sigma_bl, w))
        
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            w = w0
        else:
            w = result.x
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(w, mu_bl)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma_bl, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        weights_dict = {assets[i]: w[i] for i in range(n_assets)}
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )
    
    def hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        linkage_method: str = 'ward'
    ) -> PortfolioWeights:
        """
        Hierarchical Risk Parity (HRP) optimization.
        
        Uses hierarchical clustering to group assets and allocate risk.
        
        Args:
            returns: DataFrame of asset returns
            linkage_method: Linkage method for clustering
            
        Returns:
            PortfolioWeights object
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate correlation matrix
        corr = returns.corr().values
        
        # Convert correlation to distance
        distance = np.sqrt((1 - corr) / 2)
        
        # Hierarchical clustering
        dist_matrix = squareform(distance)
        linkage_matrix = linkage(dist_matrix, method=linkage_method)
        
        # Get cluster order
        cluster_order = self._get_cluster_order(linkage_matrix, n_assets)
        
        # Calculate variance
        var = returns.var().values
        
        # Recursive bisection
        def recursive_bisection(cluster_indices):
            if len(cluster_indices) == 1:
                return {cluster_indices[0]: 1.0}
            
            # Split cluster
            mid = len(cluster_indices) // 2
            left_cluster = cluster_indices[:mid]
            right_cluster = cluster_indices[mid:]
            
            # Calculate cluster variances
            left_var = np.sum(var[left_cluster])
            right_var = np.sum(var[right_cluster])
            
            # Allocate weights
            total_var = left_var + right_var
            left_weight = right_var / total_var
            right_weight = left_var / total_var
            
            # Recursively allocate
            left_weights = recursive_bisection(left_cluster)
            right_weights = recursive_bisection(right_cluster)
            
            # Scale weights
            for key in left_weights:
                left_weights[key] *= left_weight
            for key in right_weights:
                right_weights[key] *= right_weight
            
            return {**left_weights, **right_weights}
        
        weights_dict = recursive_bisection(cluster_order)
        
        # Normalize weights
        total_weight = sum(weights_dict.values())
        weights_dict = {k: v / total_weight for k, v in weights_dict.items()}
        
        # Map to asset names
        weights_dict = {assets[i]: weights_dict[i] for i in range(n_assets)}
        
        # Calculate portfolio statistics
        w = np.array([weights_dict[asset] for asset in assets])
        mu = returns.mean().values
        sigma = returns.cov().values
        
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )
    
    def _get_cluster_order(self, linkage_matrix, n_assets: int) -> List[int]:
        """Get cluster order from linkage matrix."""
        order = []
        for i in range(n_assets):
            order.append(i)
        return order
    
    def risk_parity(
        self,
        returns: pd.DataFrame,
        target_risk: Optional[float] = None
    ) -> PortfolioWeights:
        """
        Risk parity optimization.
        
        Equal risk contribution from each asset.
        
        Args:
            returns: DataFrame of asset returns
            target_risk: Target portfolio risk (optional)
            
        Returns:
            PortfolioWeights object
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate covariance matrix
        sigma = returns.cov().values
        
        # Objective: minimize difference in risk contributions
        def objective(w):
            portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
            marginal_risk = np.dot(sigma, w) / portfolio_risk
            risk_contributions = w * marginal_risk
            
            # Minimize variance of risk contributions
            return np.var(risk_contributions)
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            w = w0
        else:
            w = result.x
        
        # Scale to target risk if specified
        if target_risk is not None:
            portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
            if portfolio_risk > 0:
                w = w * (target_risk / portfolio_risk)
                w = w / np.sum(w)  # Re-normalize
        
        # Calculate portfolio statistics
        mu = returns.mean().values
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        weights_dict = {assets[i]: w[i] for i in range(n_assets)}
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )
    
    def minimum_variance(
        self,
        returns: pd.DataFrame
    ) -> PortfolioWeights:
        """
        Minimum variance portfolio optimization.
        
        Minimize portfolio variance subject to sum(w) = 1.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            PortfolioWeights object
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate covariance matrix
        sigma = returns.cov().values
        
        # Objective: minimize portfolio variance
        def objective(w):
            return np.dot(w, np.dot(sigma, w))
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            w = w0
        else:
            w = result.x
        
        # Calculate portfolio statistics
        mu = returns.mean().values
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        weights_dict = {assets[i]: w[i] for i in range(n_assets)}
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )
    
    def maximum_diversification(
        self,
        returns: pd.DataFrame
    ) -> PortfolioWeights:
        """
        Maximum diversification portfolio optimization.
        
        Maximize diversification ratio: (w'σ) / sqrt(w'Σw)
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            PortfolioWeights object
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate covariance and standard deviation
        sigma = returns.cov().values
        std = np.std(returns, ddof=1).values
        
        # Objective: maximize diversification ratio
        def objective(w):
            portfolio_std = np.sqrt(np.dot(w, np.dot(sigma, w)))
            weighted_avg_std = np.dot(w, std)
            return -weighted_avg_std / portfolio_std
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            w = w0
        else:
            w = result.x
        
        # Calculate portfolio statistics
        mu = returns.mean().values
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        weights_dict = {assets[i]: w[i] for i in range(n_assets)}
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )
    
    def regime_based_optimization(
        self,
        returns: pd.DataFrame,
        regime: str,
        constraints: Optional[Dict] = None
    ) -> PortfolioWeights:
        """
        Regime-based portfolio optimization.
        
        Adjusts optimization method based on market regime.
        
        Args:
            returns: DataFrame of asset returns
            regime: Market regime ('crisis', 'normal', 'bull', 'bear')
            constraints: Additional constraints
            
        Returns:
            PortfolioWeights object
        """
        if regime == 'crisis':
            # Use minimum variance in crisis
            return self.minimum_variance(returns)
        elif regime == 'normal':
            # Use mean-variance in normal markets
            return self.mean_variance_optimization(returns, constraints=constraints)
        elif regime == 'bull':
            # Use maximum diversification in bull market
            return self.maximum_diversification(returns)
        elif regime == 'bear':
            # Use risk parity in bear market
            return self.risk_parity(returns)
        else:
            # Default to HRP
            return self.hierarchical_risk_parity(returns)
    
    def constraints_optimization(
        self,
        returns: pd.DataFrame,
        constraints: Dict[str, Union[float, List[float]]]
    ) -> PortfolioWeights:
        """
        Portfolio optimization with custom constraints.
        
        Args:
            returns: DataFrame of asset returns
            constraints: Dictionary of constraints
            
        Returns:
            PortfolioWeights object
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate mean returns and covariance matrix
        mu = returns.mean().values
        sigma = returns.cov().values
        
        # Objective function (maximize Sharpe)
        def objective(w):
            portfolio_return = np.dot(w, mu)
            portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
            return -portfolio_return / portfolio_risk if portfolio_risk > 0 else -np.inf
        
        # Build constraints
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        if 'min_weight' in constraints and 'max_weight' in constraints:
            bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        elif 'max_weight' in constraints:
            bounds = [(0, constraints['max_weight']) for _ in range(n_assets)]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # Sector constraints
        if 'sector_constraints' in constraints:
            sector_mapping = constraints['sector_mapping']
            sector_limits = constraints['sector_constraints']
            
            for sector, limit in sector_limits.items():
                sector_indices = [i for i, asset in enumerate(assets) if sector_mapping.get(asset) == sector]
                if sector_indices:
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=sector_indices, lim=limit: lim - np.sum(w[idx])
                    })
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        result = optimize.minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            w = w0
        else:
            w = result.x
        
        # Calculate portfolio statistics
        portfolio_return = np.dot(w, mu)
        portfolio_risk = np.sqrt(np.dot(w, np.dot(sigma, w)))
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        weights_dict = {assets[i]: w[i] for i in range(n_assets)}
        
        return PortfolioWeights(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe,
        )

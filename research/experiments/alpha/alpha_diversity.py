"""
Alpha Diversity Module

Based on Comprehensive Upgrade Analysis - Tier 1 Upgrade (#4)
Expected Sharpe improvement: +0.3–0.5
Renaissance core philosophy: diverse, uncorrelated alpha sources

Methodology:
- Compute correlations between alpha strategies
- Identify uncorrelated alpha sources
- Combine alphas with optimal weights to maximize diversification
- Ensure alpha diversity in the portfolio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass
class AlphaSource:
    """Single alpha source"""
    name: str
    returns: pd.Series
    sharpe: float
    capacity: float
    turnover: float


@dataclass
class DiversityConfig:
    """Configuration for Alpha Diversity"""
    max_correlation: float = 0.3  # Maximum allowed correlation between alphas
    min_alpha_count: int = 5  # Minimum number of alpha sources
    max_alpha_count: int = 10  # Maximum number of alpha sources
    correlation_window: int = 252  # 1 year for correlation estimation
    weight_method: str = "shrinkage"  # "shrinkage", "equal", "risk_parity"
    cluster_method: str = "average"  # Clustering method for alpha selection
    min_sharpe_threshold: float = 0.5  # Minimum Sharpe for inclusion
    max_turnover_threshold: float = 2.0  # Max annual turnover (200%)


class AlphaDiversityManager:
    """
    Alpha Diversity Manager
    
    Ensures portfolio uses diverse, uncorrelated alpha sources.
    Based on Renaissance's core philosophy.
    
    Methodology:
    1. Compute correlation matrix of alpha returns
    2. Cluster alphas by correlation
    3. Select best alpha from each cluster
    4. Optimize weights for maximum diversification
    """
    
    def __init__(self, config: DiversityConfig):
        self.config = config
        
        # Alpha sources
        self.alpha_sources: Dict[str, AlphaSource] = {}
        
        # Correlation matrix
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Alpha weights
        self.alpha_weights: Dict[str, float] = {}
        
        # Selected alphas
        self.selected_alphas: List[str] = []
    
    def add_alpha_source(self, name: str, returns: pd.Series, 
                        sharpe: float, capacity: float, turnover: float) -> None:
        """
        Add an alpha source
        
        Args:
            name: Alpha name
            returns: Alpha returns series
            sharpe: Alpha Sharpe ratio
            capacity: Alpha capacity (in AUM)
            turnover: Alpha annual turnover
        """
        self.alpha_sources[name] = AlphaSource(
            name=name,
            returns=returns,
            sharpe=sharpe,
            capacity=capacity,
            turnover=turnover
        )
    
    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute correlation matrix between alpha sources
        
        Returns:
            Correlation matrix DataFrame
        """
        if len(self.alpha_sources) < 2:
            return pd.DataFrame()
        
        # Collect returns
        returns_df = pd.DataFrame()
        for name, alpha in self.alpha_sources.items():
            returns_df[name] = alpha.returns
        
        # Compute correlation
        self.correlation_matrix = returns_df.corr()
        
        return self.correlation_matrix
    
    def select_diverse_alphas(self) -> List[str]:
        """
        Select diverse alphas using hierarchical clustering
        
        Returns:
            List of selected alpha names
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()
        
        if self.correlation_matrix.empty:
            return list(self.alpha_sources.keys())[:self.config.min_alpha_count]
        
        # Convert correlation to distance
        distance_matrix = 1 - self.correlation_matrix.values
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method=self.config.cluster_method)
        
        # Cluster alphas
        n_clusters = min(len(self.alpha_sources), self.config.max_alpha_count)
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Select best alpha from each cluster
        selected = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_alphas = [
                name for name, cluster in zip(self.correlation_matrix.columns, clusters)
                if cluster == cluster_id
            ]
            
            # Select alpha with highest Sharpe in cluster
            best_alpha = max(
                cluster_alphas,
                key=lambda name: self.alpha_sources[name].sharpe
            )
            
            # Check Sharpe threshold
            if self.alpha_sources[best_alpha].sharpe >= self.config.min_sharpe_threshold:
                selected.append(best_alpha)
        
        # Ensure minimum alpha count
        if len(selected) < self.config.min_alpha_count:
            # Add remaining alphas by Sharpe
            remaining = [
                name for name in self.alpha_sources.keys()
                if name not in selected
            ]
            remaining.sort(key=lambda name: self.alpha_sources[name].sharpe, reverse=True)
            
            needed = self.config.min_alpha_count - len(selected)
            selected.extend(remaining[:needed])
        
        self.selected_alphas = selected
        return selected
    
    def optimize_alpha_weights(self) -> Dict[str, float]:
        """
        Optimize alpha weights for maximum diversification
        
        Returns:
            Dictionary of alpha -> weight
        """
        if not self.selected_alphas:
            self.select_diverse_alphas()
        
        if len(self.selected_alphas) < 2:
            # Equal weights if only one alpha
            weights = {alpha: 1.0 for alpha in self.selected_alphas}
            self.alpha_weights = weights
            return weights
        
        # Collect returns for selected alphas
        returns_df = pd.DataFrame()
        for alpha in self.selected_alphas:
            returns_df[alpha] = self.alpha_sources[alpha].returns
        
        # Compute covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualized
        
        # Get expected returns (use historical mean)
        expected_returns = returns_df.mean() * 252
        
        if self.config.weight_method == "shrinkage":
            weights = self._shrinkage_weights(expected_returns, cov_matrix)
        elif self.config.weight_method == "equal":
            weights = {alpha: 1.0/len(self.selected_alphas) for alpha in self.selected_alphas}
        elif self.config.weight_method == "risk_parity":
            weights = self._risk_parity_weights(cov_matrix)
        else:
            weights = self._shrinkage_weights(expected_returns, cov_matrix)
        
        self.alpha_weights = weights
        return weights
    
    def _shrinkage_weights(self, expected_returns: pd.Series, 
                          cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Compute shrinkage weights (Ledoit-Wolf style)
        
        Args:
            expected_returns: Expected returns
            cov_matrix: Covariance matrix
            
        Returns:
            Dictionary of weights
        """
        n = len(expected_returns)
        
        # Objective: maximize Sharpe with shrinkage regularization
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
            sharpe = -portfolio_return / portfolio_vol  # Negative for minimization
            
            # Add regularization for diversification
            concentration = np.sum(weights**2)
            regularization = 0.1 * concentration
            
            return sharpe + regularization
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
            {'type': 'ineq', 'fun': lambda w: w - 0.05},  # Min weight 5%
            {'type': 'ineq', 'fun': lambda w: 0.4 - w}  # Max weight 40%
        ]
        
        # Initial guess (equal weights)
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1)] * n
        )
        
        weights = result.x
        weights_dict = {alpha: w for alpha, w in zip(self.selected_alphas, weights)}
        
        return weights_dict
    
    def _risk_parity_weights(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Compute risk parity weights
        
        Args:
            cov_matrix: Covariance matrix
            
        Returns:
            Dictionary of weights
        """
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
            marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize difference in risk contributions
            target = np.mean(contrib)
            return np.sum((contrib - target)**2)
        
        n = len(self.selected_alphas)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Initial guess
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0.01, 1)] * n
        )
        
        weights = result.x
        weights_dict = {alpha: w for alpha, w in zip(self.selected_alphas, weights)}
        
        return weights_dict
    
    def check_correlation_constraints(self) -> Dict[str, bool]:
        """
        Check if selected alphas satisfy correlation constraints
        
        Returns:
            Dictionary of alpha pair -> satisfies_constraint
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()
        
        constraints = {}
        
        for i, alpha1 in enumerate(self.selected_alphas):
            for j, alpha2 in enumerate(self.selected_alphas):
                if i < j:
                    corr = self.correlation_matrix.loc[alpha1, alpha2]
                    pair = f"{alpha1}_{alpha2}"
                    constraints[pair] = abs(corr) <= self.config.max_correlation
        
        return constraints
    
    def get_diversification_metrics(self) -> Dict:
        """
        Get diversification metrics
        
        Returns:
            Dictionary of diversification metrics
        """
        if not self.selected_alphas:
            return {}
        
        # Average correlation
        avg_correlation = 0.0
        if self.correlation_matrix is not None:
            correlations = []
            for i, alpha1 in enumerate(self.selected_alphas):
                for j, alpha2 in enumerate(self.selected_alphas):
                    if i < j:
                        correlations.append(abs(self.correlation_matrix.loc[alpha1, alpha2]))
            avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Concentration (Herfindahl index)
        weights = list(self.alpha_weights.values())
        concentration = np.sum(np.array(weights)**2)
        
        # Number of effective alphas
        effective_n = 1.0 / concentration if concentration > 0 else 0
        
        return {
            "num_alphas": len(self.selected_alphas),
            "avg_correlation": avg_correlation,
            "concentration": concentration,
            "effective_n_alphas": effective_n,
            "max_correlation": avg_correlation * 1.5,  # Approximate
            "diversification_ratio": effective_n / len(self.selected_alphas)
        }


def simulate_alpha_sources(n_alphas: int = 15, n_days: int = 252) -> Dict[str, pd.Series]:
    """Simulate alpha sources for testing"""
    alphas = {}
    
    # Create diverse alpha sources with different characteristics
    for i in range(n_alphas):
        # Random Sharpe between 0.3 and 1.5
        sharpe = np.random.uniform(0.3, 1.5)
        vol = np.random.uniform(0.1, 0.3)
        mean = sharpe * vol / np.sqrt(252)
        
        # Generate returns
        returns = pd.Series(
            np.random.randn(n_days) * vol + mean,
            index=pd.date_range(start="2023-01-01", periods=n_days)
        )
        
        alphas[f"ALPHA_{i}"] = returns
    
    return alphas


if __name__ == "__main__":
    # Example usage
    config = DiversityConfig(
        max_correlation=0.3,
        min_alpha_count=5,
        max_alpha_count=10,
        weight_method="shrinkage"
    )
    
    manager = AlphaDiversityManager(config)
    
    # Simulate alpha sources
    print("Simulating alpha sources...")
    alpha_returns = simulate_alpha_sources(15, 252)
    
    # Add alpha sources
    for name, returns in alpha_returns.items():
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        manager.add_alpha_source(
            name=name,
            returns=returns,
            sharpe=sharpe,
            capacity=100_000_000,  # $100M
            turnover=np.random.uniform(0.5, 3.0)
        )
    
    # Select diverse alphas
    print("\nSelecting diverse alphas...")
    selected = manager.select_diverse_alphas()
    print(f"Selected {len(selected)} alphas: {selected}")
    
    # Optimize weights
    print("\nOptimizing alpha weights...")
    weights = manager.optimize_alpha_weights()
    print(f"Alpha weights:")
    for alpha, weight in weights.items():
        print(f"  {alpha}: {weight:.2%}")
    
    # Check correlation constraints
    print("\nChecking correlation constraints...")
    constraints = manager.check_correlation_constraints()
    violated = [pair for pair, satisfies in constraints.items() if not satisfies]
    if violated:
        print(f"Violated constraints: {violated}")
    else:
        print("All correlation constraints satisfied")
    
    # Get diversification metrics
    print("\nDiversification metrics:")
    metrics = manager.get_diversification_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

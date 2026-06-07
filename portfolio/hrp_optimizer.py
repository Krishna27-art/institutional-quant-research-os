"""
Hierarchical Risk Parity (HRP) Portfolio Optimizer

Based on machine-learning-for-trading (Stefan Jansen) Chapter 20
and QuantConnect Lean's HierarchicalRiskParityPortfolioConstructionModel.

Key Advantages of HRP:
- Does not require inverting the covariance matrix (unstable with high correlations)
- Uses hierarchical clustering to group similar assets
- Allocates risk within and between clusters
- More stable allocations when correlations spike
- Works well with fewer assumptions than mean-variance optimization

This is the recommended portfolio construction method for institutional
quantitative systems as it's robust to estimation errors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


@dataclass
class HRPResult:
    """Result of HRP optimization"""
    weights: Dict[str, float]  # Asset weights
    clusters: Dict[str, int]  # Cluster assignments
    linkage_matrix: np.ndarray  # Hierarchical linkage matrix
    cluster_order: List[str]  # Order of assets in dendrogram
    risk_contributions: Dict[str, float]  # Risk contribution per asset


class HRPOptimizer:
    """
    Hierarchical Risk Parity Portfolio Optimizer.
    
    Based on López de Prado's "Building Diversified Portfolios that
    Outperform Out of Sample" (2016).
    
    Algorithm:
    1. Compute correlation matrix from returns
    2. Convert correlation to distance matrix
    3. Perform hierarchical clustering
    4. Seriate assets to minimize cluster distance
    5. Allocate risk recursively within clusters
    6. Convert risk allocations to weight allocations
    
    This method is more robust than mean-variance optimization because:
    - It doesn't require expected returns (hard to estimate)
    - It doesn't invert the covariance matrix (unstable)
    - It naturally handles highly correlated assets
    """
    
    def __init__(
        self,
        min_weight: float = 0.01,  # Minimum 1% per asset
        max_weight: float = 0.25,  # Maximum 25% per asset
        risk_parity: bool = True,  # Use risk parity within clusters
        linkage_method: str = 'ward',  # Linkage method for clustering
        distance_metric: str = 'euclidean'  # Distance metric
    ):
        """
        Initialize HRP optimizer.
        
        Args:
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            risk_parity: If True, use risk parity within clusters
            linkage_method: Linkage method ('ward', 'single', 'complete', 'average')
            distance_metric: Distance metric ('euclidean', 'correlation')
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_parity = risk_parity
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
    
    def get_correlation_distance(self, corr: pd.DataFrame) -> pd.DataFrame:
        """
        Convert correlation matrix to distance matrix.
        
        Distance = sqrt(2 * (1 - correlation))
        
        Args:
            corr: Correlation matrix
            
        Returns:
            Distance matrix
        """
        dist = np.sqrt(2 * (1 - corr))
        return pd.DataFrame(dist, index=corr.index, columns=corr.columns)
    
    def cluster_assets(
        self,
        returns: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Perform hierarchical clustering on assets.
        
        Args:
            returns: DataFrame of returns (assets as columns)
            
        Returns:
            Tuple of (linkage_matrix, ordered_asset_list)
        """
        # Compute correlation matrix
        corr = returns.corr()
        
        # Convert to distance matrix
        if self.distance_metric == 'correlation':
            dist = self.get_correlation_distance(corr)
        else:
            # Use Euclidean distance on returns
            dist = pd.DataFrame(
                squareform(np.cdist(returns.T, returns.T, metric='euclidean')),
                index=returns.columns,
                columns=returns.columns
            )
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(
            squareform(dist.values),
            method=self.linkage_method
        )
        
        # Get ordered asset list (seriation)
        ordered_assets = self._get_seriation(linkage_matrix, dist.columns.tolist())
        
        return linkage_matrix, ordered_assets
    
    def _get_seriation(
        self,
        linkage_matrix: np.ndarray,
        leaf_order: List[str]
    ) -> List[str]:
        """
        Get optimal ordering of assets (seriation).
        
        This minimizes the sum of distances between adjacent assets
        in the dendrogram.
        
        Args:
            linkage_matrix: Linkage matrix from scipy
            leaf_order: Original order of assets
            
        Returns:
            Optimally ordered asset list
        """
        # For now, use the order from dendrogram
        # In production, could use more sophisticated seriation
        from scipy.cluster.hierarchy import leaves_list
        order = leaves_list(linkage_matrix)
        return [leaf_order[i] for i in order]
    
    def get_cluster_variances(
        self,
        cov: pd.DataFrame,
        assets: List[str]
    ) -> np.ndarray:
        """
        Calculate cluster variances for recursive bisection.
        
        Args:
            cov: Covariance matrix
            assets: List of assets in cluster
            
        Returns:
            Variance of cluster
        """
        cluster_cov = cov.loc[assets, assets]
        # Use equal weights for cluster variance
        w = np.ones(len(assets)) / len(assets)
        cluster_var = np.dot(w, np.dot(cluster_cov.values, w))
        return cluster_var
    
    def recursive_bisection(
        self,
        cov: pd.DataFrame,
        assets: List[str]
    ) -> Dict[str, float]:
        """
        Recursively bisect clusters and allocate risk.
        
        This is the core HRP algorithm:
        1. Split cluster into two sub-clusters
        2. Allocate risk based on cluster variances
        3. Recursively apply to each sub-cluster
        4. Combine allocations
        
        Args:
            cov: Covariance matrix
            assets: List of assets in current cluster
            
        Returns:
            Dictionary of asset weights
        """
        if len(assets) == 1:
            return {assets[0]: 1.0}
        
        if len(assets) == 2:
            # Simple 50/50 split for two assets
            return {assets[0]: 0.5, assets[1]: 0.5}
        
        # Split cluster into two halves
        mid = len(assets) // 2
        left_assets = assets[:mid]
        right_assets = assets[mid:]
        
        # Calculate cluster variances
        left_var = self.get_cluster_variances(cov, left_assets)
        right_var = self.get_cluster_variances(cov, right_assets)
        
        # Allocate risk inversely proportional to variance
        total_var = left_var + right_var
        left_risk = right_var / total_var  # Lower variance gets more risk
        right_risk = left_var / total_var
        
        # Recursively allocate within each cluster
        left_weights = self.recursive_bisection(cov, left_assets)
        right_weights = self.recursive_bisection(cov, right_assets)
        
        # Scale weights by risk allocation
        for asset in left_weights:
            left_weights[asset] *= left_risk
        for asset in right_weights:
            right_weights[asset] *= right_risk
        
        # Combine
        return {**left_weights, **right_weights}
    
    def optimize(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None
    ) -> HRPResult:
        """
        Perform HRP optimization.
        
        Args:
            returns: DataFrame of returns (assets as columns)
            expected_returns: Optional expected returns for tilt (not used in pure HRP)
            
        Returns:
            HRPResult with weights and metadata
        """
        assets = returns.columns.tolist()
        
        if len(assets) == 0:
            raise ValueError("No assets provided")
        
        if len(assets) == 1:
            return HRPResult(
                weights={assets[0]: 1.0},
                clusters={assets[0]: 0},
                linkage_matrix=np.array([]),
                cluster_order=assets,
                risk_contributions={assets[0]: 1.0}
            )
        
        # Compute covariance matrix
        cov = returns.cov() * 252  # Annualized
        
        # Perform clustering
        linkage_matrix, ordered_assets = self.cluster_assets(returns)
        
        # Get recursive bisection weights
        weights = self.recursive_bisection(cov, ordered_assets)
        
        # Apply weight constraints
        weights = self._apply_weight_constraints(weights)
        
        # Normalize to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(weights, cov)
        
        # Get cluster assignments (simple: cut dendrogram at 50% height)
        from scipy.cluster.hierarchy import fcluster
        max_dists = linkage_matrix[:, 2]
        if len(max_dists) > 0:
            threshold = max_dists[-1] * 0.5
            cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
            clusters = dict(zip(ordered_assets, cluster_labels))
        else:
            clusters = {asset: 0 for asset in assets}
        
        return HRPResult(
            weights=weights,
            clusters=clusters,
            linkage_matrix=linkage_matrix,
            cluster_order=ordered_assets,
            risk_contributions=risk_contributions
        )
    
    def _apply_weight_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply min/max weight constraints.
        
        Args:
            weights: Raw weights from HRP
            
        Returns:
            Constrained weights
        """
        # Clip to bounds
        constrained = {}
        for asset, weight in weights.items():
            constrained[asset] = np.clip(weight, self.min_weight, self.max_weight)
        
        return constrained
    
    def _calculate_risk_contributions(
        self,
        weights: Dict[str, float],
        cov: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset.
        
        Args:
            weights: Portfolio weights
            cov: Covariance matrix
            
        Returns:
            Dictionary of risk contributions (sum to 1)
        """
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])
        cov_matrix = cov.loc[assets, assets].values
        
        # Portfolio variance
        portfolio_var = np.dot(w, np.dot(cov_matrix, w))
        
        if portfolio_var == 0:
            return {a: 1.0 / len(assets) for a in assets}
        
        # Marginal risk contribution
        marginal_risk = np.dot(cov_matrix, w)
        
        # Risk contribution
        risk_contrib = w * marginal_risk
        
        # Normalize to sum to 1
        total_risk = risk_contrib.sum()
        if total_risk > 0:
            risk_contrib = risk_contrib / total_risk
        
        return dict(zip(assets, risk_contrib))


def get_hrp_optimizer(
    min_weight: float = 0.01,
    max_weight: float = 0.25,
    linkage_method: str = 'ward'
) -> HRPOptimizer:
    """
    Factory function to get an HRP optimizer with sensible defaults.
    
    Args:
        min_weight: Minimum weight per asset
        max_weight: Maximum weight per asset
        linkage_method: Linkage method for clustering
        
    Returns:
        HRPOptimizer instance
    """
    return HRPOptimizer(
        min_weight=min_weight,
        max_weight=max_weight,
        linkage_method=linkage_method
    )


if __name__ == "__main__":
    # Test the HRP optimizer
    print("Testing HRP Optimizer...")
    
    # Generate synthetic returns for 10 assets
    np.random.seed(42)
    n_assets = 10
    n_periods = 252
    
    # Create correlated returns
    base_returns = np.random.randn(n_periods)
    asset_returns = pd.DataFrame()
    
    for i in range(n_assets):
        # Each asset has some correlation with base plus idiosyncratic noise
        correlation = 0.3 + 0.5 * (i / n_assets)  # Varying correlation
        idiosyncratic = np.random.randn(n_periods) * 0.5
        asset_returns[f'Asset_{i}'] = (
            correlation * base_returns + 
            (1 - correlation) * idiosyncratic
        ) * 0.01  # 1% daily vol
    
    # Initialize optimizer
    optimizer = get_hrp_optimizer(
        min_weight=0.05,
        max_weight=0.30,
        linkage_method='ward'
    )
    
    # Optimize
    result = optimizer.optimize(asset_returns)
    
    print(f"\nOptimized weights for {len(result.weights)} assets:")
    for asset, weight in sorted(result.weights.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {weight:.2%}")
    
    print(f"\nRisk contributions:")
    for asset, risk in sorted(result.risk_contributions.items(), key=lambda x: -x[1]):
        print(f"  {asset}: {risk:.2%}")
    
    print(f"\nCluster assignments:")
    for asset, cluster in result.clusters.items():
        print(f"  {asset}: Cluster {cluster}")
    
    print(f"\nTotal weight: {sum(result.weights.values()):.4f}")
    print(f"Total risk contribution: {sum(result.risk_contributions.values()):.4f}")

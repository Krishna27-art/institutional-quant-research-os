"""
Institutional Capital Allocation & Inventory Engine.
Prioritizes Risk, Inventory, and Exposure constraints over pure directional prediction.
Uses cvxpy to solve for optimal allocation with advanced constraints:
- Long/Short bounded capacity
- Sector exposure limits
- 3/2 power-law transaction cost penalty
- Borrow cost penalties
"""

import numpy as np
import cvxpy as cp
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.alpha.marketplace.registry import AlphaMarketplace

logger = logging.getLogger(__name__)

@dataclass
class AllocationResult:
    alpha_id: str
    target_capital: float
    target_weight: float

class CapitalAllocationEngine:
    """Optimizes capital across multiple alphas using convex optimization."""
    
    def __init__(self, total_capital: float, max_leverage: float = 1.0):
        self.total_capital = total_capital
        self.max_leverage = max_leverage
        # Global risk constraints
        self.max_sector_exposure = 0.30
        self.target_volatility = 0.15
        self.risk_aversion = 2.0
        self.tc_lambda = 0.001  # Transaction cost penalty factor

    def allocate(
        self, 
        marketplace: AlphaMarketplace, 
        current_inventory: Dict[str, float],
        covariance_matrix: Optional[np.ndarray] = None,
        borrow_costs: Optional[np.ndarray] = None,
        sector_mappings: Optional[Dict[str, List[str]]] = None,
        previous_weights: Optional[np.ndarray] = None
    ) -> List[AllocationResult]:
        """
        Solves for optimal capital allocation using cvxpy.
        Maximizes expected return subject to inventory, capacity, and risk bounds.
        """
        active_alphas = marketplace.get_available_alphas()
        n = len(active_alphas)
        if n == 0:
            logger.warning("No active alphas available in marketplace. Retracting capital.")
            return []
            
        logger.info(f"Running cvxpy capital allocation across {n} alphas.")
        
        # Extract metadata
        expected_returns = np.array([marketplace.registered_alphas[aid].expected_sharpe for aid in active_alphas])
        capacities = np.array([marketplace.registered_alphas[aid].capacity_limit_usd for aid in active_alphas])
        capacity_weights = capacities / self.total_capital
        
        # Fallback to identity matrix if no covariance matrix provided
        if covariance_matrix is None or covariance_matrix.shape != (n, n):
            logger.warning("No valid covariance matrix provided. Falling back to identity.")
            covariance_matrix = np.eye(n)
            
        # Variables: weights for each alpha
        w = cp.Variable(n)
        
        # Objective Terms
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, covariance_matrix)
        objective_expr = portfolio_return - self.risk_aversion * portfolio_variance
        
        # Transaction Cost (3/2 power law) from Almgren-Chriss
        if previous_weights is not None and len(previous_weights) == n:
            # 3/2 power is modeled using cp.power(cp.abs(), 1.5)
            tc_penalty = self.tc_lambda * cp.sum(cp.power(cp.abs(w - previous_weights), 1.5))
            objective_expr -= tc_penalty
            
        # Borrow costs for short positions
        if borrow_costs is not None and len(borrow_costs) == n:
            borrow_penalty = borrow_costs @ cp.pos(-w)
            objective_expr -= borrow_penalty
            
        objective = cp.Maximize(objective_expr)
        
        # Base Constraints
        constraints = [
            cp.sum(cp.abs(w)) <= self.max_leverage,  # Max gross leverage
            w <= capacity_weights,                   # Long capacity limit
            w >= -capacity_weights                   # Short capacity limit
        ]
        
        # Sector/Factor constraints
        if sector_mappings:
            for sector, alpha_list in sector_mappings.items():
                sector_indices = [i for i, aid in enumerate(active_alphas) if aid in alpha_list]
                if sector_indices:
                    constraints.append(cp.sum(cp.abs(w[sector_indices])) <= self.max_sector_exposure)
        
        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            # SCS or ECOS are robust for power cones
            prob.solve(solver=cp.SCS)
        except Exception as e:
            logger.error(f"cvxpy SCS solver failed: {e}. Trying fallback.")
            try:
                prob.solve()
            except Exception as fallback_e:
                logger.error(f"Fallback solver failed: {fallback_e}")
                return []
            
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Optimization did not converge optimally. Status: {prob.status}")
            return []
            
        optimal_weights = w.value
        if optimal_weights is None:
            return []
            
        results = []
        for i, alpha_id in enumerate(active_alphas):
            weight = optimal_weights[i]
            # Prune dust (both long and short)
            if abs(weight) > 1e-4:
                results.append(AllocationResult(
                    alpha_id=alpha_id,
                    target_capital=float(weight * self.total_capital),
                    target_weight=float(weight)
                ))
                
        return results

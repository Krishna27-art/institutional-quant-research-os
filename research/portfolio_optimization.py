"""
Continuous Portfolio Optimization Engine
Optimizes alpha weights using convex optimization with transaction costs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from scipy import optimize
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    run_id: str
    timestamp: datetime
    alpha_weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    transaction_cost: float
    optimization_status: str
    solver_time_seconds: float


class PortfolioOptimizer:
    """
    Continuous Portfolio Optimization Engine
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        halflife_days: int = 20,
        transaction_cost_lambda: float = 0.1,
        max_weight_per_alpha: float = 0.3
    ):
        self.lookback_days = lookback_days
        self.halflife_days = halflife_days
        self.transaction_cost_lambda = transaction_cost_lambda
        self.max_weight_per_alpha = max_weight_per_alpha
        
        # Previous weights for transaction cost calculation
        self.previous_weights: Dict[str, float] = {}
        
        logger.info(
            f"Portfolio Optimizer initialized: lookback={lookback_days}d, "
            f"halflife={halflife_days}d, max_weight={max_weight_per_alpha}"
        )
    
    def optimize_portfolio(
        self,
        alpha_returns: pd.DataFrame,
        previous_weights: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights to maximize Sharpe ratio
        
        Args:
            alpha_returns: DataFrame of alpha returns (alpha_id x timestamp)
            previous_weights: Previous weights for transaction cost calculation
            
        Returns:
            OptimizationResult
        """
        run_id = f"OPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting portfolio optimization {run_id}")
        
        start_time = datetime.now()
        
        # Update previous weights
        if previous_weights:
            self.previous_weights = previous_weights
        
        # Calculate expected returns and covariance
        mu, Sigma = self._estimate_moments(alpha_returns)
        
        # Define optimization problem
        n_alphas = len(mu)
        alpha_names = alpha_returns.columns.tolist()
        
        # Objective function: maximize Sharpe ratio with transaction cost penalty
        def objective_function(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
            
            # Transaction cost penalty
            transaction_cost = 0.0
            if self.previous_weights:
                for i, alpha in enumerate(alpha_names):
                    if alpha in self.previous_weights:
                        transaction_cost += abs(weights[i] - self.previous_weights[alpha])
            
            # Maximize Sharpe ratio (minimize negative Sharpe)
            if portfolio_risk > 0:
                sharpe = portfolio_return / portfolio_risk
                objective = -sharpe + self.transaction_cost_lambda * transaction_cost
            else:
                objective = 1e6  # Large penalty for zero risk
            
            return objective
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]
        
        # Bounds: 0 <= w_i <= max_weight
        bounds = [(0.0, self.max_weight_per_alpha) for _ in range(n_alphas)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_alphas) / n_alphas
        
        # Optimize
        try:
            result = optimize.minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                optimization_status = "success"
            else:
                # Fallback to equal risk contribution
                optimal_weights = self._equal_risk_contribution(Sigma)
                optimization_status = "fallback_erc"
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fallback to equal weights
            optimal_weights = initial_weights
            optimization_status = "fallback_equal"
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, mu)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(Sigma, optimal_weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0
        
        # Calculate transaction cost
        transaction_cost = 0.0
        if self.previous_weights:
            for i, alpha in enumerate(alpha_names):
                if alpha in self.previous_weights:
                    transaction_cost += abs(optimal_weights[i] - self.previous_weights[alpha])
        
        # Convert to dictionary
        weights_dict = {alpha: optimal_weights[i] for i, alpha in enumerate(alpha_names)}
        
        # Update previous weights
        self.previous_weights = weights_dict
        
        solver_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            run_id=run_id,
            timestamp=datetime.now(),
            alpha_weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            transaction_cost=transaction_cost,
            optimization_status=optimization_status,
            solver_time_seconds=solver_time
        )
        
        logger.info(
            f"Optimization complete: Sharpe={sharpe_ratio:.4f}, "
            f"Risk={portfolio_risk:.4f}, Status={optimization_status}"
        )
        
        return result
    
    def _estimate_moments(
        self,
        returns: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate expected returns and covariance matrix
        
        Args:
            returns: DataFrame of returns
            
        Returns:
            Tuple of (expected returns, covariance matrix)
        """
        # Calculate exponential weights
        dates = returns.index
        n = len(dates)
        
        # Exponential decay weights
        decay_factor = np.log(2) / self.halflife_days
        weights = np.exp(-decay_factor * np.arange(n)[::-1])
        weights = weights / weights.sum()
        
        # Calculate weighted mean returns
        mu = returns.apply(lambda x: np.average(x, weights=weights)).values
        
        # Calculate weighted covariance with Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        Sigma = lw.fit(returns.values).covariance_
        
        return mu, Sigma
    
    def _equal_risk_contribution(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Calculate equal risk contribution weights as fallback
        
        Args:
            Sigma: Covariance matrix
            
        Returns:
            Optimal weights for equal risk contribution
        """
        n = Sigma.shape[0]
        
        def risk_budget_objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
            marginal_risk = np.dot(Sigma, weights) / portfolio_risk
            risk_contribution = weights * marginal_risk
            target_risk = portfolio_risk / n
            
            return np.sum((risk_contribution - target_risk) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.max_weight_per_alpha) for _ in range(n)]
        initial_weights = np.ones(n) / n
        
        result = optimize.minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x if result.success else initial_weights
    
    def optimize_with_regime_adjustment(
        self,
        alpha_returns: pd.DataFrame,
        regime_weights: Dict[str, float],
        previous_weights: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio with regime-specific adjustment
        
        Args:
            alpha_returns: DataFrame of alpha returns
            regime_weights: Regime-specific alpha weights
            previous_weights: Previous weights
            
        Returns:
            OptimizationResult
        """
        # First, optimize normally
        result = self.optimize_portfolio(alpha_returns, previous_weights)
        
        # Adjust weights based on regime
        adjusted_weights = {}
        for alpha, weight in result.alpha_weights.items():
            regime_factor = regime_weights.get(alpha, 1.0)
            adjusted_weights[alpha] = weight * regime_factor
        
        # Re-normalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        # Update result
        result.alpha_weights = adjusted_weights
        
        logger.info(f"Applied regime adjustment to portfolio weights")
        
        return result
    
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics given weights
        
        Args:
            weights: Alpha weights
            returns: Alpha returns
            
        Returns:
            Dictionary of portfolio metrics
        """
        # Convert weights to array
        alpha_names = returns.columns.tolist()
        weight_array = np.array([weights.get(alpha, 0.0) for alpha in alpha_names])
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weight_array)
        
        # Calculate metrics
        metrics = {
            'total_return': portfolio_returns.sum(),
            'mean_return': portfolio_returns.mean(),
            'std_return': portfolio_returns.std(),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'win_rate': (portfolio_returns > 0).sum() / len(portfolio_returns),
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def save_optimization_result(self, result: OptimizationResult, save_path: str) -> None:
        """Save optimization result to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            'run_id': result.run_id,
            'timestamp': result.timestamp.isoformat(),
            'alpha_weights': result.alpha_weights,
            'expected_return': result.expected_return,
            'expected_risk': result.expected_risk,
            'sharpe_ratio': result.sharpe_ratio,
            'transaction_cost': result.transaction_cost,
            'optimization_status': result.optimization_status,
            'solver_time_seconds': result.solver_time_seconds,
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved optimization result to {save_path}")
    
    def load_optimization_result(self, load_path: str) -> OptimizationResult:
        """Load optimization result from file"""
        with open(load_path, 'r') as f:
            result_dict = json.load(f)
        
        result = OptimizationResult(
            run_id=result_dict['run_id'],
            timestamp=datetime.fromisoformat(result_dict['timestamp']),
            alpha_weights=result_dict['alpha_weights'],
            expected_return=result_dict['expected_return'],
            expected_risk=result_dict['expected_risk'],
            sharpe_ratio=result_dict['sharpe_ratio'],
            transaction_cost=result_dict['transaction_cost'],
            optimization_status=result_dict['optimization_status'],
            solver_time_seconds=result_dict['solver_time_seconds']
        )
        
        logger.info(f"Loaded optimization result from {load_path}")
        
        return result


def simulate_portfolio_optimization():
    """Simulate portfolio optimization"""
    
    print("="*60)
    print("PORTFOLIO OPTIMIZATION ENGINE SIMULATION")
    print("="*60)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(
        lookback_days=60,
        halflife_days=20,
        transaction_cost_lambda=0.1,
        max_weight_per_alpha=0.3
    )
    
    # Generate sample alpha returns
    print("\n1. Generating sample alpha returns...")
    np.random.seed(42)
    dates = pd.date_range(datetime(2024, 1, 1), periods=60, freq='D')
    alpha_names = ['ORB', 'VWAP', 'MEAN_REVERSION', 'MOMENTUM']
    
    alpha_returns = pd.DataFrame(
        np.random.multivariate_normal(
            [0.001, 0.0008, 0.0012, 0.0009],
            [[0.0004, 0.0002, 0.0001, 0.00015],
             [0.0002, 0.0003, 0.00015, 0.0001],
             [0.0001, 0.00015, 0.0005, 0.0002],
             [0.00015, 0.0001, 0.0002, 0.00035]]
        ),
        index=dates,
        columns=alpha_names
    )
    
    print(f"  Generated returns for {len(alpha_names)} alphas over {len(dates)} days")
    
    # Optimize portfolio
    print("\n2. Optimizing portfolio...")
    result = optimizer.optimize_portfolio(alpha_returns)
    
    print(f"  Run ID: {result.run_id}")
    print(f"  Optimization status: {result.optimization_status}")
    print(f"  Solver time: {result.solver_time_seconds:.4f}s")
    
    # Show weights
    print("\n3. Optimal weights:")
    for alpha, weight in result.alpha_weights.items():
        print(f"  {alpha}: {weight:.2%}")
    
    # Show metrics
    print("\n4. Portfolio metrics:")
    print(f"  Expected return: {result.expected_return:.4%}")
    print(f"  Expected risk: {result.expected_risk:.4%}")
    print(f"  Sharpe ratio: {result.sharpe_ratio:.4f}")
    print(f"  Transaction cost: {result.transaction_cost:.4f}")
    
    # Calculate actual portfolio metrics
    print("\n5. Calculating actual portfolio metrics...")
    metrics = optimizer.calculate_portfolio_metrics(result.alpha_weights, alpha_returns)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Regime adjustment
    print("\n6. Testing regime adjustment...")
    regime_weights = {
        'ORB': 1.5,
        'VWAP': 0.8,
        'MEAN_REVERSION': 1.2,
        'MOMENTUM': 0.5
    }
    
    result_regime = optimizer.optimize_with_regime_adjustment(
        alpha_returns,
        regime_weights
    )
    
    print("  Adjusted weights:")
    for alpha, weight in result_regime.alpha_weights.items():
        print(f"    {alpha}: {weight:.2%}")
    
    # Save result
    print("\n7. Saving optimization result...")
    optimizer.save_optimization_result(result, "data/portfolio_optimization_result.json")
    print("  Saved to data/portfolio_optimization_result.json")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_portfolio_optimization()

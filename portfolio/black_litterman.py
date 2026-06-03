"""
Black-Litterman Portfolio Construction
Integrates subjective views with market equilibrium.

Critical for institutional portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize


@dataclass
class View:
    """Subjective view on asset returns"""
    assets: List[str]  # Assets involved in view
    pick: List[float]  # Relative weights (sum to 1)
    expected_return: float  # Expected excess return
    confidence: float  # Confidence in view (0-1)


@dataclass
class BlackLittermanConfig:
    """Configuration for Black-Litterman model"""
    # Market parameters
    risk_free_rate: float = 0.06  # 6% for India
    market_return: float = 0.12  # 12% expected market return
    market_volatility: float = 0.18  # 18% market volatility
    
    # View parameters
    tau: float = 0.05  # Uncertainty scaling parameter
    view_confidence: float = 0.5  # Default confidence for views
    
    # Optimization parameters
    max_weight: float = 0.20  # Maximum 20% per asset
    min_weight: float = 0.00  # Minimum 0% per asset
    max_leverage: float = 1.0  # Maximum 1x leverage


class BlackLittermanModel:
    """
    Black-Litterman Portfolio Model
    
    Combines market equilibrium with subjective views to create
    more stable and intuitive portfolio allocations.
    
    Formula:
    E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1Π + P'Ω^-1Q]
    
    Where:
    - Π: Equilibrium returns (CAPM)
    - P: Pick matrix (views)
    - Q: View returns
    - Ω: View uncertainty matrix
    - Σ: Covariance matrix
    - τ: Uncertainty scaling
    
    Expected Sharpe improvement: +0.1 to 0.2
    """
    
    def __init__(self, config: BlackLittermanConfig):
        self.config = config
        
        self.views: List[View] = []
        self.asset_returns: pd.DataFrame = None
        self.equilibrium_returns: pd.Series = None
        self.bl_returns: pd.Series = None
        self.optimal_weights: pd.Series = None
    
    def add_view(self, view: View):
        """Add subjective view"""
        self.views.append(view)
    
    def calculate_equilibrium_returns(self, cov_matrix: pd.DataFrame,
                                     market_caps: pd.Series) -> pd.Series:
        """
        Calculate equilibrium returns using CAPM.
        
        Formula: Π = λ * Σ * w_m
        Where λ = (E[Rm] - Rf) / σm^2
        
        Args:
            cov_matrix: Covariance matrix
            market_caps: Market capitalizations
        
        Returns:
            Equilibrium returns
        """
        # Market risk premium
        market_risk_premium = self.config.market_return - self.config.risk_free_rate
        
        # Risk aversion coefficient
        lambda_param = market_risk_premium / (self.config.market_volatility ** 2)
        
        # Market weights (proportional to market cap)
        market_weights = market_caps / market_caps.sum()
        
        # Equilibrium returns
        equilibrium_returns = lambda_param * cov_matrix @ market_weights
        
        self.equilibrium_returns = equilibrium_returns
        
        return equilibrium_returns
    
    def build_pick_matrix(self, assets: List[str]) -> np.ndarray:
        """
        Build pick matrix P from views.
        
        Args:
            assets: List of all assets
        
        Returns:
            Pick matrix P
        """
        if not self.views:
            return np.zeros((0, len(assets)))
        
        P = np.zeros((len(self.views), len(assets)))
        
        for i, view in enumerate(self.views):
            for j, asset in enumerate(assets):
                if asset in view.assets:
                    idx = view.assets.index(asset)
                    P[i, j] = view.pick[idx]
        
        return P
    
    def build_view_uncertainty_matrix(self, P: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        Build view uncertainty matrix Ω.
        
        Formula: Ω = diag(P * Σ * P') * τ
        
        Args:
            P: Pick matrix
            cov_matrix: Covariance matrix
        
        Returns:
            Uncertainty matrix Ω
        """
        n_views = P.shape[0]
        
        # Diagonal uncertainty
        omega = np.zeros((n_views, n_views))
        
        for i in range(n_views):
            # P_i * Σ * P_i'
            p_i = P[i, :]
            uncertainty = p_i @ cov_matrix.values @ p_i.T
            
            # Scale by confidence
            confidence = self.views[i].confidence if i < len(self.views) else self.config.view_confidence
            omega[i, i] = uncertainty / (confidence + 1e-6) * self.config.tau
        
        return omega
    
    def calculate_bl_returns(self, cov_matrix: pd.DataFrame,
                           market_caps: pd.Series) -> pd.Series:
        """
        Calculate Black-Litterman expected returns.
        
        Args:
            cov_matrix: Covariance matrix
            market_caps: Market capitalizations
        
        Returns:
            Black-Litterman returns
        """
        assets = cov_matrix.columns.tolist()
        
        # Calculate equilibrium returns
        pi = self.calculate_equilibrium_returns(cov_matrix, market_caps)
        
        # Build matrices
        P = self.build_pick_matrix(assets)
        Omega = self.build_view_uncertainty_matrix(P, cov_matrix)
        
        if len(self.views) == 0:
            # No views, return equilibrium returns
            self.bl_returns = pi
            return pi
        
        # View returns
        Q = np.array([view.expected_return for view in self.views])
        
        # Black-Litterman formula
        tau_sigma = self.config.tau * cov_matrix.values
        
        # Inverse matrices
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        # Combined precision
        M1 = tau_sigma_inv + P.T @ omega_inv @ P
        M1_inv = np.linalg.inv(M1)
        
        # Combined prior
        M2 = tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q
        
        # BL returns
        bl_returns = M1_inv @ M2
        
        self.bl_returns = pd.Series(bl_returns, index=assets)
        
        return self.bl_returns
    
    def optimize_portfolio(self, cov_matrix: pd.DataFrame,
                         expected_returns: pd.Series) -> pd.Series:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            cov_matrix: Covariance matrix
            expected_returns: Expected returns
        
        Returns:
            Optimal weights
        """
        n_assets = len(expected_returns)
        
        def objective(weights):
            # Maximize return - 0.5 * risk
            portfolio_return = weights @ expected_returns.values
            portfolio_risk = weights @ cov_matrix.values @ weights
            return -portfolio_return + 0.5 * portfolio_risk
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
        else:
            # Fallback to equal weights
            weights = pd.Series(np.ones(n_assets) / n_assets, index=expected_returns.index)
        
        self.optimal_weights = weights
        
        return weights
    
    def construct_portfolio(self, cov_matrix: pd.DataFrame,
                           market_caps: pd.Series) -> Dict:
        """
        Construct Black-Litterman portfolio.
        
        Args:
            cov_matrix: Covariance matrix
            market_caps: Market capitalizations
        
        Returns:
            Dictionary with portfolio details
        """
        # Calculate BL returns
        bl_returns = self.calculate_bl_returns(cov_matrix, market_caps)
        
        # Optimize portfolio
        weights = self.optimize_portfolio(cov_matrix, bl_returns)
        
        # Calculate portfolio metrics
        portfolio_return = weights @ bl_returns.values
        portfolio_risk = np.sqrt(weights @ cov_matrix.values @ weights)
        sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            "weights": weights,
            "expected_returns": bl_returns,
            "portfolio_return": portfolio_return,
            "portfolio_risk": portfolio_risk,
            "sharpe": sharpe,
            "num_views": len(self.views)
        }
    
    def generate_report(self) -> str:
        """Generate Black-Litterman report"""
        report = f"""
Black-Litterman Portfolio Report
{'=' * 50}
Risk-Free Rate: {self.config.risk_free_rate:.1%}
Market Return: {self.config.market_return:.1%}
Market Volatility: {self.config.market_volatility:.1%}
Tau (Uncertainty): {self.config.tau}
Number of Views: {len(self.views)}

Views:
{'-' * 50}
"""
        
        for i, view in enumerate(self.views):
            report += f"View {i+1}: {view.assets} -> {view.expected_return:.2%} "
            report += f"(confidence: {view.confidence:.1%})\n"
        
        if self.bl_returns is not None:
            report += f"\nEquilibrium vs BL Returns:\n{'-' * 50}\n"
            for asset in self.bl_returns.index:
                eq = self.equilibrium_returns[asset] if self.equilibrium_returns is not None else 0
                bl = self.bl_returns[asset]
                diff = bl - eq
                report += f"{asset}: Eq={eq:.2%}, BL={bl:.2%}, Diff={diff:+.2%}\n"
        
        if self.optimal_weights is not None:
            report += f"\nOptimal Weights:\n{'-' * 50}\n"
            for asset, weight in self.optimal_weights.items():
                if weight > 0.01:
                    report += f"{asset}: {weight:.2%}\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    config = BlackLittermanConfig(risk_free_rate=0.06, market_return=0.12, market_volatility=0.18)
    model = BlackLittermanModel(config)
    
    # Simulate data
    print("Simulating market data...")
    np.random.seed(42)
    n_assets = 5
    asset_names = ["NIFTY", "BANK_NIFTY", "MIDCAP", "SMALLCAP", "AUTO"]
    
    # Covariance matrix
    cov_matrix = pd.DataFrame(
        np.random.randn(n_assets, n_assets) * 0.01,
        index=asset_names,
        columns=asset_names
    )
    cov_matrix = cov_matrix @ cov_matrix.T  # Make positive semi-definite
    
    # Market caps
    market_caps = pd.Series([100, 50, 30, 20, 15], index=asset_names)
    
    # Add views
    print("Adding views...")
    model.add_view(View(
        assets=["NIFTY", "BANK_NIFTY"],
        pick=[0.6, 0.4],
        expected_return=0.08,
        confidence=0.7
    ))
    
    model.add_view(View(
        assets=["MIDCAP", "SMALLCAP"],
        pick=[0.5, 0.5],
        expected_return=0.10,
        confidence=0.5
    ))
    
    # Construct portfolio
    print("Constructing portfolio...")
    portfolio = model.construct_portfolio(cov_matrix, market_caps)
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {portfolio['portfolio_return']:.2%}")
    print(f"  Risk: {portfolio['portfolio_risk']:.2%}")
    print(f"  Sharpe: {portfolio['sharpe']:.2f}")
    
    print(model.generate_report())

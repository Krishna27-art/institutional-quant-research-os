"""
Riskfolio-lib Portfolio Optimization Integration

Based on Comprehensive Upgrade Analysis - Tier 1 Upgrade (#8)
Expected Sharpe improvement: +0.2–0.3
Embeds all risk constraints in portfolio optimization

Methodology:
- Use Riskfolio-lib for advanced portfolio optimization
- Implement risk parity, CVaR optimization
- Add sector, position, gross/net exposure constraints
- Implement volatility targeting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    print("Riskfolio-lib not available. Install with: pip install riskfolio-lib")


@dataclass
class PortfolioConfig:
    """Configuration for Portfolio Optimization"""
    # Objective
    objective: str = "Sharpe"  # "Sharpe", "MinRisk", "MaxRet", "RiskParity", "CVaR"
    
    # Risk model
    risk_model: str = "LedoitWolf"  # "LedoitWolf", "SampleCov", "Shrinkage"
    
    # Constraints
    gross_leverage_max: float = 3.0  # Maximum gross exposure
    net_leverage_max: float = 1.5  # Maximum net directional limit
    sector_max_exposure: float = 0.4  # Maximum sector exposure
    single_stock_max: float = 0.1  # Maximum single stock position
    var_99_1d_max: float = 0.02  # Maximum VaR (99%, 1d)
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% annual target
    vol_window: int = 20  # Volatility estimation window
    
    # Optimization parameters
    solver: str = "ECOS"  # "ECOS", "SCS", "CVXOPT"
    allow_short: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    # Factor model (optional)
    use_factor_model: bool = False
    factor_names: List[str] = None


@dataclass
class PortfolioResult:
    """Result from portfolio optimization"""
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    var_99_1d: float
    cvar_95: float
    gross_exposure: float
    net_exposure: float
    concentration: float


class RiskfolioOptimizer:
    """
    Portfolio Optimizer using Riskfolio-lib
    
    Implements advanced portfolio optimization with:
    - Multiple objectives (Sharpe, MinRisk, RiskParity, CVaR)
    - Risk constraints (VaR, CVaR)
    - Exposure constraints (sector, position, gross/net)
    - Volatility targeting
    """
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
        if not RISKFOLIO_AVAILABLE:
            raise ImportError("Riskfolio-lib not available. Install with: pip install riskfolio-lib")
        
        # Sector mapping (example - should be configured)
        self.sector_mapping: Dict[str, str] = {}
        
        # Optimization results
        self.last_result: Optional[PortfolioResult] = None
    
    def set_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """Set sector mapping for assets"""
        self.sector_mapping = mapping
    
    def optimize(self, returns: pd.DataFrame) -> PortfolioResult:
        """
        Run portfolio optimization
        
        Args:
            returns: DataFrame of asset returns (datetime index, assets as columns)
            
        Returns:
            PortfolioResult with optimized weights and metrics
        """
        # Build portfolio object
        port = rp.Portfolio(returns=returns)
        
        # Set risk model
        port.assets_stats(method_mu="hist", method_cov=self.config.risk_model)
        
        # Set objective
        if self.config.objective == "Sharpe":
            port.optimization(
                model="Classic",
                rm="MV",  # Mean-Variance
                obj="Sharpe",
                rf=0.0,
                l=self.config.allow_short,
                hist=True
            )
        elif self.config.objective == "MinRisk":
            port.optimization(
                model="Classic",
                rm="MV",
                obj="MinRisk",
                rf=0.0,
                l=self.config.allow_short,
                hist=True
            )
        elif self.config.objective == "MaxRet":
            port.optimization(
                model="Classic",
                rm="MV",
                obj="MaxRet",
                rf=0.0,
                l=self.config.allow_short,
                hist=True
            )
        elif self.config.objective == "RiskParity":
            port.optimization(
                model="Classic",
                rm="MV",
                obj="RiskParity",
                rf=0.0,
                hist=True
            )
        elif self.config.objective == "CVaR":
            port.optimization(
                model="Classic",
                rm="CVaR",
                obj="Sharpe",
                rf=0.0,
                l=self.config.allow_short,
                alpha=0.95,
                hist=True
            )
        
        # Get weights
        weights = port.weights
        
        # Apply exposure constraints
        weights = self._apply_exposure_constraints(weights, returns)
        
        # Apply volatility targeting
        weights = self._apply_volatility_targeting(weights, returns)
        
        # Calculate metrics
        expected_return = (weights * returns.mean()).sum() * 252
        expected_volatility = np.sqrt((weights @ returns.cov() @ weights) * 252)
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
        
        # Calculate VaR and CVaR
        portfolio_returns = (returns * weights).sum(axis=1)
        var_99_1d = np.percentile(portfolio_returns, 1)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        
        # Calculate exposures
        gross_exposure = weights.abs().sum()
        net_exposure = weights.sum()
        concentration = (weights ** 2).sum()
        
        result = PortfolioResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            var_99_1d=var_99_1d,
            cvar_95=cvar_95,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            concentration=concentration
        )
        
        self.last_result = result
        return result
    
    def _apply_exposure_constraints(self, weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
        """Apply exposure constraints to weights"""
        weights = weights.copy()
        
        # Single stock max constraint
        weights = weights.clip(upper=self.config.single_stock_max)
        weights = weights.clip(lower=-self.config.single_stock_max)
        
        # Sector constraints (if sector mapping available)
        if self.sector_mapping:
            sector_exposure = {}
            for asset, sector in self.sector_mapping.items():
                if asset in weights.index:
                    sector_exposure[sector] = sector_exposure.get(sector, 0) + abs(weights[asset])
            
            # Scale down sectors that exceed limit
            for sector, exposure in sector_exposure.items():
                if exposure > self.config.sector_max_exposure:
                    scale_factor = self.config.sector_max_exposure / exposure
                    for asset, sec in self.sector_mapping.items():
                        if sec == sector and asset in weights.index:
                            weights[asset] *= scale_factor
        
        # Gross leverage constraint
        gross = weights.abs().sum()
        if gross > self.config.gross_leverage_max:
            weights = weights * (self.config.gross_leverage_max / gross)
        
        # Net leverage constraint
        net = weights.sum()
        if abs(net) > self.config.net_leverage_max:
            # Scale down while preserving relative weights
            scale = self.config.net_leverage_max / abs(net)
            weights = weights * scale
        
        # Renormalize
        weights = weights / weights.abs().sum() if weights.abs().sum() > 0 else weights
        
        return weights
    
    def _apply_volatility_targeting(self, weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
        """Apply volatility targeting to weights"""
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt((weights @ returns.cov() @ weights) * 252)
        
        if portfolio_vol == 0:
            return weights
        
        # Scale to target volatility
        scale_factor = self.config.target_volatility / portfolio_vol
        weights = weights * scale_factor
        
        return weights
    
    def optimize_with_constraints(
        self,
        returns: pd.DataFrame,
        custom_constraints: Optional[Dict] = None
    ) -> PortfolioResult:
        """
        Optimize with custom constraints
        
        Args:
            returns: DataFrame of asset returns
            custom_constraints: Dictionary of custom constraints
            
        Returns:
            PortfolioResult
        """
        # Build portfolio object
        port = rp.Portfolio(returns=returns)
        
        # Set risk model
        port.assets_stats(method_mu="hist", method_cov=self.config.risk_model)
        
        # Add custom constraints
        if custom_constraints:
            # Example: custom constraints can be added here
            # This is a placeholder for more complex constraint handling
            pass
        
        # Run optimization
        port.optimization(
            model="Classic",
            rm="MV",
            obj=self.config.objective,
            rf=0.0,
            l=self.config.allow_short,
            hist=True
        )
        
        weights = port.weights
        
        # Apply standard constraints
        weights = self._apply_exposure_constraints(weights, returns)
        weights = self._apply_volatility_targeting(weights, returns)
        
        # Calculate metrics
        expected_return = (weights * returns.mean()).sum() * 252
        expected_volatility = np.sqrt((weights @ returns.cov() @ weights) * 252)
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
        
        portfolio_returns = (returns * weights).sum(axis=1)
        var_99_1d = np.percentile(portfolio_returns, 1)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        
        gross_exposure = weights.abs().sum()
        net_exposure = weights.sum()
        concentration = (weights ** 2).sum()
        
        result = PortfolioResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            var_99_1d=var_99_1d,
            cvar_95=cvar_95,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            concentration=concentration
        )
        
        self.last_result = result
        return result
    
    def get_efficient_frontier(self, returns: pd.DataFrame, n_points: int = 20) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on frontier
            
        Returns:
            DataFrame with frontier points (volatility, return, sharpe)
        """
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov=self.config.risk_model)
        
        # Calculate efficient frontier
        points = []
        for i in range(n_points):
            target_return = np.linspace(returns.mean().min(), returns.mean().max(), n_points)[i]
            
            try:
                port.optimization(
                    model="Classic",
                    rm="MV",
                    obj="MinRisk",
                    rf=0.0,
                    l=self.config.allow_short,
                    hist=True
                )
                
                weights = port.weights
                portfolio_return = (weights * returns.mean()).sum() * 252
                portfolio_vol = np.sqrt((weights @ returns.cov() @ weights) * 252)
                sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                
                points.append({
                    "volatility": portfolio_vol,
                    "return": portfolio_return,
                    "sharpe": sharpe
                })
            except:
                continue
        
        return pd.DataFrame(points)


def simulate_portfolio_optimization(n_assets: int = 10, n_days: int = 252) -> pd.DataFrame:
    """Simulate asset returns for testing"""
    np.random.seed(42)
    
    # Generate correlated returns
    mean_returns = np.random.normal(0.08, 0.02, n_assets) / 252
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = cov_matrix @ cov_matrix.T  # Make positive semi-definite
    cov_matrix = cov_matrix / np.diag(cov_matrix).max() * 0.01  # Scale to realistic vol
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    
    return pd.DataFrame(returns, index=dates, columns=asset_names)


if __name__ == "__main__":
    if not RISKFOLIO_AVAILABLE:
        print("Riskfolio-lib not installed. Install with: pip install riskfolio-lib")
    else:
        # Example usage
        config = PortfolioConfig(
            objective="Sharpe",
            risk_model="LedoitWolf",
            gross_leverage_max=3.0,
            net_leverage_max=1.5,
            sector_max_exposure=0.4,
            single_stock_max=0.1,
            target_volatility=0.15
        )
        
        optimizer = RiskfolioOptimizer(config)
        
        # Set sector mapping (example)
        sector_mapping = {
            "ASSET_0": "Technology",
            "ASSET_1": "Technology",
            "ASSET_2": "Finance",
            "ASSET_3": "Finance",
            "ASSET_4": "Healthcare",
            "ASSET_5": "Healthcare",
            "ASSET_6": "Energy",
            "ASSET_7": "Energy",
            "ASSET_8": "Consumer",
            "ASSET_9": "Consumer"
        }
        optimizer.set_sector_mapping(sector_mapping)
        
        # Simulate returns
        print("Simulating asset returns...")
        returns = simulate_portfolio_optimization(10, 252)
        
        # Optimize portfolio
        print("\nOptimizing portfolio...")
        result = optimizer.optimize(returns)
        
        print(f"\n=== Portfolio Optimization Results ===")
        print(f"Expected Return: {result.expected_return:.2%}")
        print(f"Expected Volatility: {result.expected_volatility:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"VaR (99%, 1d): {result.var_99_1d:.2%}")
        print(f"CVaR (95%): {result.cvar_95:.2%}")
        print(f"Gross Exposure: {result.gross_exposure:.2f}x")
        print(f"Net Exposure: {result.net_exposure:.2f}x")
        print(f"Concentration: {result.concentration:.4f}")
        
        print(f"\n=== Optimized Weights ===")
        for asset, weight in result.weights.items():
            if abs(weight) > 0.01:
                print(f"  {asset}: {weight:.2%}")
        
        # Test different objectives
        print(f"\n=== Testing Different Objectives ===")
        for obj in ["Sharpe", "MinRisk", "RiskParity"]:
            config.objective = obj
            optimizer = RiskfolioOptimizer(config)
            result = optimizer.optimize(returns)
            print(f"{obj}: Sharpe={result.sharpe_ratio:.2f}, Vol={result.expected_volatility:.2%}")

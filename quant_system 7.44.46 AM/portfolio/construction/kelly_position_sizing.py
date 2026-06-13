"""
Kelly Position Sizing (Capped)

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#14)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Kelly criterion for growth-optimal position sizing
- Capped to prevent over-betting
- Fractional Kelly for risk management
- Multi-asset Kelly portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import minimize


@dataclass
class KellyConfig:
    """Configuration for Kelly Position Sizing"""
    # Kelly parameters
    kelly_fraction: float = 0.5  # Fractional Kelly (0.5 = half-Kelly)
    max_position_pct: float = 0.10  # Maximum 10% per position
    max_total_leverage: float = 1.5  # Maximum 1.5x total leverage
    
    # Estimation parameters
    return_window: int = 252  # 1 year for return estimation
    min_observations: int = 60  # Minimum observations for estimation
    
    # Risk management
    enable_drawdown_limit: bool = True
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    enable_volatility_scaling: bool = True
    volatility_target: float = 0.15  # 15% annual volatility target
    
    # Multi-asset parameters
    use_full_kelly: bool = False  # Full Kelly portfolio optimization
    correlation_window: int = 126  # 6 months for correlation estimation


class KellyPositionSizer:
    """
    Kelly Position Sizing with Caps
    
    Implements the Kelly criterion for optimal position sizing
    with safety caps to prevent over-betting.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: KellyConfig):
        self.config = config
        
        # Position history
        self.position_history: List[Dict] = []
        
        # Current state
        self.current_positions: Dict[str, float] = {}
        self.current_leverage: float = 0.0
    
    def calculate_single_asset_kelly(self, returns: pd.Series) -> float:
        """
        Calculate Kelly fraction for single asset
        
        Formula: f* = μ/σ²
        
        Args:
            returns: Historical returns
            
        Returns:
            Kelly fraction (capped)
        """
        if len(returns) < self.config.min_observations:
            return 0.0
        
        # Calculate mean and variance
        mu = returns.mean()
        sigma2 = returns.var()
        
        if sigma2 == 0:
            return 0.0
        
        # Kelly fraction
        kelly = mu / sigma2
        
        # Apply fractional Kelly
        kelly *= self.config.kelly_fraction
        
        # Cap position
        kelly = np.clip(kelly, 0, self.config.max_position_pct)
        
        return kelly
    
    def calculate_multi_asset_kelly(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Kelly fractions for multiple assets
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of asset -> Kelly fraction
        """
        if self.config.use_full_kelly:
            return self._calculate_full_kelly(returns)
        else:
            # Simple single-asset Kelly for each asset
            kelly_fractions = {}
            for asset in returns.columns:
                kelly_fractions[asset] = self.calculate_single_asset_kelly(returns[asset])
            
            # Normalize to respect total leverage limit
            total = sum(kelly_fractions.values())
            if total > self.config.max_total_leverage:
                scale = self.config.max_total_leverage / total
                kelly_fractions = {k: v * scale for k, v in kelly_fractions.items()}
            
            return kelly_fractions
    
    def _calculate_full_kelly(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate full Kelly portfolio (multi-asset optimization)
        
        Solves: maximize w'μ - 0.5 * w'Σw
        subject to sum(w) ≤ max_leverage
        """
        if len(returns) < self.config.min_observations:
            return {asset: 0.0 for asset in returns.columns}
        
        mu = returns.mean().values
        Sigma = returns.cov().values
        
        n_assets = len(returns.columns)
        
        def objective(w):
            return -np.dot(w, mu) + 0.5 * np.dot(w, np.dot(Sigma, w))
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda w: self.config.max_total_leverage - np.sum(w)},
            {'type': 'ineq', 'fun': lambda w: w}  # Non-negative weights
        ]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) * (self.config.max_total_leverage / n_assets)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, self.config.max_position_pct)] * n_assets
        )
        
        if result.success:
            weights = result.x * self.config.kelly_fraction
            return dict(zip(returns.columns, weights))
        else:
            # Fallback to equal weights
            equal_weight = self.config.max_total_leverage / n_assets * self.config.kelly_fraction
            return {asset: equal_weight for asset in returns.columns}
    
    def scale_by_volatility(self, kelly_fractions: Dict[str, float], 
                           returns: pd.DataFrame) -> Dict[str, float]:
        """
        Scale Kelly fractions by volatility targeting
        
        Args:
            kelly_fractions: Raw Kelly fractions
            returns: Historical returns
            
        Returns:
            Volatility-scaled fractions
        """
        if not self.config.enable_volatility_scaling:
            return kelly_fractions
        
        scaled_fractions = {}
        
        for asset, fraction in kelly_fractions.items():
            if asset in returns.columns:
                asset_returns = returns[asset]
                vol = asset_returns.std() * np.sqrt(252)
                
                if vol > 0:
                    scale = self.config.volatility_target / vol
                    scaled_fractions[asset] = fraction * scale
                else:
                    scaled_fractions[asset] = fraction
            else:
                scaled_fractions[asset] = fraction
        
        # Renormalize
        total = sum(scaled_fractions.values())
        if total > self.config.max_total_leverage:
            scale = self.config.max_total_leverage / total
            scaled_fractions = {k: v * scale for k, v in scaled_fractions.items()}
        
        return scaled_fractions
    
    def apply_drawdown_limit(self, kelly_fractions: Dict[str, float], 
                           current_drawdown: float) -> Dict[str, float]:
        """
        Reduce positions if drawdown exceeds limit
        
        Args:
            kelly_fractions: Kelly fractions
            current_drawdown: Current portfolio drawdown
            
        Returns:
            Adjusted fractions
        """
        if not self.config.enable_drawdown_limit:
            return kelly_fractions
        
        if current_drawdown > self.config.max_drawdown_pct:
            # Scale down positions proportionally to drawdown
            scale = 1.0 - (current_drawdown - self.config.max_drawdown_pct) / self.config.max_drawdown_pct
            scale = max(scale, 0.5)  # Don't reduce below 50%
            
            return {k: v * scale for k, v in kelly_fractions.items()}
        
        return kelly_fractions
    
    def calculate_positions(self, returns: pd.DataFrame, 
                          current_drawdown: float = 0.0,
                          portfolio_value: float = 1_000_000) -> Dict[str, float]:
        """
        Calculate optimal positions using Kelly criterion
        
        Args:
            returns: Historical returns
            current_drawdown: Current portfolio drawdown
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary of asset -> position size (in currency)
        """
        # Calculate Kelly fractions
        kelly_fractions = self.calculate_multi_asset_kelly(returns)
        
        # Apply volatility scaling
        kelly_fractions = self.scale_by_volatility(kelly_fractions, returns)
        
        # Apply drawdown limit
        kelly_fractions = self.apply_drawdown_limit(kelly_fractions, current_drawdown)
        
        # Convert to position sizes
        positions = {asset: fraction * portfolio_value for asset, fraction in kelly_fractions.items()}
        
        # Update state
        self.current_positions = positions
        self.current_leverage = sum(kelly_fractions.values())
        
        # Record history
        self.position_history.append({
            "timestamp": datetime.now(),
            "positions": positions.copy(),
            "leverage": self.current_leverage,
            "kelly_fractions": kelly_fractions.copy()
        })
        
        return positions
    
    def get_kelly_metrics(self) -> Dict:
        """Get Kelly sizing metrics"""
        if not self.position_history:
            return {}
        
        latest = self.position_history[-1]
        
        return {
            "current_leverage": self.current_leverage,
            "num_positions": len(self.current_positions),
            "max_position_pct": max(self.current_positions.values()) / sum(self.current_positions.values()) if self.current_positions else 0,
            "position_history_length": len(self.position_history)
        }


def simulate_returns(n_assets: int = 10, n_days: int = 252) -> pd.DataFrame:
    """Simulate asset returns for testing"""
    np.random.seed(42)
    
    # Generate correlated returns
    mean_returns = np.random.normal(0.08, 0.02, n_assets) / 252
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = cov_matrix @ cov_matrix.T
    cov_matrix = cov_matrix / np.diag(cov_matrix).max() * 0.01
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    return pd.DataFrame(returns, index=dates, columns=asset_names)


if __name__ == "__main__":
    # Example usage
    config = KellyConfig(
        kelly_fraction=0.5,
        max_position_pct=0.10,
        max_total_leverage=1.5,
        use_full_kelly=False
    )
    
    sizer = KellyPositionSizer(config)
    
    # Simulate returns
    print("Simulating asset returns...")
    returns = simulate_returns(10, 252)
    
    # Calculate positions
    print("\nCalculating Kelly positions...")
    positions = sizer.calculate_positions(returns, current_drawdown=0.05, portfolio_value=1_000_000)
    
    print(f"\nOptimal Positions:")
    for asset, size in positions.items():
        if size > 1000:
            print(f"  {asset}: ₹{size:,.0f}")
    
    # Get metrics
    metrics = sizer.get_kelly_metrics()
    print(f"\nKelly Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test drawdown limit
    print(f"\nTesting drawdown limit...")
    positions_dd = sizer.calculate_positions(returns, current_drawdown=0.20, portfolio_value=1_000_000)
    print(f"  Positions with 20% drawdown:")
    for asset, size in positions_dd.items():
        if size > 1000:
            print(f"    {asset}: ₹{size:,.0f}")

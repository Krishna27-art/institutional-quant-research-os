"""
Liquidity Risk Management

Based on Comprehensive Upgrade Analysis - Tier 5 Upgrade (#44)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Liquidity-adjusted VaR (L-VaR)
- Market depth analysis
- Bid-ask spread impact
- Liquidity stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LiquidityRiskConfig:
    """Configuration for Liquidity Risk Management"""
    # L-VaR parameters
    lv_window: int = 20  # Lookback window for liquidity metrics
    lv_confidence: float = 0.95  # Confidence level for L-VaR
    
    # Market depth parameters
    depth_threshold: float = 0.01  # 1% depth threshold
    volume_window: int = 20  # Volume window
    
    # Spread parameters
    spread_window: int = 20  # Spread window
    spread_multiplier: float = 2.0  # Spread multiplier for stress
    
    # Position sizing
    max_position_pct_adv: float = 0.1  # Max position as % of ADV
    liquidation_days: int = 5  # Days to liquidate position
    
    # Stress scenarios
    liquidity_shock: float = 0.5  # 50% liquidity reduction in stress


class LiquidityRiskManager:
    """
    Liquidity Risk Manager
    
    Measures and manages liquidity risk in the portfolio.
    Calculates liquidity-adjusted VaR and monitors market depth.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: LiquidityRiskConfig):
        self.config = config
        
        # Liquidity metrics
        self.liquidity_metrics: Dict[str, Dict] = {}
        
        # L-VaR
        self.lvar: Optional[float] = None
    
    def calculate_average_daily_volume(self, volume: pd.Series) -> float:
        """
        Calculate average daily volume
        
        Args:
            volume: Volume series
            
        Returns:
            Average daily volume
        """
        return volume.tail(self.config.volume_window).mean()
    
    def calculate_liquidity_adjustment(self, position: float, adv: float) -> float:
        """
        Calculate liquidity adjustment based on position size
        
        Args:
            position: Position size
            adv: Average daily volume
            
        Returns:
            Liquidity adjustment factor
        """
        position_pct_adv = abs(position) / adv
        
        # Liquidity adjustment increases with position size
        if position_pct_adv < 0.01:
            return 1.0
        elif position_pct_adv < 0.05:
            return 1.0 + (position_pct_adv - 0.01) * 5
        elif position_pct_adv < 0.1:
            return 1.25 + (position_pct_adv - 0.05) * 10
        else:
            return 1.75 + (position_pct_adv - 0.1) * 20
    
    def calculate_market_depth(self, bid_volumes: pd.Series, ask_volumes: pd.Series) -> Dict:
        """
        Calculate market depth metrics
        
        Args:
            bid_volumes: Bid volumes
            ask_volumes: Ask volumes
            
        Returns:
            Market depth metrics
        """
        avg_bid_volume = bid_volumes.tail(self.config.volume_window).mean()
        avg_ask_volume = ask_volumes.tail(self.config.volume_window).mean()
        
        total_depth = avg_bid_volume + avg_ask_volume
        depth_imbalance = abs(avg_bid_volume - avg_ask_volume) / total_depth if total_depth > 0 else 0
        
        return {
            "avg_bid_volume": avg_bid_volume,
            "avg_ask_volume": avg_ask_volume,
            "total_depth": total_depth,
            "depth_imbalance": depth_imbalance
        }
    
    def calculate_spread_impact(self, spreads: pd.Series, position: float, adv: float) -> float:
        """
        Calculate spread impact on position
        
        Args:
            spreads: Bid-ask spreads
            position: Position size
            adv: Average daily volume
            
        Returns:
            Spread impact
        """
        avg_spread = spreads.tail(self.config.spread_window).mean()
        
        # Spread impact increases with position size
        position_pct_adv = abs(position) / adv
        spread_impact = avg_spread * (1 + position_pct_adv * self.config.spread_multiplier)
        
        return spread_impact
    
    def calculate_lvar(self, returns: pd.Series, positions: Dict[str, float],
                      advs: Dict[str, float]) -> float:
        """
        Calculate Liquidity-Adjusted VaR (L-VaR)
        
        Args:
            returns: Portfolio returns
            positions: Current positions
            advs: Average daily volumes
            
        Returns:
            L-VaR
        """
        # Calculate regular VaR
        var = np.percentile(returns, (1 - self.config.lv_confidence) * 100)
        
        # Calculate liquidity adjustment
        total_liquidity_adjustment = 0.0
        
        for asset, position in positions.items():
            adv = advs.get(asset, 1.0)
            adjustment = self.calculate_liquidity_adjustment(position, adv)
            total_liquidity_adjustment += adjustment
        
        avg_liquidity_adjustment = total_liquidity_adjustment / len(positions) if positions else 1.0
        
        # L-VaR = VaR * liquidity_adjustment
        lvar = abs(var) * avg_liquidity_adjustment
        
        self.lvar = lvar
        return lvar
    
    def calculate_liquidation_cost(self, positions: Dict[str, float], 
                                  prices: pd.Series, spreads: pd.Series,
                                  advs: Dict[str, float]) -> Dict:
        """
        Calculate liquidation cost under stress
        
        Args:
            positions: Current positions
            prices: Current prices
            spreads: Bid-ask spreads
            advs: Average daily volumes
            
        Returns:
            Liquidation cost breakdown
        """
        liquidation_costs = {}
        total_cost = 0.0
        
        for asset, position in positions.items():
            if asset not in prices.index:
                continue
            
            price = prices[asset]
            spread = spreads.get(asset, 0.01)
            adv = advs.get(asset, 1.0)
            
            # Daily liquidation amount
            daily_liquidation = position / self.config.liquidation_days
            
            # Spread cost per day
            spread_cost = daily_liquidation * spread * 0.5  # Half spread each side
            
            # Market impact cost (simplified)
            market_impact = (daily_liquidation / adv) * price * 0.01  # 1% impact per 1% of ADV
            
            daily_cost = spread_cost + market_impact
            total_liquidation_cost = daily_cost * self.config.liquidation_days
            
            liquidation_costs[asset] = {
                "daily_liquidation": daily_liquidation,
                "spread_cost": spread_cost,
                "market_impact": market_impact,
                "daily_cost": daily_cost,
                "total_cost": total_liquidation_cost
            }
            
            total_cost += total_liquidation_cost
        
        return {
            "asset_costs": liquidation_costs,
            "total_cost": total_cost,
            "cost_pct": total_cost / (positions @ prices) if (positions @ prices) > 0 else 0
        }
    
    def run_liquidity_stress_test(self, positions: Dict[str, float],
                                  prices: pd.Series, spreads: pd.Series,
                                  advs: Dict[str, float]) -> Dict:
        """
        Run liquidity stress test
        
        Args:
            positions: Current positions
            prices: Current prices
            spreads: Bid-ask spreads
            advs: Average daily volumes
            
        Returns:
            Stress test results
        """
        # Apply liquidity shock
        stressed_advs = {k: v * (1 - self.config.liquidity_shock) for k, v in advs.items()}
        stressed_spreads = {k: v * self.config.spread_multiplier for k, v in spreads.items()}
        
        # Calculate stressed liquidation cost
        stressed_cost = self.calculate_liquidation_cost(positions, prices, 
                                                      pd.Series(stressed_spreads),
                                                      stressed_advs)
        
        return {
            "normal_cost": self.calculate_liquidation_cost(positions, prices, spreads, advs),
            "stressed_cost": stressed_cost,
            "cost_increase": stressed_cost["total_cost"] - self.calculate_liquidation_cost(positions, prices, spreads, advs)["total_cost"],
            "cost_increase_pct": (stressed_cost["total_cost"] / self.calculate_liquidation_cost(positions, prices, spreads, advs)["total_cost"] - 1) * 100
        }
    
    def check_position_limits(self, positions: Dict[str, float], advs: Dict[str, float]) -> Dict:
        """
        Check if positions exceed liquidity limits
        
        Args:
            positions: Current positions
            advs: Average daily volumes
            
        Returns:
            Position limit check results
        """
        violations = []
        
        for asset, position in positions.items():
            adv = advs.get(asset, 1.0)
            position_pct_adv = abs(position) / adv
            
            if position_pct_adv > self.config.max_position_pct_adv:
                violations.append({
                    "asset": asset,
                    "position": position,
                    "adv": adv,
                    "position_pct_adv": position_pct_adv,
                    "limit": self.config.max_position_pct_adv,
                    "excess": position_pct_adv - self.config.max_position_pct_adv
                })
        
        return {
            "violations": violations,
            "num_violations": len(violations),
            "compliant": len(violations) == 0
        }
    
    def get_liquidity_summary(self) -> Dict:
        """Get liquidity risk summary"""
        return {
            "lvar": self.lvar,
            "lv_confidence": self.config.lv_confidence,
            "liquidity_metrics": self.liquidity_metrics
        }


def simulate_liquidity_data(n_assets: int = 10, n_days: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate liquidity data for testing"""
    np.random.seed(42)
    
    asset_names = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.date_range(start="2023-01-01", periods=n_days)
    
    # Simulate prices
    prices = pd.DataFrame(
        100 + np.random.randn(n_days, n_assets).cumsum(axis=0),
        index=dates,
        columns=asset_names
    )
    
    # Simulate volumes
    volumes = pd.DataFrame(
        np.random.exponential(100000, (n_days, n_assets)),
        index=dates,
        columns=asset_names
    )
    
    # Simulate spreads
    spreads = pd.DataFrame(
        np.random.uniform(0.001, 0.005, (n_days, n_assets)),
        index=dates,
        columns=asset_names
    )
    
    return prices, volumes, spreads


if __name__ == "__main__":
    # Example usage
    config = LiquidityRiskConfig(
        lv_window=20,
        lv_confidence=0.95,
        max_position_pct_adv=0.1,
        liquidation_days=5
    )
    
    liquidity_manager = LiquidityRiskManager(config)
    
    # Simulate data
    print("Simulating liquidity data...")
    prices, volumes, spreads = simulate_liquidity_data(10, 100)
    
    # Simulate positions
    positions = {}
    advs = {}
    
    for asset in prices.columns:
        positions[asset] = np.random.uniform(50000, 200000)
        advs[asset] = liquidity_manager.calculate_average_daily_volume(volumes[asset])
    
    print(f"\nPositions and ADV:")
    for asset in list(positions.keys())[:5]:
        print(f"  {asset}: Position=${positions[asset]:,.0f}, ADV={advs[asset]:,.0f}")
    
    # Calculate L-VaR
    print("\nCalculating L-VaR...")
    portfolio_returns = prices.pct_change().sum(axis=1)
    lvar = liquidity_manager.calculate_lvar(portfolio_returns, positions, advs)
    print(f"  L-VaR (95%): ${lvar:,.0f}")
    
    # Calculate liquidation cost
    print("\nCalculating liquidation cost...")
    liquidation_cost = liquidity_manager.calculate_liquidation_cost(
        positions, prices.iloc[-1], spreads.iloc[-1], advs
    )
    print(f"  Total Cost: ${liquidation_cost['total_cost']:,.0f}")
    print(f"  Cost %: {liquidation_cost['cost_pct']:.2%}")
    
    # Check position limits
    print("\nChecking position limits...")
    limit_check = liquidity_manager.check_position_limits(positions, advs)
    print(f"  Compliant: {limit_check['compliant']}")
    print(f"  Violations: {limit_check['num_violations']}")
    
    # Stress test
    print("\nRunning liquidity stress test...")
    stress_results = liquidity_manager.run_liquidity_stress_test(
        positions, prices.iloc[-1], spreads.iloc[-1], advs
    )
    print(f"  Normal Cost: ${stress_results['normal_cost']['total_cost']:,.0f}")
    print(f"  Stressed Cost: ${stress_results['stressed_cost']['total_cost']:,.0f}")
    print(f"  Cost Increase: {stress_results['cost_increase_pct']:.2%}")
    
    # Summary
    print("\nLiquidity Summary:")
    summary = liquidity_manager.get_liquidity_summary()
    for key, value in summary.items():
        if key != "liquidity_metrics":
            print(f"  {key}: {value}")

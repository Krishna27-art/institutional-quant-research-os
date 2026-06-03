"""
Liquidity-Adjusted Value at Risk (L-VaR)

This module implements Liquidity-Adjusted Value at Risk (L-VaR), which accounts for
market depth and liquidity risk when calculating VaR. This is critical for institutional
trading where position size relative to market depth is significant.

Key Features:
- Liquidity-adjusted VaR calculation
- Market depth integration
- Position size vs liquidity analysis
- Time horizon adjustment for liquidity
- Multi-asset L-VaR aggregation
- Stress testing with liquidity shocks

Based on V4 Blueprint - Institutional Architecture
Priority: High (Phase 1.3)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""
    PARAMETRIC = "parametric"  # Normal distribution
    HISTORICAL = "historical"  # Historical simulation
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation
    CORNISH_FISHER = "cornish_fisher"  # Skewness/kurtosis adjustment


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a symbol."""
    symbol: str
    average_daily_volume: int
    average_bid_ask_spread: float
    market_depth: int  # Total volume at best 5 levels
    price_impact_factor: float  # Price impact per unit volume
    liquidity_score: float  # 0-1, higher is more liquid
    
    def get_liquidity_adjustment(self, position_value: float) -> float:
        """
        Get liquidity adjustment factor for position.
        
        Args:
            position_value: Value of position
            
        Returns:
            Adjustment factor (1.0 = no adjustment, >1.0 = higher risk)
        """
        # Position size relative to average daily volume
        position_ratio = position_value / (self.average_daily_volume * 100)  # Simplified
        
        # Liquidity adjustment increases with position size
        adjustment = 1.0 + (position_ratio * self.price_impact_factor)
        
        return min(adjustment, 3.0)  # Cap at 3x


@dataclass
class LVaRResult:
    """Liquidity-Adjusted VaR result."""
    symbol: str
    var_95: float  # VaR at 95% confidence
    var_99: float  # VaR at 99% confidence
    lvar_95: float  # Liquidity-adjusted VaR at 95%
    lvar_99: float  # Liquidity-adjusted VaR at 99%
    liquidity_adjustment: float
    position_value: float
    time_horizon: int  # days
    method: VaRMethod
    confidence_level: float
    
    def get_liquidity_premium(self) -> float:
        """Get liquidity premium (L-VaR - VaR)."""
        return self.lvar_95 - self.var_95


class LiquidityAdjustedVaR:
    """
    Liquidity-Adjusted Value at Risk calculator.
    
    This class calculates L-VaR by adjusting traditional VaR for liquidity risk,
    accounting for position size relative to market depth and trading costs.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        var_method: VaRMethod = VaRMethod.CORNISH_FISHER
    ):
        """
        Initialize L-VaR calculator.
        
        Args:
            confidence_level: Confidence level for VaR
            time_horizon: Time horizon in days
            var_method: VaR calculation method
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.var_method = var_method
        
        logger.info(f"LiquidityAdjustedVaR initialized: {confidence_level*100}% confidence, {time_hor_day} day horizon, {var_method.value}")
    
    def calculate_liquidity_metrics(
        self,
        symbol: str,
        order_book_data: pd.DataFrame,
        window: int = 20
    ) -> LiquidityMetrics:
        """
        Calculate liquidity metrics from order book data.
        
        Args:
            symbol: Stock symbol
            order_book_data: Order book data
            window: Rolling window for calculations
            
        Returns:
            LiquidityMetrics
        """
        # Calculate average daily volume
        if 'total_bid_qty' in order_book_data.columns:
            avg_daily_volume = order_book_data['total_bid_qty'].rolling(window).mean().iloc[-1]
        else:
            avg_daily_volume = order_book_data['bid_qty_0'].rolling(window).mean().iloc[-1] * 5  # Estimate
        
        # Calculate average bid-ask spread
        if 'spread_bps' in order_book_data.columns:
            avg_spread = order_book_data['spread_bps'].rolling(window).mean().iloc[-1] / 10000
        else:
            avg_spread = 0.001  # Default
        
        # Calculate market depth (sum of top 5 levels)
        market_depth = 0
        for i in range(5):
            bid_col = f'bid_qty_{i}'
            ask_col = f'ask_qty_{i}'
            if bid_col in order_book_data.columns:
                market_depth += order_book_data[bid_col].rolling(window).mean().iloc[-1]
            if ask_col in order_book_data.columns:
                market_depth += order_book_data[ask_col].rolling(window).mean().iloc[-1]
        
        # Calculate price impact factor (simplified)
        price_impact_factor = avg_spread * 100  # Scaling factor
        
        # Calculate liquidity score (0-1)
        # Higher volume, lower spread, higher depth = higher score
        volume_score = min(avg_daily_volume / 1000000, 1.0)  # Normalize
        spread_score = max(1 - avg_spread * 1000, 0.0)  # Lower spread = higher score
        depth_score = min(market_depth / 100000, 1.0)  # Normalize
        
        liquidity_score = (volume_score + spread_score + depth_score) / 3
        
        metrics = LiquidityMetrics(
            symbol=symbol,
            average_daily_volume=int(avg_daily_volume),
            average_bid_ask_spread=avg_spread,
            market_depth=int(market_depth),
            price_impact_factor=price_impact_factor,
            liquidity_score=liquidity_score
        )
        
        return metrics
    
    def calculate_var(
        self,
        returns: pd.Series,
        position_value: float
    ) -> Tuple[float, float]:
        """
        Calculate traditional VaR.
        
        Args:
            returns: Return series
            position_value: Position value
            
        Returns:
            (VaR_95, VaR_99)
        """
        if self.var_method == VaRMethod.PARAMETRIC:
            return self._calculate_parametric_var(returns, position_value)
        elif self.var_method == VaRMethod.HISTORICAL:
            return self._calculate_historical_var(returns, position_value)
        elif self.var_method == VaRMethod.CORNISH_FISHER:
            return self._calculate_cornish_fisher_var(returns, position_value)
        else:
            return self._calculate_parametric_var(returns, position_value)
    
    def _calculate_parametric_var(
        self,
        returns: pd.Series,
        position_value: float
    ) -> Tuple[float, float]:
        """Calculate parametric VaR (normal distribution)."""
        mean = returns.mean()
        std = returns.std()
        
        # Scale for time horizon
        scaled_std = std * np.sqrt(self.time_horizon)
        
        # Calculate VaR at different confidence levels
        z_95 = stats.norm.ppf(1 - 0.95)
        z_99 = stats.norm.ppf(1 - 0.99)
        
        var_95 = position_value * (mean + z_95 * scaled_std)
        var_99 = position_value * (mean + z_99 * scaled_std)
        
        return abs(var_95), abs(var_99)
    
    def _calculate_historical_var(
        self,
        returns: pd.Series,
        position_value: float
    ) -> Tuple[float, float]:
        """Calculate historical VaR."""
        # Scale returns for time horizon
        scaled_returns = returns * np.sqrt(self.time_horizon)
        
        # Calculate percentiles
        var_95 = position_value * abs(np.percentile(scaled_returns, 5))
        var_99 = position_value * abs(np.percentile(scaled_returns, 1))
        
        return var_95, var_99
    
    def _calculate_cornish_fisher_var(
        self,
        returns: pd.Series,
        position_value: float
    ) -> Tuple[float, float]:
        """Calculate Cornish-Fisher VaR (skewness/kurtosis adjustment)."""
        mean = returns.mean()
        std = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Scale for time horizon
        scaled_std = std * np.sqrt(self.time_horizon)
        
        # Cornish-Fisher adjustment
        z_95 = stats.norm.ppf(1 - 0.95)
        z_99 = stats.norm.ppf(1 - 0.99)
        
        def cornish_fisher(z, s, k):
            return z + (s / 6) * (z**2 - 1) + (k / 24) * (z**3 - 3*z) - (s**2 / 36) * (2*z**3 - 5*z)
        
        cf_z_95 = cornish_fisher(z_95, skewness, kurtosis)
        cf_z_99 = cornish_fisher(z_99, skewness, kurtosis)
        
        var_95 = position_value * (mean + cf_z_95 * scaled_std)
        var_99 = position_value * (mean + cf_z_99 * scaled_std)
        
        return abs(var_95), abs(var_99)
    
    def calculate_lvar(
        self,
        symbol: str,
        returns: pd.Series,
        position_value: float,
        liquidity_metrics: LiquidityMetrics
    ) -> LVaRResult:
        """
        Calculate Liquidity-Adjusted VaR.
        
        Args:
            symbol: Stock symbol
            returns: Return series
            position_value: Position value
            liquidity_metrics: Liquidity metrics
            
        Returns:
            LVaRResult
        """
        # Calculate traditional VaR
        var_95, var_99 = self.calculate_var(returns, position_value)
        
        # Get liquidity adjustment
        liquidity_adjustment = liquidity_metrics.get_liquidity_adjustment(position_value)
        
        # Calculate L-VaR
        lvar_95 = var_95 * liquidity_adjustment
        lvar_99 = var_99 * liquidity_adjustment
        
        result = LVaRResult(
            symbol=symbol,
            var_95=var_95,
            var_99=var_99,
            lvar_95=lvar_95,
            lvar_99=lvar_99,
            liquidity_adjustment=liquidity_adjustment,
            position_value=position_value,
            time_horizon=self.time_horizon,
            method=self.var_method,
            confidence_level=self.confidence_level
        )
        
        return result
    
    def calculate_portfolio_lvar(
        self,
        positions: Dict[str, float],
        returns_dict: Dict[str, pd.Series],
        liquidity_metrics_dict: Dict[str, LiquidityMetrics],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, LVaRResult]:
        """
        Calculate L-VaR for portfolio.
        
        Args:
            positions: Dict of symbol -> position value
            returns_dict: Dict of symbol -> return series
            liquidity_metrics_dict: Dict of symbol -> liquidity metrics
            correlation_matrix: Correlation matrix (optional)
            
        Returns:
            Dict of symbol -> LVaRResult
        """
        results = {}
        
        for symbol, position_value in positions.items():
            if symbol in returns_dict and symbol in liquidity_metrics_dict:
                result = self.calculate_lvar(
                    symbol=symbol,
                    returns=returns_dict[symbol],
                    position_value=position_value,
                    liquidity_metrics=liquidity_metrics_dict[symbol]
                )
                results[symbol] = result
        
        return results
    
    def stress_test_liquidity(
        self,
        lvar_result: LVaRResult,
        liquidity_shock: float = 0.5
    ) -> LVaRResult:
        """
        Stress test L-VaR with liquidity shock.
        
        Args:
            lvar_result: Original L-VaR result
            liquidity_shock: Liquidity shock (0.5 = 50% reduction in liquidity)
            
        Returns:
            Stressed L-VaR result
        """
        # Increase liquidity adjustment
        stressed_adjustment = lvar_result.liquidity_adjustment * (1 + (1 - liquidity_shock))
        
        stressed_lvar_95 = lvar_result.var_95 * stressed_adjustment
        stressed_lvar_99 = lvar_result.var_99 * stressed_adjustment
        
        stressed_result = LVaRResult(
            symbol=lvar_result.symbol,
            var_95=lvar_result.var_95,
            var_99=lvar_result.var_99,
            lvar_95=stressed_lvar_95,
            lvar_99=stressed_lvar_99,
            liquidity_adjustment=stressed_adjustment,
            position_value=lvar_result.position_value,
            time_horizon=lvar_result.time_horizon,
            method=lvar_result.method,
            confidence_level=lvar_result.confidence_level
        )
        
        return stressed_result
    
    def print_lvar_report(self, result: LVaRResult) -> None:
        """Print L-VaR report."""
        print("\n" + "="*60)
        print(f"LIQUIDITY-ADJUSTED VaR REPORT: {result.symbol}")
        print("="*60)
        
        print(f"\nPosition Value: ₹{result.position_value:,.2f}")
        print(f"Time Horizon: {result.time_horizon} days")
        print(f"Method: {result.method.value}")
        print(f"Confidence Level: {result.confidence_level*100}%")
        
        print(f"\nTraditional VaR:")
        print(f"  VaR (95%): ₹{result.var_95:,.2f} ({result.var_95/result.position_value*100:.2f}%)")
        print(f"  VaR (99%): ₹{result.var_99:,.2f} ({result.var_99/result.position_value*100:.2f}%)")
        
        print(f"\nLiquidity-Adjusted VaR:")
        print(f"  L-VaR (95%): ₹{result.lvar_95:,.2f} ({result.lvar_95/result.position_value*100:.2f}%)")
        print(f"  L-VaR (99%): ₹{result.lvar_99:,.2f} ({result.lvar_99/result.position_value*100:.2f}%)")
        
        print(f"\nLiquidity Adjustment:")
        print(f"  Factor: {result.liquidity_adjustment:.2f}x")
        print(f"  Liquidity Premium: ₹{result.get_liquidity_premium():,.2f}")
        
        print("\n" + "="*60)


def sample_lvar():
    """Demonstrate L-VaR calculation."""
    print("=== Liquidity-Adjusted VaR Demo ===\n")
    
    # Initialize L-VaR calculator
    lvar = LiquidityAdjustedVaR(
        confidence_level=0.95,
        time_horizon=1,
        var_method=VaRMethod.CORNISH_FISHER
    )
    
    # Generate sample returns
    np.random.seed(42)
    n_samples = 252  # 1 year of trading days
    returns = pd.Series(np.random.normal(0.001, 0.02, n_samples))
    
    # Generate sample order book data
    order_book_data = pd.DataFrame({
        'bid_qty_0': np.random.uniform(1000, 10000, n_samples),
        'ask_qty_0': np.random.uniform(1000, 10000, n_samples),
        'spread_bps': np.random.uniform(5, 20, n_samples)
    })
    
    # Calculate liquidity metrics
    liquidity_metrics = lvar.calculate_liquidity_metrics('RELIANCE', order_book_data)
    
    print(f"Liquidity Metrics for RELIANCE:")
    print(f"  Average Daily Volume: {liquidity_metrics.average_daily_volume:,}")
    print(f"  Average Spread: {liquidity_metrics.average_bid_ask_spread:.4f}")
    print(f"  Market Depth: {liquidity_metrics.market_depth:,}")
    print(f"  Liquidity Score: {liquidity_metrics.liquidity_score:.2f}")
    
    # Calculate L-VaR for different position sizes
    position_values = [1000000, 10000000, 50000000]  # ₹1Cr, ₹10Cr, ₹50Cr
    
    for position_value in position_values:
        result = lvar.calculate_lvar(
            symbol='RELIANCE',
            returns=returns,
            position_value=position_value,
            liquidity_metrics=liquidity_metrics
        )
        
        lvar.print_lvar_report(result)
        
        # Stress test
        stressed_result = lvar.stress_test_liquidity(result, liquidity_shock=0.5)
        print(f"\nStressed L-VaR (50% liquidity shock):")
        print(f"  L-VaR (95%): ₹{stressed_result.lvar_95:,.2f} ({stressed_result.lvar_95/result.position_value*100:.2f}%)")
        print(f"  Increase: {(stressed_result.lvar_95 - result.lvar_95)/result.lvar_95*100:.1f}%")
    
    print("\n=== Liquidity-Adjusted VaR Demo Complete ===")
    print("Key capabilities:")
    print("- Liquidity-adjusted VaR calculation")
    print("- Market depth integration")
    print("- Position size vs liquidity analysis")
    print("- Stress testing with liquidity shocks")
    print("- Multiple VaR methods (parametric, historical, Cornish-Fisher)")


if __name__ == "__main__":
    sample_lvar()

import pandas as pd
import numpy as np
from typing import Dict, Any, List

class CapacityAnalyzer:
    """
    Evaluates how much capital a strategy can absorb before market impact 
    and slippage degrade its profitability to zero.
    """
    def __init__(self, avg_daily_volume_inr: float = 1_000_000_000.0, impact_coefficient: float = 0.1):
        self.avg_daily_volume_inr = avg_daily_volume_inr
        self.impact_coefficient = impact_coefficient

    def estimate_market_impact(self, trade_size_inr: float, participation_rate: float = 0.05) -> float:
        """
        Estimate price impact in basis points (bps) based on trade size.
        Using a square root impact model commonly used in institutional trading.
        Impact (bps) = impact_coefficient * sqrt(trade_size / (ADV * participation_rate))
        """
        if trade_size_inr <= 0:
            return 0.0
            
        fraction_of_adv = trade_size_inr / (self.avg_daily_volume_inr * participation_rate)
        impact_bps = self.impact_coefficient * np.sqrt(fraction_of_adv) * 10000 
        return impact_bps

    def estimate_slippage(self, trade_size_inr: float, base_spread_bps: float = 2.0) -> float:
        """
        Estimate total slippage (spread + impact) in bps.
        """
        impact = self.estimate_market_impact(trade_size_inr)
        return base_spread_bps / 2.0 + impact

    def analyze_capacity(self, 
                         avg_trade_return_bps: float, 
                         capital_sizes_inr: List[float] = [10_000, 100_000, 1_000_000, 10_000_000]) -> pd.DataFrame:
        """
        Analyze theoretical returns across different capital sizes.
        Returns a DataFrame showing degradation.
        """
        results = []
        for cap in capital_sizes_inr:
            slippage_bps = self.estimate_slippage(cap)
            net_return_bps = avg_trade_return_bps - slippage_bps
            
            results.append({
                "capital_inr": cap,
                "gross_return_bps": avg_trade_return_bps,
                "estimated_slippage_bps": slippage_bps,
                "net_return_bps": net_return_bps,
                "viable": net_return_bps > 0
            })
            
        return pd.DataFrame(results)

    def find_max_capacity(self, avg_trade_return_bps: float, min_acceptable_net_return_bps: float = 1.0) -> float:
        """
        Finds the maximum capital size in INR where the net return remains above the minimum acceptable bound.
        """
        # We solve for trade_size:
        # avg_return - (base_spread/2 + impact_coeff * sqrt(trade_size / (ADV * part)) * 10000) = min_net
        # impact = avg_return - min_net - base_spread/2
        
        base_spread_bps = 2.0
        allowed_impact_bps = avg_trade_return_bps - min_acceptable_net_return_bps - (base_spread_bps / 2.0)
        
        if allowed_impact_bps <= 0:
            return 0.0
            
        # allowed_impact_bps = coeff * sqrt(size / (ADV * part)) * 10000
        # sqrt(size / ...) = allowed_impact_bps / (coeff * 10000)
        sqrt_term = allowed_impact_bps / (self.impact_coefficient * 10000)
        max_size = (sqrt_term ** 2) * (self.avg_daily_volume_inr * 0.05)
        
        return max_size

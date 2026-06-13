"""
Unified Transaction Cost Model for Indian Markets
Matches paper trading simulator assumptions for consistency across all backtesters.

Components:
- Broker fee: 0.5 bps
- Exchange fee: 0.1 bps
- STT/CTT: 0.1 bps
- Total: 0.7 bps (simple model matching paper trading)

This unifies cost assumptions across:
- VectorizedBacktester
- ORB backtester
- VWAP backtester
- Put-Call Carry backtester
- All other backtesters
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class UnifiedCostConfig:
    """Unified transaction cost configuration matching paper trading simulator"""
    # Per debate: 2 bps slippage for large cap, 5 bps for mid cap
    slippage_large_cap_bps: float = 2.0
    slippage_mid_cap_bps: float = 5.0
    
    # Per paper trading simulator
    broker_fee_bps: float = 0.5  # 0.5 bps
    exchange_fee_bps: float = 0.1  # 0.1 bps
    stt_ctt_bps: float = 0.1  # 0.1 bps (STT/CTT)
    
    # Risk limits
    max_position_size_pct: float = 0.05  # 5% of AUM
    max_daily_loss_pct: float = -0.03  # -3% daily circuit breaker
    max_leverage: float = 4.0


class UnifiedCostModel:
    """
    Unified transaction cost model for Indian markets.
    
    Matches paper trading simulator assumptions for consistency.
    """
    
    def __init__(self, config: UnifiedCostConfig = None):
        self.config = config or UnifiedCostConfig()
        
        # Large cap symbols (from paper trading simulator)
        self.large_cap_symbols = set([
            "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
            "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"
        ])
    
    def get_slippage_bps(self, symbol: str) -> float:
        """Get slippage in basis points for a symbol."""
        if symbol in self.large_cap_symbols:
            return self.config.slippage_large_cap_bps
        else:
            return self.config.slippage_mid_cap_bps
    
    def calculate_total_fees_bps(self) -> float:
        """Calculate total fees in basis points."""
        return (
            self.config.broker_fee_bps +
            self.config.exchange_fee_bps +
            self.config.stt_ctt_bps
        )
    
    def calculate_transaction_cost(
        self,
        trade_value: float,
        symbol: str,
        is_buy: bool = True
    ) -> float:
        """
        Calculate total transaction cost for a trade.
        
        Args:
            trade_value: Value of the trade in ₹
            symbol: Stock symbol
            is_buy: True for buy, False for sell
            
        Returns:
            Total cost in ₹
        """
        # Slippage
        slippage_bps = self.get_slippage_bps(symbol)
        slippage_cost = trade_value * (slippage_bps / 10000.0)
        
        # Fees
        total_fees_bps = self.calculate_total_fees_bps()
        fees_cost = trade_value * (total_fees_bps / 10000.0)
        
        total_cost = slippage_cost + fees_cost
        
        return total_cost
    
    def calculate_round_trip_cost(
        self,
        entry_value: float,
        exit_value: float,
        symbol: str
    ) -> float:
        """
        Calculate round-trip transaction cost.
        
        Args:
            entry_value: Entry trade value in ₹
            exit_value: Exit trade value in ₹
            symbol: Stock symbol
            
        Returns:
            Total round-trip cost in ₹
        """
        entry_cost = self.calculate_transaction_cost(entry_value, symbol, is_buy=True)
        exit_cost = self.calculate_transaction_cost(exit_value, symbol, is_buy=False)
        
        return entry_cost + exit_cost


def get_unified_cost_model() -> UnifiedCostModel:
    """Get the unified cost model instance."""
    return UnifiedCostModel()


# Convenience function for backtesters
def calculate_unified_cost(
    trade_value: float,
    symbol: str,
    is_buy: bool = True
) -> float:
    """
    Calculate transaction cost using unified model.
    
    This function can be used by all backtesters to ensure consistency.
    """
    model = get_unified_cost_model()
    return model.calculate_transaction_cost(trade_value, symbol, is_buy)

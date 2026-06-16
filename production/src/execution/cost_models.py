"""
Transaction Cost Models - Market impact and transaction cost analysis
Based on Almgren et al (2005) and Kissell et al (2004) models
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MarketImpactModel:
    """
    Almgren et al (2005) Market Impact Model
    
    Models permanent and temporary market impact based on:
    - Relative order size (X/V)
    - Volatility (sigma)
    - Inverse turnover (Theta/V)
    - Trading time (T)
    """
    gamma: float = 0.314  # Universal coefficient for permanent impact
    eta: float = 0.142    # Universal coefficient for temporary impact
    
    def permanent_impact(self, pct_adv: float, annual_vol_pct: float = 0.25,
                       inv_turnover: float = 200) -> float:
        """
        Calculate permanent market impact in basis points
        
        Args:
            pct_adv: Trade as percentage of average daily volume
            annual_vol_pct: Annual volatility percentage
            inv_turnover: Inverse turnover (Theta/V)
            
        Returns:
            Permanent impact in bps
        """
        return 10000 * self.gamma * (annual_vol_pct / 16) * pct_adv * (inv_turnover) ** 0.25
    
    def temporary_impact(self, pct_adv: float, minutes: float,
                       annual_vol_pct: float = 0.25,
                       minutes_in_day: float = 60 * 6.5) -> float:
        """
        Calculate temporary market impact in basis points
        
        Args:
            pct_adv: Trade as percentage of average daily volume
            minutes: Time to execute trade in minutes
            annual_vol_pct: Annual volatility percentage
            minutes_in_day: Trading minutes in a day
            
        Returns:
            Temporary impact in bps
        """
        day_frac = minutes / minutes_in_day
        return 10000 * self.eta * (annual_vol_pct / 16) * abs(pct_adv / day_frac) ** 0.6
    
    def total_impact(self, pct_adv: float, minutes: float,
                    annual_vol_pct: float = 0.25,
                    inv_turnover: float = 200,
                    minutes_in_day: float = 60 * 6.5) -> float:
        """
        Calculate total transaction cost in basis points
        
        Args:
            pct_adv: Trade as percentage of average daily volume
            minutes: Time to execute trade in minutes
            annual_vol_pct: Annual volatility percentage
            inv_turnover: Inverse turnover (Theta/V)
            minutes_in_day: Trading minutes in a day
            
        Returns:
            Total transaction cost in bps
        """
        perm = self.permanent_impact(pct_adv, annual_vol_pct, inv_turnover)
        temp = self.temporary_impact(pct_adv, minutes, annual_vol_pct, minutes_in_day)
        return 0.5 * perm + temp


@dataclass
class TransactionCostModel:
    """
    Comprehensive transaction cost model including:
    - Commission
    - Slippage (bid-ask spread)
    - Market impact (Almgren model)
    - Implementation shortfall
    """
    commission_rate: float = 0.0005  # 5 bps
    slippage_bps: float = 2.0
    max_cost_bps: float = 20.0  # Maximum acceptable cost
    
    def __post_init__(self):
        self.impact_model = MarketImpactModel()
    
    def estimate_cost(self, symbol: str, quantity: float, price: float,
                     direction: str, market_data: Dict) -> Dict:
        """
        Estimate total transaction cost for an order
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Current price
            direction: 'BUY' or 'SELL'
            market_data: Dict with ADV, volatility, turnover, etc.
            
        Returns:
            Dict with cost breakdown
        """
        trade_value = quantity * price
        
        # Commission cost
        commission = trade_value * self.commission_rate
        commission_bps = self.commission_rate * 10000
        
        # Slippage cost (simplified bid-ask spread)
        slippage = trade_value * self.slippage_bps / 10000
        slippage_bps = self.slippage_bps
        
        # Market impact (if market data available)
        impact_bps = 0.0
        if market_data:
            adv = market_data.get('adv', 1000000)
            annual_vol = market_data.get('annual_volatility', 0.20)
            inv_turnover = market_data.get('inv_turnover', 200)
            
            pct_adv = quantity / adv
            execution_minutes = market_data.get('execution_minutes', 30)
            
            impact_bps = self.impact_model.total_impact(
                pct_adv=pct_adv,
                minutes=execution_minutes,
                annual_vol_pct=annual_vol,
                inv_turnover=inv_turnover
            )
        
        impact_cost = trade_value * impact_bps / 10000
        
        # Total cost
        total_cost = commission + slippage + impact_cost
        total_cost_bps = commission_bps + slippage_bps + impact_bps
        
        # Implementation shortfall (price drift between decision and execution)
        shortfall_bps = self._estimate_implementation_shortfall(
            market_data.get('volatility', 0.20),
            market_data.get('delay_minutes', 5)
        )
        
        return {
            'commission': commission,
            'commission_bps': commission_bps,
            'slippage': slippage,
            'slippage_bps': slippage_bps,
            'market_impact': impact_cost,
            'market_impact_bps': impact_bps,
            'total_cost': total_cost,
            'total_cost_bps': total_cost_bps,
            'implementation_shortfall_bps': shortfall_bps,
            'acceptable': total_cost_bps <= self.max_cost_bps
        }
    
    def _estimate_implementation_shortfall(self, volatility: float, 
                                        delay_minutes: float) -> float:
        """
        Estimate implementation shortfall due to price drift
        
        Args:
            volatility: Daily volatility
            delay_minutes: Delay between decision and execution
            
        Returns:
            Shortfall in bps
        """
        # Simple model: shortfall ~ volatility * sqrt(delay / trading_day)
        trading_day_minutes = 60 * 6.5
        delay_fraction = delay_minutes / trading_day_minutes
        shortfall = volatility * np.sqrt(delay_fraction) * 10000
        return shortfall


@dataclass
class VenueQuote:
    """Quote from a specific trading venue"""
    venue_id: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    latency_ms: float
    fee_per_share: float  # Negative means rebate


class SmartOrderRouter:
    """
    Smart Order Router - Route orders to optimal venues
    
    Implements routing strategies:
    - best_price: Route to venues with best prices
    - minimize_impact: Spread order to minimize market impact
    - lowest_cost: Consider fees and rebates
    """
    
    def __init__(self):
        self.venue_quotes: Dict[str, VenueQuote] = {}
    
    def update_quote(self, quote: VenueQuote) -> None:
        """Update quote for a venue"""
        self.venue_quotes[quote.venue_id] = quote
    
    def route_order(self, side: str, quantity: int,
                   strategy: str = "best_price") -> Dict:
        """
        Route order across venues
        
        Args:
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            strategy: Routing strategy
            
        Returns:
            Dict with venue allocations
        """
        if not self.venue_quotes:
            return {'allocations': [], 'strategy': strategy}
        
        if strategy == "best_price":
            allocations = self._route_best_price(side, quantity)
        elif strategy == "minimize_impact":
            allocations = self._route_minimize_impact(side, quantity)
        elif strategy == "lowest_cost":
            allocations = self._route_lowest_cost(side, quantity)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")
        
        return {
            'allocations': allocations,
            'strategy': strategy,
            'total_allocated': sum(qty for _, qty in allocations)
        }
    
    def _route_best_price(self, side: str, quantity: int) -> list:
        """Route to venues with best prices, sweeping through order book"""
        allocations = []
        remaining = quantity
        
        if side == "BUY":
            # Sort by ask price (lowest first), then by fee
            sorted_venues = sorted(
                self.venue_quotes.values(),
                key=lambda v: (v.ask_price, v.fee_per_share)
            )
            for venue in sorted_venues:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, venue.ask_size)
                if fill_qty > 0:
                    allocations.append((venue.venue_id, fill_qty))
                    remaining -= fill_qty
        else:
            # Sort by bid price (highest first)
            sorted_venues = sorted(
                self.venue_quotes.values(),
                key=lambda v: (-v.bid_price, v.fee_per_share)
            )
            for venue in sorted_venues:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, venue.bid_size)
                if fill_qty > 0:
                    allocations.append((venue.venue_id, fill_qty))
                    remaining -= fill_qty
        
        return allocations
    
    def _route_minimize_impact(self, side: str, quantity: int) -> list:
        """Spread order across venues to minimize market impact"""
        total_liquidity = sum(
            v.ask_size if side == "BUY" else v.bid_size
            for v in self.venue_quotes.values()
        )
        
        if total_liquidity == 0:
            return []
        
        allocations = []
        for venue in self.venue_quotes.values():
            venue_liquidity = venue.ask_size if side == "BUY" else venue.bid_size
            proportion = venue_liquidity / total_liquidity
            venue_qty = int(quantity * proportion)
            if venue_qty > 0:
                allocations.append((venue.venue_id, venue_qty))
        
        return allocations
    
    def _route_lowest_cost(self, side: str, quantity: int) -> list:
        """Route considering fees and rebates"""
        allocations = []
        remaining = quantity
        
        if side == "BUY":
            # Sort by (ask_price + fee) effective cost
            sorted_venues = sorted(
                self.venue_quotes.values(),
                key=lambda v: (v.ask_price + v.fee_per_share)
            )
            for venue in sorted_venues:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, venue.ask_size)
                if fill_qty > 0:
                    allocations.append((venue.venue_id, fill_qty))
                    remaining -= fill_qty
        else:
            # Sort by (bid_price - fee) effective proceeds
            sorted_venues = sorted(
                self.venue_quotes.values(),
                key=lambda v: -(v.bid_price - v.fee_per_share)
            )
            for venue in sorted_venues:
                if remaining <= 0:
                    break
                fill_qty = min(remaining, venue.bid_size)
                if fill_qty > 0:
                    allocations.append((venue.venue_id, fill_qty))
                    remaining -= fill_qty
        
        return allocations

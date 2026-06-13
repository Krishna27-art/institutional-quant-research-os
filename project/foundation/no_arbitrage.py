"""
No-Arbitrage Detectors - Level 2 Foundation

This module provides no-arbitrage detection for options:
- Put-call parity violation
- Convexity violation
- Calendar spread violation
- Butterfly spread violation
- Box spread violation

Based on Audit Report Priority 2: Asset Pricing Theories
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ArbitrageType(Enum):
    """Types of arbitrage opportunities."""
    PUT_CALL_PARITY = "put_call_parity"
    CONVEXITY = "convexity"
    CALENDAR = "calendar"
    BUTTERFLY = "butterfly"
    BOX = "box"


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data."""
    arbitrage_type: ArbitrageType
    symbol: str
    profit: float
    legs: Dict[str, Union[str, float]]
    timestamp: str
    
    def __post_init__(self):
        """Validate arbitrage opportunity."""
        if self.profit <= 0:
            raise ValueError("Arbitrage profit must be positive")


class NoArbitrageDetectors:
    """
    No-arbitrage detectors for options.
    
    This class detects arbitrage opportunities in option markets
    by checking no-arbitrage conditions.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize no-arbitrage detectors.
        
        Args:
            tolerance: Tolerance for arbitrage detection (as % of option price)
        """
        self.tolerance = tolerance
    
    def put_call_parity_violation(
        self,
        calls: Optional[Dict[float, float]] = None,
        puts: Optional[Dict[float, float]] = None,
        spot: float = 100.0,
        rate: float = 0.05,
        div: float = 0.0,
        T: float = 1.0,
        call_price: Optional[float] = None,
        put_price: Optional[float] = None,
        strike: Optional[float] = None,
        time_to_expiry: Optional[float] = None,
        risk_free_rate: Optional[float] = None
    ) -> Union[List[ArbitrageOpportunity], Dict[str, Union[bool, float]]]:
        """
        Detect put-call parity violations.
        
        Put-call parity: C - P = S - K * exp(-(r - q) * T)
        
        Args:
            calls: Dictionary of call prices {strike: price}
            puts: Dictionary of put prices {strike: price}
            spot: Current spot price
            rate: Risk-free rate
            div: Dividend yield
            T: Time to maturity (years)
            
        Returns:
            List of arbitrage opportunities, or Dict if scalar mode
        """
        if call_price is not None:
            c = call_price
            p = put_price if put_price is not None else 0.0
            k = strike if strike is not None else spot
            t = time_to_expiry if time_to_expiry is not None else T
            r = risk_free_rate if risk_free_rate is not None else rate
            
            theoretical_fwd = spot - k * np.exp(-(r - div) * t)
            synthetic_fwd = c - p
            diff = synthetic_fwd - theoretical_fwd
            abs_diff = abs(diff)
            
            violation = abs_diff > self.tolerance * c
            
            return {
                'violation': violation,
                'arbitrage_profit': abs_diff if violation else 0.0,
                'diff': diff,
            }
            
        opportunities = []
        if calls is None or puts is None:
            return opportunities
        
        # Check matching strikes
        common_strikes = set(calls.keys()) & set(puts.keys())
        
        for strike in common_strikes:
            call_price = calls[strike]
            put_price = puts[strike]
            
            if call_price <= 0 or put_price <= 0:
                continue
            
            # Calculate theoretical forward
            theoretical_fwd = spot - strike * np.exp(-(rate - div) * T)
            
            # Calculate synthetic forward from options
            synthetic_fwd = call_price - put_price
            
            # Calculate difference
            diff = synthetic_fwd - theoretical_fwd
            abs_diff = abs(diff)
            
            # Check for arbitrage (difference exceeds tolerance)
            if abs_diff > self.tolerance * call_price:
                # Determine arbitrage direction
                if diff > 0:
                    # Synthetic forward is overpriced
                    # Sell synthetic forward (sell call, buy put)
                    # Buy actual forward (buy spot, sell strike bond)
                    legs = {
                        'action': 'sell_synthetic_buy_actual',
                        'sell_call': strike,
                        'buy_put': strike,
                        'buy_spot': spot,
                        'sell_bond': strike * np.exp(-(rate - div) * T),
                    }
                else:
                    # Synthetic forward is underpriced
                    # Buy synthetic forward (buy call, sell put)
                    # Sell actual forward (sell spot, buy strike bond)
                    legs = {
                        'action': 'buy_synthetic_sell_actual',
                        'buy_call': strike,
                        'sell_put': strike,
                        'sell_spot': spot,
                        'buy_bond': strike * np.exp(-(rate - div) * T),
                    }
                
                opportunity = ArbitrageOpportunity(
                    arbitrage_type=ArbitrageType.PUT_CALL_PARITY,
                    symbol=f"K={strike}",
                    profit=abs_diff,
                    legs=legs,
                    timestamp=pd.Timestamp.now().isoformat(),
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def convexity_violation(
        self,
        calls: Dict[float, float],
        same_expiry: bool = True
    ) -> List[ArbitrageOpportunity]:
        """
        Detect convexity violations in option prices.
        
        Option prices should be convex in strike:
        C(K2) <= 0.5 * C(K1) + 0.5 * C(K3) for K1 < K2 < K3
        
        Args:
            calls: Dictionary of call prices {strike: price}
            same_expiry: Whether all options have same expiry
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        strikes = sorted(calls.keys())
        
        # Check all triplets
        for i in range(len(strikes) - 2):
            K1 = strikes[i]
            K2 = strikes[i + 1]
            K3 = strikes[i + 2]
            
            C1 = calls[K1]
            C2 = calls[K2]
            C3 = calls[K3]
            
            if C1 <= 0 or C2 <= 0 or C3 <= 0:
                continue
            
            # Check convexity
            theoretical_C2 = 0.5 * C1 + 0.5 * C3
            violation = C2 - theoretical_C2
            
            if violation > self.tolerance * C2:
                # Butterfly arbitrage
                # Sell middle strike, buy wings
                legs = {
                    'action': 'butterfly_arbitrage',
                    'sell_call': K2,
                    'buy_call_low': K1,
                    'buy_call_high': K3,
                }
                
                opportunity = ArbitrageOpportunity(
                    arbitrage_type=ArbitrageType.CONVEXITY,
                    symbol=f"K1={K1},K2={K2},K3={K3}",
                    profit=violation,
                    legs=legs,
                    timestamp=pd.Timestamp.now().isoformat(),
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def calendar_spread_violation(
        self,
        calls_near: Dict[float, float],
        calls_far: Dict[float, float],
        T_near: float,
        T_far: float,
        rate: float = 0.05
    ) -> List[ArbitrageOpportunity]:
        """
        Detect calendar spread violations.
        
        Calendar spread should be positive for same strike:
        C(T_far) - C(T_near) > 0 (for calls)
        
        Args:
            calls_near: Near-term call prices {strike: price}
            calls_far: Far-term call prices {strike: price}
            T_near: Near-term time to maturity
            T_far: Far-term time to maturity
            rate: Risk-free rate
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        # Check matching strikes
        common_strikes = set(calls_near.keys()) & set(calls_far.keys())
        
        for strike in common_strikes:
            call_near = calls_near[strike]
            call_far = calls_far[strike]
            
            if call_near <= 0 or call_far <= 0:
                continue
            
            # Calendar spread
            calendar_spread = call_far - call_near
            
            # Check for violation (far-term should be more expensive)
            if calendar_spread < -self.tolerance * call_near:
                # Sell near-term, buy far-term
                legs = {
                    'action': 'calendar_arbitrage',
                    'sell_call_near': strike,
                    'buy_call_far': strike,
                    'T_near': T_near,
                    'T_far': T_far,
                }
                
                opportunity = ArbitrageOpportunity(
                    arbitrage_type=ArbitrageType.CALENDAR,
                    symbol=f"K={strike}",
                    profit=abs(calendar_spread),
                    legs=legs,
                    timestamp=pd.Timestamp.now().isoformat(),
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def butterfly_spread_violation(
        self,
        calls: Dict[float, float],
        same_expiry: bool = True
    ) -> List[ArbitrageOpportunity]:
        """
        Detect butterfly spread violations.
        
        Butterfly spread should be non-negative:
        C(K1) - 2*C(K2) + C(K3) >= 0 for K1 < K2 < K3
        
        Args:
            calls: Dictionary of call prices {strike: price}
            same_expiry: Whether all options have same expiry
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        strikes = sorted(calls.keys())
        
        # Check all triplets
        for i in range(len(strikes) - 2):
            K1 = strikes[i]
            K2 = strikes[i + 1]
            K3 = strikes[i + 2]
            
            C1 = calls[K1]
            C2 = calls[K2]
            C3 = calls[K3]
            
            if C1 <= 0 or C2 <= 0 or C3 <= 0:
                continue
            
            # Butterfly spread
            butterfly = C1 - 2 * C2 + C3
            
            if butterfly < -self.tolerance * C2:
                # Buy butterfly
                legs = {
                    'action': 'buy_butterfly',
                    'buy_call_low': K1,
                    'sell_call_middle': K2,
                    'buy_call_high': K3,
                }
                
                opportunity = ArbitrageOpportunity(
                    arbitrage_type=ArbitrageType.BUTTERFLY,
                    symbol=f"K1={K1},K2={K2},K3={K3}",
                    profit=abs(butterfly),
                    legs=legs,
                    timestamp=pd.Timestamp.now().isoformat(),
                )
                opportunities.append(opportunity)
        
        return opportunities
    
    def box_spread_violation(
        self,
        calls: Dict[float, float],
        puts: Dict[float, float],
        strikes: Tuple[float, float],
        rate: float,
        T: float = 1.0
    ) -> List[ArbitrageOpportunity]:
        """
        Detect box spread violations.
        
        Box spread: C(K1) - C(K2) + P(K2) - P(K1) should equal (K2 - K1) * exp(-r * T)
        
        Args:
            calls: Dictionary of call prices {strike: price}
            puts: Dictionary of put prices {strike: price}
            strikes: Tuple of (K1, K2) where K1 < K2
            rate: Risk-free rate
            T: Time to maturity (years)
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        K1, K2 = strikes
        
        if K1 not in calls or K1 not in puts or K2 not in calls or K2 not in puts:
            return opportunities
        
        C1 = calls[K1]
        C2 = calls[K2]
        P1 = puts[K1]
        P2 = puts[K2]
        
        if C1 <= 0 or C2 <= 0 or P1 <= 0 or P2 <= 0:
            return opportunities
        
        # Box spread value
        box_value = C1 - C2 + P2 - P1
        
        # Theoretical box value
        theoretical_box = (K2 - K1) * np.exp(-rate * T)
        
        # Check for violation
        diff = box_value - theoretical_box
        abs_diff = abs(diff)
        
        if abs_diff > self.tolerance * theoretical_box:
            if diff > 0:
                # Box is overpriced - sell box
                legs = {
                    'action': 'sell_box',
                    'sell_call_K1': K1,
                    'buy_call_K2': K2,
                    'buy_put_K2': K2,
                    'sell_put_K1': K1,
                }
            else:
                # Box is underpriced - buy box
                legs = {
                    'action': 'buy_box',
                    'buy_call_K1': K1,
                    'sell_call_K2': K2,
                    'sell_put_K2': K2,
                    'buy_put_K1': K1,
                }
            
            opportunity = ArbitrageOpportunity(
                arbitrage_type=ArbitrageType.BOX,
                symbol=f"K1={K1},K2={K2}",
                profit=abs_diff,
                legs=legs,
                timestamp=pd.Timestamp.now().isoformat(),
            )
            opportunities.append(opportunity)
        
        return opportunities
    
    def detect_arbitrage_opportunities(
        self,
        option_chain: pd.DataFrame,
        spot: float,
        rate: float = 0.05,
        div: float = 0.0
    ) -> List[ArbitrageOpportunity]:
        """
        Detect all arbitrage opportunities in an option chain.
        
        Args:
            option_chain: DataFrame with option prices
            spot: Current spot price
            rate: Risk-free rate
            div: Dividend yield
            
        Returns:
            List of all arbitrage opportunities
        """
        opportunities = []
        
        # Separate calls and puts by expiry
        calls_by_expiry = {}
        puts_by_expiry = {}
        
        for _, row in option_chain.iterrows():
            expiry = row['expiry']
            option_type = row['type']
            strike = row['strike']
            price = row['price']
            
            if option_type == 'call':
                if expiry not in calls_by_expiry:
                    calls_by_expiry[expiry] = {}
                calls_by_expiry[expiry][strike] = price
            else:
                if expiry not in puts_by_expiry:
                    puts_by_expiry[expiry] = {}
                puts_by_expiry[expiry][strike] = price
        
        # Check each expiry
        for expiry in calls_by_expiry.keys():
            if expiry not in puts_by_expiry:
                continue
            
            calls = calls_by_expiry[expiry]
            puts = puts_by_expiry[expiry]
            
            # Put-call parity
            opportunities.extend(
                self.put_call_parity_violation(calls, puts, spot, rate, div, expiry)
            )
            
            # Convexity
            opportunities.extend(self.convexity_violation(calls))
            
            # Butterfly
            opportunities.extend(self.butterfly_spread_violation(calls))
        
        # Calendar spreads (between expiries)
        expiries = sorted(calls_by_expiry.keys())
        for i in range(len(expiries) - 1):
            T_near = expiries[i]
            T_far = expiries[i + 1]
            
            opportunities.extend(
                self.calendar_spread_violation(
                    calls_by_expiry[T_near],
                    calls_by_expiry[T_far],
                    T_near,
                    T_far,
                    rate
                )
            )
        
        return opportunities
    
    def calculate_arbitrage_profit(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float = 1.0
    ) -> float:
        """
        Calculate arbitrage profit for a given position size.
        
        Args:
            opportunity: Arbitrage opportunity
            position_size: Position size multiplier
            
        Returns:
            Expected profit
        """
        return opportunity.profit * position_size

"""
Signal Validity Tracker (SVT)
Integrated from institutional_quant folder

Pre-trade safety layer that answers: Does the behavioral mechanism have reason to be active?

Architecture V2 - Quantitative Trading System for Indian Markets
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class VetoReason(Enum):
    """Reasons for signal veto"""
    MACRO_FREEZE = "macro_freeze"
    EXPIRY_PINNING = "expiry_pinning"
    PARTICIPANT_FLOW = "participant_flow"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRUNCH = "liquidity_crunch"


@dataclass
class VetoEvent:
    """Veto event details"""
    reason: VetoReason
    description: str
    vetoed_strategies: List[str]
    timestamp: datetime


class SignalValidityTracker:
    """
    Signal Validity Tracker - Core veto engine.
    
    Vetoes:
    - Macro Freeze: Block trending strategies near major events (Budget, RBI MPC)
    - Expiry Pinning: Block momentum strategies near expiry with high OI clustering
    - Participant Flow: Block buying if FIIs net-short and cash fleeing
    - Volatility Spike: Block strategies during extreme volatility
    - Liquidity Crunch: Block strategies during low liquidity
    """
    
    def __init__(self, macro_events: List[datetime]):
        self.macro_events = macro_events
        self.veto_history: List[VetoEvent] = []
    
    def check_macro_freeze(
        self,
        current_date: datetime,
        strategy_regimes: List[str]
    ) -> Optional[VetoEvent]:
        """
        Check macro freeze veto.
        
        Vetoes trending/breakout strategies when major macro events
        are within 3 sessions.
        """
        for event in self.macro_events:
            days_to_event = (event - current_date.date()).days
            
            if 0 <= days_to_event <= 3:
                # Check if strategy is trend/breakout type
                trend_strategies = [s for s in strategy_regimes if "trend" in s.lower() or "breakout" in s.lower()]
                
                if trend_strategies:
                    return VetoEvent(
                        reason=VetoReason.MACRO_FREEZE,
                        description=f"Macro event {event} in {days_to_event} days",
                        vetoed_strategies=trend_strategies,
                        timestamp=datetime.now()
                    )
        
        return None
    
    def check_expiry_pinning(
        self,
        current_date: datetime,
        nifty_pcr: float,
        options_skew: float,
        strategy_regimes: List[str]
    ) -> Optional[VetoEvent]:
        """
        Check expiry pinning veto.
        
        Vetoes momentum/breakout strategies on NIFTY near weekly/monthly
        expiry when options OI clustering is high.
        """
        # Check if near expiry (Thursday)
        if current_date.weekday() == 3:  # Thursday
            # Check for extreme PCR and skew indicating pinning
            if nifty_pcr > 1.3 or abs(options_skew) > 0.02:
                momentum_strategies = [s for s in strategy_regimes if "momentum" in s.lower() or "trend" in s.lower()]
                
                if momentum_strategies:
                    return VetoEvent(
                        reason=VetoReason.EXPIRY_PINNING,
                        description=f"Expiry pinning detected: PCR={nifty_pcr:.2f}, Skew={options_skew:.3f}",
                        vetoed_strategies=momentum_strategies,
                        timestamp=datetime.now()
                    )
        
        return None
    
    def check_participant_flow(
        self,
        fii_cash_flow_cr: float,
        fii_idx_fut_ls_ratio: float,
        strategy_regimes: List[str]
    ) -> Optional[VetoEvent]:
        """
        Check participant flow veto.
        
        Vetoes buying strategies if FIIs net-short derivatives and cash fleeing.
        """
        if fii_cash_flow_cr < -500 and fii_idx_fut_ls_ratio < 0.9:
            buy_strategies = [s for s in strategy_regimes if "buy" in s.lower() or "long" in s.lower()]
            
            if buy_strategies:
                return VetoEvent(
                    reason=VetoReason.PARTICIPANT_FLOW,
                    description=f"FII flight: Cash={fii_cash_flow_cr:.0f}Cr, LS Ratio={fii_idx_fut_ls_ratio:.2f}",
                    vetoed_strategies=buy_strategies,
                    timestamp=datetime.now()
                )
        
        return None
    
    def check_volatility_spike(
        self,
        volatility: float,
        strategy_regimes: List[str]
    ) -> Optional[VetoEvent]:
        """
        Check volatility spike veto.
        
        Vetoes strategies during extreme volatility.
        """
        if volatility > 0.35:  # 35% volatility threshold
            return VetoEvent(
                reason=VetoReason.VOLATILITY_SPIKE,
                description=f"Volatility spike: {volatility:.2%}",
                vetoed_strategies=strategy_regimes,  # Veto all
                timestamp=datetime.now()
            )
        
        return None
    
    def check_liquidity_crunch(
        self,
        bid_ask_spread_pct: float,
        volume_ratio: float,
        strategy_regimes: List[str]
    ) -> Optional[VetoEvent]:
        """
        Check liquidity crunch veto.
        
        Vetoes strategies during low liquidity.
        """
        if bid_ask_spread_pct > 0.05 or volume_ratio < 0.3:
            return VetoEvent(
                reason=VetoReason.LIQUIDITY_CRUNCH,
                description=f"Liquidity crunch: Spread={bid_ask_spread_pct:.2%}, Volume Ratio={volume_ratio:.2f}",
                vetoed_strategies=strategy_regimes,
                timestamp=datetime.now()
            )
        
        return None
    
    def evaluate_vetoes(
        self,
        current_date: datetime,
        market_data: Dict,
        strategy_regimes: List[str]
    ) -> List[VetoEvent]:
        """
        Evaluate all veto conditions.
        
        Returns:
            List of active vetoes
        """
        vetoes = []
        
        # Check each veto condition
        macro_veto = self.check_macro_freeze(current_date, strategy_regimes)
        if macro_veto:
            vetoes.append(macro_veto)
        
        expiry_veto = self.check_expiry_pinning(
            current_date,
            market_data.get('nifty_pcr', 1.0),
            market_data.get('options_skew', 0.0),
            strategy_regimes
        )
        if expiry_veto:
            vetoes.append(expiry_veto)
        
        flow_veto = self.check_participant_flow(
            market_data.get('fii_cash_flow_cr', 0),
            market_data.get('fii_idx_fut_ls_ratio', 1.0),
            strategy_regimes
        )
        if flow_veto:
            vetoes.append(flow_veto)
        
        vol_veto = self.check_volatility_spike(
            market_data.get('volatility', 0.15),
            strategy_regimes
        )
        if vol_veto:
            vetoes.append(vol_veto)
        
        liquidity_veto = self.check_liquidity_crunch(
            market_data.get('bid_ask_spread_pct', 0.02),
            market_data.get('volume_ratio', 1.0),
            strategy_regimes
        )
        if liquidity_veto:
            vetoes.append(liquidity_veto)
        
        # Record veto history
        self.veto_history.extend(vetoes)
        
        return vetoes
    
    def get_vetoed_strategies(self, vetoes: List[VetoEvent]) -> List[str]:
        """Get list of vetoed strategies from veto events."""
        vetoed = set()
        for veto in vetoes:
            vetoed.update(veto.vetoed_strategies)
        return list(vetoed)

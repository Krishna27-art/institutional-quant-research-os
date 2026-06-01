"""
Behavioral Hypothesis Framework
Integrated from institutional_quant folder

Architecture V2 - Quantitative Trading System for Indian Markets
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class BehavioralRegime(Enum):
    """Market behavioral regimes"""
    VOLATILITY_EXPANSION = "volatility_expansion"
    LIQUIDITY_VACUUM = "liquidity_vacuum"
    MEAN_REVERSION = "mean_reversion"
    INVENTORY_REBALANCING = "inventory_rebalancing"
    PANIC_SQUEEZE = "panic_squeeze"


@dataclass
class BehavioralHypothesis:
    """
    Formal behavioral hypothesis for trading strategies.
    
    Each strategy must declare:
    - Target participant: Who is being exploited?
    - Exploited mistake: What behavioral error are they making?
    - Activation regimes: When is this mechanism active?
    - Decay invalidators: What market changes would invalidate this?
    """
    id: str
    name: str
    target_participant: str
    exploited_mistake: str
    activation_regimes: List[str]
    description: str
    decay_invalidators: List[str]
    
    def is_valid_in_regime(self, regime: str) -> bool:
        """Check if hypothesis is valid in current regime."""
        return regime in self.activation_regimes


class BehavioralTaxonomy:
    """
    Continuous behavioral taxonomy instead of static enums.
    
    Models market as probability mixture of behaviors:
    - 60% inventory rebalancing + 40% volatility expansion
    """
    
    @staticmethod
    def estimate_mixture(
        volatility_z: float,
        volume_z: float,
        spread_z: float,
        institutional_ratio: float,
        intraday_return: float
    ) -> Dict[str, float]:
        """
        Estimate behavioral mixture from market features.
        
        Args:
            volatility_z: Z-score of volatility
            volume_z: Z-score of volume
            spread_z: Z-score of bid-ask spread
            institutional_ratio: Ratio of institutional to retail flow
            intraday_return: Current intraday return
            
        Returns:
            Dictionary mapping behavior to probability weight
        """
        # Simplified mixture estimation
        behaviors = {
            BehavioralRegime.VOLATILITY_EXPANSION.value: max(0, min(1, 0.3 + 0.4 * volatility_z)),
            BehavioralRegime.LIQUIDITY_VACUUM.value: max(0, min(1, 0.2 + 0.5 * spread_z)),
            BehavioralRegime.MEAN_REVERSION.value: max(0, min(1, 0.2 - 0.3 * intraday_return)),
            BehavioralRegime.INVENTORY_REBALANCING.value: max(0, min(1, 0.3 + 0.4 * institutional_ratio)),
            BehavioralRegime.PANIC_SQUEEZE.value: max(0, min(1, 0.1 + 0.6 * abs(volatility_z) if volatility_z < -1 else 0))
        }
        
        # Normalize to sum to 1
        total = sum(behaviors.values())
        if total > 0:
            behaviors = {k: v/total for k, v in behaviors.items()}
        
        return behaviors
    
    @staticmethod
    def format_mixture(mixture: Dict[str, float]) -> str:
        """Format mixture for display."""
        return ", ".join([f"{k}: {v*100:.0f}%" for k, v in sorted(mixture.items(), key=lambda x: -x[1]) if v > 0.1])

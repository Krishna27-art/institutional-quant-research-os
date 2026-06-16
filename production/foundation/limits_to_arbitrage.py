"""
Limits to Arbitrage - Level 1 Foundation

This module provides constraints based on limits to arbitrage theory (Shleifer & Vishny 1997):
- Liquidity constraints
- Margin constraints
- Volatility constraints
- Correlation constraints
- Capacity-aware position sizing

Based on Audit Report Priority 1: Economics & Market Microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolatilityRegime:
    """Volatility regime classification."""
    def __init__(self, value="normal"):
        self.value = value
        
    def classify_regime(self, volatility: float) -> 'VolatilityRegime':
        if volatility < 0.15:
            return VolatilityRegime.LOW
        elif volatility < 0.25:
            return VolatilityRegime.NORMAL
        elif volatility < 0.40:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
            
    def __eq__(self, other):
        if isinstance(other, VolatilityRegime):
            return self.value == other.value
        return self.value == other
        
    def __hash__(self):
        return hash(self.value)
        
    def __str__(self):
        return self.value

VolatilityRegime.LOW = VolatilityRegime("low")
VolatilityRegime.NORMAL = VolatilityRegime("normal")
VolatilityRegime.HIGH = VolatilityRegime("high")
VolatilityRegime.EXTREME = VolatilityRegime("extreme")


@dataclass
class PositionConstraints:
    """Constraints for position sizing."""
    max_position_pct: float = 5.0  # Max % of portfolio value
    max_daily_volume_pct: float = 1.0  # Max % of daily volume
    max_margin_pct: float = 50.0  # Max % of portfolio as margin
    max_correlation: float = 0.7  # Max correlation with existing positions
    min_liquidity_score: float = 0.5  # Minimum liquidity score

    def calculate_max_position(
        self,
        daily_volume: float,
        current_price: float,
        participation_rate_cap: float = 0.01
    ) -> float:
        return daily_volume * participation_rate_cap


class LimitsToArbitrage:
    """
    Limits to arbitrage constraints.
    
    This class implements constraints based on Shleifer & Vishny (1997) theory
    that even correct trades can lose money if constraints prevent holding to convergence.
    """
    
    def __init__(self, constraints: Optional[PositionConstraints] = None):
        """
        Initialize limits to arbitrage constraints.
        
        Args:
            constraints: Position constraints configuration
        """
        self.constraints = constraints or PositionConstraints()
    
    def liquidity_constraint(
        self,
        position_size: float,
        daily_volume: float,
        max_pct: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply liquidity constraint to position size.
        
        Limits position size based on daily trading volume to avoid
        excessive market impact and ensure ability to exit.
        
        Args:
            position_size: Desired position size (in shares or currency)
            daily_volume: Average daily trading volume
            max_pct: Maximum % of daily volume (default from constraints)
            
        Returns:
            Tuple of (adjusted_position_size, constraint_info)
        """
        if max_pct is None:
            max_pct = self.constraints.max_daily_volume_pct / 100
        
        # Calculate maximum allowed position
        max_position = daily_volume * max_pct
        
        # Apply constraint
        adjusted_size = min(position_size, max_position)
        
        # Calculate constraint factor
        constraint_factor = adjusted_size / position_size if position_size > 0 else 1.0
        
        return adjusted_size, {
            'original_size': position_size,
            'adjusted_size': adjusted_size,
            'max_allowed': max_position,
            'constraint_factor': constraint_factor,
            'constraint_active': constraint_factor < 1.0,
        }
    
    def volatility_constraint(
        self,
        position_size: float,
        volatility: float,
        margin_requirement: float = 0.5,
        vol_regime: Optional[VolatilityRegime] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply volatility constraint to position size.
        
        Reduces position size in high volatility regimes to manage margin
        requirements and reduce risk of margin calls.
        
        Args:
            position_size: Desired position size
            volatility: Current volatility (annualized)
            margin_requirement: Margin requirement as % of position value
            vol_regime: Volatility regime classification
            
        Returns:
            Tuple of (adjusted_position_size, constraint_info)
        """
        # Determine volatility factor based on regime
        if vol_regime is None:
            vol_regime = self._classify_volatility_regime(volatility)
        
        vol_factors = {
            VolatilityRegime.LOW: 1.0,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 0.5,
            VolatilityRegime.EXTREME: 0.25,
        }
        
        vol_factor = vol_factors.get(vol_regime, 0.5)
        
        # Apply constraint
        adjusted_size = position_size * vol_factor
        
        return adjusted_size, {
            'original_size': position_size,
            'adjusted_size': adjusted_size,
            'volatility': volatility,
            'vol_regime': vol_regime.value,
            'vol_factor': vol_factor,
            'constraint_active': vol_factor < 1.0,
        }
    
    def correlation_constraint(
        self,
        position_size: float,
        existing_positions: Dict[str, float],
        new_position_correlation: Dict[str, float],
        max_correlation: Optional[float] = None
    ) -> Tuple[float, Dict[str, Union[float, bool]]]:
        """
        Apply correlation constraint to position size.
        
        Reduces position size when new position is highly correlated
        with existing positions to manage concentration risk.
        
        Args:
            position_size: Desired position size
            existing_positions: Dictionary of existing positions {symbol: size}
            new_position_correlation: Correlation with existing positions
            max_correlation: Maximum allowed correlation (default from constraints)
            
        Returns:
            Tuple of (adjusted_position_size, constraint_info)
        """
        if max_correlation is None:
            max_correlation = self.constraints.max_correlation
        
        # Find maximum correlation with existing positions
        max_corr = 0.0
        if new_position_correlation:
            max_corr = max(abs(c) for c in new_position_correlation.values())
        
        # Calculate correlation factor
        if max_corr > max_correlation:
            corr_factor = max_correlation / max_corr
        else:
            corr_factor = 1.0
        
        # Apply constraint
        adjusted_size = position_size * corr_factor
        
        return adjusted_size, {
            'original_size': position_size,
            'adjusted_size': adjusted_size,
            'max_correlation': max_corr,
            'threshold': max_correlation,
            'corr_factor': corr_factor,
            'constraint_active': corr_factor < 1.0,
        }
    
    def margin_constraint(
        self,
        position_size: float,
        position_value: float,
        portfolio_value: float,
        margin_requirement: float = 0.5,
        max_margin_pct: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply margin constraint to position size.
        
        Limits position size to ensure margin requirements don't exceed
        portfolio capacity.
        
        Args:
            position_size: Desired position size
            position_value: Value of position at current price
            portfolio_value: Total portfolio value
            margin_requirement: Margin requirement as % of position value
            max_margin_pct: Maximum margin as % of portfolio (default from constraints)
            
        Returns:
            Tuple of (adjusted_position_size, constraint_info)
        """
        if max_margin_pct is None:
            max_margin_pct = self.constraints.max_margin_pct / 100
        
        # Calculate margin required for position
        margin_required = position_value * margin_requirement
        
        # Calculate maximum allowed margin
        max_margin = portfolio_value * max_margin_pct
        
        # Calculate constraint factor
        if margin_required > max_margin:
            margin_factor = max_margin / margin_required
        else:
            margin_factor = 1.0
        
        # Apply constraint
        adjusted_size = position_size * margin_factor
        
        return adjusted_size, {
            'original_size': position_size,
            'adjusted_size': adjusted_size,
            'margin_required': margin_required,
            'max_margin': max_margin,
            'margin_factor': margin_factor,
            'constraint_active': margin_factor < 1.0,
        }
    
    def capacity_limited_size(
        self,
        signal: float,
        liquidity: float,
        vol_regime: VolatilityRegime = VolatilityRegime.NORMAL,
        correlation: float = 0.0,
        portfolio_value: float = 1_000_000,
        base_kelly: float = 0.02
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate capacity-limited position size.
        
        Combines all constraints to determine the maximum position size
        given market conditions and portfolio constraints.
        
        Args:
            signal: Trading signal (e.g., Kelly fraction)
            liquidity: Daily trading volume
            vol_regime: Current volatility regime
            correlation: Correlation with existing positions
            portfolio_value: Total portfolio value
            base_kelly: Base Kelly fraction for position sizing
            
        Returns:
            Tuple of (adjusted_position_size, constraint_info)
        """
        # Start with base position size
        base_size = portfolio_value * base_kelly * signal
        
        # Apply liquidity constraint
        liquidity_adj_size, liquidity_info = self.liquidity_constraint(
            base_size, liquidity
        )
        
        # Apply volatility constraint
        vol_adj_size, vol_info = self.volatility_constraint(
            liquidity_adj_size, 0.2, vol_regime=vol_regime
        )
        
        # Apply correlation constraint
        corr_adj_size, corr_info = self.correlation_constraint(
            vol_adj_size, {}, {'existing': correlation}
        )
        
        # Apply margin constraint
        margin_adj_size, margin_info = self.margin_constraint(
            corr_adj_size, corr_adj_size, portfolio_value
        )
        
        # Final adjusted size
        final_size = margin_adj_size
        
        return final_size, {
            'base_size': base_size,
            'liquidity_adjusted': liquidity_adj_size,
            'volatility_adjusted': vol_adj_size,
            'correlation_adjusted': corr_adj_size,
            'margin_adjusted': margin_adj_size,
            'final_size': final_size,
            'overall_constraint_factor': final_size / base_size if base_size > 0 else 1.0,
            'liquidity_constraint_active': liquidity_info['constraint_active'],
            'volatility_constraint_active': vol_info['constraint_active'],
            'correlation_constraint_active': corr_info['constraint_active'],
            'margin_constraint_active': margin_info['constraint_active'],
        }
    
    def regime_based_constraints(
        self,
        regime_type: str,
        base_constraints: Optional[PositionConstraints] = None
    ) -> PositionConstraints:
        """
        Get regime-based position constraints.
        
        Adjusts constraints based on market regime (crisis, normal, bull, bear).
        
        Args:
            regime_type: Type of regime ('crisis', 'normal', 'bull', 'bear')
            base_constraints: Base constraints to adjust
            
        Returns:
            Adjusted position constraints
        """
        if base_constraints is None:
            base_constraints = self.constraints
        
        if regime_type == 'crisis':
            # Reduce all constraints in crisis
            return PositionConstraints(
                max_position_pct=base_constraints.max_position_pct * 0.5,
                max_daily_volume_pct=base_constraints.max_daily_volume_pct * 0.5,
                max_margin_pct=base_constraints.max_margin_pct * 0.5,
                max_correlation=base_constraints.max_correlation * 0.8,
                min_liquidity_score=base_constraints.min_liquidity_score * 1.5,
            )
        
        elif regime_type == 'normal':
            return base_constraints
        
        elif regime_type == 'bull':
            # Can be more aggressive in bull market
            return PositionConstraints(
                max_position_pct=base_constraints.max_position_pct * 1.2,
                max_daily_volume_pct=base_constraints.max_daily_volume_pct * 1.2,
                max_margin_pct=base_constraints.max_margin_pct * 1.1,
                max_correlation=base_constraints.max_correlation,
                min_liquidity_score=base_constraints.min_liquidity_score,
            )
        
        elif regime_type == 'bear':
            # More conservative in bear market
            return PositionConstraints(
                max_position_pct=base_constraints.max_position_pct * 0.7,
                max_daily_volume_pct=base_constraints.max_daily_volume_pct * 0.7,
                max_margin_pct=base_constraints.max_margin_pct * 0.8,
                max_correlation=base_constraints.max_correlation * 0.9,
                min_liquidity_score=base_constraints.min_liquidity_score * 1.2,
            )
        
        else:
            return base_constraints
    
    def calculate_max_position_size(
        self,
        signal: float,
        constraints: Dict[str, float],
        portfolio_value: float = 1_000_000
    ) -> float:
        """
        Calculate maximum position size given constraints.
        
        Args:
            signal: Trading signal
            constraints: Dictionary of constraint values
            portfolio_value: Total portfolio value
            
        Returns:
            Maximum allowed position size
        """
        # Base size from signal
        base_size = portfolio_value * constraints.get('kelly_fraction', 0.02) * abs(signal)
        
        # Apply liquidity factor
        liquidity_factor = min(1.0, constraints.get('liquidity_factor', 1.0))
        
        # Apply volatility factor
        vol_factor = constraints.get('volatility_factor', 1.0)
        
        # Apply correlation factor
        corr_factor = constraints.get('correlation_factor', 1.0)
        
        # Calculate final size
        final_size = base_size * liquidity_factor * vol_factor * corr_factor
        
        return final_size
    
    def _classify_volatility_regime(self, volatility: float) -> VolatilityRegime:
        """
        Classify volatility into regime.
        
        Args:
            volatility: Annualized volatility
            
        Returns:
            Volatility regime
        """
        if volatility < 0.15:
            return VolatilityRegime.LOW
        elif volatility < 0.25:
            return VolatilityRegime.NORMAL
        elif volatility < 0.40:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

"""
CRITICAL FIX: Position consolidation across multiple alphas.

The review noted that when running multiple alphas simultaneously, there's no mechanism
to consolidate positions. If Alpha A says buy 100 shares of RELIANCE and Alpha B says
sell 50 shares, the system should net to 50 shares long, not execute both orders independently.

This module provides:
- Position netting across multiple alphas
- Order consolidation to reduce transaction costs
- Conflict resolution for opposing signals
- Risk-aware position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConsolidationStrategy(Enum):
    """Strategy for consolidating positions."""
    NET = "net"  # Net positions (buy 100 + sell 50 = buy 50)
    WEIGHTED_AVERAGE = "weighted_average"  # Weight by alpha strength
    MAJORITY_VOTE = "majority_vote"  # Follow majority direction
    RISK_PARITY = "risk_parity"  # Allocate by risk contribution


@dataclass
class AlphaSignal:
    """Signal from a single alpha."""
    alpha_name: str
    symbol: str
    direction: str  # "buy" or "sell"
    quantity: float
    strength: float  # Signal strength 0-1
    confidence: float  # Confidence 0-1
    timestamp: pd.Timestamp


@dataclass
class ConsolidatedPosition:
    """Consolidated position across all alphas."""
    symbol: str
    net_quantity: float
    net_direction: str
    contributing_alphas: List[str]
    total_strength: float
    avg_confidence: float
    consolidation_method: str


class PositionConsolidator:
    """
    Consolidate positions across multiple alphas.
    
    CRITICAL FIX: Prevents duplicate trades and reduces transaction costs by
    netting opposing signals from different alphas.
    """
    
    def __init__(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.NET,
        min_confidence: float = 0.5,
        enable_conflict_resolution: bool = True
    ):
        """
        Initialize position consolidator.
        
        Args:
            strategy: Consolidation strategy to use
            min_confidence: Minimum confidence to include signal
            enable_conflict_resolution: Whether to resolve conflicts
        """
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.enable_conflict_resolution = enable_conflict_resolution
        
        logger.info(
            f"Position consolidator initialized: strategy={strategy.value}, "
            f"min_confidence={min_confidence}"
        )
    
    def consolidate_signals(
        self,
        signals: List[AlphaSignal]
    ) -> Dict[str, ConsolidatedPosition]:
        """
        Consolidate signals from multiple alphas.
        
        Args:
            signals: List of alpha signals
            
        Returns:
            Dictionary of symbol -> consolidated position
        """
        # Group signals by symbol
        signals_by_symbol: Dict[str, List[AlphaSignal]] = {}
        
        for signal in signals:
            # Filter by confidence
            if signal.confidence < self.min_confidence:
                continue
            
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        # Consolidate each symbol
        consolidated = {}
        
        for symbol, symbol_signals in signals_by_symbol.items():
            if self.strategy == ConsolidationStrategy.NET:
                consolidated[symbol] = self._consolidate_net(symbol_signals)
            elif self.strategy == ConsolidationStrategy.WEIGHTED_AVERAGE:
                consolidated[symbol] = self._consolidate_weighted(symbol_signals)
            elif self.strategy == ConsolidationStrategy.MAJORITY_VOTE:
                consolidated[symbol] = self._consolidate_majority(symbol_signals)
            elif self.strategy == ConsolidationStrategy.RISK_PARITY:
                consolidated[symbol] = self._consolidate_risk_parity(symbol_signals)
        
        logger.info(
            f"Consolidated {len(signals)} signals into {len(consolidated)} positions "
            f"using {self.strategy.value} strategy"
        )
        
        return consolidated
    
    def _consolidate_net(self, signals: List[AlphaSignal]) -> ConsolidatedPosition:
        """
        Net positions: buy 100 + sell 50 = buy 50.
        
        Args:
            signals: Signals for a single symbol
            
        Returns:
            Consolidated position
        """
        buy_quantity = sum(s.quantity for s in signals if s.direction == "buy")
        sell_quantity = sum(s.quantity for s in signals if s.direction == "sell")
        
        net_quantity = abs(buy_quantity - sell_quantity)
        
        if buy_quantity > sell_quantity:
            net_direction = "buy"
        elif sell_quantity > buy_quantity:
            net_direction = "sell"
        else:
            net_direction = "flat"
            net_quantity = 0.0
        
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return ConsolidatedPosition(
            symbol=signals[0].symbol,
            net_quantity=net_quantity,
            net_direction=net_direction,
            contributing_alphas=[s.alpha_name for s in signals],
            total_strength=avg_strength,
            avg_confidence=avg_confidence,
            consolidation_method="net"
        )
    
    def _consolidate_weighted(self, signals: List[AlphaSignal]) -> ConsolidatedPosition:
        """
        Weighted average by signal strength.
        
        Args:
            signals: Signals for a single symbol
            
        Returns:
            Consolidated position
        """
        total_weight = sum(s.strength for s in signals)
        
        if total_weight == 0:
            return ConsolidatedPosition(
                symbol=signals[0].symbol,
                net_quantity=0.0,
                net_direction="flat",
                contributing_alphas=[s.alpha_name for s in signals],
                total_strength=0.0,
                avg_confidence=0.0,
                consolidation_method="weighted_average"
            )
        
        # Weight quantities by strength
        buy_weighted = sum(
            s.quantity * s.strength for s in signals if s.direction == "buy"
        )
        sell_weighted = sum(
            s.quantity * s.strength for s in signals if s.direction == "sell"
        )
        
        net_quantity = abs(buy_weighted - sell_weighted) / total_weight
        
        if buy_weighted > sell_weighted:
            net_direction = "buy"
        elif sell_weighted > buy_weighted:
            net_direction = "sell"
        else:
            net_direction = "flat"
            net_quantity = 0.0
        
        avg_strength = total_weight / len(signals)
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return ConsolidatedPosition(
            symbol=signals[0].symbol,
            net_quantity=net_quantity,
            net_direction=net_direction,
            contributing_alphas=[s.alpha_name for s in signals],
            total_strength=avg_strength,
            avg_confidence=avg_confidence,
            consolidation_method="weighted_average"
        )
    
    def _consolidate_majority(self, signals: List[AlphaSignal]) -> ConsolidatedPosition:
        """
        Majority vote: follow the direction with most alphas.
        
        Args:
            signals: Signals for a single symbol
            
        Returns:
            Consolidated position
        """
        buy_count = sum(1 for s in signals if s.direction == "buy")
        sell_count = sum(1 for s in signals if s.direction == "sell")
        
        if buy_count > sell_count:
            net_direction = "buy"
            avg_quantity = np.mean([s.quantity for s in signals if s.direction == "buy"])
        elif sell_count > buy_count:
            net_direction = "sell"
            avg_quantity = np.mean([s.quantity for s in signals if s.direction == "sell"])
        else:
            net_direction = "flat"
            avg_quantity = 0.0
        
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return ConsolidatedPosition(
            symbol=signals[0].symbol,
            net_quantity=avg_quantity,
            net_direction=net_direction,
            contributing_alphas=[s.alpha_name for s in signals],
            total_strength=avg_strength,
            avg_confidence=avg_confidence,
            consolidation_method="majority_vote"
        )
    
    def _consolidate_risk_parity(self, signals: List[AlphaSignal]) -> ConsolidatedPosition:
        """
        Risk parity: allocate based on risk contribution.
        
        Args:
            signals: Signals for a single symbol
            
        Returns:
            Consolidated position
        """
        # Simplified risk parity - use inverse of volatility as weight
        # In production, would use actual risk estimates
        
        volatilities = {s.alpha_name: 0.2 for s in signals}  # Placeholder
        weights = {s.alpha_name: 1.0 / volatilities[s.alpha_name] for s in signals}
        total_weight = sum(weights.values())
        
        normalized_weights = {
            k: v / total_weight for k, v in weights.items()
        }
        
        # Apply weights to quantities
        buy_weighted = sum(
            s.quantity * normalized_weights[s.alpha_name]
            for s in signals if s.direction == "buy"
        )
        sell_weighted = sum(
            s.quantity * normalized_weights[s.alpha_name]
            for s in signals if s.direction == "sell"
        )
        
        net_quantity = abs(buy_weighted - sell_weighted)
        
        if buy_weighted > sell_weighted:
            net_direction = "buy"
        elif sell_weighted > buy_weighted:
            net_direction = "sell"
        else:
            net_direction = "flat"
            net_quantity = 0.0
        
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        return ConsolidatedPosition(
            symbol=signals[0].symbol,
            net_quantity=net_quantity,
            net_direction=net_direction,
            contributing_alphas=[s.alpha_name for s in signals],
            total_strength=avg_strength,
            avg_confidence=avg_confidence,
            consolidation_method="risk_parity"
        )
    
    def resolve_conflicts(
        self,
        consolidated: Dict[str, ConsolidatedPosition],
        current_positions: Dict[str, float]
    ) -> Dict[str, ConsolidatedPosition]:
        """
        Resolve conflicts between consolidated signals and current positions.
        
        Args:
            consolidated: Consolidated positions
            current_positions: Current positions (symbol -> quantity)
            
        Returns:
            Resolved positions
        """
        resolved = {}
        
        for symbol, position in consolidated.items():
            current_qty = current_positions.get(symbol, 0.0)
            
            # Calculate trade needed
            if position.net_direction == "buy":
                trade_qty = position.net_quantity - current_qty
            elif position.net_direction == "sell":
                trade_qty = -position.net_quantity - current_qty
            else:
                trade_qty = -current_qty  # Close position
            
            # If trade is small, skip to reduce costs
            if abs(trade_qty) < 10:  # Minimum trade size
                logger.info(f"Skipping small trade for {symbol}: {trade_qty:.2f}")
                continue
            
            resolved[symbol] = position
        
        logger.info(
            f"Resolved conflicts: {len(consolidated)} -> {len(resolved)} positions"
        )
        
        return resolved
    
    def calculate_transaction_cost_savings(
        self,
        original_signals: List[AlphaSignal],
        consolidated: Dict[str, ConsolidatedPosition],
        cost_per_share: float = 0.001
    ) -> float:
        """
        Calculate transaction cost savings from consolidation.
        
        Args:
            original_signals: Original unconsolidated signals
            consolidated: Consolidated positions
            cost_per_share: Cost per share traded
            
        Returns:
            Cost savings in currency units
        """
        # Original cost
        original_quantity = sum(s.quantity for s in original_signals)
        original_cost = original_quantity * cost_per_share
        
        # Consolidated cost
        consolidated_quantity = sum(p.net_quantity for p in consolidated.values())
        consolidated_cost = consolidated_quantity * cost_per_share
        
        savings = original_cost - consolidated_cost
        savings_pct = (savings / original_cost * 100) if original_cost > 0 else 0
        
        logger.info(
            f"Transaction cost savings: ₹{savings:.2f} ({savings_pct:.1f}%)"
        )
        
        return savings

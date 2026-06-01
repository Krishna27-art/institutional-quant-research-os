"""
Bayesian Position Sizing
Multi-factor position sizing with confidence, regime, liquidity, and drift penalties.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from .probabilistic_forecasting import ProbabilisticPrediction


@dataclass
class SizingFactors:
    """Factors that influence position sizing"""
    confidence_score: float = 1.0  # From probabilistic forecasting
    regime_score: float = 1.0  # From market state engine
    liquidity_score: float = 1.0  # Based on capacity utilization
    feature_drift_penalty: float = 0.0  # From feature drift monitor
    
    def to_dict(self) -> Dict:
        return {
            "confidence_score": self.confidence_score,
            "regime_score": self.regime_score,
            "liquidity_score": self.liquidity_score,
            "feature_drift_penalty": self.feature_drift_penalty,
        }
    
    def compute_adjustment(self) -> float:
        """
        Compute total adjustment factor.
        adjustment = confidence * regime * liquidity * (1 - drift_penalty)
        """
        drift_factor = max(0.0, 1.0 - self.feature_drift_penalty)
        return self.confidence_score * self.regime_score * self.liquidity_score * drift_factor


@dataclass
class PositionSizingDecision:
    """Position sizing decision for a trade"""
    strategy_id: str
    symbol: str
    timestamp: datetime
    
    # Base sizing
    kelly_fraction: float = 0.0
    expected_return: float = 0.0
    variance: float = 0.0
    
    # Adjusted sizing
    base_position: float = 0.0
    adjusted_position: float = 0.0
    position_multiplier: float = 1.0
    
    # Sizing factors
    sizing_factors: Optional[SizingFactors] = None
    
    # Risk checks
    passed_risk_checks: bool = True
    risk_violations: List[str] = field(default_factory=list)
    
    # Final position
    final_position_size: float = 0.0  # In currency units
    position_pct_of_aum: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "kelly_fraction": self.kelly_fraction,
            "expected_return": self.expected_return,
            "variance": self.variance,
            "base_position": self.base_position,
            "adjusted_position": self.adjusted_position,
            "position_multiplier": self.position_multiplier,
            "sizing_factors": self.sizing_factors.to_dict() if self.sizing_factors else None,
            "passed_risk_checks": self.passed_risk_checks,
            "risk_violations": self.risk_violations,
            "final_position_size": self.final_position_size,
            "position_pct_of_aum": self.position_pct_of_aum,
        }


class BayesianPositionSizer:
    """
    Bayesian position sizing with multi-factor adjustments.
    Incorporates confidence, regime, liquidity, and feature drift penalties.
    """
    
    def __init__(
        self,
        aum: float = 25_00_00_000,  # ₹25 Cr default
        max_single_trade_risk_pct: float = 0.5,  # 0.5% of AUM
        max_portfolio_risk_pct: float = 10.0,  # 10% of AUM
        kelly_multiplier: float = 0.15,  # Use 15% of optimal Kelly
        max_kelly_multiple: float = 2.0,
        min_kelly_multiple: float = 0.1
    ):
        self.aum = aum
        self.max_single_trade_risk_pct = max_single_trade_risk_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.kelly_multiplier = kelly_multiplier
        self.max_kelly_multiple = max_kelly_multiple
        self.min_kelly_multiple = min_kelly_multiple
        
        # Current portfolio exposure
        self.current_portfolio_risk_pct: float = 0.0
        self.current_positions: Dict[str, float] = {}
        
        # Capacity limits
        self.capacity_limits: Dict[str, float] = {}  # strategy_id -> capacity_limit
        self.current_aum_by_strategy: Dict[str, float] = {}
    
    def set_capacity_limit(self, strategy_id: str, capacity_limit: float) -> None:
        """Set capacity limit for a strategy"""
        self.capacity_limits[strategy_id] = capacity_limit
    
    def set_current_aum(self, strategy_id: str, current_aum: float) -> None:
        """Set current AUM for a strategy"""
        self.current_aum_by_strategy[strategy_id] = current_aum
    
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        variance: float
    ) -> float:
        """
        Calculate Kelly fraction.
        
        Kelly = expected_return / variance
        
        Args:
            expected_return: Expected return
            variance: Variance of returns
        
        Returns:
            Kelly fraction
        """
        if variance <= 0:
            return 0.0
        
        kelly = expected_return / variance
        return max(0.0, kelly)
    
    def calculate_regime_score(self, regime: str) -> float:
        """
        Calculate regime score for position sizing.
        
        Args:
            regime: Current market regime
        
        Returns:
            Regime score (0.5 for high_vol, 1.0 for normal)
        """
        high_vol_regimes = ["high_vol", "crisis", "panic"]
        
        if regime.lower() in high_vol_regimes:
            return 0.5
        else:
            return 1.0
    
    def calculate_liquidity_score(
        self,
        strategy_id: str,
        position_size: float
    ) -> float:
        """
        Calculate liquidity score based on capacity utilization.
        
        Args:
            strategy_id: Strategy identifier
            position_size: Proposed position size
        
        Returns:
            Liquidity score (1.0 if < 70% capacity, decays otherwise)
        """
        if strategy_id not in self.capacity_limits:
            return 1.0
        
        capacity_limit = self.capacity_limits[strategy_id]
        current_aum = self.current_aum_by_strategy.get(strategy_id, 0.0)
        
        utilization = (current_aum + position_size) / capacity_limit
        
        if utilization < 0.7:
            return 1.0
        elif utilization < 0.9:
            # Linear decay from 1.0 to 0.5
            return 1.0 - (utilization - 0.7) / 0.2 * 0.5
        else:
            # Below 0.5 for > 90% utilization
            return max(0.1, 0.5 - (utilization - 0.9) * 5.0)
    
    def calculate_feature_drift_penalty(self, feature_drift_scores: Dict[str, float]) -> float:
        """
        Calculate feature drift penalty.
        
        Args:
            feature_drift_scores: Dictionary of feature_name -> PSI score
        
        Returns:
            Drift penalty (0 to 0.5, capped at 0)
        """
        if not feature_drift_scores:
            return 0.0
        
        # Use average PSI
        avg_psi = np.mean(list(feature_drift_scores.values()))
        
        # Penalty = PSI / 0.5, capped at 0
        penalty = min(0.5, max(0.0, avg_psi / 0.5))
        
        return penalty
    
    def size_position(
        self,
        strategy_id: str,
        symbol: str,
        expected_return: float,
        variance: float,
        confidence_score: float = 1.0,
        regime: str = "normal",
        feature_drift_scores: Optional[Dict[str, float]] = None
    ) -> PositionSizingDecision:
        """
        Calculate position size using Bayesian multi-factor sizing.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            expected_return: Expected return
            variance: Variance of returns
            confidence_score: Confidence score from probabilistic forecasting
            regime: Current market regime
            feature_drift_scores: Feature drift scores
        
        Returns:
            PositionSizingDecision with calculated position size
        """
        decision = PositionSizingDecision(
            strategy_id=strategy_id,
            symbol=symbol,
            timestamp=datetime.now(),
            expected_return=expected_return,
            variance=variance
        )
        
        # Calculate Kelly fraction
        kelly = self.calculate_kelly_fraction(expected_return, variance)
        decision.kelly_fraction = kelly
        
        # Calculate base position: Kelly * (expected_return / variance)
        # This simplifies to Kelly^2, but we keep the formula for clarity
        if variance > 0:
            base_position = kelly * (expected_return / variance)
        else:
            base_position = 0.0
        decision.base_position = base_position
        
        # Calculate sizing factors
        regime_score = self.calculate_regime_score(regime)
        
        # Estimate position size for liquidity score
        estimated_position = base_position * self.aum * self.kelly_multiplier
        liquidity_score = self.calculate_liquidity_score(strategy_id, estimated_position)
        
        drift_penalty = self.calculate_feature_drift_penalty(feature_drift_scores or {})
        
        sizing_factors = SizingFactors(
            confidence_score=confidence_score,
            regime_score=regime_score,
            liquidity_score=liquidity_score,
            feature_drift_penalty=drift_penalty
        )
        decision.sizing_factors = sizing_factors
        
        # Calculate adjustment
        adjustment = sizing_factors.compute_adjustment()
        decision.position_multiplier = adjustment
        
        # Calculate adjusted position
        adjusted_position = base_position * self.kelly_multiplier * adjustment
        decision.adjusted_position = adjusted_position
        
        # Apply Kelly clipping
        kelly_multiple = adjusted_position / base_position if base_position > 0 else 0.0
        
        if kelly_multiple > self.max_kelly_multiple:
            adjusted_position = base_position * self.max_kelly_multiple
            decision.risk_violations.append(f"Kelly multiple {kelly_multiple:.2f} exceeded max {self.max_kelly_multiple}")
        elif kelly_multiple < self.min_kelly_multiple and kelly_multiple > 0:
            adjusted_position = 0.0
            decision.risk_violations.append(f"Kelly multiple {kelly_multiple:.2f} below min {self.min_kelly_multiple}")
        
        decision.adjusted_position = adjusted_position
        
        # Risk checks
        single_trade_risk_pct = abs(adjusted_position) * 100
        if single_trade_risk_pct > self.max_single_trade_risk_pct:
            adjusted_position = self.max_single_trade_risk_pct / 100 * np.sign(adjusted_position)
            decision.risk_violations.append(
                f"Single trade risk {single_trade_risk_pct:.2f}% exceeded max {self.max_single_trade_risk_pct}%"
            )
        
        portfolio_risk_pct = self.current_portfolio_risk_pct + abs(adjusted_position) * 100
        if portfolio_risk_pct > self.max_portfolio_risk_pct:
            available_risk = self.max_portfolio_risk_pct - self.current_portfolio_risk_pct
            adjusted_position = (available_risk / 100) * np.sign(adjusted_position)
            decision.risk_violations.append(
                f"Portfolio risk {portfolio_risk_pct:.2f}% would exceed max {self.max_portfolio_risk_pct}%"
            )
        
        decision.adjusted_position = adjusted_position
        decision.passed_risk_checks = len(decision.risk_violations) == 0
        
        # Calculate final position size in currency
        decision.final_position_size = adjusted_position * self.aum
        decision.position_pct_of_aum = abs(adjusted_position) * 100
        
        return decision
    
    def update_portfolio_exposure(self, symbol: str, position_size: float) -> None:
        """Update current portfolio exposure after trade"""
        # Remove old position if exists
        if symbol in self.current_positions:
            self.current_portfolio_risk_pct -= abs(self.current_positions[symbol]) * 100
        
        # Add new position
        self.current_positions[symbol] = position_size / self.aum
        self.current_portfolio_risk_pct += abs(position_size / self.aum) * 100
    
    def get_current_portfolio_risk(self) -> float:
        """Get current portfolio risk percentage"""
        return self.current_portfolio_risk_pct
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions as percentage of AUM"""
        return self.current_positions.copy()
    
    def reset_portfolio(self) -> None:
        """Reset portfolio exposure (e.g., after flatten all)"""
        self.current_positions.clear()
        self.current_portfolio_risk_pct = 0.0


def calculate_position_size_from_prediction(
    prediction: ProbabilisticPrediction,
    variance: float,
    sizer: BayesianPositionSizer,
    regime: str = "normal",
    feature_drift_scores: Optional[Dict[str, float]] = None
) -> PositionSizingDecision:
    """
    Calculate position size from probabilistic prediction.
    
    Args:
        prediction: Probabilistic prediction
        variance: Variance of returns
        sizer: Bayesian position sizer
        regime: Current market regime
        feature_drift_scores: Feature drift scores
    
    Returns:
        PositionSizingDecision
    """
    return sizer.size_position(
        strategy_id=prediction.strategy_id,
        symbol=prediction.symbol,
        expected_return=prediction.expected_return,
        variance=variance,
        confidence_score=prediction.confidence,
        regime=regime,
        feature_drift_scores=feature_drift_scores
    )

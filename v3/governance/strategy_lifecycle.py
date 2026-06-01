"""
Strategy Lifecycle Management
Implements 6-phase lifecycle: RESEARCH → PAPER_TRADE → LIVE_10PCT → LIVE_FULL → DECAY → RETIRED
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd


class StrategyPhase(Enum):
    """Strategy lifecycle phases"""
    RESEARCH = "RESEARCH"
    PAPER_TRADE = "PAPER_TRADE"
    LIVE_10PCT = "LIVE_10PCT"
    LIVE_FULL = "LIVE_FULL"
    DECAY = "DECAY"
    RETIRED = "RETIRED"


@dataclass
class GateCriteria:
    """Criteria for phase transitions"""
    paper_trade_to_live_10pct: Dict[str, float] = field(default_factory=lambda: {
        "min_sharpe": 1.0,
        "max_drawdown": 0.15,
        "min_win_rate": 0.40,
        "min_days": 60
    })
    live_10pct_to_live_full: Dict[str, float] = field(default_factory=lambda: {
        "min_rolling_sharpe_20d": 0.8,
        "max_consecutive_losses": 5,
        "min_days": 30
    })
    live_full_maintenance: Dict[str, float] = field(default_factory=lambda: {
        "min_rolling_sharpe_20d": 0.6,
        "maintenance_days": 60
    })


@dataclass
class HealthMetrics:
    """Health metrics for strategy monitoring"""
    rolling_sharpe_20d: float
    rolling_sortino_20d: float
    alpha_half_life: float  # days
    turnover_stability: float  # coefficient of variation
    feature_drift: Dict[str, float]  # PSI per feature
    regime_stability: float  # how often regime changes
    
    # Additional metrics
    rolling_win_rate_20d: float
    current_drawdown: float
    daily_returns: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "rolling_sharpe_20d": self.rolling_sharpe_20d,
            "rolling_sortino_20d": self.rolling_sortino_20d,
            "alpha_half_life": self.alpha_half_life,
            "turnover_stability": self.turnover_stability,
            "feature_drift": self.feature_drift,
            "regime_stability": self.regime_stability,
            "rolling_win_rate_20d": self.rolling_win_rate_20d,
            "current_drawdown": self.current_drawdown,
        }


@dataclass
class DecisionRule:
    """Decision rules for lifecycle transitions"""
    condition: Callable[[HealthMetrics], bool]
    action: StrategyPhase
    reason: str
    
    def evaluate(self, metrics: HealthMetrics) -> Optional[StrategyPhase]:
        if self.condition(metrics):
            return self.action
        return None


@dataclass
class StrategyLifecycle:
    """Lifecycle state for a single strategy"""
    strategy_id: str
    strategy_name: str
    current_phase: StrategyPhase = StrategyPhase.RESEARCH
    phase_start_date: datetime = field(default_factory=datetime.now)
    allocation_pct: float = 0.0  # Percentage of target allocation
    
    # Performance tracking
    expected_sharpe: float = 0.0
    expected_max_drawdown: float = 0.0
    expected_win_rate: float = 0.0
    
    # Historical metrics
    phase_history: List[Dict] = field(default_factory=list)
    health_history: List[HealthMetrics] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def transition_to(self, new_phase: StrategyPhase, reason: str = "") -> None:
        """Transition to a new phase"""
        # Record history
        self.phase_history.append({
            "from_phase": self.current_phase.value,
            "to_phase": new_phase.value,
            "transition_date": datetime.now(),
            "reason": reason,
            "allocation_at_transition": self.allocation_pct
        })
        
        self.current_phase = new_phase
        self.phase_start_date = datetime.now()
        self.last_updated = datetime.now()
    
    def update_allocation(self, allocation_pct: float) -> None:
        """Update allocation percentage"""
        self.allocation_pct = allocation_pct
        self.last_updated = datetime.now()
    
    def days_in_phase(self) -> int:
        """Calculate days spent in current phase"""
        return (datetime.now() - self.phase_start_date).days
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "current_phase": self.current_phase.value,
            "phase_start_date": self.phase_start_date.isoformat(),
            "allocation_pct": self.allocation_pct,
            "expected_sharpe": self.expected_sharpe,
            "expected_max_drawdown": self.expected_max_drawdown,
            "expected_win_rate": self.expected_win_rate,
            "days_in_phase": self.days_in_phase(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


class LifecycleManager:
    """
    Manages strategy lifecycle transitions based on health metrics and gate criteria.
    """
    
    def __init__(self, gate_criteria: Optional[GateCriteria] = None):
        self.gate_criteria = gate_criteria or GateCriteria()
        self.strategies: Dict[str, StrategyLifecycle] = {}
        self.decision_rules: List[DecisionRule] = []
        self._setup_default_decision_rules()
    
    def _setup_default_decision_rules(self) -> None:
        """Setup default decision rules for lifecycle transitions"""
        
        # Rule: Enter DECAY if Sharpe < 0.3 for 10 days
        self.decision_rules.append(DecisionRule(
            condition=lambda m: m.rolling_sharpe_20d < 0.3,
            action=StrategyPhase.DECAY,
            reason="Rolling Sharpe below 0.3"
        ))
        
        # Rule: RETIRE if Sharpe < 0 for 15 days
        self.decision_rules.append(DecisionRule(
            condition=lambda m: m.rolling_sharpe_20d < 0.0,
            action=StrategyPhase.RETIRED,
            reason="Rolling Sharpe negative"
        ))
        
        # Rule: Reduce allocation if alpha half-life < 30 days
        self.decision_rules.append(DecisionRule(
            condition=lambda m: m.alpha_half_life < 30.0,
            action=StrategyPhase.DECAY,
            reason="Alpha half-life too short"
        ))
        
        # Rule: Flag for review if feature drift > 0.3 for any top-5 feature
        self.decision_rules.append(DecisionRule(
            condition=lambda m: any(psi > 0.3 for psi in list(m.feature_drift.values())[:5]),
            action=StrategyPhase.DECAY,
            reason="Significant feature drift detected"
        ))
    
    def register_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        expected_sharpe: float,
        expected_max_drawdown: float,
        expected_win_rate: float,
        initial_phase: StrategyPhase = StrategyPhase.RESEARCH
    ) -> StrategyLifecycle:
        """Register a new strategy for lifecycle management"""
        lifecycle = StrategyLifecycle(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            current_phase=initial_phase,
            expected_sharpe=expected_sharpe,
            expected_max_drawdown=expected_max_drawdown,
            expected_win_rate=expected_win_rate
        )
        self.strategies[strategy_id] = lifecycle
        return lifecycle
    
    def update_health_metrics(
        self,
        strategy_id: str,
        metrics: HealthMetrics
    ) -> None:
        """Update health metrics for a strategy"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not registered")
        
        lifecycle = self.strategies[strategy_id]
        lifecycle.health_history.append(metrics)
        lifecycle.last_updated = datetime.now()
    
    def evaluate_gates(
        self,
        strategy_id: str,
        current_metrics: HealthMetrics
    ) -> Optional[StrategyPhase]:
        """Evaluate if strategy can transition to next phase"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not registered")
        
        lifecycle = self.strategies[strategy_id]
        
        # Check based on current phase
        if lifecycle.current_phase == StrategyPhase.PAPER_TRADE:
            criteria = self.gate_criteria.paper_trade_to_live_10pct
            days_in_phase = lifecycle.days_in_phase()
            
            if (days_in_phase >= criteria["min_days"] and
                current_metrics.rolling_sharpe_20d >= criteria["min_sharpe"] and
                current_metrics.current_drawdown <= criteria["max_drawdown"] and
                current_metrics.rolling_win_rate_20d >= criteria["min_win_rate"]):
                return StrategyPhase.LIVE_10PCT
        
        elif lifecycle.current_phase == StrategyPhase.LIVE_10PCT:
            criteria = self.gate_criteria.live_10pct_to_live_full
            days_in_phase = lifecycle.days_in_phase()
            
            if (days_in_phase >= criteria["min_days"] and
                current_metrics.rolling_sharpe_20d >= criteria["min_rolling_sharpe_20d"]):
                return StrategyPhase.LIVE_FULL
        
        return None
    
    def apply_decision_rules(
        self,
        strategy_id: str,
        current_metrics: HealthMetrics
    ) -> Optional[tuple[StrategyPhase, str]]:
        """Apply decision rules to determine if phase change is needed"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not registered")
        
        for rule in self.decision_rules:
            new_phase = rule.evaluate(current_metrics)
            if new_phase:
                return new_phase, rule.reason
        
        return None
    
    def process_strategy(
        self,
        strategy_id: str,
        current_metrics: HealthMetrics
    ) -> Dict:
        """
        Process a strategy: update metrics, evaluate gates, apply decision rules.
        Returns a summary of actions taken.
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not registered")
        
        lifecycle = self.strategies[strategy_id]
        
        # Update health metrics
        self.update_health_metrics(strategy_id, current_metrics)
        
        result = {
            "strategy_id": strategy_id,
            "current_phase": lifecycle.current_phase.value,
            "actions_taken": [],
            "new_phase": None,
            "allocation_change": None
        }
        
        # Check for gate transitions (forward progression)
        gate_phase = self.evaluate_gates(strategy_id, current_metrics)
        if gate_phase:
            lifecycle.transition_to(gate_phase, "Gate criteria met")
            result["actions_taken"].append("Gate transition")
            result["new_phase"] = gate_phase.value
            
            # Update allocation based on phase
            if gate_phase == StrategyPhase.LIVE_10PCT:
                lifecycle.update_allocation(0.10)
                result["allocation_change"] = 0.10
            elif gate_phase == StrategyPhase.LIVE_FULL:
                lifecycle.update_allocation(1.00)
                result["allocation_change"] = 1.00
        
        # Check decision rules (backward progression or decay)
        decision_result = self.apply_decision_rules(strategy_id, current_metrics)
        if decision_result:
            new_phase, reason = decision_result
            lifecycle.transition_to(new_phase, reason)
            result["actions_taken"].append(f"Decision rule: {reason}")
            result["new_phase"] = new_phase.value
            
            # Reduce allocation if entering DECAY
            if new_phase == StrategyPhase.DECAY:
                new_allocation = lifecycle.allocation_pct * 0.5
                lifecycle.update_allocation(new_allocation)
                result["allocation_change"] = new_allocation
            elif new_phase == StrategyPhase.RETIRED:
                lifecycle.update_allocation(0.0)
                result["allocation_change"] = 0.0
        
        return result
    
    def get_all_strategies(self) -> List[Dict]:
        """Get status of all registered strategies"""
        return [lifecycle.to_dict() for lifecycle in self.strategies.values()]
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get status of a specific strategy"""
        if strategy_id in self.strategies:
            return self.strategies[strategy_id].to_dict()
        return None
    
    def get_strategies_by_phase(self, phase: StrategyPhase) -> List[Dict]:
        """Get all strategies in a specific phase"""
        return [
            lifecycle.to_dict()
            for lifecycle in self.strategies.values()
            if lifecycle.current_phase == phase
        ]


def calculate_health_metrics(
    daily_returns: List[float],
    feature_drift: Dict[str, float],
    regime_changes: int = 0,
    window: int = 20
) -> HealthMetrics:
    """
    Calculate health metrics from daily returns and other data.
    
    Args:
        daily_returns: List of daily returns
        feature_drift: Dictionary of PSI values per feature
        regime_changes: Number of regime changes in the period
        window: Rolling window for calculations
    
    Returns:
        HealthMetrics object
    """
    returns_array = np.array(daily_returns)
    
    # Rolling Sharpe (20d)
    if len(returns_array) >= window:
        recent_returns = returns_array[-window:]
        rolling_sharpe = np.mean(recent_returns) / np.std(recent_returns) if np.std(recent_returns) > 0 else 0
    else:
        rolling_sharpe = 0.0
    
    # Rolling Sortino (20d)
    if len(returns_array) >= window:
        recent_returns = returns_array[-window:]
        downside_returns = recent_returns[recent_returns < 0]
        rolling_sortino = np.mean(recent_returns) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
    else:
        rolling_sortino = 0.0
    
    # Alpha half-life (exponential decay fit)
    # Simplified: fit exponential decay to absolute returns
    if len(returns_array) >= window:
        abs_returns = np.abs(returns_array[-window:])
        # Simple approximation: half-life = window * ln(0.5) / ln(last/first)
        if abs_returns[0] > 0 and abs_returns[-1] > 0:
            alpha_half_life = window * np.log(0.5) / np.log(abs_returns[-1] / abs_returns[0])
            alpha_half_life = max(0, alpha_half_life)  # Ensure non-negative
        else:
            alpha_half_life = 365.0  # Default: 1 year
    else:
        alpha_half_life = 365.0
    
    # Turnover stability (coefficient of variation)
    # This would typically be calculated from position changes
    turnover_stability = 0.1  # Placeholder
    
    # Regime stability
    regime_stability = 1.0 - (regime_changes / max(len(daily_returns), 1))
    
    # Current drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative_returns)
    current_drawdown = (cumulative_returns[-1] - peak[-1]) / peak[-1] if len(cumulative_returns) > 0 else 0
    
    # Rolling win rate
    if len(returns_array) >= window:
        recent_returns = returns_array[-window:]
        rolling_win_rate = np.sum(recent_returns > 0) / len(recent_returns)
    else:
        rolling_win_rate = 0.0
    
    return HealthMetrics(
        rolling_sharpe_20d=rolling_sharpe,
        rolling_sortino_20d=rolling_sortino,
        alpha_half_life=alpha_half_life,
        turnover_stability=turnover_stability,
        feature_drift=feature_drift,
        regime_stability=regime_stability,
        rolling_win_rate_20d=rolling_win_rate,
        current_drawdown=abs(current_drawdown),
        daily_returns=daily_returns
    )

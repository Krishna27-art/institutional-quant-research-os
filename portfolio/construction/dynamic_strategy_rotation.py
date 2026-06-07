"""
Dynamic Strategy Rotation
Reallocate capital from decaying to growing edges.

Critical for institutional portfolio management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class RotationTrigger(Enum):
    """Triggers for strategy rotation"""
    SHARPE_DECAY = "sharpe_decay"
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    REGIME_CHANGE = "regime_change"
    CROWDING_DETECTED = "crowding_detected"
    SCHEDULED_REBALANCE = "scheduled_rebalance"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy"""
    strategy_id: str
    current_sharpe: float
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    max_drawdown: float
    current_drawdown: float
    last_rotation_date: datetime
    decay_rate: float  # Rate of Sharpe decline


@dataclass
class RotationDecision:
    """Decision to rotate capital"""
    strategy_id: str
    action: str  # "reduce", "increase", "maintain", "kill"
    old_weight: float
    new_weight: float
    trigger: RotationTrigger
    reason: str


class DynamicStrategyRotator:
    """
    Dynamic Strategy Rotator
    
    Reallocates capital from decaying to growing edges.
    
    Rules:
    1. Reduce weight if Sharpe decays > 20% over 90 days
    2. Reduce weight if drawdown > 15%
    3. Rotate on regime change
    4. Reduce weight if factor crowded
    5. Monthly scheduled rebalance
    """
    
    def __init__(self, sharpe_decay_threshold: float = 0.2,
                 drawdown_threshold: float = 0.15,
                 rotation_frequency_days: int = 30):
        self.sharpe_decay_threshold = sharpe_decay_threshold
        self.drawdown_threshold = drawdown_threshold
        self.rotation_frequency_days = rotation_frequency_days
        
        self.strategies: Dict[str, StrategyPerformance] = {}
        self.rotation_history: List[RotationDecision] = []
        self.last_rotation_date: Optional[datetime] = None
        self.current_regime: str = "normal"
    
    def add_strategy(self, performance: StrategyPerformance):
        """Add strategy performance"""
        self.strategies[performance.strategy_id] = performance
    
    def update_performance(self, strategy_id: str, current_sharpe: float,
                         rolling_sharpe_30d: float, rolling_sharpe_90d: float,
                         current_drawdown: float):
        """Update strategy performance"""
        if strategy_id not in self.strategies:
            return
        
        perf = self.strategies[strategy_id]
        perf.current_sharpe = current_sharpe
        perf.rolling_sharpe_30d = rolling_sharpe_30d
        perf.rolling_sharpe_90d = rolling_sharpe_90d
        perf.current_drawdown = current_drawdown
        
        # Calculate decay rate
        if perf.rolling_sharpe_90d > 0:
            perf.decay_rate = (perf.rolling_sharpe_90d - current_sharpe) / perf.rolling_sharpe_90d
        else:
            perf.decay_rate = 0.0
    
    def set_regime(self, regime: str):
        """Set current market regime"""
        self.current_regime = regime
    
    def should_rotate(self) -> bool:
        """Check if rotation is needed"""
        now = datetime.now()
        
        # Check scheduled rebalance
        if self.last_rotation_date is None:
            return True
        
        days_since_rotation = (now - self.last_rotation_date).days
        if days_since_rotation >= self.rotation_frequency_days:
            return True
        
        # Check if any strategy needs immediate rotation
        for strategy_id, perf in self.strategies.items():
            if perf.decay_rate > self.sharpe_decay_threshold:
                return True
            if perf.current_drawdown > self.drawdown_threshold:
                return True
        
        return False
    
    def calculate_rotations(self, current_weights: Dict[str, float],
                          regime_suitability: Optional[Dict[str, float]] = None) -> List[RotationDecision]:
        """
        Calculate rotation decisions.
        
        Args:
            current_weights: Current strategy weights
            regime_suitability: Optional regime suitability scores
        
        Returns:
            List of rotation decisions
        """
        decisions = []
        
        if regime_suitability is None:
            regime_suitability = {sid: 0.5 for sid in self.strategies.keys()}
        
        for strategy_id, perf in self.strategies.items():
            current_weight = current_weights.get(strategy_id, 0)
            new_weight = current_weight
            action = "maintain"
            trigger = RotationTrigger.SCHEDULED_REBALANCE
            reason = ""
            
            # Check Sharpe decay
            if perf.decay_rate > self.sharpe_decay_threshold:
                new_weight = current_weight * 0.5  # Reduce by 50%
                action = "reduce"
                trigger = RotationTrigger.SHARPE_DECAY
                reason = f"Sharpe decayed by {perf.decay_rate:.1%}"
            
            # Check drawdown
            elif perf.current_drawdown > self.drawdown_threshold:
                new_weight = current_weight * 0.3  # Reduce by 70%
                action = "reduce"
                trigger = RotationTrigger.DRAWDOWN_EXCEEDED
                reason = f"Drawdown {perf.current_drawdown:.1%} exceeded threshold"
            
            # Check regime suitability
            elif regime_suitability.get(strategy_id, 0.5) < 0.3:
                new_weight = current_weight * 0.5
                action = "reduce"
                trigger = RotationTrigger.REGIME_CHANGE
                reason = f"Low suitability for {self.current_regime} regime"
            
            # Check for growing edge (increase weight)
            elif perf.decay_rate < -0.1 and perf.current_sharpe > 1.0:
                new_weight = min(current_weight * 1.5, 0.4)  # Increase by 50%, max 40%
                action = "increase"
                trigger = RotationTrigger.SCHEDULED_REBALANCE
                reason = "Growing edge detected"
            
            # Kill strategy if severely degraded
            if perf.current_sharpe < 0.2 and perf.decay_rate > 0.5:
                new_weight = 0.0
                action = "kill"
                trigger = RotationTrigger.SHARPE_DECAY
                reason = "Strategy severely degraded"
            
            decision = RotationDecision(
                strategy_id=strategy_id,
                action=action,
                old_weight=current_weight,
                new_weight=new_weight,
                trigger=trigger,
                reason=reason
            )
            
            decisions.append(decision)
        
        # Normalize weights to sum to 1
        total_weight = sum(d.new_weight for d in decisions)
        if total_weight > 0:
            for decision in decisions:
                decision.new_weight /= total_weight
        
        self.rotation_history.extend(decisions)
        self.last_rotation_date = datetime.now()
        
        return decisions
    
    def get_rotation_summary(self) -> Dict:
        """Get summary of recent rotations"""
        recent_rotations = self.rotation_history[-20:] if self.rotation_history else []
        
        summary = {
            "total_rotations": len(self.rotation_history),
            "recent_rotations": len(recent_rotations),
            "reductions": sum(1 for r in recent_rotations if r.action == "reduce"),
            "increases": sum(1 for r in recent_rotations if r.action == "increase"),
            "kills": sum(1 for r in recent_rotations if r.action == "kill"),
            "last_rotation_date": self.last_rotation_date
        }
        
        return summary
    
    def generate_report(self) -> str:
        """Generate rotation report"""
        summary = self.get_rotation_summary()
        
        report = f"""
Dynamic Strategy Rotation Report
{'=' * 50}
Sharpe Decay Threshold: {self.sharpe_decay_threshold:.1%}
Drawdown Threshold: {self.drawdown_threshold:.1%}
Rotation Frequency: {self.rotation_frequency_days} days
Current Regime: {self.current_regime}

Rotation Summary:
{'-' * 50}
Total Rotations: {summary['total_rotations']}
Recent Rotations: {summary['recent_rotations']}
Reductions: {summary['reductions']}
Increases: {summary['increases']}
Kills: {summary['kills']}
Last Rotation: {summary['last_rotation_date']}

Recent Decisions:
{'-' * 50}
"""
        
        recent_rotations = self.rotation_history[-10:] if self.rotation_history else []
        for decision in recent_rotations:
            report += f"{decision.strategy_id}: {decision.action.upper()}\n"
            report += f"  {decision.old_weight:.2%} -> {decision.new_weight:.2%}\n"
            report += f"  Trigger: {decision.trigger.value}\n"
            report += f"  Reason: {decision.reason}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    rotator = DynamicStrategyRotator(sharpe_decay_threshold=0.2, drawdown_threshold=0.15)
    
    # Add strategies
    rotator.add_strategy(StrategyPerformance(
        strategy_id="momentum",
        current_sharpe=0.75,
        rolling_sharpe_30d=0.8,
        rolling_sharpe_90d=1.0,
        max_drawdown=0.10,
        current_drawdown=0.05,
        last_rotation_date=datetime.now() - timedelta(days=40),
        decay_rate=0.25
    ))
    
    rotator.add_strategy(StrategyPerformance(
        strategy_id="mean_reversion",
        current_sharpe=0.6,
        rolling_sharpe_30d=0.7,
        rolling_sharpe_90d=0.8,
        max_drawdown=0.12,
        current_drawdown=0.08,
        last_rotation_date=datetime.now() - timedelta(days=40),
        decay_rate=0.25
    ))
    
    rotator.add_strategy(StrategyPerformance(
        strategy_id="stat_arb",
        current_sharpe=1.2,
        rolling_sharpe_30d=1.1,
        rolling_sharpe_90d=1.0,
        max_drawdown=0.05,
        current_drawdown=0.02,
        last_rotation_date=datetime.now() - timedelta(days=40),
        decay_rate=-0.2
    ))
    
    # Current weights
    current_weights = {"momentum": 0.33, "mean_reversion": 0.33, "stat_arb": 0.34}
    
    # Calculate rotations
    decisions = rotator.calculate_rotations(current_weights)
    
    print("Rotation Decisions:")
    for decision in decisions:
        print(f"{decision.strategy_id}: {decision.action} ({decision.old_weight:.2%} -> {decision.new_weight:.2%})")
    
    print(rotator.generate_report())

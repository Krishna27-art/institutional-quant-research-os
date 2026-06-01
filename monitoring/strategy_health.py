"""
Strategy Auto-Deactivation on Decay
Based on Institutional Audit Recommendations

Key findings from audit:
- No live tracking of rolling 20-day Sharpe vs. historical
- Strategy becomes unprofitable but still trades
- Need: Strategy health metric + auto-deactivation

Architecture V2 Upgrade - 90-Day Plan Item #5
Priority: P0 (Critical)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json


@dataclass
class StrategyHealth:
    """Health metrics for a strategy"""
    strategy_name: str
    status: str  # "active", "warning", "deactivated"
    rolling_20d_sharpe: float
    backtest_sharpe: float
    sharpe_ratio: float  # rolling / backtest
    consecutive_negative_days: int
    last_30d_pnl: float
    last_30d_drawdown: float
    last_update: datetime


@dataclass
class DeactivationEvent:
    """Record of strategy deactivation"""
    strategy_name: str
    deactivation_time: datetime
    reason: str
    rolling_sharpe: float
    consecutive_negative_days: int
    reactivation_eligible: datetime


class StrategyHealthMonitor:
    """
    Strategy Health Monitor for auto-deactivation.
    
    Deactivation Rules:
    - Rolling 20-day Sharpe < 0 for 10 consecutive days → deactivate
    - Rolling 20-day Sharpe < 0.5 * backtest Sharpe → warning
    - Reactivation eligible after 30 days of deactivated status
    """
    
    def __init__(self):
        self.strategy_health: Dict[str, StrategyHealth] = {}
        self.deactivation_history: List[DeactivationEvent] = []
        self.pnl_history: Dict[str, List[float]] = {}
        self.backtest_sharpe: Dict[str, float] = {}
    
    def register_strategy(self, strategy_name: str, backtest_sharpe: float) -> None:
        """
        Register a strategy with its backtest Sharpe.
        
        Args:
            strategy_name: Strategy name
            backtest_sharpe: Sharpe ratio from backtest
        """
        self.backtest_sharpe[strategy_name] = backtest_sharpe
        self.pnl_history[strategy_name] = []
        
        self.strategy_health[strategy_name] = StrategyHealth(
            strategy_name=strategy_name,
            status="active",
            rolling_20d_sharpe=0.0,
            backtest_sharpe=backtest_sharpe,
            sharpe_ratio=0.0,
            consecutive_negative_days=0,
            last_30d_pnl=0.0,
            last_30d_drawdown=0.0,
            last_update=datetime.now()
        )
    
    def update_pnl(self, strategy_name: str, daily_pnl: float) -> None:
        """
        Update daily PnL for a strategy.
        
        Args:
            strategy_name: Strategy name
            daily_pnl: Daily PnL (currency units)
        """
        if strategy_name not in self.pnl_history:
            self.register_strategy(strategy_name, 1.0)  # Default backtest Sharpe
        
        self.pnl_history[strategy_name].append(daily_pnl)
        
        # Keep only last 30 days
        if len(self.pnl_history[strategy_name]) > 30:
            self.pnl_history[strategy_name].pop(0)
    
    def calculate_rolling_sharpe(self, strategy_name: str, window: int = 20) -> float:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            strategy_name: Strategy name
            window: Rolling window in days
            
        Returns:
            Rolling Sharpe ratio
        """
        if strategy_name not in self.pnl_history:
            return 0.0
        
        pnl = self.pnl_history[strategy_name]
        
        if len(pnl) < window:
            return 0.0
        
        # Use last 'window' days
        recent_pnl = pnl[-window:]
        
        # Calculate returns (assuming equal capital allocation)
        returns = np.array(recent_pnl) / 1e7  # Normalize by ₹1 Cr
        
        # Calculate Sharpe (annualized)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return sharpe
    
    def check_deactivation_criteria(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Check if strategy should be deactivated.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            (should_deactivate, reason) tuple
        """
        if strategy_name not in self.strategy_health:
            return False, "Strategy not registered"
        
        health = self.strategy_health[strategy_name]
        
        if health.status == "deactivated":
            # Check if reactivation is eligible
            reactivation_eligible = health.last_update + timedelta(days=30)
            if datetime.now() >= reactivation_eligible:
                return False, "Reactivation eligible"
            return False, "Already deactivated"
        
        # Check rolling Sharpe < 0 for 10 consecutive days
        if health.rolling_20d_sharpe < 0:
            health.consecutive_negative_days += 1
            if health.consecutive_negative_days >= 10:
                return True, f"Rolling Sharpe < 0 for {health.consecutive_negative_days} days"
        else:
            health.consecutive_negative_days = 0
        
        # Check Sharpe ratio < 0.5
        if health.sharpe_ratio < 0.5:
            return True, f"Sharpe ratio ({health.sharpe_ratio:.2f}) below 0.5 threshold"
        
        return False, "Healthy"
    
    def update_health(self, strategy_name: str) -> StrategyHealth:
        """
        Update health metrics for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Updated StrategyHealth
        """
        if strategy_name not in self.strategy_health:
            self.register_strategy(strategy_name, 1.0)
        
        health = self.strategy_health[strategy_name]
        
        # Calculate rolling Sharpe
        rolling_sharpe = self.calculate_rolling_sharpe(strategy_name)
        health.rolling_20d_sharpe = rolling_sharpe
        
        # Calculate Sharpe ratio
        if health.backtest_sharpe > 0:
            health.sharpe_ratio = rolling_sharpe / health.backtest_sharpe
        
        # Calculate last 30d PnL
        if strategy_name in self.pnl_history:
            health.last_30d_pnl = sum(self.pnl_history[strategy_name])
            
            # Calculate drawdown
            cumulative = np.cumsum(self.pnl_history[strategy_name])
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak * 100 if len(peak) > 0 else 0
            health.last_30d_drawdown = drawdown[-1] if len(drawdown) > 0 else 0
        
        # Determine status
        should_deactivate, reason = self.check_deactivation_criteria(strategy_name)
        
        if should_deactivate:
            health.status = "deactivated"
            
            # Record deactivation event
            event = DeactivationEvent(
                strategy_name=strategy_name,
                deactivation_time=datetime.now(),
                reason=reason,
                rolling_sharpe=health.rolling_20d_sharpe,
                consecutive_negative_days=health.consecutive_negative_days,
                reactivation_eligible=datetime.now() + timedelta(days=30)
            )
            self.deactivation_history.append(event)
        elif health.sharpe_ratio < 0.5:
            health.status = "warning"
        else:
            health.status = "active"
        
        health.last_update = datetime.now()
        
        return health
    
    def reactivate_strategy(self, strategy_name: str) -> bool:
        """
        Attempt to reactivate a deactivated strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            True if reactivated, False otherwise
        """
        if strategy_name not in self.strategy_health:
            return False
        
        health = self.strategy_health[strategy_name]
        
        if health.status != "deactivated":
            return False
        
        # Check if reactivation is eligible
        last_deactivation = [e for e in self.deactivation_history if e.strategy_name == strategy_name]
        if not last_deactivation:
            return False
        
        last_event = last_deactivation[-1]
        if datetime.now() < last_event.reactivation_eligible:
            return False
        
        # Reactivate
        health.status = "active"
        health.consecutive_negative_days = 0
        health.last_update = datetime.now()
        
        return True
    
    def get_health_report(self) -> Dict[str, StrategyHealth]:
        """Get health report for all strategies."""
        for strategy_name in self.strategy_health:
            self.update_health(strategy_name)
        
        return self.strategy_health
    
    def print_health_report(self) -> None:
        """Print health report for all strategies."""
        health_report = self.get_health_report()
        
        print("\n" + "="*60)
        print("STRATEGY HEALTH MONITOR REPORT")
        print("="*60)
        
        for strategy_name, health in health_report.items():
            status_icon = "✅" if health.status == "active" else "⚠️" if health.status == "warning" else "❌"
            print(f"\n{status_icon} {strategy_name}")
            print(f"  Status: {health.status.upper()}")
            print(f"  Rolling 20d Sharpe: {health.rolling_20d_sharpe:.2f}")
            print(f"  Backtest Sharpe: {health.backtest_sharpe:.2f}")
            print(f"  Sharpe Ratio: {health.sharpe_ratio:.2f}")
            print(f"  Consecutive Negative Days: {health.consecutive_negative_days}")
            print(f"  Last 30d PnL: ₹{health.last_30d_pnl:,.2f}")
            print(f"  Last 30d Drawdown: {health.last_30d_drawdown:.2f}%")
        
        print("\nDeactivation History:")
        for event in self.deactivation_history[-5:]:  # Last 5 events
            print(f"  {event.strategy_name}: {event.reason} on {event.deactivation_time}")
        
        print("="*60)
    
    def to_json(self) -> str:
        """Convert health report to JSON."""
        health_report = self.get_health_report()
        
        report_dict = {
            "timestamp": datetime.now().isoformat(),
            "strategies": {
                name: {
                    "status": health.status,
                    "rolling_20d_sharpe": health.rolling_20d_sharpe,
                    "backtest_sharpe": health.backtest_sharpe,
                    "sharpe_ratio": health.sharpe_ratio,
                    "consecutive_negative_days": health.consecutive_negative_days,
                    "last_30d_pnl": health.last_30d_pnl,
                    "last_30d_drawdown": health.last_30d_drawdown,
                    "last_update": health.last_update.isoformat()
                }
                for name, health in health_report.items()
            },
            "deactivation_history": [
                {
                    "strategy_name": e.strategy_name,
                    "deactivation_time": e.deactivation_time.isoformat(),
                    "reason": e.reason,
                    "rolling_sharpe": e.rolling_sharpe,
                    "consecutive_negative_days": e.consecutive_negative_days,
                    "reactivation_eligible": e.reactivation_eligible.isoformat()
                }
                for e in self.deactivation_history
            ]
        }
        
        return json.dumps(report_dict, indent=2)


def run_sample_monitoring():
    """Run sample strategy health monitoring."""
    monitor = StrategyHealthMonitor()
    
    # Register strategies with backtest Sharpe
    monitor.register_strategy("ORB", 1.1)
    monitor.register_strategy("VWAP", 0.9)
    monitor.register_strategy("PCP", 0.7)
    
    # Simulate 30 days of PnL data
    np.random.seed(42)
    
    # ORB: Good performance initially, then decay
    orb_pnl = list(np.random.normal(50000, 100000, 15))  # Good 15 days
    orb_pnl += list(np.random.normal(-50000, 50000, 15))  # Decay 15 days
    
    # VWAP: Consistent performance
    vwap_pnl = list(np.random.normal(30000, 80000, 30))
    
    # PCP: Poor performance
    pcp_pnl = list(np.random.normal(-20000, 60000, 30))
    
    # Update PnL
    for i in range(30):
        monitor.update_pnl("ORB", orb_pnl[i])
        monitor.update_pnl("VWAP", vwap_pnl[i])
        monitor.update_pnl("PCP", pcp_pnl[i])
    
    # Update health and print report
    monitor.print_health_report()
    
    # Export JSON
    print("\nJSON Report:")
    print(monitor.to_json())
    
    return monitor


if __name__ == "__main__":
    run_sample_monitoring()

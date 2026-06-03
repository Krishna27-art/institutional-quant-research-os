"""
Circuit Breaker System
Automatically halts trading when losses exceed thresholds.

Critical for institutional risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class BreakerType(Enum):
    """Types of circuit breakers"""
    DAILY_LOSS = "daily_loss"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    DRAWDOWN = "drawdown"
    VOLATILITY_SPIKE = "volatility_spike"


class BreakerAction(Enum):
    """Actions taken when breaker triggers"""
    HALT_TRADING = "halt_trading"
    REDUCE_POSITION = "reduce_position"
    KILL_STRATEGY = "kill_strategy"
    ALERT_ONLY = "alert_only"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers"""
    # CRITICAL FIX: Adjusted thresholds based on diagnostic report
    # Daily loss breaker: increased from 2% to 5% to prevent premature halting
    daily_loss_threshold_bps: float = 500.0  # 5% daily loss (was 2%)
    daily_loss_action: BreakerAction = BreakerAction.HALT_TRADING
    
    # Consecutive losses breaker: reduced from 5 to 3 to catch failing strategies faster
    consecutive_loss_threshold: int = 3  # 3 consecutive losing days (was 5)
    consecutive_loss_action: BreakerAction = BreakerAction.KILL_STRATEGY
    
    # Drawdown breaker
    max_drawdown_bps: float = 1000.0  # 10% max drawdown
    drawdown_action: BreakerAction = BreakerAction.REDUCE_POSITION
    
    # Volatility spike breaker
    vol_spike_multiplier: float = 3.0  # 3x normal volatility
    vol_spike_action: BreakerAction = BreakerAction.HALT_TRADING
    
    # Reset conditions
    auto_reset_hours: int = 24  # Auto-reset after 24 hours
    manual_reset_required: bool = True  # Require manual reset for kill


@dataclass
class BreakerEvent:
    """Record of a circuit breaker event"""
    timestamp: datetime
    breaker_type: BreakerType
    trigger_value: float
    threshold: float
    action_taken: BreakerAction
    description: str


class CircuitBreakerSystem:
    """
    Circuit Breaker System
    
    Monitors portfolio performance and automatically triggers risk controls
    when thresholds are exceeded.
    
    Rules:
    1. Daily loss > 2% → halt trading
    2. 5 consecutive losing days → kill strategy
    3. Drawdown > 10% → reduce position
    4. Volatility > 3x normal → halt trading
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        
        # State
        self.is_trading_halted: bool = False
        self.is_strategy_killed: bool = False
        self.halt_start_time: Optional[datetime] = None
        self.consecutive_losses: int = 0
        self.current_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        
        # Event history
        self.events: List[BreakerEvent] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
    
    def update(self, current_equity: float, daily_return: float, 
               current_volatility: float, normal_volatility: float) -> Dict[str, bool]:
        """
        Update circuit breaker system with current performance.
        
        Args:
            current_equity: Current portfolio equity
            daily_return: Today's return (in bps)
            current_volatility: Current volatility (annualized)
            normal_volatility: Normal/baseline volatility (annualized)
        
        Returns:
            Dictionary of breaker trigger status
        """
        triggers = {}
        
        # Check if strategy is killed
        if self.is_strategy_killed:
            return {"strategy_killed": True}
        
        # Check if trading is halted
        if self.is_trading_halted:
            # Check if auto-reset time has passed
            if self.halt_start_time and (datetime.now() - self.halt_start_time).total_seconds() > self.config.auto_reset_hours * 3600:
                if not self.config.manual_reset_required:
                    self.reset_halt()
            else:
                return {"trading_halted": True}
        
        # Update performance tracking
        self.daily_returns.append(daily_return)
        self.equity_curve.append(current_equity)
        
        # Update peak equity and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.consecutive_losses = 0  # Reset on new high
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 10000  # bps
        
        # Check daily loss breaker
        triggers["daily_loss"] = self._check_daily_loss(daily_return)
        
        # Check consecutive losses breaker
        triggers["consecutive_losses"] = self._check_consecutive_losses(daily_return)
        
        # Check drawdown breaker
        triggers["drawdown"] = self._check_drawdown()
        
        # Check volatility spike breaker
        triggers["volatility_spike"] = self._check_volatility_spike(current_volatility, normal_volatility)
        
        return triggers
    
    def _check_daily_loss(self, daily_return: float) -> bool:
        """Check if daily loss exceeds threshold"""
        if daily_return < -self.config.daily_loss_threshold_bps:
            self._trigger_breaker(
                BreakerType.DAILY_LOSS,
                abs(daily_return),
                self.config.daily_loss_threshold_bps,
                self.config.daily_loss_action,
                f"Daily loss {daily_return:.2f} bps exceeds threshold {self.config.daily_loss_threshold_bps} bps"
            )
            return True
        return False
    
    def _check_consecutive_losses(self, daily_return: float) -> bool:
        """Check if consecutive losses exceed threshold"""
        if daily_return < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.consecutive_losses >= self.config.consecutive_loss_threshold:
            self._trigger_breaker(
                BreakerType.CONSECUTIVE_LOSSES,
                self.consecutive_losses,
                self.config.consecutive_loss_threshold,
                self.config.consecutive_loss_action,
                f"{self.consecutive_losses} consecutive losing days exceeds threshold {self.config.consecutive_loss_threshold}"
            )
            return True
        return False
    
    def _check_drawdown(self) -> bool:
        """Check if drawdown exceeds threshold"""
        if self.current_drawdown > self.config.max_drawdown_bps:
            self._trigger_breaker(
                BreakerType.DRAWDOWN,
                self.current_drawdown,
                self.config.max_drawdown_bps,
                self.config.drawdown_action,
                f"Drawdown {self.current_drawdown:.2f} bps exceeds threshold {self.config.max_drawdown_bps} bps"
            )
            return True
        return False
    
    def _check_volatility_spike(self, current_vol: float, normal_vol: float) -> bool:
        """Check if volatility spike exceeds threshold"""
        if normal_vol > 0 and current_vol > self.config.vol_spike_multiplier * normal_vol:
            self._trigger_breaker(
                BreakerType.VOLATILITY_SPIKE,
                current_vol,
                self.config.vol_spike_multiplier * normal_vol,
                self.config.vol_spike_action,
                f"Volatility {current_vol:.2f}% exceeds {self.config.vol_spike_multiplier}x normal {normal_vol:.2f}%"
            )
            return True
        return False
    
    def _trigger_breaker(self, breaker_type: BreakerType, trigger_value: float,
                        threshold: float, action: BreakerAction, description: str):
        """Trigger circuit breaker"""
        event = BreakerEvent(
            timestamp=datetime.now(),
            breaker_type=breaker_type,
            trigger_value=trigger_value,
            threshold=threshold,
            action_taken=action,
            description=description
        )
        
        self.events.append(event)
        
        # Execute action
        if action == BreakerAction.HALT_TRADING:
            self.is_trading_halted = True
            self.halt_start_time = datetime.now()
        elif action == BreakerAction.KILL_STRATEGY:
            self.is_strategy_killed = True
            self.is_trading_halted = True
        elif action == BreakerAction.REDUCE_POSITION:
            # Signal to reduce position (handled externally)
            pass
    
    def reset_halt(self):
        """Reset trading halt"""
        self.is_trading_halted = False
        self.halt_start_time = None
    
    def manual_reset(self):
        """Manual reset (for killed strategies)"""
        self.is_strategy_killed = False
        self.is_trading_halted = False
        self.halt_start_time = None
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
    
    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "is_trading_halted": self.is_trading_halted,
            "is_strategy_killed": self.is_strategy_killed,
            "consecutive_losses": self.consecutive_losses,
            "current_drawdown_bps": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "total_events": len(self.events),
            "halt_duration_hours": (datetime.now() - self.halt_start_time).total_seconds() / 3600 if self.halt_start_time else 0
        }
    
    def get_event_history(self, n_recent: int = 10) -> List[BreakerEvent]:
        """Get recent breaker events"""
        return self.events[-n_recent:]
    
    def generate_report(self) -> str:
        """Generate circuit breaker report"""
        status = self.get_status()
        
        report = f"""
Circuit Breaker Status Report
{'=' * 50}
Trading Halted: {status['is_trading_halted']}
Strategy Killed: {status['is_strategy_killed']}
Consecutive Losses: {status['consecutive_losses']}
Current Drawdown: {status['current_drawdown_bps']:.2f} bps
Peak Equity: {status['peak_equity']:.2f}
Total Events: {status['total_events']}

Recent Events:
{'-' * 50}
"""
        
        for event in self.get_event_history():
            report += f"{event.timestamp}: {event.breaker_type.value} - {event.description}\n"
            report += f"  Action: {event.action_taken.value}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    config = CircuitBreakerConfig(
        daily_loss_threshold_bps=200.0,
        consecutive_loss_threshold=5,
        max_drawdown_bps=1000.0,
        vol_spike_multiplier=3.0
    )
    
    breaker = CircuitBreakerSystem(config)
    
    # Simulate trading days
    print("Simulating trading days...")
    for day in range(20):
        equity = 1000000 + np.random.randn() * 10000
        daily_return = np.random.randn() * 50  # Random daily return in bps
        current_vol = 15 + np.random.randn() * 5  # Random volatility
        normal_vol = 15.0
        
        triggers = breaker.update(equity, daily_return, current_vol, normal_vol)
        
        if any(triggers.values()):
            print(f"Day {day+1}: Breaker triggered - {triggers}")
            print(breaker.generate_report())
            break
    
    print("\nFinal Status:")
    print(breaker.generate_report())

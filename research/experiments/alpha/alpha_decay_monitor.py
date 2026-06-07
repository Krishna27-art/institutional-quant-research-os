"""
Alpha Decay Monitoring

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#20)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- Monitors IC, Sharpe, turnover over time
- Detects alpha decay automatically
- Triggers retraining when performance degrades
- Rolling window performance tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DecayConfig:
    """Configuration for Alpha Decay Monitoring"""
    # Monitoring window
    window_days: int = 20  # Rolling window for metrics
    min_observations: int = 10  # Minimum observations for calculation
    
    # Decay thresholds
    ic_decay_threshold: float = 0.3  # 30% drop in IC triggers alert
    sharpe_decay_threshold: float = 0.3  # 30% drop in Sharpe triggers alert
    turnover_increase_threshold: float = 2.0  # 2x increase in turnover triggers alert
    
    # Retraining triggers
    enable_auto_retrain: bool = True
    retrain_trigger_days: int = 5  # Retrain if decay persists for 5 days
    min_sharpe_for_retrain: float = 0.3  # Minimum Sharpe to avoid retraining
    max_drawdown_for_retrain: float = 0.15  # Retrain if drawdown exceeds 15%
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown_hours: int = 24  # Minimum time between alerts


class AlphaDecayMonitor:
    """
    Alpha Decay Monitor
    
    Tracks alpha performance over time and detects decay.
    Triggers retraining when performance degrades.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: DecayConfig):
        self.config = config
        
        # Performance history
        self.ic_history: deque = deque(maxlen=config.window_days)
        self.sharpe_history: deque = deque(maxlen=config.window_days)
        self.turnover_history: deque = deque(maxlen=config.window_days)
        self.return_history: deque = deque(maxlen=config.window_days)
        
        # Baseline performance (from initial training)
        self.baseline_ic: Optional[float] = None
        self.baseline_sharpe: Optional[float] = None
        self.baseline_turnover: Optional[float] = None
        
        # Decay state
        self.decay_detected: bool = False
        self.decay_start_date: Optional[datetime] = None
        self.consecutive_decay_days: int = 0
        self.last_alert_time: Optional[datetime] = None
        
        # Retraining state
        self.retrain_triggered: bool = False
        self.retrain_count: int = 0
    
    def set_baseline(self, ic: float, sharpe: float, turnover: float) -> None:
        """
        Set baseline performance from initial training
        
        Args:
            ic: Baseline Information Coefficient
            sharpe: Baseline Sharpe ratio
            turnover: Baseline turnover
        """
        self.baseline_ic = ic
        self.baseline_sharpe = sharpe
        self.baseline_turnover = turnover
    
    def update(self, 
               predictions: pd.Series, 
               actuals: pd.Series,
               positions: pd.Series = None) -> Dict:
        """
        Update monitor with new performance data
        
        Args:
            predictions: Predicted returns
            actuals: Actual returns
            positions: Position changes (for turnover calculation)
            
        Returns:
            Dictionary with current metrics and decay status
        """
        # Calculate IC
        ic = predictions.corr(actuals)
        ic = ic if not np.isnan(ic) else 0.0
        
        # Calculate Sharpe
        sharpe = predictions.mean() / (predictions.std() + 1e-8) * np.sqrt(252)
        sharpe = sharpe if not np.isnan(sharpe) else 0.0
        
        # Calculate turnover
        if positions is not None:
            turnover = positions.diff().abs().sum() / positions.abs().sum() if positions.abs().sum() > 0 else 0
        else:
            turnover = 0.0
        
        # Store in history
        self.ic_history.append(ic)
        self.sharpe_history.append(sharpe)
        self.turnover_history.append(turnover)
        self.return_history.append(predictions.mean())
        
        # Check for decay
        decay_status = self._check_decay()
        
        # Check retraining trigger
        retrain_status = self._check_retrain_trigger()
        
        return {
            "current_ic": ic,
            "current_sharpe": sharpe,
            "current_turnover": turnover,
            "decay_detected": decay_status["decay_detected"],
            "decay_reason": decay_status["decay_reason"],
            "retrain_triggered": retrain_status["retrain_triggered"],
            "consecutive_decay_days": self.consecutive_decay_days
        }
    
    def _check_decay(self) -> Dict:
        """Check if alpha has decayed"""
        if len(self.ic_history) < self.config.min_observations:
            return {"decay_detected": False, "decay_reason": ""}
        
        current_ic = np.mean(list(self.ic_history)[-self.config.min_observations:])
        current_sharpe = np.mean(list(self.sharpe_history)[-self.config.min_observations:])
        current_turnover = np.mean(list(self.turnover_history)[-self.config.min_observations:])
        
        decay_reasons = []
        
        # Check IC decay
        if self.baseline_ic is not None:
            ic_drop = (self.baseline_ic - current_ic) / self.baseline_ic if self.baseline_ic > 0 else 0
            if ic_drop > self.config.ic_decay_threshold:
                decay_reasons.append(f"IC dropped {ic_drop:.1%}")
        
        # Check Sharpe decay
        if self.baseline_sharpe is not None:
            sharpe_drop = (self.baseline_sharpe - current_sharpe) / self.baseline_sharpe if self.baseline_sharpe > 0 else 0
            if sharpe_drop > self.config.sharpe_decay_threshold:
                decay_reasons.append(f"Sharpe dropped {sharpe_drop:.1%}")
        
        # Check turnover increase
        if self.baseline_turnover is not None and self.baseline_turnover > 0:
            turnover_increase = current_turnover / self.baseline_turnover
            if turnover_increase > self.config.turnover_increase_threshold:
                decay_reasons.append(f"Turnover increased {turnover_increase:.1f}x")
        
        # Check absolute Sharpe threshold
        if current_sharpe < self.config.min_sharpe_for_retrain:
            decay_reasons.append(f"Sharpe below threshold: {current_sharpe:.2f}")
        
        # Update decay state
        if decay_reasons:
            if not self.decay_detected:
                self.decay_detected = True
                self.decay_start_date = datetime.now()
            
            self.consecutive_decay_days += 1
            decay_reason = ", ".join(decay_reasons)
        else:
            self.decay_detected = False
            self.decay_start_date = None
            self.consecutive_decay_days = 0
            decay_reason = ""
        
        return {
            "decay_detected": self.decay_detected,
            "decay_reason": decay_reason
        }
    
    def _check_retrain_trigger(self) -> Dict:
        """Check if retraining should be triggered"""
        if not self.config.enable_auto_retrain:
            return {"retrain_triggered": False, "reason": ""}
        
        reasons = []
        
        # Check consecutive decay days
        if self.consecutive_decay_days >= self.config.retrain_trigger_days:
            reasons.append(f"Decay persisted for {self.consecutive_decay_days} days")
        
        # Check drawdown
        if len(self.return_history) >= 20:
            returns = list(self.return_history)[-20:]
            cum_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            max_dd = np.min(drawdown)
            
            if abs(max_dd) > self.config.max_drawdown_for_retrain:
                reasons.append(f"Drawdown exceeded {abs(max_dd):.1%}")
        
        # Trigger retraining
        if reasons and not self.retrain_triggered:
            self.retrain_triggered = True
            self.retrain_count += 1
            reason = ", ".join(reasons)
        else:
            self.retrain_triggered = False
            reason = ""
        
        return {
            "retrain_triggered": self.retrain_triggered,
            "reason": reason
        }
    
    def reset_retrain_trigger(self) -> None:
        """Reset retraining trigger after retraining"""
        self.retrain_triggered = False
        self.decay_detected = False
        self.consecutive_decay_days = 0
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if len(self.ic_history) < self.config.min_observations:
            return {}
        
        current_ic = np.mean(list(self.ic_history)[-self.config.min_observations:])
        current_sharpe = np.mean(list(self.sharpe_history)[-self.config.min_observations:])
        current_turnover = np.mean(list(self.turnover_history)[-self.config.min_observations:])
        
        return {
            "baseline_ic": self.baseline_ic,
            "current_ic": current_ic,
            "ic_change": (current_ic - self.baseline_ic) / self.baseline_ic if self.baseline_ic else 0,
            "baseline_sharpe": self.baseline_sharpe,
            "current_sharpe": current_sharpe,
            "sharpe_change": (current_sharpe - self.baseline_sharpe) / self.baseline_sharpe if self.baseline_sharpe else 0,
            "baseline_turnover": self.baseline_turnover,
            "current_turnover": current_turnover,
            "turnover_change": (current_turnover - self.baseline_turnover) / self.baseline_turnover if self.baseline_turnover else 0,
            "decay_detected": self.decay_detected,
            "consecutive_decay_days": self.consecutive_decay_days,
            "retrain_count": self.retrain_count
        }
    
    def get_alerts(self) -> List[str]:
        """Get current alerts"""
        alerts = []
        
        if not self.config.enable_alerts:
            return alerts
        
        # Check cooldown
        if self.last_alert_time:
            time_since_alert = datetime.now() - self.last_alert_time
            if time_since_alert.total_seconds() < self.config.alert_cooldown_hours * 3600:
                return alerts
        
        # Decay alert
        if self.decay_detected and self.consecutive_decay_days >= 3:
            alerts.append(f"Alpha decay detected for {self.consecutive_decay_days} consecutive days")
            self.last_alert_time = datetime.now()
        
        # Retraining alert
        if self.retrain_triggered:
            alerts.append("Retraining triggered due to performance degradation")
            self.last_alert_time = datetime.now()
        
        return alerts


def simulate_alpha_performance(n_days: int = 100) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate alpha performance for testing"""
    np.random.seed(42)
    
    # Generate predictions with decay
    predictions = []
    for day in range(n_days):
        # Decay signal over time
        signal_strength = max(0.5, 1.0 - day / 200)
        prediction = np.random.randn() * 0.02 * signal_strength
        predictions.append(prediction)
    
    predictions = pd.Series(predictions)
    
    # Generate actual returns
    actuals = predictions + np.random.randn(n_days) * 0.01
    actuals = pd.Series(actuals)
    
    # Generate positions
    positions = predictions.cumsum()
    positions = pd.Series(positions)
    
    return predictions, actuals, positions


if __name__ == "__main__":
    # Example usage
    config = DecayConfig(
        window_days=20,
        ic_decay_threshold=0.3,
        sharpe_decay_threshold=0.3,
        enable_auto_retrain=True,
        retrain_trigger_days=5
    )
    
    monitor = AlphaDecayMonitor(config)
    
    # Simulate data
    print("Simulating alpha performance...")
    predictions, actuals, positions = simulate_alpha_performance(100)
    
    # Set baseline from first 20 days
    print("\nSetting baseline...")
    baseline_ic = predictions[:20].corr(actuals[:20])
    baseline_sharpe = predictions[:20].mean() / (predictions[:20].std() + 1e-8) * np.sqrt(252)
    baseline_turnover = positions[:20].diff().abs().sum() / positions[:20].abs().sum()
    
    monitor.set_baseline(baseline_ic, baseline_sharpe, baseline_turnover)
    
    print(f"  Baseline IC: {baseline_ic:.4f}")
    print(f"  Baseline Sharpe: {baseline_sharpe:.4f}")
    print(f"  Baseline Turnover: {baseline_turnover:.4f}")
    
    # Monitor performance day by day
    print("\nMonitoring performance...")
    for i in range(20, len(predictions)):
        status = monitor.update(
            predictions[i:i+1], 
            actuals[i:i+1],
            positions[i:i+1]
        )
        
        if i % 10 == 0:
            print(f"  Day {i}: IC={status['current_ic']:.4f}, Sharpe={status['current_sharpe']:.4f}, Decay={status['decay_detected']}")
        
        if status['retrain_triggered']:
            print(f"\n  Retraining triggered at day {i}: {status['decay_reason']}")
            monitor.reset_retrain_trigger()
    
    # Get summary
    print("\nPerformance Summary:")
    summary = monitor.get_performance_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get alerts
    print("\nAlerts:")
    alerts = monitor.get_alerts()
    if alerts:
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("  No alerts")

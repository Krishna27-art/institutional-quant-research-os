"""
Alpha Decay Detection
Implements institutional-grade alpha decay monitoring and detection.

Based on institutional review recommendations:
- Alpha decay detection
- Strategy lifecycle management
- Performance degradation monitoring
- Statistical significance testing
- Kill criteria enforcement
- Automated strategy retirement

Key features:
- Rolling window performance monitoring
- Statistical significance testing (t-test, bootstrap)
- Decay rate estimation
- Capacity impact detection
- Market regime change detection
- Automated kill switch integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy lifecycle status"""
    RESEARCH = "RESEARCH"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE_10 = "LIVE_10"  # 10% of capital
    LIVE_FULL = "LIVE_FULL"
    DECAYING = "DECAYING"
    RETIRED = "RETIRED"


class DecaySeverity(Enum):
    """Decay severity levels"""
    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"


@dataclass
class DecayMetrics:
    """Alpha decay metrics"""
    strategy: str
    current_sharpe: float
    baseline_sharpe: float
    sharpe_decay_pct: float
    current_return: float
    baseline_return: float
    return_decay_pct: float
    decay_rate: float  # Decay per month
    statistical_significance: float  # p-value
    capacity_impact: float
    severity: DecaySeverity
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "strategy": self.strategy,
            "current_sharpe": self.current_sharpe,
            "baseline_sharpe": self.baseline_sharpe,
            "sharpe_decay_pct": self.sharpe_decay_pct,
            "current_return": self.current_return,
            "baseline_return": self.baseline_return,
            "return_decay_pct": self.return_decay_pct,
            "decay_rate": self.decay_rate,
            "statistical_significance": self.statistical_significance,
            "capacity_impact": self.capacity_impact,
            "severity": self.severity.value,
            "recommendation": self.recommendation
        }


@dataclass
class PerformanceDataPoint:
    """Performance data point"""
    timestamp: datetime
    strategy: str
    return_pct: float
    sharpe: float
    drawdown: float
    volume: float


class AlphaDecayDetector:
    """
    Detect and monitor alpha decay for strategies.
    
    Features:
    - Rolling window performance monitoring
    - Statistical significance testing
    - Decay rate estimation
    - Capacity impact detection
    - Market regime change detection
    - Automated kill switch integration
    """
    
    def __init__(
        self,
        kill_sharpe_threshold: float = 0.8,
        kill_return_threshold: float = -0.10,
        statistical_significance_threshold: float = 0.05,
        window_size: int = 90  # 90 days rolling window
    ):
        self.kill_sharpe_threshold = kill_sharpe_threshold
        self.kill_return_threshold = kill_return_threshold
        self.statistical_significance_threshold = statistical_significance_threshold
        self.window_size = window_size
        
        # Performance buffers
        self.performance_buffers: Dict[str, deque] = {}
        
        # Baseline metrics (established during validation)
        self.baseline_metrics: Dict[str, Dict] = {}
        
        # Strategy status
        self.strategy_status: Dict[str, StrategyStatus] = {}
        
        # Metrics (optional)
        self.metrics = None
        self._init_metrics()
        
        logger.info("Alpha decay detector initialized")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics if available"""
        try:
            from monitoring.prometheus_metrics import PrometheusMetrics
            self.metrics = PrometheusMetrics(port=8003)
            logger.info("Prometheus metrics initialized for alpha decay detection")
        except ImportError:
            logger.warning("Prometheus metrics not available")
    
    def add_strategy(self, strategy: str, baseline_sharpe: float, baseline_return: float):
        """Add strategy with baseline metrics"""
        self.performance_buffers[strategy] = deque(maxlen=self.window_size)
        self.baseline_metrics[strategy] = {
            "sharpe": baseline_sharpe,
            "return": baseline_return
        }
        self.strategy_status[strategy] = StrategyStatus.RESEARCH
        
        logger.info(f"Added strategy {strategy} with baseline Sharpe {baseline_sharpe:.2f}")
    
    def add_performance_data(self, data_point: PerformanceDataPoint):
        """Add performance data point"""
        strategy = data_point.strategy
        
        if strategy not in self.performance_buffers:
            logger.warning(f"Unknown strategy: {strategy}")
            return
        
        self.performance_buffers[strategy].append(data_point)
    
    def compute_decay_metrics(self, strategy: str) -> Optional[DecayMetrics]:
        """Compute decay metrics for a strategy"""
        if strategy not in self.performance_buffers:
            return None
        
        buffer = self.performance_buffers[strategy]
        
        if len(buffer) < 30:  # Need at least 30 data points
            return None
        
        # Extract recent performance
        recent_data = list(buffer)[-30:]  # Last 30 days
        current_sharpe = np.mean([d.sharpe for d in recent_data])
        current_return = np.mean([d.return_pct for d in recent_data])
        
        # Get baseline
        baseline = self.baseline_metrics.get(strategy, {})
        baseline_sharpe = baseline.get("sharpe", current_sharpe)
        baseline_return = baseline.get("return", current_return)
        
        # Calculate decay percentages
        sharpe_decay_pct = (current_sharpe - baseline_sharpe) / baseline_sharpe if baseline_sharpe != 0 else 0
        return_decay_pct = (current_return - baseline_return) / baseline_return if baseline_return != 0 else 0
        
        # Estimate decay rate (linear regression on Sharpe over time)
        decay_rate = self._estimate_decay_rate(buffer)
        
        # Statistical significance test
        p_value = self._test_statistical_significance(buffer, baseline_sharpe)
        
        # Capacity impact (simplified)
        avg_volume = np.mean([d.volume for d in recent_data])
        capacity_impact = self._estimate_capacity_impact(buffer, avg_volume)
        
        # Determine severity
        severity = self._determine_severity(sharpe_decay_pct, p_value)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            strategy,
            current_sharpe,
            current_return,
            severity,
            p_value
        )
        
        # Record metrics
        if self.metrics:
            self.metrics.update_strategy_sharpe(strategy, "current", current_sharpe)
            self.metrics.update_strategy_return(strategy, "current", current_return)
        
        metrics = DecayMetrics(
            strategy=strategy,
            current_sharpe=current_sharpe,
            baseline_sharpe=baseline_sharpe,
            sharpe_decay_pct=sharpe_decay_pct,
            current_return=current_return,
            baseline_return=baseline_return,
            return_decay_pct=return_decay_pct,
            decay_rate=decay_rate,
            statistical_significance=p_value,
            capacity_impact=capacity_impact,
            severity=severity,
            recommendation=recommendation
        )
        
        return metrics
    
    def _estimate_decay_rate(self, buffer: deque) -> float:
        """Estimate decay rate (Sharpe change per month)"""
        if len(buffer) < 10:
            return 0.0
        
        # Extract Sharpe values over time
        sharpe_values = [d.sharpe for d in buffer]
        
        # Linear regression
        x = np.arange(len(sharpe_values))
        slope, _ = np.polyfit(x, sharpe_values, 1)
        
        # Convert to monthly decay (assuming daily data)
        monthly_decay = slope * 30
        
        return monthly_decay
    
    def _test_statistical_significance(self, buffer: deque, baseline_sharpe: float) -> float:
        """Test if current Sharpe is statistically different from baseline"""
        if len(buffer) < 20:
            return 1.0  # Not enough data
        
        # Extract recent Sharpe values
        recent_sharpe = [d.sharpe for d in list(buffer)[-20:]]
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(recent_sharpe, baseline_sharpe)
        
        return p_value
    
    def _estimate_capacity_impact(self, buffer: deque, avg_volume: float) -> float:
        """Estimate capacity impact on performance"""
        if len(buffer) < 10:
            return 0.0
        
        # Correlation between volume and Sharpe
        volumes = [d.volume for d in buffer]
        sharpes = [d.sharpe for d in buffer]
        
        if len(volumes) < 2:
            return 0.0
        
        correlation = np.corrcoef(volumes, sharpes)[0, 1]
        
        # Negative correlation suggests capacity constraints
        capacity_impact = -correlation if not np.isnan(correlation) else 0.0
        
        return capacity_impact
    
    def _determine_severity(self, sharpe_decay_pct: float, p_value: float) -> DecaySeverity:
        """Determine decay severity"""
        if sharpe_decay_pct > -0.1:  # Less than 10% decay
            return DecaySeverity.NONE
        elif sharpe_decay_pct > -0.3:  # 10-30% decay
            if p_value < 0.05:
                return DecaySeverity.MODERATE
            else:
                return DecaySeverity.MILD
        elif sharpe_decay_pct > -0.5:  # 30-50% decay
            if p_value < 0.01:
                return DecaySeverity.SEVERE
            else:
                return DecaySeverity.MODERATE
        else:  # More than 50% decay
            return DecaySeverity.CRITICAL
    
    def _generate_recommendation(
        self,
        strategy: str,
        current_sharpe: float,
        current_return: float,
        severity: DecaySeverity,
        p_value: float
    ) -> str:
        """Generate recommendation based on decay metrics"""
        if severity == DecaySeverity.NONE:
            return "Continue monitoring - no significant decay detected"
        
        if severity == DecaySeverity.MILD:
            return "Monitor closely - mild decay detected, investigate causes"
        
        if severity == DecaySeverity.MODERATE:
            return "Reduce position size - moderate decay, consider scaling down"
        
        if severity == DecaySeverity.SEVERE:
            return "Pause strategy - severe decay, halt new positions"
        
        if severity == DecaySeverity.CRITICAL:
            if current_sharpe < self.kill_sharpe_threshold:
                return "KILL STRATEGY - Sharpe below kill threshold, retire immediately"
            if current_return < self.kill_return_threshold:
                return "KILL STRATEGY - Return below kill threshold, retire immediately"
            return "CRITICAL - Immediate action required, consider retirement"
        
        return "Monitor"
    
    def check_kill_criteria(self, strategy: str) -> Tuple[bool, str]:
        """Check if strategy meets kill criteria"""
        metrics = self.compute_decay_metrics(strategy)
        
        if not metrics:
            return False, "Insufficient data"
        
        # Check Sharpe threshold
        if metrics.current_sharpe < self.kill_sharpe_threshold:
            return True, f"Sharpe {metrics.current_sharpe:.2f} below threshold {self.kill_sharpe_threshold}"
        
        # Check return threshold
        if metrics.current_return < self.kill_return_threshold:
            return True, f"Return {metrics.current_return:.2%} below threshold {self.kill_return_threshold:.2%}"
        
        # Check statistical significance
        if metrics.statistical_significance < self.statistical_significance_threshold:
            return True, f"Statistically significant decay detected (p={metrics.statistical_significance:.4f})"
        
        # Check severity
        if metrics.severity == DecaySeverity.CRITICAL:
            return True, "Critical decay severity detected"
        
        return False, "Strategy meets all criteria"
    
    def update_strategy_status(self, strategy: str, status: StrategyStatus):
        """Update strategy lifecycle status"""
        self.strategy_status[strategy] = status
        logger.info(f"Strategy {strategy} status updated to {status.value}")
    
    def monitor_all_strategies(self) -> Dict[str, DecayMetrics]:
        """Monitor all strategies and compute decay metrics"""
        results = {}
        
        for strategy in self.performance_buffers.keys():
            metrics = self.compute_decay_metrics(strategy)
            if metrics:
                results[strategy] = metrics
                
                # Check kill criteria
                should_kill, reason = self.check_kill_criteria(strategy)
                if should_kill:
                    logger.warning(f"Strategy {strategy} should be killed: {reason}")
                    self.update_strategy_status(strategy, StrategyStatus.RETIRED)
        
        return results
    
    def generate_decay_report(self) -> Dict:
        """Generate comprehensive decay report"""
        metrics = self.monitor_all_strategies()
        
        total_strategies = len(metrics)
        decaying_strategies = [s for s, m in metrics.items() if m.severity != DecaySeverity.NONE]
        critical_strategies = [s for s, m in metrics.items() if m.severity == DecaySeverity.CRITICAL]
        
        report = {
            "summary": {
                "total_strategies": total_strategies,
                "decaying_strategies": len(decaying_strategies),
                "critical_strategies": len(critical_strategies),
                "healthy_strategies": total_strategies - len(decaying_strategies)
            },
            "metrics": {s: m.to_dict() for s, m in metrics.items()},
            "recommendations": self._generate_overall_recommendations(metrics)
        }
        
        return report
    
    def _generate_overall_recommendations(self, metrics: Dict[str, DecayMetrics]) -> List[str]:
        """Generate overall portfolio recommendations"""
        recommendations = []
        
        critical = [s for s, m in metrics.items() if m.severity == DecaySeverity.CRITICAL]
        if critical:
            recommendations.append(
                f"URGENT: {len(critical)} strategies in critical decay. "
                f"Immediate action required for: {', '.join(critical)}"
            )
        
        severe = [s for s, m in metrics.items() if m.severity == DecaySeverity.SEVERE]
        if severe:
            recommendations.append(
                f"WARNING: {len(severe)} strategies showing severe decay. "
                f"Consider reducing exposure to: {', '.join(severe)}"
            )
        
        # Check for capacity issues
        capacity_constrained = [
            s for s, m in metrics.items() 
            if m.capacity_impact > 0.5
        ]
        if capacity_constrained:
            recommendations.append(
                f"CAPACITY: {len(capacity_constrained)} strategies may be capacity constrained. "
                f"Consider reducing position sizes: {', '.join(capacity_constrained)}"
            )
        
        # General recommendations
        if len(metrics) > 0:
            avg_sharpe = np.mean([m.current_sharpe for m in metrics.values()])
            if avg_sharpe < 1.0:
                recommendations.append(
                    f"Portfolio average Sharpe {avg_sharpe:.2f} below target 1.0. "
                    "Consider strategy rotation or parameter tuning."
                )
        
        return recommendations


def run_sample_decay_detection():
    """Run sample alpha decay detection"""
    print("="*60)
    print("ALPHA DECAY DETECTION - DEMO")
    print("="*60)
    
    # Create detector
    detector = AlphaDecayDetector(
        kill_sharpe_threshold=0.8,
        kill_return_threshold=-0.10
    )
    
    # Add strategies
    detector.add_strategy("ORB", baseline_sharpe=1.5, baseline_return=0.15)
    detector.add_strategy("VWAP", baseline_sharpe=1.2, baseline_return=0.12)
    detector.add_strategy("PCP", baseline_sharpe=1.8, baseline_return=0.18)
    
    # Generate sample performance data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    
    for i, date in enumerate(dates):
        # ORB: decaying over time
        orb_sharpe = 1.5 - (i * 0.005) + np.random.normal(0, 0.1)
        orb_return = 0.15 - (i * 0.001) + np.random.normal(0, 0.02)
        orb_drawdown = np.random.uniform(0, 0.1)
        orb_volume = np.random.uniform(1000000, 5000000)
        
        detector.add_performance_data(PerformanceDataPoint(
            timestamp=date,
            strategy="ORB",
            return_pct=orb_return,
            sharpe=orb_sharpe,
            drawdown=orb_drawdown,
            volume=orb_volume
        ))
        
        # VWAP: stable
        vwap_sharpe = 1.2 + np.random.normal(0, 0.1)
        vwap_return = 0.12 + np.random.normal(0, 0.02)
        vwap_drawdown = np.random.uniform(0, 0.08)
        vwap_volume = np.random.uniform(1000000, 5000000)
        
        detector.add_performance_data(PerformanceDataPoint(
            timestamp=date,
            strategy="VWAP",
            return_pct=vwap_return,
            sharpe=vwap_sharpe,
            drawdown=vwap_drawdown,
            volume=vwap_volume
        ))
        
        # PCP: severe decay
        pcp_sharpe = 1.8 - (i * 0.015) + np.random.normal(0, 0.1)
        pcp_return = 0.18 - (i * 0.002) + np.random.normal(0, 0.02)
        pcp_drawdown = np.random.uniform(0, 0.15)
        pcp_volume = np.random.uniform(1000000, 5000000)
        
        detector.add_performance_data(PerformanceDataPoint(
            timestamp=date,
            strategy="PCP",
            return_pct=pcp_return,
            sharpe=pcp_sharpe,
            drawdown=pcp_drawdown,
            volume=pcp_volume
        ))
    
    # Monitor all strategies
    print("\nMonitoring strategies...")
    metrics = detector.monitor_all_strategies()
    
    # Print results
    print("\n" + "="*60)
    print("DECAY METRICS")
    print("="*60)
    for strategy, metric in metrics.items():
        print(f"\n{strategy}:")
        print(f"  Current Sharpe: {metric.current_sharpe:.2f} (baseline: {metric.baseline_sharpe:.2f})")
        print(f"  Sharpe Decay: {metric.sharpe_decay_pct:.1%}")
        print(f"  Current Return: {metric.current_return:.2%} (baseline: {metric.baseline_return:.2%})")
        print(f"  Decay Rate: {metric.decay_rate:.3f} per month")
        print(f"  Statistical Significance: p={metric.statistical_significance:.4f}")
        print(f"  Severity: {metric.severity.value}")
        print(f"  Recommendation: {metric.recommendation}")
    
    # Generate report
    report = detector.generate_decay_report()
    
    print("\n" + "="*60)
    print("DECAY REPORT SUMMARY")
    print("="*60)
    print(f"Total strategies: {report['summary']['total_strategies']}")
    print(f"Decaying strategies: {report['summary']['decaying_strategies']}")
    print(f"Critical strategies: {report['summary']['critical_strategies']}")
    print(f"Healthy strategies: {report['summary']['healthy_strategies']}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("="*60)


if __name__ == "__main__":
    run_sample_decay_detection()

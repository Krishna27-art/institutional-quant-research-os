"""
Performance Monitoring with Prometheus/Grafana

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#11)
Expected Sharpe improvement: +0.15–0.25

Methodology:
- Prometheus metrics collection for all system components
- Grafana dashboards for real-time monitoring
- Alerting on performance degradation
- Strategy health tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from collections import deque

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus client not available. Install with: pip install prometheus_client")


@dataclass
class MonitoringConfig:
    """Configuration for Performance Monitoring"""
    prometheus_port: int = 9090
    metrics_update_interval_seconds: int = 5
    alert_sharpe_threshold: float = 0.5  # Alert if Sharpe drops below 0.5
    alert_drawdown_threshold: float = 0.10  # Alert if drawdown exceeds 10%
    alert_latency_threshold_ms: float = 100  # Alert if latency exceeds 100ms
    
    # Metrics retention
    return_history_size: int = 1000
    position_history_size: int = 100


class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # Metrics history
        self.return_history: deque = deque(maxlen=config.return_history_size)
        self.position_history: deque = deque(maxlen=config.position_history_size)
        self.latency_history: deque = deque(maxlen=100)
        
        # Current state
        self.current_pnl: float = 0.0
        self.current_positions: Dict[str, float] = {}
        self.current_drawdown: float = 0.0
        self.current_sharpe: float = 0.0
        
        # Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics"""
        # Portfolio metrics
        self.prom_pnl = Gauge('quant_pnl_total', 'Total PnL')
        self.prom_sharpe = Gauge('quant_sharpe_ratio', 'Sharpe ratio')
        self.prom_drawdown = Gauge('quant_drawdown', 'Current drawdown')
        self.prom_return_1d = Gauge('quant_return_1d', '1-day return')
        
        # Position metrics
        self.prom_gross_exposure = Gauge('quant_gross_exposure', 'Gross exposure')
        self.prom_net_exposure = Gauge('quant_net_exposure', 'Net exposure')
        self.prom_num_positions = Gauge('quant_num_positions', 'Number of positions')
        
        # Latency metrics
        self.prom_signal_latency = Histogram('quant_signal_latency_seconds', 'Signal generation latency')
        self.prom_execution_latency = Histogram('quant_execution_latency_seconds', 'Execution latency')
        self.prom_total_latency = Histogram('quant_total_latency_seconds', 'End-to-end latency')
        
        # Trade metrics
        self.prom_trades_total = Counter('quant_trades_total', 'Total number of trades')
        self.prom_trades_successful = Counter('quant_trades_successful', 'Successful trades')
        self.prom_trades_failed = Counter('quant_trades_failed', 'Failed trades')
        
        # Strategy metrics
        self.prom_strategy_return = Gauge('quant_strategy_return', 'Strategy return', ['strategy'])
        self.prom_strategy_sharpe = Gauge('quant_strategy_sharpe', 'Strategy Sharpe', ['strategy'])
    
    def update_pnl(self, pnl: float) -> None:
        """Update PnL"""
        self.current_pnl = pnl
        self.return_history.append(pnl)
        
        if PROMETHEUS_AVAILABLE:
            self.prom_pnl.set(pnl)
            
            # Calculate 1-day return
            if len(self.return_history) >= 2:
                return_1d = self.return_history[-1] - self.return_history[-2]
                self.prom_return_1d.set(return_1d)
    
    def update_positions(self, positions: Dict[str, float]) -> None:
        """Update positions"""
        self.current_positions = positions
        self.position_history.append(positions.copy())
        
        if PROMETHEUS_AVAILABLE:
            gross = sum(abs(v) for v in positions.values())
            net = sum(positions.values())
            num = len(positions)
            
            self.prom_gross_exposure.set(gross)
            self.prom_net_exposure.set(net)
            self.prom_num_positions.set(num)
    
    def update_sharpe(self, sharpe: float) -> None:
        """Update Sharpe ratio"""
        self.current_sharpe = sharpe
        
        if PROMETHEUS_AVAILABLE:
            self.prom_sharpe.set(sharpe)
    
    def update_drawdown(self, drawdown: float) -> None:
        """Update drawdown"""
        self.current_drawdown = drawdown
        
        if PROMETHEUS_AVAILABLE:
            self.prom_drawdown.set(drawdown)
    
    def record_latency(self, signal_latency: float, execution_latency: float) -> None:
        """Record latency metrics"""
        total_latency = signal_latency + execution_latency
        self.latency_history.append(total_latency)
        
        if PROMETHEUS_AVAILABLE:
            self.prom_signal_latency.observe(signal_latency)
            self.prom_execution_latency.observe(execution_latency)
            self.prom_total_latency.observe(total_latency)
    
    def record_trade(self, success: bool) -> None:
        """Record trade"""
        if PROMETHEUS_AVAILABLE:
            self.prom_trades_total.inc()
            if success:
                self.prom_trades_successful.inc()
            else:
                self.prom_trades_failed.inc()
    
    def update_strategy_metrics(self, strategy_name: str, returns: pd.Series) -> None:
        """Update strategy-specific metrics"""
        if len(returns) < 2:
            return
        
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        total_return = returns.sum()
        
        if PROMETHEUS_AVAILABLE:
            self.prom_strategy_return.labels(strategy=strategy_name).set(total_return)
            self.prom_strategy_sharpe.labels(strategy=strategy_name).set(sharpe)
    
    def get_alerts(self) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        if self.current_sharpe < self.config.alert_sharpe_threshold:
            alerts.append(f"Sharpe ratio below threshold: {self.current_sharpe:.2f} < {self.config.alert_sharpe_threshold}")
        
        if abs(self.current_drawdown) > self.config.alert_drawdown_threshold:
            alerts.append(f"Drawdown exceeds threshold: {abs(self.current_drawdown):.2%} > {self.config.alert_drawdown_threshold:.0%}")
        
        if self.latency_history:
            avg_latency = np.mean(self.latency_history) * 1000  # Convert to ms
            if avg_latency > self.config.alert_latency_threshold_ms:
                alerts.append(f"Average latency exceeds threshold: {avg_latency:.1f}ms > {self.config.alert_latency_threshold_ms}ms")
        
        return alerts
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics"""
        returns_array = np.array(self.return_history) if self.return_history else np.array([])
        
        return {
            "current_pnl": self.current_pnl,
            "current_sharpe": self.current_sharpe,
            "current_drawdown": self.current_drawdown,
            "num_positions": len(self.current_positions),
            "gross_exposure": sum(abs(v) for v in self.current_positions.values()),
            "net_exposure": sum(self.current_positions.values()),
            "avg_latency_ms": np.mean(self.latency_history) * 1000 if self.latency_history else 0,
            "num_returns": len(returns_array),
            "return_mean": returns_array.mean() if len(returns_array) > 0 else 0,
            "return_std": returns_array.std() if len(returns_array) > 0 else 0
        }


class GrafanaDashboardConfig:
    """Configuration for Grafana dashboard"""
    
    @staticmethod
    def get_dashboard_config() -> Dict:
        """Get Grafana dashboard configuration"""
        return {
            "dashboard": {
                "title": "Quant Research OS - Performance Dashboard",
                "panels": [
                    {
                        "title": "PnL Over Time",
                        "targets": [
                            {
                                "expr": "quant_pnl_total",
                                "legendFormat": "Total PnL"
                            }
                        ]
                    },
                    {
                        "title": "Sharpe Ratio",
                        "targets": [
                            {
                                "expr": "quant_sharpe_ratio",
                                "legendFormat": "Sharpe"
                            }
                        ]
                    },
                    {
                        "title": "Drawdown",
                        "targets": [
                            {
                                "expr": "quant_drawdown",
                                "legendFormat": "Drawdown"
                            }
                        ]
                    },
                    {
                        "title": "Exposure",
                        "targets": [
                            {
                                "expr": "quant_gross_exposure",
                                "legendFormat": "Gross"
                            },
                            {
                                "expr": "quant_net_exposure",
                                "legendFormat": "Net"
                            }
                        ]
                    },
                    {
                        "title": "Latency Distribution",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(quant_total_latency_seconds[5m]))",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.99, rate(quant_total_latency_seconds[5m]))",
                                "legendFormat": "99th percentile"
                            }
                        ]
                    },
                    {
                        "title": "Trade Rate",
                        "targets": [
                            {
                                "expr": "rate(quant_trades_total[1m])",
                                "legendFormat": "Trades/sec"
                            }
                        ]
                    },
                    {
                        "title": "Strategy Returns",
                        "targets": [
                            {
                                "expr": "quant_strategy_return",
                                "legendFormat": "{{strategy}}"
                            }
                        ]
                    },
                    {
                        "title": "Strategy Sharpe",
                        "targets": [
                            {
                                "expr": "quant_strategy_sharpe",
                                "legendFormat": "{{strategy}}"
                            }
                        ]
                    }
                ]
            }
        }


class PerformanceMonitor:
    """
    Performance Monitor with Prometheus/Grafana
    
    Collects and exposes metrics for real-time monitoring.
    Provides alerting on performance degradation.
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics = PerformanceMetrics(config)
        
        # Start Prometheus server
        if PROMETHEUS_AVAILABLE:
            start_http_server(config.prometheus_port)
            print(f"Prometheus metrics server started on port {config.prometheus_port}")
    
    def update(self, 
               pnl: float,
               positions: Dict[str, float],
               sharpe: float,
               drawdown: float) -> None:
        """Update all metrics"""
        self.metrics.update_pnl(pnl)
        self.metrics.update_positions(positions)
        self.metrics.update_sharpe(sharpe)
        self.metrics.update_drawdown(drawdown)
    
    def check_alerts(self) -> List[str]:
        """Check for alerts"""
        return self.metrics.get_alerts()
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        return self.metrics.get_metrics_summary()


if __name__ == "__main__":
    # Example usage
    config = MonitoringConfig(
        prometheus_port=9090,
        alert_sharpe_threshold=0.5,
        alert_drawdown_threshold=0.10
    )
    
    monitor = PerformanceMonitor(config)
    
    # Simulate updates
    print("Simulating performance updates...")
    for i in range(100):
        pnl = np.random.randn() * 1000
        positions = {"RELIANCE": np.random.randn() * 100, "TCS": np.random.randn() * 50}
        sharpe = np.random.uniform(0, 2)
        drawdown = -np.random.uniform(0, 0.15)
        
        monitor.update(pnl, positions, sharpe, drawdown)
        
        # Record latency
        signal_latency = np.random.exponential(0.01)
        execution_latency = np.random.exponential(0.02)
        monitor.metrics.record_latency(signal_latency, execution_latency)
        
        # Record trade
        monitor.metrics.record_trade(np.random.random() > 0.1)
        
        time.sleep(0.1)
    
    # Check alerts
    alerts = monitor.check_alerts()
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\nNo alerts")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"\nMetrics Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

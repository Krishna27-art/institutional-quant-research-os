"""
Monitoring with Prometheus/Grafana Structure
Architecture V2 - Quantitative Trading System for Indian Markets

Components:
- Prometheus metrics collection
- Custom metrics for trading system
- Alerting rules
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from typing import Dict, Optional
from datetime import datetime
import time


class TradingMetrics:
    """
    Prometheus metrics for trading system.
    
    Metrics:
    - Counters: Total orders, signals, trades
    - Gauges: Current PnL, positions, latency
    - Histograms: Order execution time, signal generation time
    - Summaries: Feature computation time
    """
    
    def __init__(self):
        # Counter metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders',
            ['symbol', 'side', 'status']
        )
        
        self.signals_total = Counter(
            'trading_signals_total',
            'Total number of signals generated',
            ['alpha_name', 'direction']
        )
        
        self.trades_total = Counter(
            'trading_trades_total',
            'Total number of trades executed',
            ['symbol', 'strategy']
        )
        
        self.risk_checks_total = Counter(
            'trading_risk_checks_total',
            'Total number of risk checks performed',
            ['result']  # 'passed', 'failed'
        )
        
        # Gauge metrics
        self.portfolio_pnl = Gauge(
            'trading_portfolio_pnl',
            'Current portfolio PnL',
            ['type']  # 'total', 'daily', 'unrealized'
        )
        
        self.positions_count = Gauge(
            'trading_positions_count',
            'Current number of positions',
            ['symbol']
        )
        
        self.leverage = Gauge(
            'trading_leverage',
            'Current portfolio leverage'
        )
        
        self.var_99 = Gauge(
            'trading_var_99',
            '99% Value at Risk'
        )
        
        self.regime = Gauge(
            'trading_regime',
            'Current market regime (encoded as int)',
            ['regime_name']
        )
        
        self.alpha_confidence = Gauge(
            'trading_alpha_confidence',
            'Alpha signal confidence',
            ['alpha_name']
        )
        
        # Histogram metrics
        self.order_execution_time = Histogram(
            'trading_order_execution_time_seconds',
            'Order execution time in seconds',
            ['symbol', 'order_type']
        )
        
        self.signal_generation_time = Histogram(
            'trading_signal_generation_time_seconds',
            'Signal generation time in seconds',
            ['alpha_name']
        )
        
        self.feature_computation_time = Histogram(
            'trading_feature_computation_time_seconds',
            'Feature computation time in seconds',
            ['symbol']
        )
        
        self.risk_check_time = Histogram(
            'trading_risk_check_time_seconds',
            'Risk check time in seconds'
        )
        
        # Summary metrics
        self.latency = Summary(
            'trading_latency_seconds',
            'End-to-end latency in seconds'
        )
        
        self.market_data_latency = Summary(
            'trading_market_data_latency_seconds',
            'Market data processing latency in seconds'
        )
    
    def record_order(self, symbol: str, side: str, status: str):
        """Record an order."""
        self.orders_total.labels(symbol=symbol, side=side, status=status).inc()
    
    def record_signal(self, alpha_name: str, direction: str):
        """Record a signal."""
        self.signals_total.labels(alpha_name=alpha_name, direction=direction).inc()
    
    def record_trade(self, symbol: str, strategy: str):
        """Record a trade."""
        self.trades_total.labels(symbol=symbol, strategy=strategy).inc()
    
    def record_risk_check(self, result: str):
        """Record a risk check."""
        self.risk_checks_total.labels(result=result).inc()
    
    def update_portfolio_pnl(self, pnl_type: str, value: float):
        """Update portfolio PnL."""
        self.portfolio_pnl.labels(type=pnl_type).set(value)
    
    def update_positions_count(self, symbol: str, count: int):
        """Update positions count."""
        self.positions_count.labels(symbol=symbol).set(count)
    
    def update_leverage(self, value: float):
        """Update leverage."""
        self.leverage.set(value)
    
    def update_var(self, value: float):
        """Update VaR."""
        self.var_99.set(value)
    
    def update_regime(self, regime_name: str, value: int):
        """Update regime."""
        self.regime.labels(regime_name=regime_name).set(value)
    
    def update_alpha_confidence(self, alpha_name: str, value: float):
        """Update alpha confidence."""
        self.alpha_confidence.labels(alpha_name=alpha_name).set(value)
    
    def time_order_execution(self, symbol: str, order_type: str):
        """Context manager for timing order execution."""
        return self.order_execution_time.labels(symbol=symbol, order_type=order_type).time()
    
    def time_signal_generation(self, alpha_name: str):
        """Context manager for timing signal generation."""
        return self.signal_generation_time.labels(alpha_name=alpha_name).time()
    
    def time_feature_computation(self, symbol: str):
        """Context manager for timing feature computation."""
        return self.feature_computation_time.labels(symbol=symbol).time()
    
    def time_risk_check(self):
        """Context manager for timing risk check."""
        return self.risk_check_time.time()
    
    def observe_latency(self, value: float):
        """Observe latency."""
        self.latency.observe(value)
    
    def observe_market_data_latency(self, value: float):
        """Observe market data latency."""
        self.market_data_latency.observe(value)


class AlertManager:
    """
    Alert manager for trading system.
    
    Alert conditions:
    - Latency spike
    - Circuit breaker hit
    - VaR exceeded
    - Leverage exceeded
    - Position limit exceeded
    """
    
    def __init__(self, metrics: TradingMetrics):
        self.metrics = metrics
        self.alert_history: List[Dict] = []
    
    def check_latency_spike(self, threshold_ms: float = 1000):
        """Check for latency spike."""
        # Get current latency from metrics
        # (In production, this would query the actual metric)
        current_latency = 500  # Placeholder
        
        if current_latency > threshold_ms:
            alert = {
                "type": "latency_spike",
                "severity": "warning",
                "message": f"Latency spike: {current_latency}ms > {threshold_ms}ms",
                "timestamp": datetime.now().isoformat()
            }
            self.alert_history.append(alert)
            return alert
        return None
    
    def check_circuit_breaker(self, daily_pnl_pct: float, threshold: float = -0.03):
        """Check if circuit breaker triggered."""
        if daily_pnl_pct <= threshold:
            alert = {
                "type": "circuit_breaker",
                "severity": "critical",
                "message": f"Circuit breaker triggered: {daily_pnl_pct:.2%} <= {threshold:.2%}",
                "timestamp": datetime.now().isoformat()
            }
            self.alert_history.append(alert)
            return alert
        return None
    
    def check_var_exceeded(self, var: float, threshold: float = 0.02):
        """Check if VaR exceeded."""
        if var > threshold:
            alert = {
                "type": "var_exceeded",
                "severity": "warning",
                "message": f"VaR exceeded: {var:.2%} > {threshold:.2%}",
                "timestamp": datetime.now().isoformat()
            }
            self.alert_history.append(alert)
            return alert
        return None
    
    def check_leverage_exceeded(self, leverage: float, threshold: float = 4.0):
        """Check if leverage exceeded."""
        if leverage >= threshold:
            alert = {
                "type": "leverage_exceeded",
                "severity": "critical",
                "message": f"Leverage exceeded: {leverage:.2f}x >= {threshold:.2f}x",
                "timestamp": datetime.now().isoformat()
            }
            self.alert_history.append(alert)
            return alert
        return None
    
    def check_position_limit(self, position_size_pct: float, threshold: float = 0.05):
        """Check if position limit exceeded."""
        if position_size_pct > threshold:
            alert = {
                "type": "position_limit_exceeded",
                "severity": "warning",
                "message": f"Position limit exceeded: {position_size_pct:.2%} > {threshold:.2%}",
                "timestamp": datetime.now().isoformat()
            }
            self.alert_history.append(alert)
            return alert
        return None
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get recent alerts."""
        return self.alert_history[-n:]
    
    def clear_old_alerts(self, hours: int = 24):
        """Clear alerts older than specified hours."""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        self.alert_history = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]).timestamp() > cutoff
        ]


def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")


# Global metrics instance
metrics = TradingMetrics()
alert_manager = AlertManager(metrics)

"""
Metrics Collector - Prometheus metrics collection
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from typing import Dict, Optional
import time


class MetricsCollector:
    """Collect and expose Prometheus metrics"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        
        # Counters
        self.trade_counter = Counter('trades_total', 'Total number of trades', ['symbol', 'side'])
        self.order_counter = Counter('orders_total', 'Total number of orders', ['status'])
        self.signal_counter = Counter('signals_total', 'Total number of signals generated', ['alpha_id'])
        
        # Gauges
        self.portfolio_value_gauge = Gauge('portfolio_value', 'Current portfolio value')
        self.position_gauge = Gauge('position_size', 'Current position size', ['symbol'])
        self.pnl_gauge = Gauge('daily_pnl', 'Daily PnL')
        self.drawdown_gauge = Gauge('drawdown', 'Current drawdown')
        self.var_gauge = Gauge('var_1d', '1-day VaR')
        self.leverage_gauge = Gauge('leverage', 'Current leverage')
        
        # Histograms
        self.latency_histogram = Histogram('signal_latency_seconds', 'Signal generation latency')
        self.slippage_histogram = Histogram('slippage_bps', 'Slippage in basis points', ['symbol'])
        self.fill_time_histogram = Histogram('fill_time_seconds', 'Order fill time')
        
        # Alpha-specific metrics
        self.alpha_sharpe_gauge = Gauge('alpha_sharpe', 'Alpha Sharpe ratio', ['alpha_id'])
        self.alpha_turnover_gauge = Gauge('alpha_turnover', 'Alpha turnover rate', ['alpha_id'])
        self.alpha_hit_rate_gauge = Gauge('alpha_hit_rate', 'Alpha hit rate', ['alpha_id'])
    
    def start_server(self) -> None:
        """Start Prometheus HTTP server"""
        start_http_server(self.port)
    
    def record_trade(self, symbol: str, side: str) -> None:
        """Record a trade"""
        self.trade_counter.labels(symbol=symbol, side=side).inc()
    
    def record_order(self, status: str) -> None:
        """Record an order"""
        self.order_counter.labels(status=status).inc()
    
    def record_signal(self, alpha_id: str) -> None:
        """Record a signal generation"""
        self.signal_counter.labels(alpha_id=alpha_id).inc()
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value"""
        self.portfolio_value_gauge.set(value)
    
    def update_position(self, symbol: str, size: float) -> None:
        """Update position size"""
        self.position_gauge.labels(symbol=symbol).set(size)
    
    def update_pnl(self, pnl: float) -> None:
        """Update daily PnL"""
        self.pnl_gauge.set(pnl)
    
    def update_drawdown(self, drawdown: float) -> None:
        """Update drawdown"""
        self.drawdown_gauge.set(drawdown)
    
    def update_var(self, var: float) -> None:
        """Update VaR"""
        self.var_gauge.set(var)
    
    def update_leverage(self, leverage: float) -> None:
        """Update leverage"""
        self.leverage_gauge.set(leverage)
    
    def record_latency(self, latency_seconds: float) -> None:
        """Record signal generation latency"""
        self.latency_histogram.observe(latency_seconds)
    
    def record_slippage(self, symbol: str, slippage_bps: float) -> None:
        """Record slippage"""
        self.slippage_histogram.labels(symbol=symbol).observe(slippage_bps)
    
    def record_fill_time(self, fill_time_seconds: float) -> None:
        """Record order fill time"""
        self.fill_time_histogram.observe(fill_time_seconds)
    
    def update_alpha_metrics(self, alpha_id: str, sharpe: float, 
                           turnover: float, hit_rate: float) -> None:
        """Update alpha-specific metrics"""
        self.alpha_sharpe_gauge.labels(alpha_id=alpha_id).set(sharpe)
        self.alpha_turnover_gauge.labels(alpha_id=alpha_id).set(turnover)
        self.alpha_hit_rate_gauge.labels(alpha_id=alpha_id).set(hit_rate)
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        class Timer:
            def __init__(self, collector, name):
                self.collector = collector
                self.name = name
            
            def __enter__(self):
                self.start = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start
                self.collector.latency_histogram.observe(elapsed)
        
        return Timer(self, operation_name)

"""
Structured logging for the institutional quant research platform.

Provides JSON-formatted logging with context for:
- Debugging and troubleshooting
- Audit trail
- Performance monitoring
- Error tracking
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted log messages.
    
    Each log entry includes:
    - timestamp: ISO 8601 format
    - level: INFO, WARNING, ERROR, DEBUG
    - message: Human-readable message
    - context: Additional key-value pairs for context
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        enable_console: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            level: Logging level (default: INFO)
            log_file: Optional file path for log output
            enable_console: Whether to log to console (default: True)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter('%(message)s')
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _log(self, level: str, message: str, **context: Any) -> None:
        """
        Internal logging method that formats as JSON.
        
        Args:
            level: Log level (info, warning, error, debug)
            message: Human-readable message
            **context: Additional key-value pairs for context
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.upper(),
            'message': message,
            **context
        }
        getattr(self.logger, level)(json.dumps(record))
    
    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        self._log('info', message, **context)
    
    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        self._log('warning', message, **context)
    
    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        self._log('error', message, **context)
    
    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        self._log('debug', message, **context)
    
    def exception(self, message: str, **context: Any) -> None:
        """Log exception with traceback."""
        import traceback
        context['traceback'] = traceback.format_exc()
        self._log('error', message, **context)


class PerformanceLogger:
    """
    Performance logger for tracking execution times and metrics.
    
    Useful for:
    - Identifying bottlenecks
    - Monitoring system health
    - Benchmarking
    """
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize performance logger.
        
        Args:
            logger: StructuredLogger instance
        """
        self.logger = logger
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str, **context: Any) -> float:
        """
        End timing an operation and log the duration.
        
        Args:
            operation: Operation name
            **context: Additional context to log
            
        Returns:
            Duration in seconds
        """
        import time
        if operation not in self.start_times:
            self.logger.warning(f"Timer not started for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        self.logger.info(
            f"Operation completed: {operation}",
            operation=operation,
            duration_seconds=duration,
            **context
        )
        return duration
    
    def log_metric(self, metric_name: str, value: float, **context: Any) -> None:
        """
        Log a performance metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            **context: Additional context
        """
        self.logger.info(
            f"Metric: {metric_name}",
            metric_name=metric_name,
            value=value,
            **context
        )


class AuditLogger:
    """
    Audit logger for tracking critical events and decisions.
    
    Used for:
    - Regulatory compliance
    - Trade audit trail
    - Risk management decisions
    - System state changes
    """
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize audit logger.
        
        Args:
            logger: StructuredLogger instance
        """
        self.logger = logger
    
    def log_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: str,
        **context: Any
    ) -> None:
        """Log a trade execution."""
        self.logger.info(
            f"Trade executed: {order_id}",
            event_type="trade",
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy=strategy,
            **context
        )
    
    def log_risk_decision(
        self,
        decision_type: str,
        action: str,
        reason: str,
        **context: Any
    ) -> None:
        """Log a risk management decision."""
        self.logger.info(
            f"Risk decision: {decision_type}",
            event_type="risk_decision",
            decision_type=decision_type,
            action=action,
            reason=reason,
            **context
        )
    
    def log_system_state(
        self,
        component: str,
        state: str,
        **context: Any
    ) -> None:
        """Log a system state change."""
        self.logger.info(
            f"System state change: {component}",
            event_type="system_state",
            component=component,
            state=state,
            **context
        )
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        severity: str = "high",
        **context: Any
    ) -> None:
        """Log an error event."""
        self.logger.error(
            f"Error: {error_type}",
            event_type="error",
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            **context
        )


# Convenience function to get a logger
def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, level=level)

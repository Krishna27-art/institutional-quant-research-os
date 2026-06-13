"""
CRITICAL FIX: Structured logging configuration.

The review noted that logging is inconsistent across modules - some use print(),
some use basic logging, some have no logging at all. This makes debugging and
monitoring difficult in production.

This module provides:
- Centralized logging configuration
- Structured JSON logging for production
- Log level management
- Log rotation and retention
- Context-aware logging (correlation IDs, user tracking)
"""

import logging
import logging.config
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class ContextFilter(logging.Filter):
    """
    Add context information to log records.
    
    Adds correlation IDs, timestamps, and other contextual information
    to all log records for traceability.
    """
    
    def __init__(self):
        super().__init__()
        self.correlation_id = None
        
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracing."""
        self.correlation_id = correlation_id
    
    def filter(self, record):
        """Add context to log record."""
        record.correlation_id = self.correlation_id or str(uuid.uuid4())[:8]
        record.timestamp = datetime.utcnow().isoformat()
        return True


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "json",
    enable_console: bool = True
) -> None:
    """
    Setup structured logging configuration.
    
    CRITICAL FIX: Centralized logging setup for consistent logging across all modules.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        log_format: Log format - "json" for production, "text" for development
        enable_console: Whether to enable console output
    """
    # Create logs directory if file logging is enabled
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Context filter for correlation IDs
    context_filter = ContextFilter()
    
    # Define logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context": {
                "()": "core.logging_config.ContextFilter"
            }
        },
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s %(timestamp)s"
            },
            "text": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": log_format,
                "filters": ["context"],
                "stream": sys.stdout
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"] if enable_console else []
        },
        "loggers": {
            # Module-specific loggers with appropriate levels
            "regime": {"level": "INFO", "propagate": True},
            "backtest": {"level": "INFO", "propagate": True},
            "risk": {"level": "INFO", "propagate": True},
            "execution": {"level": "INFO", "propagate": True},
            "data": {"level": "WARNING", "propagate": True},
            "alpha": {"level": "INFO", "propagate": True},
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": log_format,
            "filters": ["context"],
            "filename": log_file,
            "maxBytes": 100 * 1024 * 1024,  # 100 MB
            "backupCount": 10,
            "encoding": "utf-8"
        }
        config["root"]["handlers"].append("file")
    
    # Apply configuration
    try:
        logging.config.dictConfig(config)
    except Exception as e:
        # Fallback to basic config if JSON formatter not available
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logging.warning(f"Failed to apply structured logging config: {e}. Using basic config.")
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, format={log_format}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                self.logger = self.get_logger()
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(self.__class__.__module__ + "." + self.__class__.__name__)


# Pre-configured logger instances for common modules
def get_regime_logger() -> logging.Logger:
    """Get logger for regime detection module."""
    return logging.getLogger("regime")


def get_backtest_logger() -> logging.Logger:
    """Get logger for backtesting module."""
    return logging.getLogger("backtest")


def get_risk_logger() -> logging.Logger:
    """Get logger for risk management module."""
    return logging.getLogger("risk")


def get_execution_logger() -> logging.Logger:
    """Get logger for execution module."""
    return logging.getLogger("execution")


def get_data_logger() -> logging.Logger:
    """Get logger for data module."""
    return logging.getLogger("data")


def get_alpha_logger() -> logging.Logger:
    """Get logger for alpha module."""
    return logging.getLogger("alpha")


# Initialize logging on import if not already configured
if not logging.getLogger().handlers:
    setup_logging(log_level="INFO", log_format="text")

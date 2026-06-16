"""
CRITICAL FIX: Comprehensive error handling and retry logic.

The review noted that none of the code stubs include try-except blocks, retry logic
for API calls, or circuit breakers for downstream services. In production, broker APIs
fail, network partitions occur, and Kafka messages are lost.

This module provides:
- Decorator-based retry logic with exponential backoff
- Circuit breaker pattern for downstream services
- Structured exception handling with logging
- Timeout handling for long-running operations
"""

import time
import logging
from functools import wraps
from typing import Callable, Optional, Type, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    timeout_seconds: int = 60  # How long to stay open
    success_threshold: int = 2  # Successes to close from half-open


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    
    Tracks failures and opens the circuit when threshold is exceeded.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()
        
    def record_success(self):
        """Record a successful operation."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after recovery")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset on success in closed state
    
    def record_failure(self):
        """Record a failed operation."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
                )
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened after failure in half-open state")
    
    def can_execute(self) -> bool:
        """Check if operation can execute (circuit is not open)."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if (datetime.now() - self.last_failure_time).total_seconds() > self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to half-open state")
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
        return False


def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_base: Initial backoff time in seconds
        backoff_factor: Multiplier for backoff time
        exceptions: Tuple of exception types to catch
        on_retry: Optional callback called on each retry (exception, attempt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        backoff = backoff_base * (backoff_factor ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {backoff:.2f}s..."
                        )
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        
                        time.sleep(backoff)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator for timeout handling.
    
    Note: This uses threading and is not accurate for CPU-bound tasks.
    For production, use asyncio or signal-based timeout.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = None
            exception = None
            
            def target():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            if exception:
                raise exception
            
            return result
        
        return wrapper
    return decorator


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """
    Decorator for circuit breaker pattern.
    
    Args:
        circuit_breaker: CircuitBreaker instance
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not circuit_breaker.can_execute():
                raise Exception(
                    f"Circuit breaker '{circuit_breaker.name}' is {circuit_breaker.state.value}. "
                    "Operation rejected."
                )
            
            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_return: Any = None,
    log_error: bool = True,
    raise_on_error: bool = False
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        log_error: Whether to log errors
        raise_on_error: Whether to raise exception on error
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
        if raise_on_error:
            raise
        return default_return


class RetryPolicy:
    """
    Configurable retry policy for different scenarios.
    """
    
    # Broker API calls - should retry aggressively
    BROKER_API = {
        'max_attempts': 5,
        'backoff_base': 0.5,
        'backoff_factor': 2.0,
        'exceptions': (ConnectionError, TimeoutError, OSError)
    }
    
    # Database operations - moderate retry
    DATABASE = {
        'max_attempts': 3,
        'backoff_base': 1.0,
        'backoff_factor': 2.0,
        'exceptions': (ConnectionError, TimeoutError)
    }
    
    # External API calls - conservative retry
    EXTERNAL_API = {
        'max_attempts': 3,
        'backoff_base': 2.0,
        'backoff_factor': 1.5,
        'exceptions': (ConnectionError, TimeoutError)
    }
    
    # Market data feeds - fast retry
    MARKET_DATA = {
        'max_attempts': 2,
        'backoff_base': 0.1,
        'backoff_factor': 2.0,
        'exceptions': (ConnectionError, TimeoutError)
    }


def get_retry_decorator(policy_name: str) -> Callable:
    """
    Get retry decorator from policy name.
    
    Args:
        policy_name: Name of retry policy (e.g., 'BROKER_API')
        
    Returns:
        Retry decorator
    """
    policy = getattr(RetryPolicy, policy_name, RetryPolicy.EXTERNAL_API)
    return with_retry(**policy)


# Example circuit breakers for common services
BROKER_API_CIRCUIT = CircuitBreaker("broker_api")
DATABASE_CIRCUIT = CircuitBreaker("database")
MARKET_DATA_CIRCUIT = CircuitBreaker("market_data")

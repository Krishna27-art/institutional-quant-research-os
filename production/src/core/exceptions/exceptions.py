"""
Custom exception hierarchy for the institutional quant research platform.

Provides structured error handling for different failure modes:
- Data errors (missing, invalid, out-of-range)
- Risk errors (limits breached, VaR exceeded)
- Execution errors (order failures, connectivity)
- Configuration errors (invalid settings)
"""

class QuantPlatformError(Exception):
    """Base exception for all platform errors."""
    pass


class DataError(QuantPlatformError):
    """Base class for data-related errors."""
    pass


class MissingDataError(DataError):
    """Raised when required data is missing."""
    pass


class InvalidDataError(DataError):
    """Raised when data is invalid (e.g., negative prices, zero volume)."""
    pass


class DataValidationError(DataError):
    """Raised when data fails validation checks."""
    pass


class DataStaleError(DataError):
    """Raised when data is too old to be useful."""
    pass


class RiskError(QuantPlatformError):
    """Base class for risk-related errors."""
    pass


class RiskLimitExceededError(RiskError):
    """Raised when a risk limit is breached."""
    def __init__(self, limit_type: str, current_value: float, limit_value: float):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        super().__init__(
            f"{limit_type} exceeded: {current_value} > {limit_value}"
        )


class VaRExceededError(RiskError):
    """Raised when VaR exceeds allowed threshold."""
    def __init__(self, var_value: float, threshold: float):
        self.var_value = var_value
        self.threshold = threshold
        super().__init__(
            f"VaR exceeded: {var_value} > {threshold}"
        )


class CircuitBreakerTriggeredError(RiskError):
    """Raised when circuit breaker is triggered."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Circuit breaker triggered: {reason}")


class ExecutionError(QuantPlatformError):
    """Base class for execution-related errors."""
    pass


class OrderError(ExecutionError):
    """Raised when order placement/modification/cancellation fails."""
    pass


class OrderRejectedError(OrderError):
    """Raised when order is rejected by exchange."""
    def __init__(self, order_id: str, reason: str):
        self.order_id = order_id
        self.reason = reason
        super().__init__(f"Order {order_id} rejected: {reason}")


class BrokerConnectivityError(ExecutionError):
    """Raised when broker connection fails."""
    pass


class InsufficientLiquidityError(ExecutionError):
    """Raised when insufficient liquidity to fill order."""
    def __init__(self, symbol: str, requested_qty: float, available_qty: float):
        self.symbol = symbol
        self.requested_qty = requested_qty
        self.available_qty = available_qty
        super().__init__(
            f"Insufficient liquidity for {symbol}: "
            f"requested {requested_qty}, available {available_qty}"
        )


class ConfigurationError(QuantPlatformError):
    """Raised when configuration is invalid."""
    pass


class InvalidParameterError(ConfigurationError):
    """Raised when a parameter is invalid."""
    def __init__(self, parameter_name: str, parameter_value: any, reason: str):
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.reason = reason
        super().__init__(
            f"Invalid parameter {parameter_name}={parameter_value}: {reason}"
        )


class PortfolioError(QuantPlatformError):
    """Base class for portfolio-related errors."""
    pass


class InsufficientCapitalError(PortfolioError):
    """Raised when insufficient capital for trade."""
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient capital: required {required}, available {available}"
        )


class PositionLimitExceededError(PortfolioError):
    """Raised when position limit is exceeded."""
    def __init__(self, symbol: str, current_qty: float, limit_qty: float):
        self.symbol = symbol
        self.current_qty = current_qty
        self.limit_qty = limit_qty
        super().__init__(
            f"Position limit exceeded for {symbol}: "
            f"{current_qty} > {limit_qty}"
        )


class BacktestError(QuantPlatformError):
    """Base class for backtest-related errors."""
    pass


class InsufficientDataError(BacktestError):
    """Raised when insufficient data for backtest."""
    def __init__(self, required_days: int, available_days: int):
        self.required_days = required_days
        self.available_days = available_days
        super().__init__(
            f"Insufficient data: required {required_days} days, "
            f"available {available_days} days"
        )


class LookaheadBiasError(BacktestError):
    """Raised when lookahead bias is detected."""
    pass

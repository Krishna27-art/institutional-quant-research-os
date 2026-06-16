"""
Data Validation Pipeline Architecture

This module implements institutional-grade data validation pipeline that sits between
data ingestion and downstream consumers (feature store, research, execution).

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                          │
│  (Zerodha, Yahoo, Arctic, WebSocket feeds)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Data Validation Pipeline                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Schema       │  │ Freshness    │  │ Consistency  │          │
│  │ Validation   │  │ Monitoring   │  │ Checks       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Statistical  │  │ Business     │  │ Circuit      │          │
│  │ Validation   │  │ Rules        │  │ Breakers     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Truth Database                                 │
│  (Single source of truth for all validated data)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Downstream Consumers                            │
│  (Feature Store, Research Layer, Execution Engine)               │
└─────────────────────────────────────────────────────────────────┘

Key Principles:
1. Validate at every boundary - never trust external data
2. Single source of truth - all consumers read from validated data
3. Fail fast - block bad data before it propagates
4. Immutable audit trail - every validation decision is logged
5. Circuit breakers - automatically block failing data sources
6. Schema-first - enforce strict schema validation
7. Freshness monitoring - detect stale data automatically
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from threading import Lock
import json

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataSourceType(Enum):
    """Types of data sources."""
    ZERODHA = "zerodha"
    YAHOO = "yahoo"
    ARCTIC = "arctic"
    WEBSOCKET = "websocket"
    MANUAL = "manual"


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    is_valid: bool
    severity: ValidationSeverity
    check_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidatedData:
    """Data that has passed validation pipeline."""
    symbol: str
    data: pd.DataFrame
    source: DataSourceType
    validation_results: List[ValidationResult]
    validated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSchema:
    """Schema definition for market data."""
    required_columns: List[str]
    column_types: Dict[str, str]
    value_ranges: Dict[str, tuple] = field(default_factory=dict)
    nullable_columns: List[str] = field(default_factory=list)

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate data against schema."""
        issues = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check column types
        for col, expected_type in self.column_types.items():
            if col in data.columns:
                if expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        issues.append(f"Column {col} should be numeric")
                elif expected_type == "datetime":
                    if not pd.api.types.is_datetime64_any_dtype(data[col]):
                        issues.append(f"Column {col} should be datetime")
        
        # Check value ranges
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in data.columns:
                if (data[col] < min_val).any() or (data[col] > max_val).any():
                    issues.append(f"Column {col} has values outside range [{min_val}, {max_val}]")
        
        # Check null values in non-nullable columns
        non_nullable = set(self.required_columns) - set(self.nullable_columns)
        for col in non_nullable:
            if col in data.columns and data[col].isnull().any():
                issues.append(f"Column {col} has null values but is not nullable")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                check_name="schema_validation",
                message="Schema validation failed",
                details={"issues": issues}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            check_name="schema_validation",
            message="Schema validation passed"
        )


class DataValidator(ABC):
    """Abstract base class for data validators."""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> ValidationResult:
        """Validate data and return result."""
        pass


class SchemaValidator(DataValidator):
    """Validates data against predefined schema."""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
    
    def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> ValidationResult:
        return self.schema.validate(data)


class FreshnessValidator(DataValidator):
    """Validates data freshness."""
    
    def __init__(self, max_staleness_seconds: Dict[str, int]):
        """
        Args:
            max_staleness_seconds: Max staleness per data type
                e.g., {'tick': 5, '1min': 60, '5min': 300}
        """
        self.max_staleness_seconds = max_staleness_seconds
    
    def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> ValidationResult:
        data_type = context.get('data_type', '1min')
        max_staleness = self.max_staleness_seconds.get(data_type, 300)
        
        # Get last timestamp
        if isinstance(data.index, pd.DatetimeIndex):
            last_timestamp = data.index[-1]
        elif 'timestamp' in data.columns:
            last_timestamp = pd.to_datetime(data['timestamp'].iloc[-1])
        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                check_name="freshness_validation",
                message="Cannot determine data timestamp"
            )
        
        staleness = (datetime.now() - last_timestamp).total_seconds()
        
        if staleness > max_staleness:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                check_name="freshness_validation",
                message=f"Data is stale: {staleness:.0f}s old (max: {max_staleness}s)",
                details={"staleness_seconds": staleness, "max_staleness": max_staleness}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            check_name="freshness_validation",
            message=f"Data is fresh: {staleness:.0f}s old"
        )


class ConsistencyValidator(DataValidator):
    """Validates data consistency (OHLC relationships, etc.)."""
    
    def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> ValidationResult:
        issues = []
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                check_name="consistency_validation",
                message=f"Missing required columns: {required_cols}"
            )
        
        # Check OHLC consistency
        if not (data['high'] >= data['low']).all():
            issues.append("High < Low violation")
        if not (data['high'] >= data['open']).all():
            issues.append("High < Open violation")
        if not (data['high'] >= data['close']).all():
            issues.append("High < Close violation")
        if not (data['low'] <= data['open']).all():
            issues.append("Low > Open violation")
        if not (data['low'] <= data['close']).all():
            issues.append("Low > Close violation")
        
        # Check for non-positive prices
        for col in required_cols:
            if (data[col] <= 0).any():
                issues.append(f"Non-positive values in {col}")
        
        # Check for extreme moves
        if 'volume' in data.columns:
            data['price_change'] = data['close'].pct_change()
            extreme_moves = data[abs(data['price_change']) > 0.20]  # 20% move
            if len(extreme_moves) > 0:
                issues.append(f"Extreme price moves detected: {len(extreme_moves)} bars")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                check_name="consistency_validation",
                message="Consistency validation failed",
                details={"issues": issues}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            check_name="consistency_validation",
            message="Consistency validation passed"
        )


class StatisticalValidator(DataValidator):
    """Statistical validation using distribution checks."""
    
    def __init__(self, enable_normality_test: bool = True, enable_outlier_detection: bool = True):
        self.enable_normality_test = enable_normality_test
        self.enable_outlier_detection = enable_outlier_detection
    
    def validate(self, data: pd.DataFrame, context: Dict[str, Any]) -> ValidationResult:
        issues = []
        
        if 'close' not in data.columns or len(data) < 20:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                check_name="statistical_validation",
                message="Insufficient data for statistical validation"
            )
        
        try:
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    check_name="statistical_validation",
                    message="Insufficient returns for statistical validation"
                )
            
            # Check for extreme skewness
            skewness = returns.skew()
            if abs(skewness) > 5:
                issues.append(f"Extreme skewness: {skewness:.2f}")
            
            # Check for extreme kurtosis
            kurtosis = returns.kurtosis()
            if kurtosis > 20:
                issues.append(f"Extreme kurtosis: {kurtosis:.2f}")
            
            # Check for outliers using IQR method
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            outliers = returns[(returns < Q1 - 3 * IQR) | (returns > Q3 + 3 * IQR)]
            if len(outliers) > len(returns) * 0.05:  # More than 5% outliers
                issues.append(f"High outlier count: {len(outliers)}/{len(returns)}")
            
        except Exception as e:
            logger.warning(f"Statistical validation error: {e}")
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                check_name="statistical_validation",
                message=f"Statistical validation error: {e}"
            )
        
        if issues:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                check_name="statistical_validation",
                message="Statistical validation found anomalies",
                details={"issues": issues}
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            check_name="statistical_validation",
            message="Statistical validation passed"
        )


class CircuitBreaker:
    """Circuit breaker for failing data sources."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 300):
        """
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            timeout_seconds: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, datetime] = {}
        self.circuit_open: Dict[str, bool] = {}
        self._lock = Lock()
    
    def record_failure(self, source: str) -> None:
        """Record a failure for a data source."""
        with self._lock:
            self.failure_counts[source] = self.failure_counts.get(source, 0) + 1
            self.last_failure_time[source] = datetime.now()
            
            if self.failure_counts[source] >= self.failure_threshold:
                self.circuit_open[source] = True
                logger.error(f"Circuit breaker OPEN for {source} after {self.failure_counts[source]} failures")
    
    def record_success(self, source: str) -> None:
        """Record a success for a data source."""
        with self._lock:
            self.failure_counts[source] = 0
            if source in self.circuit_open:
                del self.circuit_open[source]
                logger.info(f"Circuit breaker CLOSED for {source}")
    
    def is_circuit_open(self, source: str) -> bool:
        """Check if circuit is open for a data source."""
        with self._lock:
            if source not in self.circuit_open:
                return False
            
            # Check if timeout has elapsed
            last_failure = self.last_failure_time.get(source)
            if last_failure and (datetime.now() - last_failure).total_seconds() > self.timeout_seconds:
                # Attempt recovery
                logger.info(f"Circuit breaker attempting recovery for {source}")
                del self.circuit_open[source]
                return False
            
            return True
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        with self._lock:
            return {
                source: {
                    "circuit_open": self.circuit_open.get(source, False),
                    "failure_count": self.failure_counts.get(source, 0),
                    "last_failure": self.last_failure_time.get(source, None)
                }
                for source in set(list(self.circuit_open.keys()) + list(self.failure_counts.keys()))
            }


class DataValidationPipeline:
    """
    Main data validation pipeline that orchestrates all validators.
    
    This is the single entry point for all data entering the system.
    All downstream consumers (feature store, research, execution) must
    read data that has passed through this pipeline.
    """
    
    # Default schema for OHLCV data
    DEFAULT_OHLCV_SCHEMA = DataSchema(
        required_columns=['open', 'high', 'low', 'close', 'volume'],
        column_types={
            'open': 'numeric',
            'high': 'numeric',
            'low': 'numeric',
            'close': 'numeric',
            'volume': 'numeric'
        },
        value_ranges={
            'open': (0, 1e9),
            'high': (0, 1e9),
            'low': (0, 1e9),
            'close': (0, 1e9),
            'volume': (0, 1e12)
        },
        nullable_columns=[]
    )
    
    def __init__(
        self,
        validators: Optional[List[DataValidator]] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        block_on_failure: bool = True
    ):
        """
        Args:
            validators: List of validators to apply (default: all standard validators)
            circuit_breaker: Circuit breaker for failing sources
            block_on_failure: Whether to block data that fails validation
        """
        self.validators = validators or self._default_validators()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.block_on_failure = block_on_failure
        self.validation_history: List[ValidatedData] = []
        self._lock = Lock()
        
        logger.info(f"DataValidationPipeline initialized with {len(self.validators)} validators")
    
    def _default_validators(self) -> List[DataValidator]:
        """Create default set of validators."""
        return [
            SchemaValidator(self.DEFAULT_OHLCV_SCHEMA),
            FreshnessValidator({
                'tick': 5,
                '1min': 60,
                '5min': 300,
                '15min': 900,
                '1hour': 3600,
                '1day': 86400
            }),
            ConsistencyValidator(),
            StatisticalValidator()
        ]
    
    def validate(
        self,
        symbol: str,
        data: pd.DataFrame,
        source: DataSourceType,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidatedData:
        """
        Validate data through the pipeline.
        
        Args:
            symbol: Stock/index symbol
            data: DataFrame to validate
            source: Data source type
            context: Additional context (data_type, etc.)
        
        Returns:
            ValidatedData with validation results
        """
        context = context or {}
        validation_results = []
        
        # Check circuit breaker
        source_str = source.value if isinstance(source, DataSourceType) else str(source)
        if self.circuit_breaker.is_circuit_open(source_str):
            logger.error(f"Circuit breaker open for {source_str}, rejecting data for {symbol}")
            validation_results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                check_name="circuit_breaker",
                message=f"Circuit breaker open for source {source_str}"
            ))
            return ValidatedData(
                symbol=symbol,
                data=pd.DataFrame(),  # Return empty data
                source=source,
                validation_results=validation_results,
                validated_at=datetime.now(),
                metadata={"blocked_by_circuit_breaker": True}
            )
        
        # Run all validators
        for validator in self.validators:
            try:
                result = validator.validate(data, context)
                validation_results.append(result)
                
                # Log failures
                if not result.is_valid:
                    logger.warning(
                        f"Validation failed for {symbol} from {source}: "
                        f"{result.check_name} - {result.message}"
                    )
                    
            except Exception as e:
                logger.error(f"Validator {validator.__class__.__name__} failed for {symbol}: {e}")
                validation_results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    check_name=validator.__class__.__name__,
                    message=f"Validator error: {e}"
                ))
        
        # Determine overall validity
        has_critical = any(r.severity == ValidationSeverity.CRITICAL for r in validation_results)
        has_error = any(r.severity == ValidationSeverity.ERROR for r in validation_results)
        
        is_valid = not (has_critical or (has_error and self.block_on_failure))
        
        # Record success/failure in circuit breaker
        if is_valid:
            self.circuit_breaker.record_success(source_str)
        else:
            self.circuit_breaker.record_failure(source_str)
        
        # Create validated data
        validated_data = ValidatedData(
            symbol=symbol,
            data=data if is_valid else pd.DataFrame(),
            source=source,
            validation_results=validation_results,
            validated_at=datetime.now(),
            metadata={
                "is_valid": is_valid,
                "has_critical": has_critical,
                "has_error": has_error
            }
        )
        
        # Store in history
        with self._lock:
            self.validation_history.append(validated_data)
            # Keep only last 1000 validations
            if len(self.validation_history) > 1000:
                self.validation_history = self.validation_history[-1000:]
        
        return validated_data
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        with self._lock:
            total = len(self.validation_history)
            if total == 0:
                return {"total": 0}
            
            valid = sum(1 for v in self.validation_history if v.metadata.get("is_valid", False))
            invalid = total - valid
            
            # Count by severity
            severity_counts = {}
            for v in self.validation_history:
                for result in v.validation_results:
                    severity = result.severity.value
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            return {
                "total": total,
                "valid": valid,
                "invalid": invalid,
                "validity_rate": valid / total if total > 0 else 0,
                "severity_counts": severity_counts,
                "circuit_breaker_status": self.circuit_breaker.get_status()
            }
    
    def add_validator(self, validator: DataValidator) -> None:
        """Add a custom validator to the pipeline."""
        self.validators.append(validator)
        logger.info(f"Added validator: {validator.__class__.__name__}")


# Singleton instance
_pipeline: Optional[DataValidationPipeline] = None


def get_validation_pipeline() -> DataValidationPipeline:
    """Get the singleton validation pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataValidationPipeline()
    return _pipeline


if __name__ == "__main__":
    # Test the validation pipeline
    print("Testing Data Validation Pipeline...")
    
    pipeline = DataValidationPipeline()
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    good_data = pd.DataFrame({
        'open': np.random.uniform(1000, 1100, 100),
        'high': np.random.uniform(1100, 1200, 100),
        'low': np.random.uniform(900, 1000, 100),
        'close': np.random.uniform(1000, 1100, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Validate good data
    result = pipeline.validate('RELIANCE', good_data, DataSourceType.YAHOO, {'data_type': '1min'})
    print(f"Good data validation: {result.metadata['is_valid']}")
    
    # Create bad data (missing columns)
    bad_data = pd.DataFrame({
        'open': np.random.uniform(1000, 1100, 100),
        'close': np.random.uniform(1000, 1100, 100)
    }, index=dates)
    
    # Validate bad data
    result = pipeline.validate('RELIANCE', bad_data, DataSourceType.YAHOO, {'data_type': '1min'})
    print(f"Bad data validation: {result.metadata['is_valid']}")
    
    # Print summary
    summary = pipeline.get_validation_summary()
    print(f"\nValidation Summary: {summary}")

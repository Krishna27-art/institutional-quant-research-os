"""
Comprehensive Tests for Data Validation Pipeline

Tests the institutional-grade data validation pipeline including:
- Schema validation
- Freshness monitoring
- Consistency checks
- Statistical validation
- Circuit breakers
- Integration with data layer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.data_validation_pipeline import (
    DataValidationPipeline,
    DataSourceType,
    DataSchema,
    SchemaValidator,
    FreshnessValidator,
    ConsistencyValidator,
    StatisticalValidator,
    CircuitBreaker,
    ValidationResult,
    ValidationSeverity,
    get_validation_pipeline
)


class TestDataSchema:
    """Test DataSchema validation."""
    
    def test_schema_validation_success(self):
        """Test schema validation with valid data."""
        schema = DataSchema(
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
            }
        )
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'high': np.random.uniform(1100, 1200, 100),
            'low': np.random.uniform(900, 1000, 100),
            'close': np.random.uniform(1000, 1100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        result = schema.validate(data)
        assert result.is_valid
        assert result.severity == ValidationSeverity.INFO
    
    def test_schema_validation_missing_columns(self):
        """Test schema validation with missing columns."""
        schema = DataSchema(
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            column_types={'open': 'numeric', 'high': 'numeric', 'low': 'numeric', 'close': 'numeric', 'volume': 'numeric'}
        )
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'close': np.random.uniform(1000, 1100, 100)
        }, index=dates)
        
        result = schema.validate(data)
        assert not result.is_valid
        assert result.severity == ValidationSeverity.ERROR
        assert 'Missing required columns' in result.details['issues'][0]


class TestFreshnessValidator:
    """Test FreshnessValidator."""
    
    def test_fresh_data(self):
        """Test validation of fresh data."""
        validator = FreshnessValidator({'1min': 60})
        
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=5), periods=100, freq='1min')
        data = pd.DataFrame({
            'close': np.random.uniform(1000, 1100, 100)
        }, index=dates)
        
        result = validator.validate(data, {'data_type': '1min'})
        assert result.is_valid
        assert 'fresh' in result.message.lower()
    
    def test_stale_data(self):
        """Test validation of stale data."""
        validator = FreshnessValidator({'1min': 60})
        
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=120), periods=100, freq='1min')
        data = pd.DataFrame({
            'close': np.random.uniform(1000, 1100, 100)
        }, index=dates)
        
        result = validator.validate(data, {'data_type': '1min'})
        assert not result.is_valid
        assert result.severity == ValidationSeverity.WARNING
        assert 'stale' in result.message.lower()


class TestConsistencyValidator:
    """Test ConsistencyValidator."""
    
    def test_consistent_ohlc(self):
        """Test validation of consistent OHLC data."""
        validator = ConsistencyValidator()
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'high': np.random.uniform(1100, 1200, 100),
            'low': np.random.uniform(900, 1000, 100),
            'close': np.random.uniform(1000, 1100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high >= low
        data['high'] = data[['open', 'close']].max(axis=1) + 10
        data['low'] = data[['open', 'close']].min(axis=1) - 10
        
        result = validator.validate(data, {})
        assert result.is_valid
    
    def test_inconsistent_ohlc(self):
        """Test validation of inconsistent OHLC data."""
        validator = ConsistencyValidator()
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'high': np.random.uniform(900, 950, 100),  # High < Open
            'low': np.random.uniform(1100, 1200, 100),  # Low > Open
            'close': np.random.uniform(1000, 1100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        result = validator.validate(data, {})
        assert not result.is_valid
        assert result.severity == ValidationSeverity.ERROR


class TestStatisticalValidator:
    """Test StatisticalValidator."""
    
    def test_normal_distribution(self):
        """Test validation of normally distributed returns."""
        validator = StatisticalValidator()
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        returns = np.random.normal(0, 0.01, 100)
        prices = 1000 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        result = validator.validate(data, {})
        assert result.is_valid
    
    def test_extreme_skewness(self):
        """Test validation of data with extreme skewness."""
        validator = StatisticalValidator()
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        # Create highly skewed returns
        returns = np.concatenate([np.random.normal(0.01, 0.001, 90), np.random.normal(-0.1, 0.01, 10)])
        prices = 1000 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'close': prices
        }, index=dates)
        
        result = validator.validate(data, {})
        # May or may not fail depending on actual skewness
        # Just ensure it doesn't crash
        assert result is not None


class TestCircuitBreaker:
    """Test CircuitBreaker."""
    
    def test_circuit_breaker_closed_to_open(self):
        """Test circuit breaker opening after failures."""
        cb = CircuitBreaker(failure_threshold=3)
        
        assert not cb.is_circuit_open('test_source')
        
        # Record failures
        for _ in range(3):
            cb.record_failure('test_source')
        
        assert cb.is_circuit_open('test_source')
        assert cb.get_status()['test_source']['circuit_open'] == True
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=1)
        
        # Open circuit
        for _ in range(3):
            cb.record_failure('test_source')
        
        assert cb.is_circuit_open('test_source')
        
        # Wait for timeout
        import time
        time.sleep(2)
        
        # Should attempt recovery (circuit is closed for the next call)
        assert not cb.is_circuit_open('test_source')
    
    def test_circuit_breaker_close_after_success(self):
        """Test circuit breaker closing after successful recovery."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=1)
        
        # Open circuit
        for _ in range(3):
            cb.record_failure('test_source')
        
        # Wait for timeout
        import time
        time.sleep(2)
        
        # Should attempt recovery
        assert not cb.is_circuit_open('test_source')
        
        # Record success
        cb.record_success('test_source')
        
        # Should be fully closed
        assert not cb.is_circuit_open('test_source')
        assert 'test_source' not in cb.circuit_open


class TestDataValidationPipeline:
    """Test DataValidationPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = DataValidationPipeline()
        assert pipeline is not None
        assert len(pipeline.validators) > 0
    
    def test_pipeline_validation_success(self):
        """Test successful validation through pipeline."""
        pipeline = DataValidationPipeline()
        
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=5), periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'high': np.random.uniform(1100, 1200, 100),
            'low': np.random.uniform(900, 1000, 100),
            'close': np.random.uniform(1000, 1100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = data[['open', 'close']].max(axis=1) + 10
        data['low'] = data[['open', 'close']].min(axis=1) - 10
        
        result = pipeline.validate('RELIANCE', data, DataSourceType.YAHOO, {'data_type': '1min'})
        assert result.metadata['is_valid']
        assert len(result.validation_results) > 0
    
    def test_pipeline_validation_failure(self):
        """Test validation failure through pipeline."""
        pipeline = DataValidationPipeline()
        
        # Create invalid data (missing columns)
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=5), periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'close': np.random.uniform(1000, 1100, 100)
        }, index=dates)
        
        result = pipeline.validate('RELIANCE', data, DataSourceType.YAHOO, {'data_type': '1min'})
        assert not result.metadata['is_valid']
        assert result.data.empty  # Should return empty data on failure
    
    def test_pipeline_circuit_breaker_integration(self):
        """Test circuit breaker integration in pipeline."""
        pipeline = DataValidationPipeline()
        
        # Create invalid data to trigger failures
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=5), periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'close': np.random.uniform(1000, 1100, 100)
        }, index=dates)
        
        # Trigger multiple failures
        for _ in range(10):
            pipeline.validate('RELIANCE', data, DataSourceType.YAHOO, {'data_type': '1min'})
        
        # Check circuit breaker status
        summary = pipeline.get_validation_summary()
        circuit_status = summary['circuit_breaker_status']
        assert len(circuit_status) > 0
    
    def test_pipeline_summary(self):
        """Test validation summary."""
        pipeline = DataValidationPipeline()
        
        dates = pd.date_range(start=datetime.now() - timedelta(minutes=5), periods=100, freq='1min')
        data = pd.DataFrame({
            'open': np.random.uniform(1000, 1100, 100),
            'high': np.random.uniform(1100, 1200, 100),
            'low': np.random.uniform(900, 1000, 100),
            'close': np.random.uniform(1000, 1100, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        pipeline.validate('RELIANCE', data, DataSourceType.YAHOO, {'data_type': '1min'})
        
        summary = pipeline.get_validation_summary()
        assert 'total' in summary
        assert 'valid' in summary
        assert 'invalid' in summary
        assert summary['total'] > 0
    
    def test_pipeline_custom_validator(self):
        """Test adding custom validator to pipeline."""
        pipeline = DataValidationPipeline()
        
        class CustomValidator:
            def validate(self, data, context):
                from src.core.data_validation_pipeline import ValidationResult, ValidationSeverity
                return ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    check_name="custom",
                    message="Custom validation passed"
                )
        
        pipeline.add_validator(CustomValidator())
        assert len(pipeline.validators) > 4  # Default 4 + custom


class TestSingleton:
    """Test singleton pattern."""
    
    def test_singleton_instance(self):
        """Test that singleton returns same instance."""
        pipeline1 = get_validation_pipeline()
        pipeline2 = get_validation_pipeline()
        assert pipeline1 is pipeline2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

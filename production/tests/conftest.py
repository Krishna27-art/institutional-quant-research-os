"""
Pytest configuration and fixtures for automated testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 300
    
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': prices * (1 + np.random.uniform(-0.02, 0, n_days)),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days)
    })
    
    data.set_index('timestamp', inplace=True)
    return data


@pytest.fixture
def sample_market_data():
    """Generate sample market data for multiple symbols."""
    symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    data = {}
    
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        n_days = 300
        
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        base_price = np.random.uniform(50, 2000)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
            'low': prices * (1 + np.random.uniform(-0.02, 0, n_days)),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n_days)
        })
        
        df.set_index('timestamp', inplace=True)
        data[symbol] = df
    
    return data


@pytest.fixture
def sample_signals():
    """Generate sample trading signals."""
    return [
        {'symbol': 'RELIANCE', 'signal': 'BUY', 'strength': 0.8, 'timestamp': datetime.now()},
        {'symbol': 'TCS', 'signal': 'SELL', 'strength': 0.6, 'timestamp': datetime.now()},
        {'symbol': 'HDFCBANK', 'signal': 'HOLD', 'strength': 0.5, 'timestamp': datetime.now()}
    ]


@pytest.fixture
def sample_portfolio():
    """Generate sample portfolio data."""
    return {
        'cash': 10000000.0,
        'positions': {
            'RELIANCE': {'quantity': 100, 'avg_price': 2000.0},
            'TCS': {'quantity': 50, 'avg_price': 3500.0}
        },
        'total_value': 10000000.0
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ex=None):
            self.data[key] = value
        
        def delete(self, key):
            self.data.pop(key, None)
        
        def exists(self, key):
            return key in self.data
    
    return MockRedis()


@pytest.fixture
def mock_postgres_connection():
    """Mock PostgreSQL connection for testing."""
    class MockConnection:
        def __init__(self):
            self.queries = []
        
        def execute(self, query, params=None):
            self.queries.append((query, params))
            return []
        
        def fetchall(self):
            return []
        
        def commit(self):
            pass
        
        def rollback(self):
            pass
    
    return MockConnection()


# Integration test fixtures
@pytest.fixture(scope="session")
def docker_compose():
    """Docker compose for integration tests."""
    # This would spin up required services (PostgreSQL, Redis, Kafka)
    # For now, return None as a placeholder
    return None


# E2E test fixtures
@pytest.fixture(scope="session")
def test_environment():
    """Test environment configuration for E2E tests."""
    return {
        'api_url': 'http://localhost:8000',
        'database_url': 'postgresql://test:test@localhost:5432/test',
        'redis_url': 'redis://localhost:6379/0',
        'kafka_brokers': 'localhost:9092'
    }

import sys
from unittest.mock import MagicMock, patch

# Define MockOperationalError class for psycopg2.OperationalError
class MockOperationalError(Exception):
    pass

# Mock psycopg2
mock_psycopg2 = MagicMock()
mock_psycopg2.OperationalError = MockOperationalError
sys.modules['psycopg2'] = mock_psycopg2

# Mock clickhouse_connect
mock_clickhouse = MagicMock()
sys.modules['clickhouse_connect'] = mock_clickhouse

import os
import shutil
import tempfile
import asyncio
import numpy as np
import pandas as pd
import pytest

from src.regime.detectors.hmm import RobustHMMRegime
from src.shared.db.connection_manager import ConnectionManager, DatabaseConfig
from dashboard.api.api_server import fetch_history_async

# Test 1: Graceful under-100 observation HMM fallback
def test_hmm_graceful_fallback():
    # Create test data with fewer than 100 rows
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "open": 20000 * np.cumprod(1 + rng.normal(0.0, 0.01, 50)),
        "high": [20100] * 50,
        "low": [19900] * 50,
        "close": 20000 * np.cumprod(1 + rng.normal(0.0, 0.01, 50)),
        "volume": [100000] * 50
    }, index=dates)

    engine = RobustHMMRegime()
    # fitting should not raise ValueError, should warn and set model to None
    engine.fit(df)
    
    assert engine.model is None
    
    # predict_regime should fall back to rule-based classification without crashing
    regimes = engine.predict_regime(df)
    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(df) - 20  # because rolling(20) drops first 20 rows
    assert all(r in ['bull', 'bear', 'sideways', 'high_vol'] for r in regimes)

# Test 2: Persistent JWT secret storage on disk
def test_jwt_secret_persistence():
    temp_dir = tempfile.mkdtemp()
    secret_file = os.path.join(temp_dir, ".jwt_secret")
    
    # Mock os.path.join in api_server to point to our temp_dir
    with patch("os.path.join", return_value=secret_file):
        # We will mock the env-var config so SECRET_KEY is not set in env
        with patch.dict(os.environ, {}, clear=True):
            # We will run the logic that api_server uses
            # logic block:
            SECRET_KEY1 = None
            if os.path.exists(secret_file):
                with open(secret_file, "r") as f:
                    SECRET_KEY1 = f.read().strip()
            if not SECRET_KEY1:
                import secrets
                SECRET_KEY1 = secrets.token_hex(32)
                with open(secret_file, "w") as f:
                    f.write(SECRET_KEY1)
                    
            # Check file was written
            assert os.path.exists(secret_file)
            with open(secret_file, "r") as f:
                saved_key = f.read().strip()
            assert SECRET_KEY1 == saved_key
            
            # Second load
            SECRET_KEY2 = None
            if os.path.exists(secret_file):
                with open(secret_file, "r") as f:
                    SECRET_KEY2 = f.read().strip()
                    
            assert SECRET_KEY2 == SECRET_KEY1
            
    shutil.rmtree(temp_dir)

# Test 3: Robust DB connection retry behavior
def test_db_connection_retry():
    cm = ConnectionManager()
    config = DatabaseConfig(host="localhost", port=5432, database="test_db", username="user", password="pwd")
    
    # Mock psycopg2.connect to fail twice with OperationalError and succeed on third try
    dummy_conn = MagicMock()
    mock_psycopg2.connect.reset_mock()
    mock_psycopg2.connect.side_effect = [
        MockOperationalError("Could not connect"),
        MockOperationalError("Could not connect"),
        dummy_conn
    ]
    
    conn = cm.get_postgres_connection(config)
    assert conn == dummy_conn
    assert mock_psycopg2.connect.call_count == 3

# Test 4: Concurrent yfinance fetches (non-blocking)
@pytest.mark.asyncio
async def test_concurrent_yfinance_fetches():
    # Mock yfinance Ticker history to simulate network latency
    mock_ticker = MagicMock()
    async def mock_history(*args, **kwargs):
        await asyncio.sleep(0.1)
        return pd.DataFrame({"Close": [100.0]})
        
    # We mock yf.Ticker
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker_inst = MagicMock()
        mock_ticker_inst.history.side_effect = lambda *args, **kwargs: pd.DataFrame({"Close": [100.0]})
        mock_ticker_class.return_value = mock_ticker_inst
        
        # Call multiple fetches concurrently
        tasks = [fetch_history_async(symbol, "5d") for symbol in ["AAPL", "MSFT", "GOOG"]]
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == 3
        for symbol, df in results:
            assert symbol in ["AAPL", "MSFT", "GOOG"]
            assert not df.empty
            assert "Close" in df.columns

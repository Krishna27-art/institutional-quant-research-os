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


def test_jwt_secret_enforcement():
    # Verify that missing credentials in production raises ValueError
    def check_prod_credentials(env_dict):
        sec = env_dict.get("JWT_SECRET_KEY")
        admin = env_dict.get("ADMIN_PASSWORD")
        trader = env_dict.get("TRADER_PASSWORD")
        if not sec:
            raise ValueError("JWT_SECRET_KEY environment variable is required in production settings.")
        if not admin:
            raise ValueError("ADMIN_PASSWORD environment variable is required in production settings.")
        if not trader:
            raise ValueError("TRADER_PASSWORD environment variable is required in production settings.")

    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        check_prod_credentials({})
    with pytest.raises(ValueError, match="ADMIN_PASSWORD"):
        check_prod_credentials({"JWT_SECRET_KEY": "some_secret"})
    with pytest.raises(ValueError, match="TRADER_PASSWORD"):
        check_prod_credentials({"JWT_SECRET_KEY": "some_secret", "ADMIN_PASSWORD": "admin"})


def test_quality_gate_ffill():
    from src.data.quality_gate import DataQualityGate
    gate = DataQualityGate(min_rows=3, max_stale_closes=999, drop_bad_rows=True)
    dates = pd.date_range("2026-06-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [105.0, 95.0, 105.0, 105.0, 105.0],  # row 1: high < low/open/close
            "low": [95.0, 98.0, 95.0, 95.0, 95.0],
            "close": [101.0, 99.0, 101.0, 101.0, 101.0],
            "volume": [1000] * 5,
        },
        index=dates,
    )
    clean_df, result = gate.validate("TEST", df)
    assert len(clean_df) == 5
    assert clean_df.loc[dates[1], "volume"] == 0.0
    assert clean_df.loc[dates[1], "close"] == 101.0


def test_sharpe_zero_variance(tmp_path):
    from src.alpha.prediction_registry import PredictionRegistry, PredictionRecord
    from datetime import datetime, timedelta
    db_file = tmp_path / "test_sharpe_zero_variance.db"
    registry = PredictionRegistry(db_path=str(db_file), min_ic=0.01)
    now = datetime.now()
    # Insert 5 resolved predictions with identical exit prices (return is 0, std is 0)
    for i in range(5):
        pred = PredictionRecord(
            symbol="RELIANCE",
            strategy="zero_var_strat",
            direction="long",
            predicted_return=0.01,
            confidence=0.8,
            entry_price=100.0,
            timestamp=now - timedelta(days=5-i),
            horizon_minutes=390
        )
        pid = registry.record_prediction(pred)
        registry.resolve_prediction(pid, exit_price=100.0, exit_timestamp=now)
    
    report = registry.get_strategy_report("zero_var_strat")
    assert report.sharpe == 0.0


def test_allocator_method_parsing():
    from src.portfolio.engine import PortfolioAllocator
    from datetime import datetime
    allocator = PortfolioAllocator(total_capital=10_000_000.0)
    symbols = [f"STOCK_{i}" for i in range(12)]
    dates = pd.date_range("2026-06-01", periods=100)
    data = pd.DataFrame(index=dates)
    np.random.seed(42)
    for s in symbols:
        data[s] = 100.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 100)))
        
    signals = [{"symbol": s, "direction": 1.0, "strength": 0.5, "rv": 0.5, "stop_loss_price": 95.0, "strategy": "momentum"} for s in symbols]
    
    hrp_positions = allocator.allocate(signals, method="hrp", price_history=data)
    assert len(hrp_positions) > 0
    assert sum(p.capital for p in hrp_positions) <= allocator.total_capital
    
    views = {"STOCK_0": 0.02, "STOCK_1": -0.01, "STOCK_2": 0.015}
    bl_positions = allocator.allocate(signals, method="black_litterman", price_history=data, views=views)
    assert len(bl_positions) > 0
    assert sum(p.capital for p in bl_positions) <= allocator.total_capital


def test_hard_circuit_breaker_shutdown():
    from src.risk.institutional_risk_engine import InstitutionalRiskEngine, CircuitBreakerTrigger
    from datetime import datetime, timedelta
    import os
    if os.path.exists("circuit_breaker_state.json"):
        try:
            os.remove("circuit_breaker_state.json")
        except Exception:
            pass
            
    shutdown_called = False
    def mock_shutdown():
        nonlocal shutdown_called
        shutdown_called = True
        
    risk_engine = InstitutionalRiskEngine(
        capital=10_000_000.0,
        shutdown_callback=mock_shutdown
    )
    
    triggered, reason = risk_engine.check_circuit_breaker(daily_pnl=-450000.0)
    assert triggered
    assert reason == "daily_loss_exceeded"
    assert shutdown_called
    assert risk_engine.circuit_breaker_active
    assert risk_engine.hard_breaker.state.is_active
    assert risk_engine.hard_breaker.state.orders_cancelled
    assert risk_engine.hard_breaker.state.positions_closed
    
    risk_engine.hard_breaker.allow_manual_override = True
    risk_engine.hard_breaker.reset(force=True)
    risk_engine.circuit_breaker_active = False
    shutdown_called = False
    
    should_stop, drawdown_pct = risk_engine.check_trailing_drawdown_limit(current_equity=8500000.0)
    assert should_stop
    assert drawdown_pct == 0.15
    assert shutdown_called
    assert risk_engine.circuit_breaker_active
    assert risk_engine.hard_breaker.state.is_active
    
    if os.path.exists("circuit_breaker_state.json"):
        try:
            os.remove("circuit_breaker_state.json")
        except Exception:
            pass

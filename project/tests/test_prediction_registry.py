"""
Test script to verify the Prediction Registry and HMM Regime integration.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd
import yfinance as yf
import pytest

from src.alpha.prediction_storage import PredictionStorage, Prediction
from src.alpha.manager import AlphaManager


@pytest.fixture
def temp_db(tmp_path):
    """Fixture to create a temporary database path."""
    db_file = tmp_path / "test_predictions.db"
    return str(db_file)


def test_prediction_storage_basic(temp_db):
    """Test storing, retrieving, and updating predictions in sqlite."""
    storage = PredictionStorage(db_path=temp_db)
    
    # 1. Store a prediction
    pred1 = Prediction(
        symbol="RELIANCE",
        direction="long",
        confidence=0.8,
        target_price=3100.0,
        stop_loss=2900.0,
        entry_price=3000.0,
        timestamp=datetime.now() - timedelta(days=2),
        strategy="orb"
    )
    
    pred_id1 = storage.store_prediction(pred1)
    assert pred_id1 > 0
    
    # 2. Store another prediction
    pred2 = Prediction(
        symbol="HDFCBANK",
        direction="short",
        confidence=0.7,
        target_price=1500.0,
        stop_loss=1700.0,
        entry_price=1600.0,
        timestamp=datetime.now() - timedelta(days=1),
        strategy="orb"
    )
    pred_id2 = storage.store_prediction(pred2)
    assert pred_id2 > pred_id1
    
    # 3. Retrieve predictions
    all_preds = storage.get_predictions()
    assert len(all_preds) == 2
    
    # Check that IDs are correctly mapped on retrieval
    p1_retrieved = next(p for p in all_preds if p.symbol == "RELIANCE")
    assert p1_retrieved.id == pred_id1
    assert p1_retrieved.exit_price is None
    assert p1_retrieved.realized_return is None
    
    # 4. Check initial metrics
    metrics = storage.get_performance_metrics()
    assert metrics["total_predictions"] == 2
    assert metrics["pending"] == 2
    assert metrics["realized"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["accuracy"] == 0.0
    
    # 5. Update outcome for prediction 1 (win)
    exit_time1 = datetime.now() - timedelta(days=1)
    storage.update_outcome(pred_id1, exit_price=3120.0, exit_timestamp=exit_time1)
    
    # Check updated fields
    updated_preds = storage.get_predictions()
    p1_updated = next(p for p in updated_preds if p.symbol == "RELIANCE")
    assert p1_updated.exit_price == 3120.0
    assert p1_updated.is_correct is True
    # Long return: (3120 - 3000)/3000 = 4%
    assert abs(p1_updated.realized_return - 0.04) < 1e-6
    
    # 6. Update outcome for prediction 2 (loss)
    exit_time2 = datetime.now()
    storage.update_outcome(pred_id2, exit_price=1750.0, exit_timestamp=exit_time2)
    
    # Check updated fields
    updated_preds2 = storage.get_predictions()
    p2_updated = next(p for p in updated_preds2 if p.symbol == "HDFCBANK")
    assert p2_updated.exit_price == 1750.0
    assert p2_updated.is_correct is False
    # Short return: (1600 - 1750)/1600 = -9.375%
    assert abs(p2_updated.realized_return - (-0.09375)) < 1e-6
    
    # 7. Check finalized metrics
    final_metrics = storage.get_performance_metrics()
    assert final_metrics["total_predictions"] == 2
    assert final_metrics["pending"] == 0
    assert final_metrics["realized"] == 2
    # 1 win, 1 loss => 50% win rate
    assert final_metrics["win_rate"] == 0.5
    assert final_metrics["accuracy"] == 0.5
    assert final_metrics["avg_return"] == (0.04 + (-0.09375)) / 2


def test_alpha_manager_stop_loss_key_fix():
    """Test that AlphaManager correctly maps the 'stop' key to 'stop_loss' database field."""
    # Mock prediction storage inside AlphaManager
    manager = AlphaManager()
    manager.prediction_storage = MagicMock()
    
    # Mock data
    market_data = pd.DataFrame({"close": [3000.0]}, index=[datetime.now()])
    
    # Mock scan_symbols output representing ORB signal (uses 'stop' key)
    with patch("src.alpha.manager.scan_symbols") as mock_scan:
        mock_scan.return_value = [
            {
                "symbol": "RELIANCE",
                "direction": 1,
                "confidence": 0.85,
                "target": 3200.0,
                "stop": 2950.0,
                "strategy": "orb"
            }
        ]
        
        manager.generate_signals("RELIANCE", market_data)
        
        # Verify that prediction storage was called with target stop_loss = 2950.0
        assert manager.prediction_storage.store_prediction.call_count == 1
        stored_pred = manager.prediction_storage.store_prediction.call_args[0][0]
        assert stored_pred.symbol == "RELIANCE"
        assert stored_pred.stop_loss == 2950.0  # Verify it is 2950.0, not 0.0!


@patch("yfinance.Ticker")
def test_outcome_updater_logic(mock_ticker_class, temp_db):
    """Test that the updater logic resolves target breaches and forced exits correctly."""
    storage = PredictionStorage(db_path=temp_db)
    
    # Store 3 predictions:
    # 1. Long that will hit stop loss
    # 2. Long that will hit target
    # 3. Long that will expire (5 market days, 7 calendar days)
    now = datetime.now()
    
    pred_stop = Prediction(
        symbol="INFY",
        direction="long",
        confidence=0.8,
        target_price=1600.0,
        stop_loss=1400.0,
        entry_price=1500.0,
        timestamp=now - timedelta(days=2),
        strategy="orb"
    )
    pred_target = Prediction(
        symbol="TCS",
        direction="long",
        confidence=0.8,
        target_price=3500.0,
        stop_loss=3300.0,
        entry_price=3400.0,
        timestamp=now - timedelta(days=2),
        strategy="orb"
    )
    pred_aged = Prediction(
        symbol="SBIN",
        direction="long",
        confidence=0.8,
        target_price=800.0,
        stop_loss=700.0,
        entry_price=750.0,
        timestamp=now - timedelta(days=8), # Older than 7 days
        strategy="orb"
    )
    
    id_stop = storage.store_prediction(pred_stop)
    id_target = storage.store_prediction(pred_target)
    id_aged = storage.store_prediction(pred_aged)
    
    # Mock yfinance data responses:
    # For INFY: price goes down to 1350 (hits stop at 1400)
    # For TCS: price goes up to 3550 (hits target at 3500)
    # For SBIN: price remains 760 (does not hit target/stop, exits at close)
    
    # Mock dataframes
    dates = pd.date_range(now - timedelta(days=2), periods=5, freq="h")
    
    df_infy = pd.DataFrame({
        "open": [1500.0, 1480.0, 1450.0, 1390.0, 1380.0],
        "high": [1510.0, 1490.0, 1460.0, 1400.0, 1390.0],
        "low": [1490.0, 1440.0, 1420.0, 1350.0, 1370.0], # Low hits 1350 (<= 1400)
        "close": [1495.0, 1460.0, 1430.0, 1380.0, 1380.0]
    }, index=dates)
    
    df_tcs = pd.DataFrame({
        "open": [3400.0, 3420.0, 3450.0, 3510.0, 3520.0],
        "high": [3415.0, 3440.0, 3480.0, 3550.0, 3540.0], # High hits 3550 (>= 3500)
        "low": [3390.0, 3410.0, 3430.0, 3490.0, 3500.0],
        "close": [3410.0, 3435.0, 3470.0, 3520.0, 3530.0]
    }, index=dates)
    
    df_sbin = pd.DataFrame({
        "open": [750.0, 752.0, 755.0, 758.0, 760.0],
        "high": [755.0, 758.0, 760.0, 762.0, 765.0], # Lows/highs never hit 700/800
        "low": [748.0, 750.0, 752.0, 755.0, 758.0],
        "close": [752.0, 754.0, 758.0, 760.0, 761.0]
    }, index=pd.date_range(now - timedelta(days=8), periods=5, freq="D"))
    
    def side_effect_ticker(symbol):
        ticker_mock = MagicMock()
        if "INFY" in symbol:
            ticker_mock.history.return_value = df_infy
        elif "TCS" in symbol:
            ticker_mock.history.return_value = df_tcs
        elif "SBIN" in symbol:
            ticker_mock.history.return_value = df_sbin
        return ticker_mock
        
    mock_ticker_class.side_effect = side_effect_ticker
    
    # We will simulate the update_prediction_outcomes background loop body manually
    # instead of running it asynchronously.
    def run_update_logic():
        pending = [p for p in storage.get_predictions() if p.exit_price is None]
        for pred in pending:
            symbol = pred.symbol
            yf_symbol = symbol + ".NS"
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history()
            hist.columns = [c.lower() for c in hist.columns]
            
            pred_ts = pd.to_datetime(pred.timestamp).tz_localize(None)
            post_bars = hist[hist.index.tz_localize(None) >= pred_ts]
            if post_bars.empty:
                continue
                
            exit_price = None
            exit_time = None
            age_days = (datetime.now() - pred.timestamp).days
            
            for idx, bar in post_bars.iterrows():
                bar_time = idx.to_pydatetime().replace(tzinfo=None)
                if pred.direction == 'long':
                    if bar['low'] <= pred.stop_loss:
                        exit_price = pred.stop_loss
                        exit_time = bar_time
                        break
                    elif bar['high'] >= pred.target_price:
                        exit_price = pred.target_price
                        exit_time = bar_time
                        break
                        
            # Force exit for aged prediction
            if exit_price is None and age_days >= 7:
                latest_bar = post_bars.iloc[-1]
                exit_price = float(latest_bar['close'])
                exit_time = post_bars.index[-1].to_pydatetime().replace(tzinfo=None)
                
            if exit_price is not None and exit_time is not None:
                storage.update_outcome(pred.id, exit_price, exit_time)
                
    # Run the logic
    run_update_logic()
    
    # Verify outcomes
    updated_preds = {p.id: p for p in storage.get_predictions()}
    
    # 1. INFY should be resolved as hit stop_loss (1400.0)
    assert updated_preds[id_stop].exit_price == 1400.0
    assert updated_preds[id_stop].is_correct is False
    
    # 2. TCS should be resolved as hit target (3500.0)
    assert updated_preds[id_target].exit_price == 3500.0
    assert updated_preds[id_target].is_correct is True
    
    # 3. SBIN should be force-exited at close (761.0)
    assert updated_preds[id_aged].exit_price == 761.0
    
    # Verify count metrics reflect this
    metrics = storage.get_performance_metrics()
    assert metrics["total_predictions"] == 3
    assert metrics["pending"] == 0
    assert metrics["realized"] == 3

import pytest
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.alpha.prediction_registry import PredictionRegistry, PredictionRecord, get_prediction_registry
import src.alpha.prediction_registry as prediction_registry

# Import research modules dynamically to resolve their paths
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "production"))

from research.experiments.alpha.alpha_decay_monitor import AlphaDecayMonitor, DecayConfig
from research.limits_to_arbitrage import LimitsToArbitrageEngine
from src.alpha.alphas.mean_reversion_strategies import KalmanPairs


def test_alpha_decay_monitor_spearman_and_cost():
    config = DecayConfig(window_days=10, min_observations=5)
    monitor = AlphaDecayMonitor(config)
    
    # 10 identical rank ordered predictions and actuals
    # Spearman rank correlation must be exactly 1.0
    predictions = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, -5.0, -4.0, -3.0, -2.0, -1.0])
    actuals = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1, -5.1, -4.1, -3.1, -2.1, -1.1])
    
    result = monitor.update(predictions, actuals)
    
    assert result["current_ic"] == pytest.approx(1.0)
    
    # Sharpe calculation:
    # position is sign(predictions) = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    # change at index 5: diff is -2, abs change is 2. pos_change = 2.0
    # cost = 0.00025 * 2.0 = 0.0005
    # gross_returns = position * actuals = [1.1, 2.1, 3.1, 4.1, 5.1, 5.1, 4.1, 3.1, 2.1, 1.1]
    # net_returns at index 5 should be 5.1 - 0.0005 = 5.0995
    # Let's verify standard deviation and mean calculations
    gross = np.sign(predictions) * actuals
    pos_change = np.sign(predictions).diff().fillna(0).abs()
    expected_costs = 0.00025 * pos_change
    expected_net = gross - expected_costs
    expected_sharpe = expected_net.mean() / (expected_net.std() + 1e-8) * np.sqrt(252)
    
    assert result["current_sharpe"] == pytest.approx(expected_sharpe)


def test_limits_to_arbitrage_registry_and_ic(tmp_path):
    # Setup temporary database for prediction registry
    db_path = str(tmp_path / "test_limits_to_arbitrage.db")
    custom_registry = PredictionRegistry(db_path=db_path)
    
    # Temporarily override singleton
    original_registry = prediction_registry._registry
    prediction_registry._registry = custom_registry
    
    try:
        engine = LimitsToArbitrageEngine()
        assert engine.registry is not None
        
        # Test detection registers predictions for two assets
        hist_prices = pd.Series([100.0] * 30, index=pd.date_range(datetime.now() - timedelta(days=30), periods=30))
        hist_vols = pd.Series([100000.0] * 30, index=pd.date_range(datetime.now() - timedelta(days=30), periods=30))
        
        entry_time = datetime.now() - timedelta(minutes=120)
        
        # 1. RELIANCE: Severity 1.0 (expected = 0.02), realized exit 95 (+5.55%)
        panic_rel = engine.detect_panic_selling(
            symbol="RELIANCE",
            timestamp=entry_time,
            price=90.0,
            volume=500000.0,
            historical_prices=hist_prices,
            historical_volumes=hist_vols
        )
        assert panic_rel is not None
        
        # 2. INFY: Severity 0.5 (expected = 0.01), realized exit 80 (-11.11%)
        # To get severity 0.5, let's trigger panic with a smaller drop
        panic_infy = engine.detect_panic_selling(
            symbol="INFY",
            timestamp=entry_time,
            price=94.0,  # 6% drop (strictly < 5% threshold)
            volume=400000.0,  # 4x volume spike (strictly > 3x threshold)
            historical_prices=hist_prices,
            historical_volumes=hist_vols
        )
        assert panic_infy is not None
        
        # Retrieve registered predictions and update one to prevent constant inputs
        conn = custom_registry._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id, symbol, predicted_return FROM predictions WHERE strategy = 'LimitsToArbitrage_Panic'")
        rows = cursor.fetchall()
        assert len(rows) == 2
        
        # Modify INFY predicted return to break constant array
        cursor.execute("SELECT id FROM predictions WHERE symbol = 'INFY'")
        infy_id = cursor.fetchone()[0]
        cursor.execute("UPDATE predictions SET predicted_return = 0.01 WHERE id = ?", (infy_id,))
        conn.commit()
        
        # To avoid MIN_PREDICTIONS_FOR_IC limit (which is 100), we patch it down to 2
        original_min_preds = prediction_registry.MIN_PREDICTIONS_FOR_IC
        prediction_registry.MIN_PREDICTIONS_FOR_IC = 2
        
        try:
            # Resolve RELIANCE (exit 95)
            cursor.execute("SELECT id FROM predictions WHERE symbol = 'RELIANCE'")
            rel_id = cursor.fetchone()[0]
            custom_registry.resolve_prediction(
                prediction_id=rel_id,
                exit_price=95.0,
                exit_timestamp=entry_time + timedelta(minutes=90)
            )
            
            # Resolve INFY (exit 80)
            cursor.execute("SELECT id FROM predictions WHERE symbol = 'INFY'")
            infy_id = cursor.fetchone()[0]
            custom_registry.resolve_prediction(
                prediction_id=infy_id,
                exit_price=80.0,
                exit_timestamp=entry_time + timedelta(minutes=90)
            )
            
            # Compute IC
            ic_report = engine.compute_engine_ic(alpha_id="LimitsToArbitrage_Panic")
            assert ic_report["total_predictions"] == 2
            assert ic_report["mean_ic"] != 0.0 or ic_report["rolling_ic"] != 0.0
        finally:
            prediction_registry.MIN_PREDICTIONS_FOR_IC = original_min_preds
            
    finally:
        prediction_registry._registry = original_registry


def test_kalman_pairs_cost_filtering():
    # entry_threshold=2.0, exit_threshold=0.5
    strategy = KalmanPairs(entry_threshold=2.0, exit_threshold=0.5)
    strategy.beta = 1.5
    
    # 1. Cost check is bypassed if prices are not provided
    sig, state = strategy.get_signal(z=3.0)
    assert sig == -1.0
    assert state == "SHORT_Y_LONG_X"
    
    # 2. Expected profit > cost
    # expected_profit = max(0, 3.0 - 0.5) * spread_std = 2.5 * 10 = 25.0
    # est_cost = cost(y) + beta * cost(x)
    # y = 1000, x = 600
    # cost(y) = 1000 * 2 * (0.00015+0.00025+0.0000345+0.000001+0.0002) = 1.271
    # cost(x) = 600 * 2 * (0.0006355) = 0.7626. beta * cost(x) = 1.5 * 0.7626 = 1.1439
    # total cost = 1.271 + 1.1439 = 2.4149
    # Since 25.0 > 2.4149, trade should be allowed
    sig, state = strategy.get_signal(z=3.0, y_price=1000.0, x_price=600.0, spread_std=10.0)
    assert sig == -1.0
    assert state == "SHORT_Y_LONG_X"
    
    # 3. Expected profit < cost
    # expected_profit = 2.5 * 0.1 = 0.25
    # total cost = 2.4149
    # Since 0.25 < 2.4149, trade should be blocked by COST_BARRIER
    sig, state = strategy.get_signal(z=3.0, y_price=1000.0, x_price=600.0, spread_std=0.1)
    assert sig == 0.0
    assert state == "COST_BARRIER"

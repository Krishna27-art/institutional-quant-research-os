"""
Tests for DataQualityGate and PredictionRegistry.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from src.data.quality_gate import DataQualityGate, get_quality_gate
from src.alpha_factory.prediction_registry import PredictionRegistry, PredictionRecord, get_prediction_registry


def test_data_quality_gate_empty():
    gate = DataQualityGate(min_rows=5)
    df = pd.DataFrame()
    clean_df, result = gate.validate("TEST", df)
    assert not result.passed
    assert "empty_dataframe" in result.violations


def test_data_quality_gate_missing_columns():
    gate = DataQualityGate(min_rows=5)
    df = pd.DataFrame({"open": [100.0], "close": [101.0]})
    clean_df, result = gate.validate("TEST", df)
    assert not result.passed
    assert any("missing_columns" in v for v in result.violations)


def test_data_quality_gate_rule2_ohlc_inconsistency():
    gate = DataQualityGate(min_rows=3, max_stale_closes=999, drop_bad_rows=True)
    dates = pd.date_range(datetime.now(), periods=5, freq="D")
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
    assert not result.passed
    assert len(clean_df) == 4
    assert "ohlc_bad" in result.violation_counts
    assert result.violation_counts["ohlc_bad"] == 1


def test_data_quality_gate_rule3_stale_prices():
    gate = DataQualityGate(min_rows=3, max_stale_closes=2, drop_bad_rows=True)
    dates = pd.date_range(datetime.now(), periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [105.0] * 5,
            "low": [95.0] * 5,
            "close": [100.0, 100.0, 100.0, 100.0, 102.0],  # 4 stale closes -> last 2 are marked stale
            "volume": [1000] * 5,
        },
        index=dates,
    )
    clean_df, result = gate.validate("TEST", df)
    assert not result.passed
    assert len(clean_df) == 3  # rows 2 and 3 dropped (0-indexed 0, 1 are fine, 2 and 3 dropped)
    assert "stale_prices" in result.violation_counts


def test_data_quality_gate_rule4_volume_sanity():
    gate = DataQualityGate(min_rows=3, max_stale_closes=999, drop_bad_rows=True)
    dates = pd.date_range(datetime.now(), periods=5, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [105.0] * 5,
            "low": [95.0] * 5,
            "close": [101.0] * 5,
            "volume": [1000, 0, 1000, -500, 1000],  # row 1: 0, row 3: -500
        },
        index=dates,
    )
    clean_df, result = gate.validate("TEST", df)
    assert not result.passed
    assert len(clean_df) == 3
    assert result.violation_counts["zero_volume"] == 2


def test_data_quality_gate_rule5_overnight_gap_soft():
    # overnight gap is a soft violation; it should flag it but not fail the pass status if other rules are OK
    gate = DataQualityGate(min_rows=3, max_overnight_gap_pct=0.10)
    dates = pd.date_range(datetime.now(), periods=3, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0, 120.0, 120.0],  # > 10% overnight gap on day 1
            "high": [105.0, 125.0, 125.0],
            "low": [95.0, 115.0, 115.0],
            "close": [100.0, 120.0, 120.0],
            "volume": [1000] * 3,
        },
        index=dates,
    )
    clean_df, result = gate.validate("TEST", df)
    assert result.passed  # Passes since overnight_gap is soft
    assert "overnight_gap" in result.violation_counts
    assert result.violation_counts["overnight_gap"] == 1


def test_prediction_registry_basic(tmp_path):
    db_file = tmp_path / "test_prediction_registry.db"
    registry = PredictionRegistry(db_path=str(db_file), min_ic=0.02, ic_lookback_days=30)
    
    # Check table existence
    assert registry.db_path == str(db_file)
    
    now = datetime.now()
    pred1 = PredictionRecord(
        symbol="RELIANCE",
        strategy="orb",
        direction="long",
        predicted_return=0.03,
        confidence=0.8,
        entry_price=3000.0,
        timestamp=now - timedelta(hours=8),
        horizon_minutes=390
    )
    
    pred_id = registry.record_prediction(pred1)
    assert pred_id == 1
    
    summary = registry.get_summary()
    assert summary["total_predictions"] == 1
    assert summary["pending"] == 1
    assert summary["resolved"] == 0
    
    # Resolve prediction
    realized_ret = registry.resolve_prediction(pred_id, exit_price=3090.0, exit_timestamp=now)
    assert realized_ret == 0.03  # (3090-3000)/3000 = 0.03
    
    summary = registry.get_summary()
    assert summary["pending"] == 0
    assert summary["resolved"] == 1


def test_prediction_registry_ic_and_demotion(tmp_path):
    db_file = tmp_path / "test_prediction_registry_ic.db"
    # Set lookback very short and min predictions to 5 for testing
    registry = PredictionRegistry(db_path=str(db_file), min_ic=0.10)
    
    # We must patch MIN_PREDICTIONS_FOR_IC to test with fewer predictions
    from src.alpha_factory import prediction_registry
    original_min = prediction_registry.MIN_PREDICTIONS_FOR_IC
    prediction_registry.MIN_PREDICTIONS_FOR_IC = 5
    
    try:
        now = datetime.now()
        # Create 5 predictions for strategy 'strat1' that are correct
        for i in range(5):
            pred = PredictionRecord(
                symbol="RELIANCE",
                strategy="strat1",
                direction="long",
                predicted_return=0.01 + 0.01 * i,
                confidence=0.8,
                entry_price=100.0,
                timestamp=now - timedelta(days=2),
                horizon_minutes=390
            )
            pid = registry.record_prediction(pred)
            # Correct outcome: higher predicted return correlates with higher exit price
            registry.resolve_prediction(pid, exit_price=100.0 + 1.0 * (i + 1), exit_timestamp=now)
            
        # Create 5 predictions for strategy 'strat2' that are incorrect (negatively correlated)
        for i in range(5):
            pred = PredictionRecord(
                symbol="RELIANCE",
                strategy="strat2",
                direction="long",
                predicted_return=0.01 + 0.01 * i,
                confidence=0.8,
                entry_price=100.0,
                timestamp=now - timedelta(days=2),
                horizon_minutes=390
            )
            pid = registry.record_prediction(pred)
            # Inverse outcome: higher predicted return correlates with lower exit price
            registry.resolve_prediction(pid, exit_price=100.0 - 1.0 * (i + 1), exit_timestamp=now)
            
        ic_strat1 = registry.compute_ic("strat1")
        ic_strat2 = registry.compute_ic("strat2")
        
        # strat1 should have highly positive correlation (IC = 1.0)
        assert ic_strat1 > 0.8
        # strat2 should have highly negative correlation (IC = -1.0)
        assert ic_strat2 < -0.8
        
        demoted = registry.check_demotions()
        assert "strat2" in demoted
        assert "strat1" not in demoted
        
        report_strat2 = registry.get_strategy_report("strat2")
        assert not report_strat2.is_active
        
    finally:
        prediction_registry.MIN_PREDICTIONS_FOR_IC = original_min

"""
Unit tests for new architectural additions:
- Portfolio construction models (HRP, Black-Litterman)
- Tail Risk Metrics (EVT, Cornish-Fisher VaR)
- Point-in-time Universe Tracker
- Data Quality Gate Halt Triggers
- NSE Transaction Cost Model
- CLI Performance Dashboard
"""

from io import StringIO
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.portfolio.engine import PortfolioAllocator
from src.risk.metrics import calculate_var, calculate_var_historical, calculate_var_evt
from src.data.universe_tracker import UniverseTracker
from src.data.quality_gate import DataQualityGate
from src.execution.cost_model import NSETransactionCostModel
from src.monitoring.dashboard_cli import AlphaPerformanceCLIDashboard


def test_evt_and_cornish_fisher_var():
    """Verify Cornish-Fisher and EVT VaR calculations return expected results."""
    # Generate mock returns (mixture of normal and fat-tailed shocks)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, 200)
    returns[::20] = -0.075  # Add extreme negative tail events
    
    capital = 1_000_000.0
    
    # 1. Cornish-Fisher VaR
    var_cf = calculate_var(returns, capital, confidence=0.99, use_cornish_fisher=True)
    var_param = calculate_var(returns, capital, confidence=0.99, use_cornish_fisher=False)
    
    # Cornish-Fisher should reflect fat tails (higher VaR)
    assert var_cf > 0.0
    assert var_param > 0.0
    assert var_cf > var_param
    
    # 2. EVT VaR (Generalized Pareto Distribution tail fit)
    var_evt = calculate_var_evt(returns, capital, confidence=0.99)
    assert var_evt > 0.0
    
    # Compare with standard historical simulation
    var_hist = calculate_var_historical(returns, capital, confidence=0.99)
    assert abs(var_evt - var_hist) < 0.20 * capital  # Sane bounds


def test_hrp_and_black_litterman_allocation():
    """Verify Hierarchical Risk Parity (HRP) and Black-Litterman sizing algorithms."""
    # Setup allocator
    allocator = PortfolioAllocator(total_capital=10_000_000.0)
    
    # Generate mock prices for 12 assets (HRP runs for 10+ assets)
    np.random.seed(42)
    symbols = [f"STOCK_{i}" for i in range(12)]
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=100)
    data = pd.DataFrame(index=dates)
    for s in symbols:
        data[s] = 100.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 100)))
        
    signals = [{"symbol": s, "direction": 1.0, "strength": 0.5, "rv": 0.5, "stop_loss_price": 95.0, "strategy": "momentum"} for s in symbols]
    
    # 1. Test HRP allocation
    hrp_positions = allocator.allocate(signals, method="hrp", price_history=data)
    assert len(hrp_positions) > 0
    total_hrp_cap = sum(pos.capital for pos in hrp_positions)
    assert total_hrp_cap <= allocator.total_capital
    
    # 2. Test Black-Litterman allocation
    # BL blends views with market equilibrium returns. Provide views for first 3 stocks.
    views = {"STOCK_0": 0.02, "STOCK_1": -0.01, "STOCK_2": 0.015}
    bl_positions = allocator.allocate(signals, method="black_litterman", price_history=data, views=views)
    assert len(bl_positions) > 0
    total_bl_cap = sum(pos.capital for pos in bl_positions)
    assert total_bl_cap <= allocator.total_capital


def test_universe_tracker():
    """Verify Point-in-Time Universe Tracker computes correct constituents."""
    tracker = UniverseTracker()
    index_name = "NIFTY50"
    
    t0 = datetime(2025, 1, 1)
    t1 = datetime(2025, 3, 1)
    t2 = datetime(2025, 6, 1)
    
    # Set base universe
    base_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK"]
    tracker.set_initial_universe(index_name, base_symbols)
    
    # Record constituent change (addition & removal)
    tracker.add_change(index_name, t1, added=["ITC"], removed=["INFY"])
    
    # Query at t0 (should equal base)
    univ_t0 = tracker.get_universe(index_name, t0)
    assert univ_t0 == sorted(base_symbols)
    
    # Query at t1 + 1 day (should have ITC and lack INFY)
    univ_t1 = tracker.get_universe(index_name, t1 + timedelta(days=1))
    assert "ITC" in univ_t1
    assert "INFY" not in univ_t1
    assert len(univ_t1) == 4


def test_quality_gate_halts():
    """Verify quality gate automated halts trigger under stress conditions."""
    gate = DataQualityGate()
    
    dates = pd.date_range("2025-06-01", periods=10)
    
    # 1. Normal DataFrame
    df_normal = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "volume": [1000.0] * 10
    }, index=dates)
    
    halt, reason = gate.should_halt_signals(df_normal)
    assert not halt
    
    # 2. Extreme Price Movement (> 25% drop)
    df_drop = df_normal.copy()
    df_drop.iloc[-1, df_drop.columns.get_loc("close")] = 70.0  # 30% drop
    halt_price, reason_price = gate.should_halt_signals(df_drop)
    assert halt_price
    assert "Extreme price movement" in reason_price
    # Fast forward time to simulate stale data (> 15 mins)
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    halt_stale, reason_stale = gate.should_halt_signals(df_normal, last_update_time=stale_time)
    assert halt_stale
    assert "stale" in reason_stale


def test_nse_transaction_cost_model():
    """Verify STT, Stamp Duty, GST, and Zerodha Brokerage calculations for NSE."""
    model = NSETransactionCostModel()
    
    # 1. Delivery buy order
    costs_del = model.calculate_cost(price=1000.0, quantity=100.0, side="buy", product_type="delivery")
    assert costs_del["brokerage"] == 0.0  # Zerodha delivery is free
    assert costs_del["stt"] == 100.0      # 0.1% on 100,000
    assert costs_del["stamp_duty"] == pytest.approx(15.0) # 0.015% on buy
    
    # 2. Intraday sell order
    costs_int = model.calculate_cost(price=1000.0, quantity=100.0, side="sell", product_type="intraday")
    assert costs_int["brokerage"] == 20.0  # Cap at flat 20.0
    assert costs_int["stt"] == 25.0        # 0.025% on sell
    assert costs_int["stamp_duty"] == 0.0  # 0% stamp duty on sell


def test_cli_dashboard():
    """Verify CLI alpha performance dashboard output rendering works."""
    dashboard = AlphaPerformanceCLIDashboard()
    stream = StringIO()
    dashboard.render(stream)
    report = stream.getvalue()
    
    assert "INSTITUTIONAL QUANT OS: ALPHA PERFORMANCE DASHBOARD" in report
    assert "SYSTEM OVERVIEW" in report
    assert "STRATEGY PERFORMANCE" in report

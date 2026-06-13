import pytest
from fastapi.testclient import TestClient
from dashboard.api.api_server import app

client = TestClient(app)

def test_stock_profile():
    response = client.get("/api/stocks/RELIANCE/profile")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "RELIANCE"
    assert "price" in data
    assert "sector" in data

def test_stock_history():
    response = client.get("/api/stocks/RELIANCE/history?period=5d")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        assert "open" in data[0]

def test_stock_predictions():
    response = client.get("/api/stocks/RELIANCE/predictions")
    assert response.status_code == 200
    data = response.json()
    assert "horizons" in data
    assert "1D" in data["horizons"]

def test_stock_factors():
    response = client.get("/api/stocks/RELIANCE/factors")
    assert response.status_code == 200
    data = response.json()
    assert "momentum" in data
    assert "combined" in data

def test_stock_options():
    response = client.get("/api/stocks/RELIANCE/options")
    assert response.status_code == 200
    data = response.json()
    assert "underlying_price" in data
    assert "option_chain" in data

def test_alpha_lab_metrics():
    response = client.get("/api/alpha-lab/metrics")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_strategies():
    response = client.get("/api/strategies")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_strategy_performance():
    response = client.get("/api/strategies/orb/performance")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_strategy_trades():
    response = client.get("/api/strategies/orb/trades")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_current_regime():
    response = client.get("/api/regime/current")
    assert response.status_code == 200
    data = response.json()
    assert "current_regime" in data

def test_regime_history():
    response = client.get("/api/regime/history?days=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_risk_portfolio():
    response = client.get("/api/risk/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert "var_99" in data

def test_risk_exposures():
    response = client.get("/api/risk/exposures")
    assert response.status_code == 200
    data = response.json()
    assert "sector_exposure" in data

def test_risk_stress_tests():
    response = client.get("/api/risk/stress-tests")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_run_backtest():
    payload = {
        "strategy": "orb",
        "capital": 250000000.0,
        "start_date": "2024-01-01",
        "end_date": "2024-01-05",
        "slippage_bps": 2.0,
        "universe": ["RELIANCE", "HDFCBANK"]
    }
    response = client.post("/api/backtests/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "sharpe_ratio" in data

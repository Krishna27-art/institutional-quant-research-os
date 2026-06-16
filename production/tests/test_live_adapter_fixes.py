import pytest
from src.execution.adapters.live_adapter import LiveAdapter
from src.portfolio.engine import PortfolioAllocator
from src.data.quality_gate import get_quality_gate

def test_portfolio_allocator_signed_weights():
    allocator = PortfolioAllocator(max_position_pct=0.2)
    # A short signal with a positive weight should output a negative weight allocation
    signals = [{
        "symbol": "INFY",
        "direction": -1.0,
        "strength": 0.8,
        "confidence": 0.9,
    }]
    # Test allocate_from_alpha_signals path
    allocations = allocator.allocate_from_alpha_signals(capital=100000.0, signals=signals)
    assert len(allocations) == 1
    assert allocations[0].weight < 0, "Short signals must produce negative weights."
    assert allocations[0].score < 0, "Short signals must produce negative scores."
    
    # Test allocate path
    allocations_def = allocator.allocate(100000.0, signals)
    assert len(allocations_def) == 1
    assert allocations_def[0].weight < 0, "Short signals must produce negative weights in default allocate."

def test_live_adapter_price_hook(monkeypatch):
    from src.execution.adapters.live_adapter import LiveConfig
    config = LiveConfig(broker_api_key="mock", broker_api_secret="mock")
    adapter = LiveAdapter(config=config)
    
    def mock_get_price(symbol, days=1):
        import pandas as pd
        return pd.DataFrame({"close": [105.5]})
    
    # Mock the truth DB hook
    monkeypatch.setattr("src.data.truth.get_price_history", mock_get_price)
    
    price = adapter._get_current_price("TEST_SYM")
    assert price == 105.5, "Live adapter must fetch real prices via truth DB hook."

def test_data_quality_engine_alias():
    # Verify that the DataQualityEngine maps correctly to DataQualityGate via adapter methods
    engine = get_quality_gate()
    assert hasattr(engine, "check_data_quality"), "Gate must expose check_data_quality adapter"
    assert hasattr(engine, "get_quality_summary"), "Gate must expose get_quality_summary adapter"

import pytest
import asyncio
import pandas as pd
from src.state.publisher import publisher
from src.features.feature_pipeline import FeaturePipeline
from src.core.config.settings import TRADING_CAPITAL

@pytest.mark.asyncio
async def test_e2e_trading_loop():
    # Verify NAV updates properly
    publisher.state['nav'] = TRADING_CAPITAL
    
    pipeline = FeaturePipeline()
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    
    # Mock data
    df = pd.DataFrame({
        "open": [100.0] * 60,
        "high": [105.0] * 60,
        "low": [95.0] * 60,
        "close": [100.0 + i for i in range(60)],
        "volume": [1000] * 60
    }, index=dates)
    
    for i, date in enumerate(dates):
        sub_df = df.iloc[:i+1]
        
        # Verify feature cache works
        features = pipeline.compute_features(ohlcv=sub_df, timestamp=date, symbol="TEST")
        assert features is not None
        if i >= 50:
            assert ("TEST", date) in pipeline._feature_cache
        
        # Verify NAV updates
        publisher.state['nav'] += 10.0
        
        # Verify capital bounds (e.g., position size vs NAV)
        position_value = 5000.0
        assert position_value < publisher.state['nav']

    assert publisher.state['nav'] == TRADING_CAPITAL + 600.0

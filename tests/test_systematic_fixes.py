import numpy as np
import pandas as pd

from alpha.orb_zarattini import scan_symbols
from features.indicators import compute_atr, parkinson_volatility
from live.market_stream import NSEWebSocketStream
from src.data.data_loader import NSEDataLoader
from portfolio.allocator import PortfolioAllocator
from risk.institutional_risk_engine import InstitutionalRiskEngine as FullRiskEngine
from risk.risk_engine import InstitutionalRiskEngine
from main import QuantResearchOS


def test_var_is_positive_left_tail_loss() -> None:
    engine = InstitutionalRiskEngine(confidence_level=0.99)
    returns = pd.Series(np.r_[np.full(250, 0.001), [-0.03, -0.025, -0.02]])

    var = engine.compute_var(10_000_000, returns)
    historical_var = engine.compute_historical_var(10_000_000, returns)
    cvar = engine.compute_cvar(10_000_000, returns)

    assert var > 0
    assert historical_var > 0
    assert cvar >= historical_var


def test_full_risk_engine_respects_max_leverage_and_positive_var() -> None:
    engine = FullRiskEngine(capital=10_000_000, max_leverage=1.0)
    returns = np.r_[np.full(250, 0.001), [-0.04, -0.03, -0.02]]

    assert engine.max_leverage == 1.0
    assert engine.calculate_var(returns) > 0
    assert engine.calculate_tail_risk(returns, percentile=0.15) > 0


def test_atr_and_parkinson_volatility_are_price_scale_sane() -> None:
    df = pd.DataFrame(
        {
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
        }
    )

    atr = compute_atr(df, period=3)
    park = parkinson_volatility(df, period=3)

    assert atr.iloc[-1] == 2.0
    assert 0 < park.iloc[-1] < 0.05


def test_orb_scan_symbols_uses_real_breakout_and_relative_volume() -> None:
    day1 = pd.date_range("2024-01-01 09:15", periods=8, freq="1min")
    day2 = pd.date_range("2024-01-02 09:15", periods=8, freq="1min")
    idx = day1.append(day2)
    close = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 101, 102, 103, 104, 106, 107, 108], dtype=float)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [100] * 8 + [500] * 8,
        },
        index=idx,
    )

    signals = scan_symbols({"RELIANCE": df}, day2[-1].to_pydatetime())

    assert len(signals) == 1
    assert signals[0]["symbol"] == "RELIANCE"
    assert signals[0]["direction"] == 1
    assert signals[0]["rv"] > 1.0


def test_market_stream_aggregates_ticks_to_completed_bar() -> None:
    emitted = []
    stream = NSEWebSocketStream(["NIFTY"], emitted.append)

    stream._process_tick({"symbol": "NIFTY", "price": 100.0, "timestamp": "2024-01-01T09:15:01", "volume": 10})
    stream._process_tick({"symbol": "NIFTY", "price": 102.0, "timestamp": "2024-01-01T09:15:20", "volume": 5})
    stream._process_tick({"symbol": "NIFTY", "price": 101.0, "timestamp": "2024-01-01T09:16:00", "volume": 7})

    assert emitted == [
        {
            "symbol": "NIFTY",
            "timestamp": pd.Timestamp("2024-01-01T09:15:00").to_pydatetime(),
            "open": 100.0,
            "high": 102.0,
            "low": 100.0,
            "close": 102.0,
            "volume": 15.0,
        }
    ]


def test_backtest_data_gate_rejects_empty_required_data() -> None:
    system = QuantResearchOS(config_path="missing-config.yaml")
    system.market_data = {
        "NIFTY": pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    }

    assert not system._has_required_backtest_data()


def test_data_loader_standardizes_multiindex_ohlcv() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            ("Open", "RELIANCE.NS"): [100, 101, 102],
            ("High", "RELIANCE.NS"): [101, 102, 103],
            ("Low", "RELIANCE.NS"): [99, 100, 101],
            ("Close", "RELIANCE.NS"): [100.5, 101.5, 102.5],
            ("Volume", "RELIANCE.NS"): [1000, 1100, 1200],
        },
        index=index,
    )

    loader = NSEDataLoader(data_dir="/tmp/missing-data")
    out = loader._standardize_ohlcv(raw, "RELIANCE.NS")

    assert list(out.columns) == ["open", "high", "low", "close", "volume", "symbol"]
    assert out["symbol"].unique().tolist() == ["RELIANCE.NS"]
    assert out["close"].iloc[-1] == 102.5


def test_allocator_reads_dict_alpha_signals() -> None:
    allocator = PortfolioAllocator(total_capital=1_000_000)
    signals = [
        {
            "symbol": "RELIANCE",
            "direction": 1,
            "rv": 3.0,
            "confidence": 0.8,
            "expected_return": 0.02,
        }
    ]

    allocations = allocator.allocate(signals)

    assert len(allocations) == 1
    assert allocations[0].symbol == "RELIANCE"
    assert allocations[0].capital > 0
    assert allocations[0].score > 0

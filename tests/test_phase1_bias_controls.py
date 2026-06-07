import numpy as np
import pandas as pd

from alpha.orb_zarattini import ORBBacktesterZarattini, ORBConfig
from alpha.put_call_carry_shin import PCPCarryBacktesterShin, PCPCarryConfig
from alpha.vwap_trend_zarattini import VWAPTrendBacktesterZarattini, VWAPConfig
from backtesting.backtest_orb import BacktestConfig as ORBPathConfig
from backtesting.backtest_orb import ORBBacktester
from backtesting.backtest_pcp import PCPBacktester, PCPBacktestConfig
from features.feature_pipeline import FeatureConfig, FeaturePipeline
from regime.hmm_engine import HMMRegimeEngine, Regime


def _ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.linspace(100, 120, len(index))
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    )


def test_vwap_uses_open_not_same_bar_high_low_close() -> None:
    index = pd.date_range("2024-01-01 09:15", periods=3, freq="1min")
    data = _ohlcv(index)

    changed = data.copy()
    changed.loc[index[-1], ["high", "low", "close"]] = [10_000.0, 1.0, 9_000.0]

    backtester = VWAPTrendBacktesterZarattini(VWAPConfig())

    assert backtester.calculate_vwap(data).iloc[-1] == backtester.calculate_vwap(changed).iloc[-1]


def test_orb_short_transaction_costs_include_buy_and_sell_legs() -> None:
    config = ORBConfig()
    backtester = ORBBacktesterZarattini(config)

    short_cost = backtester._calculate_transaction_costs(
        entry_price=100.0,
        exit_price=95.0,
        quantity=10,
        side="SHORT",
    )

    buy_value = 95.0 * 10
    sell_value = 100.0 * 10
    expected = (
        config.brokerage_per_order * 2
        + buy_value * config.stamp_duty_rate
        + sell_value * config.stt_rate
        + (buy_value + sell_value) * config.exchange_rate
        + (buy_value + sell_value) * config.sebi_fees_rate
        + config.brokerage_per_order * 2 * config.gst_rate
    )

    assert short_cost == expected


def test_feature_pipeline_ignores_current_bar_close_high_low() -> None:
    index = pd.date_range("2024-01-01 09:15", periods=80, freq="1min")
    data = _ohlcv(index)

    changed = data.copy()
    changed.loc[index[-1], ["high", "low", "close"]] = [10_000.0, 1.0, 9_000.0]

    pipeline = FeaturePipeline(
        FeatureConfig(
            enable_leakage_detection=False,
            enable_psi_detection=False,
            enable_future_info_check=False,
        )
    )

    features = pipeline.compute_features("NIFTY", data, timestamp=index[-1].to_pydatetime())
    changed_features = pipeline.compute_features("NIFTY", changed, timestamp=index[-1].to_pydatetime())

    assert features == changed_features
    assert set(features) == set(pipeline.feature_names)


def test_pcp_short_option_costs_charge_entry_sell_leg() -> None:
    config = PCPCarryConfig()
    backtester = PCPCarryBacktesterShin(config)

    costs = backtester._calculate_option_transaction_costs(
        entry_premium=10.0,
        exit_premium=1.0,
        quantity=50,
        side="SHORT",
    )

    entry_value = 10.0 * 50
    exit_value = 1.0 * 50
    expected = (
        config.brokerage_per_order * 2
        + entry_value * config.stt_rate
        + (entry_value + exit_value) * config.exchange_rate
        + (entry_value + exit_value) * config.sebi_fees_rate
        + config.brokerage_per_order * 2 * config.gst_rate
    )

    assert costs == expected


def test_pcp_margin_sizing_prevents_premium_based_leverage() -> None:
    config = PCPBacktestConfig(
        initial_capital=10_000_000,
        max_position_pct=0.02,
        lot_size=50,
        margin_pct_notional=0.15,
    )
    backtester = PCPBacktester(config)

    backtester._execute_strangle_trade(
        call_strike=10_500.0,
        call_entry=10.0,
        call_exit=1.0,
        put_strike=9_500.0,
        put_entry=10.0,
        put_exit=1.0,
        iv_entry=0.2,
        iv_exit=0.1,
        entry_time=index_time("2024-01-03"),
        exit_time=index_time("2024-01-04"),
    )

    assert {trade.side for trade in backtester.trades} == {"SHORT"}
    assert {trade.quantity for trade in backtester.trades} == {100}
    assert all(trade.pnl > 0 for trade in backtester.trades)


def test_orb_path_exit_can_stop_out_long() -> None:
    backtester = ORBBacktester(ORBPathConfig())
    path = pd.DataFrame(
        {
            "open": [101.0, 100.0],
            "high": [101.2, 100.2],
            "low": [100.5, 99.0],
            "close": [100.8, 99.5],
            "volume": [1000, 1000],
        },
        index=pd.date_range("2024-01-01 09:21", periods=2, freq="1min"),
    )

    exit_price, exit_time, reason = backtester._simulate_path_exit(
        path,
        side="long",
        stop_loss=99.5,
        target=103.0,
    )

    assert exit_price == 99.5
    assert reason == "stop_loss"
    assert exit_time == path.index[-1]


def test_hmm_positive_return_fallback_is_not_bearish() -> None:
    engine = HMMRegimeEngine(config={"min_samples": 1})
    state = engine.predict_regime(
        {
            "realized_vol_5d": 0.12,
            "implied_vol": 0.14,
            "nifty_return_5d": 0.036,
            "turnover_ratio_5d": 1.0,
            "india_vix": 12.0,
        },
        index_time("2024-01-01"),
    )

    assert state.regime == Regime.BULL_TREND


def index_time(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)

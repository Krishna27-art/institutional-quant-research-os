import numpy as np
import pandas as pd
import pytest

from src.alpha.alphas import (
    BiLevelChaoticFusionGCN,
    GameStockAlpha,
    InvestorType,
    classify_investor,
    correlation_graph,
)
from src.alpha.evolution import MadEvolveAlphaEngine
from src.execution.signal_adaptive import SignalAdaptiveExecutor
from src.risk import (
    FIGARCHVolatility,
    MirroredWeibullVaR,
    PurgedEmbargoTimeSeriesSplit,
    algometric_feedback_gap,
    deflated_sharpe_ratio,
    prediction_interval_coverage,
)
from market_data.feature_generation.institutional_feature_factory import InstitutionalFeatureFactory


def _market_frame(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1000, 5000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


def test_gamestock_classifies_and_scores_heterogeneous_flow() -> None:
    assert classify_investor(75_000, 0.0005) == InvestorType.INSTITUTIONAL
    assert classify_investor(15_000, 0.006) == InvestorType.HOT_MONEY
    assert classify_investor(500, 0.003) == InvestorType.RETAIL

    trades = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "volume": 80_000, "signed_volume": -80_000, "price_impact": 0.0008},
            {"symbol": "RELIANCE", "volume": 25_000, "signed_volume": 25_000, "price_impact": 0.0060},
            {"symbol": "RELIANCE", "volume": 1_000, "signed_volume": 1_000, "price_impact": 0.0020},
        ]
    )
    alpha = GameStockAlpha(reversal_threshold=0.5)
    signals = alpha.compute_signals(alpha.aggregate_flows(trades))

    assert signals.loc["RELIANCE", "regime"] == "hot_money_momentum_reversal_risk"
    assert signals.loc["RELIANCE", "signal"] < 0.0
    assert signals.loc["RELIANCE", "confidence"] > 0.5


def test_correlation_graph_uses_absolute_correlation_threshold() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03],
            "B": [0.02, 0.04, -0.02, 0.06],
            "C": [-0.01, -0.02, 0.01, -0.03],
            "D": [0.03, -0.01, 0.02, -0.02],
        }
    )
    adj = correlation_graph(returns, threshold=0.9)

    assert adj.loc["A", "B"] == 1.0
    assert adj.loc["A", "C"] == 1.0
    assert adj.loc["A", "D"] == 0.0
    assert np.diag(adj).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_bcf_gcn_outputs_prediction_intervals() -> None:
    torch = pytest.importorskip("torch")
    model = BiLevelChaoticFusionGCN(n_features=4, hidden=8)
    x = torch.randn(5, 4)
    adjacency = torch.eye(5)

    out = model(x, adjacency)

    assert set(out) == {"prediction", "lower", "upper", "width"}
    assert out["prediction"].shape == (5,)
    assert torch.all(out["upper"] >= out["lower"])
    assert torch.all(out["width"] > 0)


def test_advanced_metrics_penalize_trials_and_measure_intervals() -> None:
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, -0.002, 0.011] * 20)

    raw = deflated_sharpe_ratio(returns, n_trials=1)
    deflated = deflated_sharpe_ratio(returns, n_trials=100)
    coverage = prediction_interval_coverage([0.0, 0.5, 1.5], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    assert deflated < raw
    assert coverage["coverage"] == pytest.approx(2 / 3)
    assert coverage["avg_width"] > 0.0


def test_purged_embargo_split_removes_nearby_training_rows() -> None:
    splitter = PurgedEmbargoTimeSeriesSplit(n_splits=2, test_size=10, purge=3, embargo=4)
    splits = list(splitter.split(np.arange(50)))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        assert np.all((train_idx < test_idx.min() - 3) | (train_idx >= test_idx.max() + 1 + 4))


def test_weibull_var_figarch_and_feedback_are_positive() -> None:
    rng = np.random.default_rng(3)
    returns = pd.Series(np.r_[rng.normal(0.001, 0.01, 250), [-0.04, -0.035, -0.03, -0.025]])

    var = MirroredWeibullVaR(confidence=0.99).var(10_000_000, returns)
    vol = FIGARCHVolatility(d=0.4).fit(returns).forecast(horizon=5)
    feedback = algometric_feedback_gap([9.0, 8.0, 7.0, 6.0], [10.0, 10.0, 10.0, 10.0], [1.0, 2.0, 3.0, 4.0])

    assert var > 0.0
    assert vol > 0.0
    assert feedback < 0.0


def test_madevolve_evaluates_and_mutates_alpha_population() -> None:
    data = _market_frame()
    target = data["close"].pct_change().shift(-1).fillna(0.0)
    engine = MadEvolveAlphaEngine()

    candidates = engine.evolve(data, target, generations=2)

    assert len(candidates) >= 2
    assert candidates[0].fitness >= candidates[-1].fitness
    assert "deflated_sharpe" in candidates[0].diagnostics


def test_signal_adaptive_executor_changes_depth_with_inventory_and_signal() -> None:
    executor = SignalAdaptiveExecutor(max_participation=0.25)

    buy_pressure = executor.compute_quote(
        remaining_inventory=-50,
        alpha_signal=0.8,
        time_left_fraction=0.5,
        volatility=0.2,
        inventory_limit=100,
    )
    sell_pressure = executor.compute_quote(
        remaining_inventory=50,
        alpha_signal=-0.8,
        time_left_fraction=0.5,
        volatility=0.2,
        inventory_limit=100,
    )

    assert buy_pressure.bid_depth < buy_pressure.ask_depth
    assert sell_pressure.ask_depth < sell_pressure.bid_depth
    assert 0.0 <= buy_pressure.participation_rate <= 0.25


def test_institutional_fractal_features_are_deterministic_not_placeholder_random() -> None:
    data = _market_frame(140)
    factory = InstitutionalFeatureFactory()

    first = factory.compute_fractal_features(data, windows=[60])
    second = factory.compute_fractal_features(data, windows=[60])

    pd.testing.assert_frame_equal(first, second)
    assert first["hurst_60d"].dropna().between(0.0, 1.0).all()

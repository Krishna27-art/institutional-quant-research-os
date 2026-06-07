import numpy as np
import pandas as pd

from src.feature_store.compute import FeatureComputer, ResearchFeatures


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=index,
    )


def test_fractional_weights_follow_lopez_de_prado_signs() -> None:
    features = ResearchFeatures()

    weights = features._fractional_weights(d=0.4, threshold=0.0, max_weights=4)

    np.testing.assert_allclose(weights, np.array([1.0, -0.4, -0.12, -0.064]), atol=1e-12)


def test_feature_computer_exposes_phase1_research_features() -> None:
    close = np.linspace(100.0, 140.0, 180) + np.sin(np.arange(180) / 3.0)
    data = _ohlcv(close)
    computer = FeatureComputer()

    out = computer.compute_all(
        data,
        [
            "fracdiff_close_d04",
            "chaos_logistic_return",
            "chaos_tent_return",
            "hurst_60d",
            "rough_vol_regime_60d",
        ],
    )

    assert list(out.columns) == [
        "fracdiff_close_d04",
        "chaos_logistic_return",
        "chaos_tent_return",
        "hurst_60d",
        "rough_vol_regime_60d",
    ]
    assert out["fracdiff_close_d04"].first_valid_index() is not None
    assert out["hurst_60d"].dropna().between(0.0, 1.0).all()
    assert set(out["rough_vol_regime_60d"].dropna().unique()).issubset({0.0, 1.0})


def test_chaotic_return_features_are_trailing_and_bounded() -> None:
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 160)))
    data = _ohlcv(close)
    computer = FeatureComputer()

    logistic = computer.compute_feature("chaos_logistic_return", data)
    tent = computer.compute_feature("chaos_tent_return", data)

    assert logistic.dropna().between(0.0, 0.95).all()
    assert tent.dropna().between(0.0, 0.9).all()

    changed = data.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 10.0
    original_until_previous = computer.compute_feature("chaos_logistic_return", data).iloc[:-1]
    changed_until_previous = computer.compute_feature("chaos_logistic_return", changed).iloc[:-1]

    pd.testing.assert_series_equal(original_until_previous, changed_until_previous)


def test_rough_vol_regime_flags_low_hurst_windows() -> None:
    pattern = np.tile(np.array([1.0, -1.0]), 90)
    close = 100.0 + np.cumsum(pattern)
    data = _ohlcv(close)
    computer = FeatureComputer()

    hurst = computer.compute_feature("hurst_60d", data)
    rough = computer.compute_feature("rough_vol_regime_60d", data)

    assert hurst.dropna().iloc[-1] < 0.45
    assert rough.dropna().iloc[-1] == 1.0

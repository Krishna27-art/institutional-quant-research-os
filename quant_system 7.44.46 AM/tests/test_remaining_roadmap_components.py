import numpy as np
import pandas as pd
import pytest

from market_data.feature_generation import ChaoticFeatureSelector
from market_data.options import RoughBergomiPricer, black_scholes_price


def test_chaotic_feature_selector_finds_predictive_feature() -> None:
    rng = np.random.default_rng(123)
    n = 160
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    target = 0.8 * signal + 0.05 * rng.normal(size=n)
    X = pd.DataFrame(
        {
            "predictive": signal,
            "redundant": signal + 0.01 * rng.normal(size=n),
            "noise": noise,
        }
    )

    selector = ChaoticFeatureSelector(population_size=18, generations=18, random_state=7)
    result = selector.select(X, pd.Series(target))

    assert "predictive" in result.selected_features or "redundant" in result.selected_features
    assert "noise" not in result.selected_features or result.relevance > 0.5
    assert result.score > 0.4


def test_rough_bergomi_prices_options_and_generates_mispricing_signal() -> None:
    pricer = RoughBergomiPricer(
        hurst=0.12,
        eta=1.2,
        rho=-0.5,
        xi0=0.04,
        steps=16,
        paths=512,
        random_state=11,
    )
    model_price = pricer.price(spot=100.0, strike=100.0, maturity=0.5, rate=0.02, option_type="call")
    bs_price = black_scholes_price(100.0, 100.0, 0.5, 0.02, 0.2, "call")
    signal = pricer.mispricing_signal(
        spot=100.0,
        strike=100.0,
        maturity=0.5,
        market_price=model_price.price * 0.8,
        rate=0.02,
        option_type="call",
        min_edge=0.05,
    )

    assert model_price.price > 0.0
    assert model_price.standard_error > 0.0
    assert model_price.implied_vol_proxy > 0.0
    assert bs_price > 0.0
    assert signal.action == "buy_option"
    assert signal.signal > 0.0

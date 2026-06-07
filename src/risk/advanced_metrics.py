"""Advanced risk, validation, and volatility tools from the roadmap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import stats


def annualized_sharpe(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe with zero risk-free rate."""
    values = pd.Series(returns, dtype=float).dropna()
    if len(values) < 2:
        return 0.0
    std = values.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(values.mean() / std * np.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> float:
    """Bailey/Lopez de Prado-style deflated Sharpe approximation.

    The observed Sharpe is penalized by the expected best Sharpe inflation from
    trying many variants, adjusted for skew and kurtosis.
    """
    values = pd.Series(returns, dtype=float).dropna()
    observed = annualized_sharpe(values, periods_per_year=periods_per_year)
    if len(values) < 3 or n_trials <= 1:
        return observed

    skew = float(stats.skew(values, bias=False, nan_policy="omit"))
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False, nan_policy="omit"))
    sr_periodic = observed / np.sqrt(periods_per_year)
    var_sr = (
        1
        + 0.5 * sr_periodic**2
        - skew * sr_periodic
        + ((kurtosis - 3.0) / 4.0) * sr_periodic**2
    ) / max(len(values) - 1, 1)
    inflation = stats.norm.ppf(1.0 - 1.0 / max(n_trials, 2)) * np.sqrt(max(var_sr, 0.0))
    return float(observed - inflation * np.sqrt(periods_per_year))


def prediction_interval_coverage(
    realized: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return PICP and normalized interval width for prediction intervals."""
    frame = pd.DataFrame({"realized": realized, "lower": lower, "upper": upper}).dropna()
    if frame.empty:
        return {"coverage": 0.0, "avg_width": 0.0, "miss_rate": 1.0}
    inside = (frame["realized"] >= frame["lower"]) & (frame["realized"] <= frame["upper"])
    avg_width = float((frame["upper"] - frame["lower"]).mean())
    return {
        "coverage": float(inside.mean()),
        "avg_width": avg_width if np.isfinite(avg_width) else 0.0,
        "miss_rate": float(1.0 - inside.mean()),
    }


def algometric_feedback_gap(
    realized_pnl: pd.Series | np.ndarray,
    predicted_pnl: pd.Series | np.ndarray,
    action: pd.Series | np.ndarray,
) -> float:
    """Estimate self-impact slope: realized-predicted PnL regressed on action."""
    frame = pd.DataFrame({"gap": np.asarray(realized_pnl) - np.asarray(predicted_pnl), "action": action}).dropna()
    if len(frame) < 2:
        return 0.0
    x = frame["action"].to_numpy(dtype=float)
    y = frame["gap"].to_numpy(dtype=float)
    x_var = np.var(x)
    if x_var == 0 or not np.isfinite(x_var):
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / x_var)


@dataclass(frozen=True)
class ArbitrageConstraintResult:
    """Limits-to-arbitrage constraint report."""

    passed: bool
    constraints: dict[str, bool]
    utilization: dict[str, float]


def limits_to_arbitrage(
    position_size: float,
    price: float,
    daily_volume: float,
    volatility: float,
    capital: float,
    margin_requirement: float = 0.2,
    max_correlation: float = 0.0,
    liquidity_fraction: float = 0.01,
    max_margin_fraction: float = 0.5,
    max_volatility: float = 0.5,
    max_allowed_correlation: float = 0.8,
) -> ArbitrageConstraintResult:
    """Check Shleifer-Vishny-style practical arbitrage limits."""
    notional = abs(float(position_size) * float(price))
    liquidity_cap = max(float(daily_volume) * liquidity_fraction * float(price), 1e-12)
    margin_cap = max(float(capital) * max_margin_fraction, 1e-12)
    margin = notional * margin_requirement
    constraints = {
        "liquidity": notional <= liquidity_cap,
        "margin": margin <= margin_cap,
        "volatility": float(volatility) <= max_volatility,
        "correlation": float(max_correlation) <= max_allowed_correlation,
    }
    utilization = {
        "liquidity": float(notional / liquidity_cap),
        "margin": float(margin / margin_cap),
        "volatility": float(volatility / max(max_volatility, 1e-12)),
        "correlation": float(max_correlation / max(max_allowed_correlation, 1e-12)),
    }
    return ArbitrageConstraintResult(passed=all(constraints.values()), constraints=constraints, utilization=utilization)


@dataclass(slots=True)
class PurgedEmbargoTimeSeriesSplit:
    """Time-series CV with a purge before and embargo after each test fold."""

    n_splits: int = 5
    test_size: int | None = None
    purge: int = 0
    embargo: int = 0

    def split(self, X: pd.DataFrame | pd.Series | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        test_size = self.test_size or n // (self.n_splits + 1)
        if test_size <= 0:
            raise ValueError("test_size must be positive")

        for fold in range(self.n_splits):
            test_start = n - (self.n_splits - fold) * test_size
            test_end = min(test_start + test_size, n)
            if test_start <= 0 or test_start >= n:
                continue

            left_train_end = max(0, test_start - self.purge)
            right_train_start = min(n, test_end + self.embargo)
            train_idx = np.r_[0:left_train_end, right_train_start:n]
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) and len(test_idx):
                yield train_idx.astype(int), test_idx.astype(int)


@dataclass(slots=True)
class MirroredWeibullVaR:
    """Left-tail VaR/ES with mirrored Weibull tail fitting."""

    confidence: float = 0.99
    min_tail_observations: int = 20

    def fit_tail(self, returns: pd.Series | np.ndarray) -> tuple[float, float]:
        """Fit Weibull shape/scale to positive loss magnitudes."""
        values = pd.Series(returns, dtype=float).dropna()
        losses = (-values[values < 0.0]).to_numpy(dtype=float)
        if len(losses) < self.min_tail_observations:
            losses = (-values).clip(lower=0.0).to_numpy(dtype=float)
            losses = losses[losses > 0.0]
        if len(losses) < 3:
            return 1.0, float(np.std(values) if len(values) else 0.0)
        shape, _, scale = stats.weibull_min.fit(losses, floc=0.0)
        return float(shape), float(scale)

    def var(self, portfolio_value: float, returns: pd.Series | np.ndarray) -> float:
        """Positive currency loss at configured confidence."""
        shape, scale = self.fit_tail(returns)
        if scale <= 0:
            return 0.0
        loss_fraction = stats.weibull_min.ppf(self.confidence, shape, loc=0.0, scale=scale)
        return float(max(loss_fraction, 0.0) * portfolio_value)

    def expected_shortfall(self, portfolio_value: float, returns: pd.Series | np.ndarray) -> float:
        """Expected shortfall by integrating fitted Weibull quantiles."""
        shape, scale = self.fit_tail(returns)
        if scale <= 0:
            return 0.0
        qs = np.linspace(self.confidence, 0.999, 128)
        tail_losses = stats.weibull_min.ppf(qs, shape, loc=0.0, scale=scale)
        return float(np.nanmean(tail_losses) * portfolio_value)


@dataclass(slots=True)
class FIGARCHVolatility:
    """Lightweight FIGARCH-style long-memory variance forecaster.

    This is a deterministic fractional-kernel forecaster suitable for feature
    generation and risk overlays. It is intentionally lighter than full MLE
    FIGARCH, but captures persistent variance via fractional weights.
    """

    d: float = 0.4
    threshold: float = 1e-4
    max_lags: int = 256
    annualization: int = 252
    mean_: float = field(default=0.0, init=False)
    realized_variance_: pd.Series = field(default_factory=pd.Series, init=False)
    unconditional_variance_: float = field(default=0.0, init=False)
    weights_: np.ndarray = field(default_factory=lambda: np.array([1.0]), init=False)

    def fit(self, returns: pd.Series | np.ndarray) -> "FIGARCHVolatility":
        values = pd.Series(returns, dtype=float).dropna()
        self.mean_ = float(values.mean()) if len(values) else 0.0
        residuals = values - self.mean_
        self.realized_variance_ = residuals.pow(2)
        self.unconditional_variance_ = float(self.realized_variance_.mean()) if len(values) else 0.0
        self.weights_ = self._fractional_weights()
        return self

    def forecast(self, horizon: int = 1, annualized: bool = True) -> float:
        if not hasattr(self, "realized_variance_"):
            raise ValueError("fit must be called before forecast")
        realized = self.realized_variance_.dropna().to_numpy(dtype=float)
        if len(realized) == 0:
            return 0.0
        weights = self.weights_[: min(len(self.weights_), len(realized))]
        recent = realized[-len(weights) :][::-1]
        variance = float(np.dot(weights, recent) / max(weights.sum(), 1e-12))
        variance = max(variance, self.unconditional_variance_ * 0.05)
        horizon_variance = variance * max(horizon, 1)
        if annualized:
            return float(np.sqrt(horizon_variance * self.annualization / max(horizon, 1)))
        return horizon_variance

    def _fractional_weights(self) -> np.ndarray:
        weights = [1.0]
        for k in range(1, self.max_lags):
            weight = weights[-1] * (k - 1 + self.d) / k
            if abs(weight) < self.threshold:
                break
            weights.append(weight)
        return np.asarray(weights, dtype=float)

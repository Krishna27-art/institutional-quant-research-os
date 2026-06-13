"""
Research-driven features from the advanced quant roadmap.

The functions here are intentionally deterministic and trailing-only so they
can be used safely in walk-forward research and production feature snapshots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ResearchFeatures:
    """Fractional-memory, chaotic-map, and rough-volatility features."""

    def compute(self, feature_name: str, data: pd.DataFrame) -> pd.Series:
        """Compute a research feature."""
        if "close" not in data.columns:
            raise ValueError("Data must contain 'close' column")

        close = data["close"].astype(float)

        if feature_name == "fracdiff_close_d04":
            return self._fractional_difference(close, d=0.4)
        if feature_name == "chaos_logistic_return":
            returns = np.log(close / close.shift(1))
            normalized = self._rolling_minmax(returns, window=63)
            return self._logistic_map(normalized)
        if feature_name == "chaos_tent_return":
            returns = np.log(close / close.shift(1))
            normalized = self._rolling_minmax(returns, window=63)
            return self._tent_map(normalized)
        if feature_name == "hurst_60d":
            returns = np.log(close / close.shift(1))
            return self._rolling_hurst(returns, window=60)
        if feature_name == "rough_vol_regime_60d":
            hurst = self.compute("hurst_60d", data)
            return pd.Series(np.where(hurst < 0.45, 1.0, 0.0), index=data.index).where(hurst.notna())

        raise ValueError(f"Unknown research feature: {feature_name}")

    def _fractional_difference(
        self,
        series: pd.Series,
        d: float,
        threshold: float = 1e-3,
        max_weights: int = 256,
    ) -> pd.Series:
        """Apply fixed-width fractional differencing while preserving index."""
        weights = self._fractional_weights(d=d, threshold=threshold, max_weights=max_weights)
        out = pd.Series(np.nan, index=series.index, dtype=float)
        values = series.astype(float)

        if len(weights) > len(values):
            return out

        reversed_weights = weights[::-1]
        for end in range(len(weights) - 1, len(values)):
            window = values.iloc[end - len(weights) + 1 : end + 1]
            if window.isna().any():
                continue
            out.iloc[end] = float(np.dot(reversed_weights, window.to_numpy(dtype=float)))

        return out

    def _fractional_weights(self, d: float, threshold: float, max_weights: int) -> np.ndarray:
        """Return Lopez de Prado-style fractional differencing weights."""
        weights = [1.0]
        for k in range(1, max_weights):
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
        return np.array(weights, dtype=float)

    def _rolling_minmax(self, series: pd.Series, window: int) -> pd.Series:
        """Normalize values to [0, 1] using only trailing observations."""
        rolling_min = series.rolling(window=window, min_periods=window).min()
        rolling_max = series.rolling(window=window, min_periods=window).max()
        spread = rolling_max - rolling_min
        normalized = (series - rolling_min) / spread.replace(0.0, np.nan)
        return normalized.clip(lower=0.0, upper=1.0)

    def _logistic_map(self, x: pd.Series, r: float = 3.8) -> pd.Series:
        """Chaotic logistic-map transform on normalized returns."""
        return r * x * (1.0 - x)

    def _tent_map(self, x: pd.Series, mu: float = 1.8) -> pd.Series:
        """Chaotic tent-map transform on normalized returns."""
        left = mu * x
        right = mu * (1.0 - x)
        return pd.Series(np.where(x < 0.5, left, right), index=x.index).where(x.notna())

    def _rolling_hurst(self, returns: pd.Series, window: int, max_lag: int = 20) -> pd.Series:
        """Estimate rolling Hurst exponent by log-log variance scaling."""
        hurst = pd.Series(np.nan, index=returns.index, dtype=float)

        for end in range(window - 1, len(returns)):
            sample = returns.iloc[end - window + 1 : end + 1].dropna()
            if len(sample) < window - 1:
                continue
            hurst.iloc[end] = self._hurst(sample, max_lag=max_lag)

        return hurst

    def _hurst(self, series: pd.Series, max_lag: int) -> float:
        """Estimate H where std(x[t+lag]-x[t]) scales as lag**H."""
        values = series.to_numpy(dtype=float)
        max_lag = min(max_lag, max(2, len(values) // 2))
        lags = np.arange(2, max_lag + 1)
        tau = []

        for lag in lags:
            diff = values[lag:] - values[:-lag]
            scale = np.std(diff)
            if np.isfinite(scale) and scale > 0:
                tau.append(scale)
            else:
                tau.append(np.nan)

        tau_array = np.asarray(tau, dtype=float)
        valid = np.isfinite(tau_array) & (tau_array > 0)
        if valid.sum() < 2:
            return np.nan

        slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau_array[valid]), 1)
        return float(np.clip(slope, 0.0, 1.0))

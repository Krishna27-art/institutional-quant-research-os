"""
Feature engineering layer: microstructure features,
chaotic transformations, and sector graph construction.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Computes all features for the alpha strategies.
    Optimized for vectorized operations on pandas DataFrames.
    """

    @staticmethod
    def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute candlestick microstructure features:
        - Body ratio, range ratio, upper/lower shadow ratios
        - Log returns, realized volatility
        - Price persistence signals
        """
        df = df.copy()

        # Basic candlestick features
        df["body"] = abs(df["Close"] - df["Open"])
        df["range"] = df["High"] - df["Low"]
        df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

        df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
        df["upper_shadow_ratio"] = df["upper_shadow"] / df["range"].replace(0, np.nan)
        df["lower_shadow_ratio"] = df["lower_shadow"] / df["range"].replace(0, np.nan)

        # Direction
        df["is_bullish"] = (df["Close"] > df["Open"]).astype(int)

        # Log returns
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Realized volatility (rolling)
        for window in [5, 14, 21, 60]:
            df[f"realized_vol_{window}"] = (
                df["log_return"].rolling(window).std() * np.sqrt(252)
            )

        # ATR (Average True Range)
        df["tr"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1))
            )
        )
        for window in [5, 14, 21]:
            df[f"atr_{window}"] = df["tr"].rolling(window).mean()

        # Persistence signal (dt) - tendency to continue in same direction
        df["persistence"] = (
            df["is_bullish"].rolling(5).mean() - 0.5
        ) * 2  # Normalized to [-1, 1]

        return df

    @staticmethod
    def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features: relative volume, volume imbalance,
        volume-weighted momentum.
        """
        df = df.copy()

        # Relative volume (vs 20-day average)
        vol_ma = df["Volume"].rolling(20).mean()
        df["relative_volume"] = df["Volume"] / vol_ma.replace(0, np.nan)

        # Short-term relative volume (vs 5-day)
        vol_ma_5 = df["Volume"].rolling(5).mean()
        df["relative_volume_5"] = df["Volume"] / vol_ma_5.replace(0, np.nan)

        # Volume z-score
        df["volume_zscore"] = (
            (df["Volume"] - df["Volume"].rolling(50).mean()) /
            df["Volume"].rolling(50).std().replace(0, np.nan)
        )

        # Bid-ask imbalance proxy (using OHLCV)
        df["volume_imbalance"] = (
            2 * df["Close"] - df["High"] - df["Low"]
        ) / (df["High"] - df["Low"]).replace(0, np.nan)

        # Volume-price confirmation
        df["vol_price_confirm"] = (
            df["is_bullish"] * df["relative_volume"]
        )

        # VWAP distance
        if "Vwap" in df.columns:
            df["vwap_distance"] = (df["Close"] - df["Vwap"]) / df["Vwap"]
            df["vwap_distance_zscore"] = (
                df["vwap_distance"].rolling(50)
                .apply(lambda x: stats.zscore(x)[-1] if len(x) > 1 else 0,
                       raw=False)
            )
        else:
            typical = (df["High"] + df["Low"] + df["Close"]) / 3
            cum_vol = df["Volume"].cumsum()
            cum_vol_price = (typical * df["Volume"]).cumsum()
            vwap = cum_vol_price / cum_vol.replace(0, np.nan)
            df["vwap_distance"] = (df["Close"] - vwap) / vwap

        return df

    @staticmethod
    def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-scale momentum and mean-reversion features.
        """
        df = df.copy()

        # Multi-scale returns
        for window in [1, 3, 5, 10, 21, 60]:
            df[f"return_{window}"] = df["Close"].pct_change(window)

        # RSI
        for window in [14, 21]:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f"rsi_{window}"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["Close"].ewm(span=12).mean()
        ema_26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Band position
        for window in [20]:
            sma = df["Close"].rolling(window).mean()
            std = df["Close"].rolling(window).std()
            df["bb_position"] = (df["Close"] - sma) / (2 * std).replace(0, np.nan)

        # Mean reversion score
        df["mean_reversion_score"] = -df["bb_position"]

        return df

    @staticmethod
    def compute_chaotic_features(
        df: pd.DataFrame,
        map_type: str = "logistic",
        r: float = 3.99,
        mu: float = 0.5,
        n_iterations: int = 100,
        n_bins: int = 10
    ) -> pd.DataFrame:
        """
        Apply chaotic map transformations to financial time series.
        Based on the Bi-Level Chaotic Fusion GCN paper.

        The idea: chaotic maps can detect hidden non-linear structures
        in financial noise that linear methods miss.

        Args:
            df: OHLCV DataFrame
            map_type: 'logistic' or 'tent'
            r: Logistic map parameter (3.57 < r <= 4.0 for chaos)
            mu: Tent map parameter
            n_iterations: Number of chaotic iterations
            n_bins: Number of bins for chaotic partitioning
        """
        df = df.copy()

        returns = df["log_return"].dropna().values
        if len(returns) < 50:
            return df

        # Normalize returns to [0, 1] for chaotic map input
        ret_min, ret_max = returns.min(), returns.max()
        if ret_max == ret_min:
            df["chaotic_entropy"] = 0.0
            return df

        normalized = (returns - ret_min) / (ret_max - ret_min)

        # Apply chaotic map iteration
        chaotic_series = np.zeros(len(normalized))
        chaotic_series[0] = normalized[0]

        for i in range(1, len(normalized)):
            x = chaotic_series[i - 1]
            if map_type == "logistic":
                # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
                chaotic_series[i] = r * x * (1 - x)
            else:
                # Tent map: x_{n+1} = mu * min(x_n, 1 - x_n)
                chaotic_series[i] = mu * min(x, 1 - x)

        # Compute chaotic entropy (Shannon entropy of chaotic partitions)
        def chaotic_entropy(series, bins):
            hist, _ = np.histogram(series, bins=bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist / hist.sum()))

        # Rolling chaotic entropy
        window = 20
        entropies = []
        for i in range(len(chaotic_series)):
            if i < window:
                entropies.append(np.nan)
            else:
                entropies.append(
                    chaotic_entropy(chaotic_series[i - window:i], n_bins)
                )

        df["chaotic_entropy"] = np.nan
        df.loc[df["log_return"].notna(), "chaotic_entropy"] = entropies

        # Chaotic Lyapunov exponent (sensitivity to initial conditions)
        lyapunov = []
        for i in range(len(chaotic_series)):
            if i < window:
                lyapunov.append(np.nan)
            else:
                segment = chaotic_series[i - window:i]
                if map_type == "logistic":
                    # For logistic map: lambda = ln|r - 2rx|
                    lambdas = np.log(np.abs(r - 2 * r * segment))
                else:
                    # For tent map: lambda = ln(mu) when in chaotic regime
                    lambdas = np.full(len(segment), np.log(mu))
                lyapunov.append(np.mean(lambdas))

        df["chaotic_lyapunov"] = np.nan
        df.loc[df["log_return"].notna(), "chaotic_lyapunov"] = lyapunov

        # Chaotic deviation score (distance between actual and chaotic trajectory)
        deviation = np.abs(normalized - chaotic_series)
        deviation_rolling = pd.Series(deviation).rolling(window).mean().values
        df["chaotic_deviation"] = np.nan
        df.loc[df["log_return"].notna(), "chaotic_deviation"] = deviation_rolling

        return df

    @staticmethod
    def build_sector_correlation_matrix(
        data_dict: Dict[str, pd.DataFrame],
        lookback: int = 60
    ) -> pd.DataFrame:
        """
        Build cross-sectional correlation matrix for sector graph.

        Args:
            data_dict: {symbol: DataFrame with 'log_return'}
            lookback: Rolling window for correlation

        Returns:
            Correlation matrix DataFrame
        """
        returns = pd.DataFrame({
            sym: df["log_return"] for sym, df in data_dict.items()
            if "log_return" in df.columns
        })

        return returns.tail(lookback).corr()

    @staticmethod
    def build_sector_graph(
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.3,
        sector_map: Dict[str, str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build adjacency matrix and feature matrix for GCN.

        Args:
            correlation_matrix: Stock correlation matrix
            threshold: Minimum correlation for edge
            sector_map: {symbol: sector} mapping

        Returns:
            (adjacency_matrix, feature_matrix)
        """
        n = len(correlation_matrix)

        # Adjacency: threshold-based edges
        adj = (correlation_matrix.abs() > threshold).astype(float).values
        np.fill_diagonal(adj, 0)  # No self-loops

        # Feature matrix: correlation values as edge weights
        features = correlation_matrix.values

        return adj, features

    def compute_all_features(
        self,
        df: pd.DataFrame,
        include_chaotic: bool = True,
        chaotic_config: dict = None
    ) -> pd.DataFrame:
        """Compute all features in one pass."""
        df = self.compute_candlestick_features(df)
        df = self.compute_volume_features(df)
        df = self.compute_momentum_features(df)

        if include_chaotic and chaotic_config:
            df = self.compute_chaotic_features(df, **chaotic_config)

        return df

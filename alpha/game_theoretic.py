"""
Game-Theoretic Alpha for Indian Markets.
Models market participants as strategic agents in a game.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GameTheoreticSignal:
    symbol: str
    direction: int  # 0=short, 1=neutral, 2=long
    confidence: float
    hot_money_score: float
    institutional_score: float
    equilibrium_deviation: float


class GameTheoreticAlpha:
    """
    Game-theoretic alpha generation.

    Models:
    1. Hot money (retail, HFT) as fast-moving agents
    2. Institutional money as slow-moving agents
    3. Equilibrium analysis to detect mispricings

    Based on the idea that markets are strategic games where
    different participant types have different time horizons
    and information advantages.
    """

    def __init__(self, config: dict):
        self.config = config
        gt_config = config.get("alpha", {}).get("game_theoretic", {})

        self.hot_money_window = gt_config.get("hot_money_window", 5)
        self.institutional_window = gt_config.get("institutional_window", 20)
        self.equilibrium_tolerance = gt_config.get("equilibrium_tolerance", 0.01)

        self._hot_money_positions: Dict[str, float] = {}
        self._institutional_positions: Dict[str, float] = {}

    def _compute_hot_money_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> float:
        """
        Compute hot money (fast-moving) signal.

        Uses short-term momentum and volume patterns.
        """
        if len(df) < self.hot_money_window:
            return 0.0

        recent = df.tail(self.hot_money_window)

        # Short-term momentum
        momentum = (recent["Close"].iloc[-1] / recent["Close"].iloc[0]) - 1

        # Volume acceleration
        vol_avg = recent["Volume"].mean()
        vol_current = recent["Volume"].iloc[-1]
        vol_acceleration = (vol_current / vol_avg) - 1 if vol_avg > 0 else 0

        # Price volatility
        volatility = recent["Close"].pct_change().std()

        # Combine signals
        hot_money_score = momentum * 0.5 + vol_acceleration * 0.3 - volatility * 0.2

        return hot_money_score

    def _compute_institutional_signal(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> float:
        """
        Compute institutional (slow-moving) signal.

        Uses longer-term trends and value metrics.
        """
        if len(df) < self.institutional_window:
            return 0.0

        recent = df.tail(self.institutional_window)

        # Long-term momentum
        momentum = (recent["Close"].iloc[-1] / recent["Close"].iloc[0]) - 1

        # Trend strength (regression R^2)
        x = np.arange(len(recent))
        y = recent["Close"].values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Volume consistency
        vol_cv = recent["Volume"].std() / recent["Volume"].mean() if recent["Volume"].mean() > 0 else 0
        vol_consistency = 1 - min(vol_cv, 1.0)

        # Combine signals
        institutional_score = momentum * 0.4 + r_squared * 0.4 + vol_consistency * 0.2

        return institutional_score

    def _compute_equilibrium(
        self,
        hot_money_score: float,
        institutional_score: float
    ) -> Tuple[float, float]:
        """
        Compute equilibrium and deviation.

        Equilibrium is where hot money and institutional signals align.
        Deviation measures how far current state is from equilibrium.
        """
        # Equilibrium is the average of the two signals
        equilibrium = (hot_money_score + institutional_score) / 2

        # Deviation is the distance from equilibrium
        deviation = abs(hot_money_score - institutional_score)

        return equilibrium, deviation

    def generate_signal(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict:
        """
        Generate game-theoretic signal.

        Args:
            data_dict: {symbol: DataFrame}
            symbol: Symbol to generate signal for

        Returns:
            Signal dictionary with direction, confidence, etc.
        """
        if symbol not in data_dict:
            return {
                "direction": 1,
                "confidence": 0.0,
                "hot_money_score": 0.0,
                "institutional_score": 0.0,
                "equilibrium_deviation": 0.0,
            }

        df = data_dict[symbol]

        # Compute signals
        hot_money_score = self._compute_hot_money_signal(df, symbol)
        institutional_score = self._compute_institutional_signal(df, symbol)

        # Compute equilibrium
        equilibrium, deviation = self._compute_equilibrium(
            hot_money_score, institutional_score
        )

        # Store positions for tracking
        self._hot_money_positions[symbol] = hot_money_score
        self._institutional_positions[symbol] = institutional_score

        # Generate direction based on equilibrium and deviation
        if deviation < self.equilibrium_tolerance:
            # Near equilibrium - follow the trend
            if equilibrium > 0.01:
                direction = 2  # Long
                confidence = min(abs(equilibrium) * 10, 0.8)
            elif equilibrium < -0.01:
                direction = 0  # Short
                confidence = min(abs(equilibrium) * 10, 0.8)
            else:
                direction = 1  # Neutral
                confidence = 0.0
        else:
            # Far from equilibrium - potential mean reversion
            # If hot money is leading institutional, fade the hot money
            if hot_money_score > institutional_score + self.equilibrium_tolerance:
                direction = 0  # Short (fade hot money long)
                confidence = min(deviation * 5, 0.7)
            elif institutional_score > hot_money_score + self.equilibrium_tolerance:
                direction = 2  # Long (fade hot money short)
                confidence = min(deviation * 5, 0.7)
            else:
                direction = 1  # Neutral
                confidence = 0.0

        return {
            "direction": direction,
            "confidence": confidence,
            "hot_money_score": hot_money_score,
            "institutional_score": institutional_score,
            "equilibrium_deviation": deviation,
        }

    def get_market_state(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Get overall market state from game-theoretic perspective.

        Args:
            data_dict: {symbol: DataFrame}

        Returns:
            Market state dictionary
        """
        hot_money_scores = []
        institutional_scores = []
        deviations = []

        for symbol, df in data_dict.items():
            hot = self._compute_hot_money_signal(df, symbol)
            inst = self._compute_institutional_signal(df, symbol)
            _, dev = self._compute_equilibrium(hot, inst)

            hot_money_scores.append(hot)
            institutional_scores.append(inst)
            deviations.append(dev)

        return {
            "avg_hot_money_score": np.mean(hot_money_scores) if hot_money_scores else 0.0,
            "avg_institutional_score": np.mean(institutional_scores) if institutional_scores else 0.0,
            "avg_deviation": np.mean(deviations) if deviations else 0.0,
            "hot_money_bullish_pct": np.mean([s > 0 for s in hot_money_scores]) if hot_money_scores else 0.0,
            "institutional_bullish_pct": np.mean([s > 0 for s in institutional_scores]) if institutional_scores else 0.0,
        }

    def reset(self) -> None:
        """Reset strategy state."""
        self._hot_money_positions.clear()
        self._institutional_positions.clear()

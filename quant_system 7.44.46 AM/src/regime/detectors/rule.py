"""
Rule-Based Regime Detector
"""

import numpy as np
import pandas as pd
from typing import Tuple
from ..states import MarketRegimeState


class RuleBasedRegimeDetector:
    """Classifies regimes based on technical rule-based criteria."""

    def __init__(self, sma_period: int = 200, vol_period: int = 21) -> None:
        self.sma_period = sma_period
        self.vol_period = vol_period

    def predict(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Predict regime and volatility levels.
        """
        close = data["close"].astype(float)
        returns = close.pct_change()
        sma = close.rolling(self.sma_period).mean()
        vol = returns.rolling(self.vol_period).std() * np.sqrt(252)

        regimes = []
        vols = []

        for t in range(len(close)):
            current_close = close.iloc[t]
            current_sma = sma.iloc[t]
            current_vol = vol.iloc[t]

            # 1. Volatility
            if pd.isna(current_vol):
                vol_state = "normal_vol"
            elif current_vol > 0.25:
                vol_state = "high_vol"
            elif current_vol < 0.12:
                vol_state = "low_vol"
            else:
                vol_state = "normal_vol"

            # 2. Trend Regime
            if pd.isna(current_sma):
                regime = MarketRegimeState.SIDEWAYS.value
            elif current_close > current_sma * 1.02:
                regime = MarketRegimeState.TREND_UP.value
            elif current_close < current_sma * 0.98:
                regime = MarketRegimeState.TREND_DOWN.value
            else:
                regime = MarketRegimeState.SIDEWAYS.value

            regimes.append(regime)
            vols.append(vol_state)

        return pd.Series(regimes, index=data.index), pd.Series(vols, index=data.index)

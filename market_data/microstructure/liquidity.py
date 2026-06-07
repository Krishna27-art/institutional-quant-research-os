"""Liquidity analytics and tiering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class LiquidityTierResult:
    symbol: str
    tier: str
    adv_inr: float
    amihud_illiquidity: float
    spread_proxy: float


class LiquidityAnalyzer:
    def turnover_inr(self, frame: pd.DataFrame) -> pd.Series:
        return frame["close"] * frame["volume"]

    def amihud_illiquidity(self, frame: pd.DataFrame) -> pd.Series:
        turnover = self.turnover_inr(frame).replace(0, np.nan)
        return frame["close"].pct_change().abs() / turnover

    def spread_proxy(self, frame: pd.DataFrame) -> pd.Series:
        return (frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)

    def classify_tier(self, symbol: str, frame: pd.DataFrame) -> LiquidityTierResult:
        adv = float(self.turnover_inr(frame).rolling(20).mean().iloc[-1])
        illiquidity = float(self.amihud_illiquidity(frame).rolling(20).mean().iloc[-1])
        spread = float(self.spread_proxy(frame).rolling(20).mean().iloc[-1])
        if adv >= 1_000_000_000:
            tier = "high"
        elif adv >= 250_000_000:
            tier = "medium"
        else:
            tier = "low"
        return LiquidityTierResult(symbol=symbol.upper(), tier=tier, adv_inr=adv, amihud_illiquidity=illiquidity, spread_proxy=spread)

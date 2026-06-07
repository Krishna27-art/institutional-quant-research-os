"""Market state representation and lightweight state composition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Literal


VolatilityRegime = Literal["low", "medium", "high"]
BreadthRegime = Literal["bullish", "neutral", "bearish"]
LiquidityRegime = Literal["high", "medium", "low"]
ParticipationRegime = Literal["fii", "dii", "mixed", "unknown"]
CorrelationRegime = Literal["low", "medium", "high"]
MacroRegime = Literal["trend", "mean_revert", "volatile", "risk_on", "risk_off", "unknown"]


@dataclass(frozen=True, slots=True)
class MarketState:
    """Compact, machine-readable market context."""

    trend_strength: float
    volatility_regime: VolatilityRegime
    breadth: BreadthRegime
    liquidity_quality: LiquidityRegime
    participation: ParticipationRegime
    correlation_regime: CorrelationRegime
    regime: MacroRegime
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketStateEngine:
    """Rule-based market state composer for the first production slice."""

    def build(self, features: Mapping[str, Any]) -> MarketState:
        trend_strength = float(features.get("trend_strength", 0.0))
        daily_vol = float(features.get("daily_volatility", 0.0))
        breadth_score = float(features.get("breadth_score", 0.0))
        liquidity_score = float(features.get("liquidity_score", 0.0))
        participation_score = float(features.get("participation_score", 0.0))
        correlation_score = float(features.get("correlation_score", 0.0))

        volatility_regime: VolatilityRegime
        if daily_vol < 0.008:
            volatility_regime = "low"
        elif daily_vol < 0.015:
            volatility_regime = "medium"
        else:
            volatility_regime = "high"

        breadth: BreadthRegime
        if breadth_score > 0.2:
            breadth = "bullish"
        elif breadth_score < -0.2:
            breadth = "bearish"
        else:
            breadth = "neutral"

        liquidity_quality: LiquidityRegime
        if liquidity_score > 0.7:
            liquidity_quality = "high"
        elif liquidity_score > 0.4:
            liquidity_quality = "medium"
        else:
            liquidity_quality = "low"

        participation: ParticipationRegime
        if participation_score > 0.4:
            participation = "fii"
        elif participation_score < -0.4:
            participation = "dii"
        elif participation_score != 0.0:
            participation = "mixed"
        else:
            participation = "unknown"

        correlation_regime: CorrelationRegime
        if correlation_score > 0.7:
            correlation_regime = "high"
        elif correlation_score > 0.35:
            correlation_regime = "medium"
        else:
            correlation_regime = "low"

        if trend_strength > 0.55 and breadth == "bullish":
            regime: MacroRegime = "trend"
        elif trend_strength < -0.35 and breadth == "bearish":
            regime = "mean_revert"
        elif volatility_regime == "high":
            regime = "volatile"
        elif participation == "fii":
            regime = "risk_on"
        elif participation == "dii":
            regime = "risk_off"
        else:
            regime = "unknown"

        confidence = max(0.0, min(1.0, 0.25 + abs(trend_strength) * 0.4 + abs(breadth_score) * 0.2))

        return MarketState(
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            breadth=breadth,
            liquidity_quality=liquidity_quality,
            participation=participation,
            correlation_regime=correlation_regime,
            regime=regime,
            confidence=confidence,
        )


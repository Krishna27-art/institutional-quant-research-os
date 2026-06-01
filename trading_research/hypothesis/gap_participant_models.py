"""Participant models for gap events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ParticipantRegime:
    regime: str
    confidence: float
    diagnostics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {"regime": self.regime, "confidence": self.confidence, **self.diagnostics}


class RetailPanicDetector:
    def __init__(self, min_gap_pct: float = -1.5, min_volume_shock: float = 1.5, max_delivery_pct: float = 35.0):
        self.min_gap_pct = min_gap_pct
        self.min_volume_shock = min_volume_shock
        self.max_delivery_pct = max_delivery_pct

    def score(self, gap_pct: float, volume_ratio: float, delivery_pct: Optional[float] = None, prior_trend: Optional[float] = None) -> float:
        if gap_pct > self.min_gap_pct:
            return 0.0
        score = 0.0
        score += 0.30 * min(abs(gap_pct) / 5.0, 1.0)
        if 0.5 <= volume_ratio <= 10.0:
            score += 0.25 * min((volume_ratio - 1.0) / 4.0, 1.0)
        if delivery_pct is not None and delivery_pct < self.max_delivery_pct:
            score += 0.25 * (1.0 - (delivery_pct / 50.0))
        if prior_trend is not None:
            score += 0.20 * (1.0 - min(abs(prior_trend) / 10.0, 1.0))
        return float(min(score, 1.0))


class InstitutionalAbsorptionDetector:
    def __init__(self, min_delivery_pct: float = 50.0):
        self.min_delivery_pct = min_delivery_pct

    def score(self, delivery_pct: Optional[float] = None, auction_imbalance: Optional[float] = None, open_vs_prev_close: Optional[float] = None) -> float:
        score = 0.0
        if delivery_pct is not None and delivery_pct >= self.min_delivery_pct:
            score += 0.50 * min((delivery_pct - 50.0) / 30.0, 1.0)
        if auction_imbalance is not None:
            score += 0.30 * max(float(auction_imbalance), 0.0)
        if open_vs_prev_close is not None and -2.0 <= open_vs_prev_close <= 0.0:
            score += 0.20 * (1.0 - abs(open_vs_prev_close) / 2.0)
        return float(min(score, 1.0))


class ParticipantRegimeClassifier:
    def __init__(self):
        self.retail_panic = RetailPanicDetector()
        self.institutional_absorption = InstitutionalAbsorptionDetector()

    def classify(
        self,
        gap_pct: float,
        volume_ratio: float,
        delivery_pct: Optional[float] = None,
        prior_trend: Optional[float] = None,
        auction_imbalance: Optional[float] = None,
        open_vs_prev_close: Optional[float] = None,
    ) -> ParticipantRegime:
        retail_score = self.retail_panic.score(gap_pct, volume_ratio, delivery_pct, prior_trend)
        absorption_score = self.institutional_absorption.score(delivery_pct, auction_imbalance, open_vs_prev_close)
        scores = {"retail_panic": retail_score, "institutional_absorption": absorption_score}
        regime, confidence = max(scores.items(), key=lambda item: item[1])
        diagnostics = {
            "retail_panic_score": retail_score,
            "institutional_absorption_score": absorption_score,
            "gap_pct": gap_pct,
            "volume_ratio": volume_ratio,
            "delivery_pct": delivery_pct if delivery_pct is not None else 0.0,
        }
        return ParticipantRegime(regime=regime, confidence=float(confidence), diagnostics=diagnostics)

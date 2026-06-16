"""
Alpha Manager Facade — Orchestrates all active alpha strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping
from datetime import datetime

import pandas as pd

# Dummy scan_symbols since research/alpha was removed
def scan_symbols(data_dict, timestamp):
    import numpy as np
    signals = []
    for sym, df in data_dict.items():
        if df.empty: continue
        direction = np.random.choice([-1, 1])
        signals.append({
            "symbol": sym,
            "direction": float(direction),
            "strength": 0.05,
            "confidence": 0.8,
            "strategy": "orb"
        })
    return signals

from .prediction_storage import PredictionStorage, Prediction
from .prediction_registry import get_prediction_registry, PredictionRecord

logger = logging.getLogger(__name__)


class AlphaManager:
    """
    Orchestrates alpha signal generation, prediction tracking,
    and rolling performance evaluation/demotion.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.registry = get_prediction_registry()
        self.prediction_storage = PredictionStorage()

    def generate_signals(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Generate signals and record predictions for tracking."""
        if market_data.empty:
            return []

        # Run scanner
        signals = scan_symbols({symbol: market_data}, market_data.index[-1])
        
        # Log to prediction registry
        for signal in signals:
            direction = signal.get("direction")
            if direction != 0:
                pred_rec = PredictionRecord(
                    symbol=symbol,
                    strategy=signal.get("strategy", "orb"),
                    direction="long" if direction > 0 else "short",
                    predicted_return=float(signal.get("strength", 0.01)),
                    confidence=float(signal.get("confidence", 0.5)),
                    entry_price=float(market_data.iloc[-1]["close"]),
                    timestamp=datetime.now(),
                    horizon_minutes=390,  # 1 trading day
                )
                pred_id = self.registry.record_prediction(pred_rec)
                signal["prediction_id"] = pred_id

                # Backward compatibility storage
                compat_pred = Prediction(
                    symbol=symbol,
                    direction="long" if direction > 0 else "short",
                    confidence=float(signal.get("confidence", 0.5)),
                    target_price=float(signal.get("target", 0)),
                    stop_loss=float(signal.get("stop", signal.get("stop_loss", 0))),
                    entry_price=float(market_data.iloc[-1]["close"]),
                    timestamp=datetime.now(),
                    strategy=signal.get("strategy", "orb")
                )
                self.prediction_storage.store_prediction(compat_pred)

        return list(signals)

    def combine_signals(self, signals, **_: Any) -> dict[str, Any]:
        """Combine signals from multiple strategies."""
        signal_list = list(signals)
        if not signal_list:
            return {"combined_signal": 0.0, "weights": {}, "signals": []}

        combined = 0.0
        weights: dict[str, float] = {}
        for signal in signal_list:
            strategy = str(signal.get("strategy", "orb"))
            direction = float(signal.get("direction", 0.0))
            strength = float(signal.get("strength", signal.get("rv", 0.0)))
            confidence = float(signal.get("confidence", 0.5))
            combined += direction * strength * confidence
            weights[strategy] = weights.get(strategy, 0.0) + confidence

        total = sum(weights.values()) or 1.0
        weights = {name: value / total for name, value in weights.items()}
        return {"combined_signal": combined / len(signal_list), "weights": weights, "signals": signal_list}
    
    def get_prediction_metrics(self, strategy: str = None) -> dict:
        """Retrieve performance metrics from registry."""
        if strategy:
            report = self.registry.get_strategy_report(strategy)
            return {
                "total_predictions": report.total_predictions,
                "resolved_predictions": report.resolved_predictions,
                "hit_rate": report.hit_rate,
                "mean_ic": report.mean_ic,
                "rolling_ic": report.rolling_ic,
                "avg_return": report.avg_return,
                "sharpe": report.sharpe,
                "is_active": report.is_active,
            }
        else:
            return self.registry.get_summary()
    
    def update_prediction_outcome(self, prediction_id: int, exit_price: float, exit_timestamp: datetime):
        """Update a prediction with its realized outcome."""
        self.registry.resolve_prediction(prediction_id, exit_price, exit_timestamp)

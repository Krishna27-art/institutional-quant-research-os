"""Backward compatibility facade for the alpha manager.

Delegates completely to the unified `src/alpha/manager.py`.
"""

from __future__ import annotations

from typing import Any, Mapping
from datetime import datetime
import pandas as pd

from src.alpha.manager import AlphaManager as UnifiedAlphaManager
from alpha.orb_zarattini import scan_symbols  # Expose for mocking in tests


class AlphaManager:
    """Compatibility wrapper that delegates to src.alpha.manager.AlphaManager."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._manager = UnifiedAlphaManager(config)

    @property
    def prediction_storage(self):
        return self._manager.prediction_storage

    @prediction_storage.setter
    def prediction_storage(self, val):
        self._manager.prediction_storage = val

    def generate_signals(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return self._manager.generate_signals(symbol, market_data, **kwargs)

    def combine_signals(self, signals, **kwargs: Any) -> dict[str, Any]:
        return self._manager.combine_signals(signals, **kwargs)
    
    def get_prediction_metrics(self, strategy: str = None) -> dict:
        return self._manager.get_prediction_metrics(strategy)
    
    def update_prediction_outcome(self, prediction_id: int, exit_price: float, exit_timestamp: datetime):
        self._manager.update_prediction_outcome(prediction_id, exit_price, exit_timestamp)

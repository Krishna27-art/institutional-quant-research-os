"""Backward compatibility facade for the feature store.

Delegates completely to the unified `src/features/store.py`.
"""

from __future__ import annotations

from typing import Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd

from src.features.store import (
    FeatureStore as UnifiedFeatureStore,
    FeatureStorageBackend,
    FeatureMetadata,
    FeatureSet,
    get_feature_store as get_unified_feature_store,
)


class FeatureStore:
    """Compatibility wrapper that delegates to src.features.store.FeatureStore."""

    def __init__(
        self,
        storage_backend: FeatureStorageBackend = FeatureStorageBackend.MEMORY,
        storage_path: str = None
    ) -> None:
        self._store = UnifiedFeatureStore(storage_backend, storage_path)

    @property
    def feature_registry(self):
        return self._store.feature_registry

    @feature_registry.setter
    def feature_registry(self, val):
        self._store.feature_registry = val

    def store_features(
        self,
        features: pd.DataFrame,
        feature_name: str,
        version: int = 1,
        metadata: FeatureMetadata = None,
        overwrite: bool = False
    ) -> str:
        return self._store.store_features(features, feature_name, version, metadata, overwrite)

    def get_features(
        self,
        feature_name: str,
        version: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        return self._store.get_features(feature_name, version, timestamp)

    def get_feature_metadata(
        self,
        feature_name: str,
        version: Optional[int] = None
    ) -> Optional[FeatureMetadata]:
        return self._store.get_feature_metadata(feature_name, version)

    def list_versions(self, feature_name: str) -> List[int]:
        return self._store.list_versions(feature_name)

    def check_consistency(
        self,
        feature_name: str,
        training_version: int,
        serving_version: int
    ) -> Tuple[bool, str]:
        return self._store.check_consistency(feature_name, training_version, serving_version)

    def backfill_features(
        self,
        feature_name: str,
        start_date: datetime,
        end_date: datetime,
        compute_fn: callable,
        version: int = 1
    ) -> int:
        return self._store.backfill_features(feature_name, start_date, end_date, compute_fn, version)

    def get_feature_summary(self) -> dict:
        return self._store.get_feature_summary()

    def print_summary(self) -> None:
        self._store.print_summary()


def get_feature_store() -> FeatureStore:
    """Get the singleton feature store instance wrapper."""
    unified_store = get_unified_feature_store()
    wrapper = FeatureStore()
    wrapper._store = unified_store
    return wrapper

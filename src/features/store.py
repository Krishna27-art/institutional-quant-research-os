"""
Feature Store with Versioning
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class FeatureStorageBackend(Enum):
    """Storage backends for features."""
    REDIS = "redis"
    TIMESCALEDB = "timescaledb"
    PARQUET = "parquet"
    MEMORY = "memory"


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    description: str
    data_type: str
    version: int
    created_at: datetime
    created_by: str
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureSet:
    """A set of features for a specific time and version."""
    features: pd.DataFrame
    metadata: FeatureMetadata
    timestamp: datetime
    version: int
    hash: str


class FeatureStore:
    """
    Feature store with versioning.
    
    This class provides a centralized repository for features with versioning,
    lineage tracking, and consistency checks.
    """
    
    def __init__(
        self,
        storage_backend: FeatureStorageBackend = FeatureStorageBackend.MEMORY,
        storage_path: str = None
    ):
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path) if storage_path else Path(__file__).parent / "feature_store"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._memory_store: Dict[str, Dict[int, FeatureSet]] = {}
        self.feature_registry: Dict[str, FeatureMetadata] = {}
        
        logger.info(f"FeatureStore initialized with backend: {storage_backend.value}")
    
    def store_features(
        self,
        features: pd.DataFrame,
        feature_name: str,
        version: int = 1,
        metadata: FeatureMetadata = None,
        overwrite: bool = False
    ) -> str:
        feature_hash = self._generate_hash(features, version)
        
        if metadata is None:
            metadata = FeatureMetadata(
                name=feature_name,
                description=f"Feature set {feature_name}",
                data_type="numeric",
                version=version,
                created_at=datetime.now(),
                created_by="system"
            )
        
        feature_set = FeatureSet(
            features=features,
            metadata=metadata,
            timestamp=datetime.now(),
            version=version,
            hash=feature_hash
        )
        
        if self.storage_backend == FeatureStorageBackend.MEMORY:
            self._store_memory(feature_name, feature_set, overwrite)
        elif self.storage_backend == FeatureStorageBackend.PARQUET:
            self._store_parquet(feature_name, feature_set, overwrite)
        else:
            logger.warning(f"Backend {self.storage_backend.value} not implemented, using memory")
            self._store_memory(feature_name, feature_set, overwrite)
        
        self.feature_registry[f"{feature_name}_v{version}"] = metadata
        
        logger.info(f"Stored features {feature_name} v{version} with hash {feature_hash}")
        return feature_hash
    
    def get_features(
        self,
        feature_name: str,
        version: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        if version is None:
            version = self._get_latest_version(feature_name)
            if version is None:
                logger.warning(f"No versions found for {feature_name}")
                return None
        
        if self.storage_backend == FeatureStorageBackend.MEMORY:
            feature_set = self._get_memory(feature_name, version)
        elif self.storage_backend == FeatureStorageBackend.PARQUET:
            feature_set = self._get_parquet(feature_name, version)
        else:
            feature_set = self._get_memory(feature_name, version)
        
        if feature_set is None:
            logger.warning(f"Features {feature_name} v{version} not found")
            return None
        
        return feature_set.features
    
    def get_feature_metadata(
        self,
        feature_name: str,
        version: Optional[int] = None
    ) -> Optional[FeatureMetadata]:
        if version is None:
            version = self._get_latest_version(feature_name)
            if version is None:
                return None
        
        key = f"{feature_name}_v{version}"
        return self.feature_registry.get(key)
    
    def list_versions(self, feature_name: str) -> List[int]:
        versions = []
        
        if self.storage_backend == FeatureStorageBackend.MEMORY:
            if feature_name in self._memory_store:
                versions = list(self._memory_store[feature_name].keys())
        elif self.storage_backend == FeatureStorageBackend.PARQUET:
            feature_dir = self.storage_path / feature_name
            if feature_dir.exists():
                for version_file in feature_dir.glob("v*.parquet"):
                    version = int(version_file.stem[1:])
                    versions.append(version)
        
        return sorted(versions)
    
    def check_consistency(
        self,
        feature_name: str,
        training_version: int,
        serving_version: int
    ) -> Tuple[bool, str]:
        training_meta = self.get_feature_metadata(feature_name, training_version)
        serving_meta = self.get_feature_metadata(feature_name, serving_version)
        
        if training_meta is None or serving_meta is None:
            return False, "One or both versions not found"
        
        if training_version != serving_version:
            return False, f"Version mismatch: training v{training_version}, serving v{serving_version}"
        
        training_features = self.get_features(feature_name, training_version)
        serving_features = self.get_features(feature_name, serving_version)
        
        if training_features is None or serving_features is None:
            return False, "Cannot retrieve features for comparison"
        
        if not training_features.columns.equals(serving_features.columns):
            return False, "Schema mismatch between training and serving"
        
        if not self._check_statistics_consistency(training_features, serving_features):
            return False, "Statistics drift detected"
        
        return True, "Consistent"
    
    def backfill_features(
        self,
        feature_name: str,
        start_date: datetime,
        end_date: datetime,
        compute_fn: callable,
        version: int = 1
    ) -> int:
        dates = pd.date_range(start=start_date, end=end_date, freq='1D')
        backfilled = 0
        
        for date in dates:
            try:
                features = compute_fn(date)
                
                if features is not None and not features.empty:
                    self.store_features(
                        features,
                        f"{feature_name}_{date.strftime('%Y%m%d')}",
                        version
                    )
                    backfilled += 1
            except Exception as e:
                logger.error(f"Failed to backfill for {date}: {e}")
        
        logger.info(f"Backfilled {backfilled} dates for {feature_name}")
        return backfilled
    
    def _generate_hash(self, features: pd.DataFrame, version: int) -> str:
        hash_input = f"{features.columns.tolist()}_{features.shape}_{version}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_latest_version(self, feature_name: str) -> Optional[int]:
        versions = self.list_versions(feature_name)
        return max(versions) if versions else None
    
    def _store_memory(self, feature_name: str, feature_set: FeatureSet, overwrite: bool) -> None:
        if feature_name not in self._memory_store:
            self._memory_store[feature_name] = {}
        
        if not overwrite and feature_set.version in self._memory_store[feature_name]:
            logger.warning(f"Version {feature_set.version} already exists, not overwriting")
            return
        
        self._memory_store[feature_name][feature_set.version] = feature_set
    
    def _get_memory(self, feature_name: str, version: int) -> Optional[FeatureSet]:
        if feature_name not in self._memory_store:
            return None
        return self._memory_store[feature_name].get(version)
    
    def _store_parquet(self, feature_name: str, feature_set: FeatureSet, overwrite: bool) -> None:
        feature_dir = self.storage_path / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = feature_dir / f"v{feature_set.version}.parquet"
        
        if not overwrite and file_path.exists():
            logger.warning(f"File {file_path} already exists, not overwriting")
            return
        
        feature_set.features.to_parquet(file_path)
        
        meta_path = feature_dir / f"v{feature_set.version}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'name': feature_set.metadata.name,
                'description': feature_set.metadata.description,
                'version': feature_set.version,
                'created_at': feature_set.metadata.created_at.isoformat(),
                'created_by': feature_set.metadata.created_by,
                'dependencies': feature_set.metadata.dependencies,
                'tags': feature_set.metadata.tags,
                'parameters': feature_set.metadata.parameters,
                'statistics': feature_set.metadata.statistics
            }, f)
    
    def _get_parquet(self, feature_name: str, version: int) -> Optional[FeatureSet]:
        feature_dir = self.storage_path / feature_name
        file_path = feature_dir / f"v{version}.parquet"
        meta_path = feature_dir / f"v{version}_metadata.json"
        
        if not file_path.exists():
            return None
        
        features = pd.read_parquet(file_path)
        
        metadata = None
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
                metadata = FeatureMetadata(
                    name=meta_data['name'],
                    description=meta_data['description'],
                    data_type="numeric",
                    version=meta_data['version'],
                    created_at=pd.to_datetime(meta_data['created_at']),
                    created_by=meta_data['created_by'],
                    dependencies=meta_data.get('dependencies', []),
                    tags=meta_data.get('tags', {}),
                    parameters=meta_data.get('parameters', {}),
                    statistics=meta_data.get('statistics', {})
                )
        
        return FeatureSet(
            features=features,
            metadata=metadata,
            timestamp=datetime.now(),
            version=version,
            hash=self._generate_hash(features, version)
        )
    
    def _check_statistics_consistency(
        self,
        training_features: pd.DataFrame,
        serving_features: pd.DataFrame
    ) -> bool:
        for col in training_features.columns:
            if col not in serving_features.columns:
                return False
            
            train_mean = training_features[col].mean()
            serving_mean = serving_features[col].mean()
            
            if abs(train_mean - serving_mean) / abs(train_mean) > 0.05:
                return False
        
        return True
    
    def get_feature_summary(self) -> Dict:
        summary = {
            'total_features': len(self.feature_registry),
            'storage_backend': self.storage_backend.value,
            'features': {}
        }
        
        for key, metadata in self.feature_registry.items():
            summary['features'][key] = {
                'name': metadata.name,
                'version': metadata.version,
                'created_at': metadata.created_at.isoformat(),
                'description': metadata.description
            }
        
        return summary
    
    def print_summary(self) -> None:
        summary = self.get_feature_summary()
        
        print("\n" + "="*60)
        print("FEATURE STORE SUMMARY")
        print("="*60)
        print(f"\nStorage Backend: {summary['storage_backend']}")
        print(f"Total Features: {summary['total_features']}")
        
        if summary['features']:
            print(f"\nFeature Sets:")
            for key, info in summary['features'].items():
                print(f"  {key}: {info['description']} (v{info['version']})")
        
        print("\n" + "="*60)


_feature_store = None

def get_feature_store() -> FeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store

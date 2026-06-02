"""
Time-Machine Simulator API
Provides deterministic, immutable snapshots for any timestamp
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import pickle
import hashlib
import json

from point_in_time_reconstruction import (
    PointInTimeReconstructor,
    TimeSnapshot,
    DataType,
    CorporateActionAdjuster,
    UniverseManager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnapshotCache:
    """Cache for time snapshots to improve performance"""
    
    def __init__(self, cache_path: str = "data/time_machine_cache"):
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, TimeSnapshot] = {}
        
    def _get_cache_key(self, timestamp: datetime, symbols: tuple, data_types: tuple) -> str:
        """Generate cache key"""
        key_str = f"{timestamp.isoformat()}_{sorted(symbols)}_{sorted(data_types)}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(
        self,
        timestamp: datetime,
        symbols: tuple,
        data_types: tuple
    ) -> Optional[TimeSnapshot]:
        """Get snapshot from cache"""
        cache_key = self._get_cache_key(timestamp, symbols, data_types)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_path / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    snapshot = pickle.load(f)
                self.memory_cache[cache_key] = snapshot
                return snapshot
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
        
        return None
    
    def put(
        self,
        timestamp: datetime,
        symbols: tuple,
        data_types: tuple,
        snapshot: TimeSnapshot
    ) -> None:
        """Store snapshot in cache"""
        cache_key = self._get_cache_key(timestamp, symbols, data_types)
        
        # Store in memory
        self.memory_cache[cache_key] = snapshot
        
        # Store on disk
        cache_file = self.cache_path / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(snapshot, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def clear(self) -> None:
        """Clear cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_path.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cache cleared")


class TimeMachineSimulator:
    """
    Time-Machine Simulator for deterministic point-in-time snapshots
    """
    
    def __init__(
        self,
        data_path: str = "data/historical",
        cache_enabled: bool = True
    ):
        self.reconstructor = PointInTimeReconstructor(data_path, cache_enabled)
        self.cache = SnapshotCache() if cache_enabled else None
        self.snapshot_history: List[TimeSnapshot] = []
        
        logger.info("Time-Machine Simulator initialized")
    
    def get_snapshot(
        self,
        timestamp: datetime,
        symbols: List[str],
        data_types: List[DataType],
        lookback_days: int = 0,
        use_cache: bool = True
    ) -> TimeSnapshot:
        """
        Get point-in-time snapshot at timestamp
        Deterministic: same inputs always produce same output
        
        Args:
            timestamp: Point-in-time timestamp
            symbols: List of symbols to include
            data_types: Types of data to include
            lookback_days: Number of days to look back for features
            use_cache: Whether to use cache
            
        Returns:
            TimeSnapshot with frozen data
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(timestamp, tuple(symbols), tuple(data_types))
            if cached:
                logger.debug(f"Cache hit for {timestamp}")
                return cached
        
        # Get snapshot from reconstructor
        snapshot = self.reconstructor.get_snapshot(
            timestamp=timestamp,
            symbols=symbols,
            data_types=data_types,
            lookback_days=lookback_days
        )
        
        # Store in cache
        if use_cache and self.cache:
            self.cache.put(timestamp, tuple(symbols), tuple(data_types), snapshot)
        
        # Store in history
        self.snapshot_history.append(snapshot)
        
        return snapshot
    
    def get_snapshot_range(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = '1D',
        symbols: Optional[List[str]] = None,
        data_types: Optional[List[DataType]] = None,
        lookback_days: int = 0,
        use_cache: bool = True
    ) -> List[TimeSnapshot]:
        """
        Get range of snapshots for backtesting
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Frequency of snapshots
            symbols: Symbols to include
            data_types: Data types to include
            lookback_days: Lookback window for features
            use_cache: Whether to use cache
            
        Returns:
            List of TimeSnapshots
        """
        if symbols is None:
            symbols = ['NIFTY', 'BANKNIFTY']
        
        if data_types is None:
            data_types = [DataType.OHLCV]
        
        timestamps = pd.date_range(start_date, end_date, freq=frequency)
        snapshots = []
        
        for timestamp in timestamps:
            snapshot = self.get_snapshot(
                timestamp=timestamp,
                symbols=symbols,
                data_types=data_types,
                lookback_days=lookback_days,
                use_cache=use_cache
            )
            snapshots.append(snapshot)
        
        logger.info(f"Generated {len(snapshots)} snapshots from {start_date} to {end_date}")
        
        return snapshots
    
    def get_feature_matrix(
        self,
        snapshots: List[TimeSnapshot],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert snapshots to feature matrix for ML
        
        Args:
            snapshots: List of TimeSnapshots
            feature_names: Specific features to extract (None = all)
            
        Returns:
            DataFrame with features indexed by timestamp and symbol
        """
        feature_rows = []
        
        for snapshot in snapshots:
            if snapshot.features.empty:
                continue
            
            for symbol, row in snapshot.features.iterrows():
                feature_row = {
                    'timestamp': snapshot.timestamp,
                    'symbol': symbol,
                }
                
                # Add features
                if feature_names:
                    for feat in feature_names:
                        if feat in row:
                            feature_row[feat] = row[feat]
                else:
                    for feat, val in row.items():
                        feature_row[feat] = val
                
                feature_rows.append(feature_row)
        
        if feature_rows:
            df = pd.DataFrame(feature_rows)
            df = df.set_index(['timestamp', 'symbol'])
        else:
            df = pd.DataFrame()
        
        return df
    
    def get_labels(
        self,
        snapshots: List[TimeSnapshot],
        forward_periods: int = 1,
        target_col: str = 'returns'
    ) -> pd.DataFrame:
        """
        Get labels for ML training (forward returns)
        
        Args:
            snapshots: List of TimeSnapshots
            forward_periods: Number of periods forward
            target_col: Target column for returns
            
        Returns:
            DataFrame with labels indexed by timestamp and symbol
        """
        label_rows = []
        
        # Get feature matrix first
        feature_matrix = self.get_feature_matrix(snapshots)
        
        if feature_matrix.empty:
            return pd.DataFrame()
        
        # Calculate forward returns
        for (timestamp, symbol), row in feature_matrix.iterrows():
            # Find snapshot forward_periods ahead
            current_idx = next(
                i for i, s in enumerate(snapshots) 
                if s.timestamp == timestamp
            )
            
            if current_idx + forward_periods < len(snapshots):
                future_snapshot = snapshots[current_idx + forward_periods]
                
                if symbol in future_snapshot.features.index:
                    future_price = future_snapshot.features.loc[symbol, 'close']
                    current_price = row['close']
                    
                    forward_return = (future_price / current_price) - 1
                    
                    label_rows.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'label': forward_return,
                    })
        
        if label_rows:
            df = pd.DataFrame(label_rows)
            df = df.set_index(['timestamp', 'symbol'])
        else:
            df = pd.DataFrame()
        
        return df
    
    def validate_determinism(
        self,
        timestamp: datetime,
        symbols: List[str],
        data_types: List[DataType],
        num_trials: int = 5
    ) -> Dict[str, bool]:
        """
        Validate that simulator is deterministic
        Same inputs should always produce same outputs
        
        Args:
            timestamp: Timestamp to test
            symbols: Symbols to test
            data_types: Data types to test
            num_trials: Number of trials to run
            
        Returns:
            Validation results
        """
        results = {
            'deterministic': True,
            'hash_consistent': True,
            'features_consistent': True,
        }
        
        snapshots = []
        hashes = []
        
        for _ in range(num_trials):
            snapshot = self.get_snapshot(
                timestamp=timestamp,
                symbols=symbols,
                data_types=data_types,
                use_cache=False  # Disable cache for testing
            )
            snapshots.append(snapshot)
            
            # Compute hash
            feature_hash = hashlib.sha256(
                snapshot.features.to_string().encode()
            ).hexdigest()
            hashes.append(feature_hash)
        
        # Check if all hashes are the same
        if len(set(hashes)) != 1:
            results['hash_consistent'] = False
            results['deterministic'] = False
            logger.error("Hash inconsistency detected")
        
        # Check if features are the same
        for i in range(1, len(snapshots)):
            if not snapshots[0].features.equals(snapshots[i].features):
                results['features_consistent'] = False
                results['deterministic'] = False
                logger.error("Feature inconsistency detected")
        
        return results
    
    def get_snapshot_metadata(
        self,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Get metadata about snapshot at timestamp
        Useful for debugging and validation
        
        Args:
            timestamp: Timestamp to query
            
        Returns:
            Metadata dictionary
        """
        # Get snapshot
        snapshot = self.get_snapshot(
            timestamp=timestamp,
            symbols=['NIFTY'],
            data_types=[DataType.OHLCV],
            lookback_days=0
        )
        
        metadata = {
            'timestamp': timestamp.isoformat(),
            'universe_size': len(snapshot.universe),
            'data_versions': {
                data_id: {
                    'version_hash': version.version_hash,
                    'record_date': version.record_date.isoformat(),
                }
                for data_id, version in snapshot.data_versions.items()
            },
            'feature_count': len(snapshot.features.columns) if not snapshot.features.empty else 0,
            'row_count': len(snapshot.features) if not snapshot.features.empty else 0,
        }
        
        return metadata
    
    def export_snapshot(
        self,
        snapshot: TimeSnapshot,
        export_path: str
    ) -> None:
        """
        Export snapshot to file for reproducibility
        
        Args:
            snapshot: Snapshot to export
            export_path: Path to export to
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export as JSON
        export_data = {
            'timestamp': snapshot.timestamp.isoformat(),
            'universe': snapshot.universe,
            'features': snapshot.features.to_dict() if not snapshot.features.empty else {},
            'metadata': snapshot.metadata,
            'data_versions': {
                data_id: {
                    'version_hash': version.version_hash,
                    'record_date': version.record_date.isoformat(),
                    'data_type': version.data_type.value,
                }
                for data_id, version in snapshot.data_versions.items()
            }
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported snapshot to {export_path}")
    
    def import_snapshot(
        self,
        import_path: str
    ) -> TimeSnapshot:
        """
        Import snapshot from file for reproducibility
        
        Args:
            import_path: Path to import from
            
        Returns:
            TimeSnapshot
        """
        import_path = Path(import_path)
        
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        # Reconstruct snapshot
        snapshot = TimeSnapshot(
            timestamp=datetime.fromisoformat(import_data['timestamp']),
            data_versions={},
            universe=import_data['universe'],
            features=pd.DataFrame(import_data['features']),
            metadata=import_data['metadata'],
        )
        
        logger.info(f"Imported snapshot from {import_path}")
        
        return snapshot


def simulate_time_machine():
    """Simulate time machine operations"""
    
    print("="*60)
    print("TIME-MACHINE SIMULATOR SIMULATION")
    print("="*60)
    
    # Initialize simulator
    simulator = TimeMachineSimulator(cache_enabled=True)
    
    # Get single snapshot
    print("\n1. Getting snapshot at 2022-01-15...")
    snapshot = simulator.get_snapshot(
        timestamp=datetime(2022, 1, 15),
        symbols=['NIFTY', 'BANKNIFTY'],
        data_types=[DataType.OHLCV],
        lookback_days=20
    )
    
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  Universe: {snapshot.universe}")
    print(f"  Features shape: {snapshot.features.shape}")
    
    # Get snapshot range
    print("\n2. Getting snapshot range...")
    snapshots = simulator.get_snapshot_range(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        frequency='1D',
        symbols=['NIFTY'],
        data_types=[DataType.OHLCV],
        lookback_days=5
    )
    print(f"  Generated {len(snapshots)} snapshots")
    
    # Get feature matrix
    print("\n3. Getting feature matrix...")
    feature_matrix = simulator.get_feature_matrix(snapshots)
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    if not feature_matrix.empty:
        print(f"  Features: {list(feature_matrix.columns)}")
    
    # Get labels
    print("\n4. Getting labels...")
    labels = simulator.get_labels(snapshots, forward_periods=1)
    print(f"  Labels shape: {labels.shape}")
    
    # Validate determinism
    print("\n5. Validating determinism...")
    validation = simulator.validate_determinism(
        timestamp=datetime(2022, 1, 15),
        symbols=['NIFTY'],
        data_types=[DataType.OHLCV],
        num_trials=3
    )
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Get metadata
    print("\n6. Getting snapshot metadata...")
    metadata = simulator.get_snapshot_metadata(datetime(2022, 1, 15))
    print(f"  Universe size: {metadata['universe_size']}")
    print(f"  Feature count: {metadata['feature_count']}")
    print(f"  Row count: {metadata['row_count']}")
    
    # Export/import snapshot
    print("\n7. Testing export/import...")
    export_path = "data/test_snapshot.json"
    simulator.export_snapshot(snapshot, export_path)
    imported_snapshot = simulator.import_snapshot(export_path)
    print(f"  Exported and imported snapshot")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_time_machine()

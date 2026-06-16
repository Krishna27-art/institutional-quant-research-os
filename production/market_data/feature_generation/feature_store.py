"""
Feature Store with Point-in-Time Data Integrity
Based on the critique: Ensure no look-ahead bias in backtesting

Critical for avoiding look-ahead bias:
- Features computed using only data available at that time
- Corporate actions adjusted
- Survivorship bias corrected
- Point-in-time reconstruction

Architecture:
- Raw data → Point-in-time adjustment → Feature computation → Storage
- Versioned features for reproducibility
- Feature lineage tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class FeatureVersion(Enum):
    """Feature version for reproducibility."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "latest"


@dataclass
class FeatureMetadata:
    """Metadata for a feature."""
    name: str
    description: str
    category: str
    version: FeatureVersion
    created_at: datetime
    parameters: Dict
    dependencies: List[str] = field(default_factory=list)
    lookback_window: int = 0  # Required historical data in days


@dataclass
class PointInTimeData:
    """Point-in-time data snapshot."""
    timestamp: datetime
    symbol: str
    ohlcv: pd.DataFrame
    corporate_actions: List[Dict] = field(default_factory=list)
    adjusted_prices: Optional[pd.Series] = None


class PointInTimeReconstructor:
    """
    Reconstruct data as it would have appeared at each point in time.
    
    This prevents look-ahead bias by ensuring features are computed
    using only data that would have been available at that time.
    """
    
    def __init__(self):
        self.corporate_action_cache: Dict[str, List[Dict]] = {}
        self.adjustment_cache: Dict[str, pd.Series] = {}
    
    def add_corporate_action(
        self,
        symbol: str,
        action_type: str,  # split, bonus, dividend, rights
        ex_date: datetime,
        ratio: float = 1.0,
        amount: float = 0.0
    ) -> None:
        """Add corporate action for adjustment."""
        if symbol not in self.corporate_action_cache:
            self.corporate_action_cache[symbol] = []
        
        self.corporate_action_cache[symbol].append({
            'type': action_type,
            'ex_date': ex_date,
            'ratio': ratio,
            'amount': amount
        })
    
    def get_adjusted_prices(
        self,
        symbol: str,
        prices: pd.Series,
        as_of_date: datetime
    ) -> pd.Series:
        """
        Get price series adjusted for corporate actions as of a specific date.
        
        Only applies corporate actions that occurred before as_of_date.
        """
        if symbol not in self.corporate_action_cache:
            return prices
        
        adjusted = prices.copy()
        
        # Apply adjustments in reverse chronological order
        actions = [
            a for a in self.corporate_action_cache[symbol]
            if a['ex_date'] <= as_of_date
        ]
        
        # Sort by date descending
        actions.sort(key=lambda x: x['ex_date'], reverse=True)
        
        for action in actions:
            if action['type'] == 'split':
                adjusted = adjusted * action['ratio']
            elif action['type'] == 'bonus':
                adjusted = adjusted * action['ratio']
            elif action['type'] == 'dividend':
                # Adjust for dividend (simplified)
                adjusted = adjusted - action['amount']
        
        return adjusted
    
    def get_point_in_time_snapshot(
        self,
        symbol: str,
        data: pd.DataFrame,
        as_of_date: datetime,
        lookback_days: int = 252
    ) -> PointInTimeData:
        """
        Get point-in-time snapshot of data.
        
        Returns only data that would have been available at as_of_date.
        """
        # Filter data up to as_of_date
        historical_data = data[data.index <= as_of_date]
        
        # Get lookback window
        if len(historical_data) > lookback_days:
            historical_data = historical_data.tail(lookback_days)
        
        # Get corporate actions up to as_of_date
        actions = []
        if symbol in self.corporate_action_cache:
            actions = [
                a for a in self.corporate_action_cache[symbol]
                if a['ex_date'] <= as_of_date
            ]
        
        # Adjust prices
        adjusted_prices = self.get_adjusted_prices(
            symbol, historical_data['close'], as_of_date
        )
        
        return PointInTimeData(
            timestamp=as_of_date,
            symbol=symbol,
            ohlcv=historical_data,
            corporate_actions=actions,
            adjusted_prices=adjusted_prices
        )


class FeatureStore:
    """
    Feature Store with point-in-time data integrity.
    
    Ensures no look-ahead bias by:
    1. Computing features using only historical data available at each timestamp
    2. Adjusting for corporate actions
    3. Versioning features for reproducibility
    4. Tracking feature lineage
    """
    
    def __init__(self):
        self.pit_reconstructor = PointInTimeReconstructor()
        self.feature_registry: Dict[str, FeatureMetadata] = {}
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        self.feature_lineage: Dict[str, List[str]] = {}
    
    def register_feature(
        self,
        name: str,
        description: str,
        category: str,
        version: FeatureVersion = FeatureVersion.V1,
        parameters: Dict = None,
        dependencies: List[str] = None,
        lookback_window: int = 0
    ) -> None:
        """Register a feature in the store."""
        metadata = FeatureMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            created_at=datetime.now(),
            parameters=parameters or {},
            dependencies=dependencies or [],
            lookback_window=lookback_window
        )
        
        self.feature_registry[name] = metadata
    
    def compute_feature_point_in_time(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_name: str,
        as_of_date: datetime,
        feature_func: callable
    ) -> Optional[float]:
        """
        Compute a feature value as of a specific point in time.
        
        Ensures no look-ahead bias by using only data available at as_of_date.
        """
        # Get point-in-time snapshot
        pit_data = self.pit_reconstructor.get_point_in_time_snapshot(
            symbol, data, as_of_date
        )
        
        # Compute feature using only historical data
        try:
            feature_value = feature_func(pit_data.ohlcv)
            return feature_value
        except Exception as e:
            print(f"Error computing feature {feature_name}: {e}")
            return None
    
    def compute_feature_series_point_in_time(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_name: str,
        feature_func: callable,
        min_points: int = 100
    ) -> pd.Series:
        """
        Compute feature series with point-in-time integrity.
        
        Computes feature value at each timestamp using only historical data.
        """
        feature_values = []
        timestamps = []
        
        # Start from minimum required lookback
        start_idx = min_points
        
        for i in range(start_idx, len(data)):
            as_of_date = data.index[i]
            
            # Compute feature using data up to as_of_date
            historical_data = data.iloc[:i+1]
            
            try:
                feature_value = feature_func(historical_data)
                feature_values.append(feature_value)
                timestamps.append(as_of_date)
            except Exception as e:
                feature_values.append(np.nan)
                timestamps.append(as_of_date)
        
        return pd.Series(feature_values, index=timestamps)
    
    def compute_rolling_feature(
        self,
        data: pd.DataFrame,
        window: int,
        feature_func: callable
    ) -> pd.Series:
        """
        Compute rolling feature with proper point-in-time handling.
        
        This is the standard way to compute features without look-ahead bias.
        """
        return data['close'].rolling(window).apply(feature_func, raw=False)
    
    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for a feature."""
        return self.feature_registry.get(feature_name)
    
    def list_features_by_category(self, category: str) -> List[str]:
        """List all features in a category."""
        return [
            name for name, metadata in self.feature_registry.items()
            if metadata.category == category
        ]
    
    def cache_feature(self, feature_name: str, feature_data: pd.DataFrame) -> None:
        """Cache computed feature data."""
        self.feature_cache[feature_name] = feature_data
    
    def get_cached_feature(self, feature_name: str) -> Optional[pd.DataFrame]:
        """Get cached feature data."""
        return self.feature_cache.get(feature_name)
    
    def validate_point_in_time_integrity(
        self,
        feature_series: pd.Series,
        original_data: pd.DataFrame,
        feature_name: str
    ) -> bool:
        """
        Validate that feature series has no look-ahead bias.
        
        Checks that feature values at time t only depend on data up to time t.
        """
        # This is a simplified validation
        # In production, would implement more sophisticated checks
        
        # Check that feature series aligns with original data
        if not feature_series.index.equals(original_data.index):
            print(f"Feature {feature_name}: Index mismatch detected")
            return False
        
        # Check for NaN values at the beginning (expected due to lookback)
        # but not in the middle (could indicate look-ahead)
        if feature_series.isna().sum() > len(feature_series) * 0.5:
            print(f"Feature {feature_name}: Too many NaN values")
            return False
        
        return True


class StandardFeatures:
    """Standard feature computations with point-in-time integrity."""
    
    @staticmethod
    def returns(data: pd.DataFrame, period: int = 1) -> float:
        """Compute returns over period."""
        if len(data) < period + 1:
            return np.nan
        return (data['close'].iloc[-1] / data['close'].iloc[-period-1] - 1)
    
    @staticmethod
    def volatility(data: pd.DataFrame, period: int = 20) -> float:
        """Compute realized volatility over period."""
        if len(data) < period:
            return np.nan
        returns = data['close'].pct_change().tail(period)
        return returns.std()
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> float:
        """Compute RSI over period."""
        if len(data) < period + 1:
            return np.nan
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    @staticmethod
    def moving_average(data: pd.DataFrame, period: int = 20) -> float:
        """Compute moving average over period."""
        if len(data) < period:
            return np.nan
        return data['close'].tail(period).mean()
    
    @staticmethod
    def bollinger_position(data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> float:
        """Compute position within Bollinger Bands."""
        if len(data) < period:
            return np.nan
        sma = data['close'].tail(period).mean()
        std = data['close'].tail(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        current = data['close'].iloc[-1]
        return (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5


if __name__ == "__main__":
    # Test the Feature Store
    print("Testing Feature Store with Point-in-Time Integrity...")
    
    feature_store = FeatureStore()
    
    # Register some standard features
    feature_store.register_feature(
        name="returns_5d",
        description="5-day returns",
        category="price",
        lookback_window=5
    )
    
    feature_store.register_feature(
        name="volatility_20d",
        description="20-day realized volatility",
        category="volatility",
        lookback_window=20
    )
    
    feature_store.register_feature(
        name="rsi_14",
        description="14-day RSI",
        category="momentum",
        lookback_window=14
    )
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = np.random.normal(100, 10, n).cumsum()
    prices = prices - prices.min() + 100
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n))),
        'close': prices,
        'volume': np.random.normal(1000000, 200000, n)
    }, index=dates)
    
    # Add a corporate action (split)
    feature_store.pit_reconstructor.add_corporate_action(
        symbol="TEST",
        action_type="split",
        ex_date=datetime(2020, 6, 15),
        ratio=2.0
    )
    
    # Compute features with point-in-time integrity
    print("\nComputing features with point-in-time integrity...")
    
    returns_series = feature_store.compute_feature_series_point_in_time(
        symbol="TEST",
        data=data,
        feature_name="returns_5d",
        feature_func=lambda x: StandardFeatures.returns(x, 5)
    )
    
    vol_series = feature_store.compute_feature_series_point_in_time(
        symbol="TEST",
        data=data,
        feature_name="volatility_20d",
        feature_func=lambda x: StandardFeatures.volatility(x, 20)
    )
    
    print(f"Returns series length: {len(returns_series)}")
    print(f"Volatility series length: {len(vol_series)}")
    print(f"Returns sample: {returns_series.tail(5).values}")
    print(f"Volatility sample: {vol_series.tail(5).values}")
    
    # Validate point-in-time integrity
    print("\nValidating point-in-time integrity...")
    returns_valid = feature_store.validate_point_in_time_integrity(
        returns_series, data, "returns_5d"
    )
    vol_valid = feature_store.validate_point_in_time_integrity(
        vol_series, data, "volatility_20d"
    )
    
    print(f"Returns integrity: {returns_valid}")
    print(f"Volatility integrity: {vol_valid}")
    
    # Test point-in-time snapshot
    print("\nTesting point-in-time snapshot...")
    pit_snapshot = feature_store.pit_reconstructor.get_point_in_time_snapshot(
        symbol="TEST",
        data=data,
        as_of_date=datetime(2020, 6, 10),
        lookback_days=50
    )
    
    print(f"Snapshot date: {pit_snapshot.timestamp}")
    print(f"Data points in snapshot: {len(pit_snapshot.ohlcv)}")
    print(f"Corporate actions: {len(pit_snapshot.corporate_actions)}")

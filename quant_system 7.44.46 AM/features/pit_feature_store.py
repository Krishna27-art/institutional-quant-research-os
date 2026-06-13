"""
Point-in-Time (PIT) Feature Store

Based on Qlib's DatasetHandler pattern from repository analysis.
This prevents look-ahead bias by ensuring every feature request
specifies an as_of timestamp and the system guarantees no future data leaks.

Key Concept:
- Traditional feature stores: "Give me RSI for RELIANCE" → uses latest data
- PIT feature store: "Give me RSI for RELIANCE as of 2024-01-15" → uses only data available on that date

This is CRITICAL for accurate backtesting. Without PIT, your backtest
Sharpe ratio is overstated because the model "sees" future data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Type of feature"""
    PRICE = "price"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    DERIVED = "derived"


@dataclass
class FeatureRequest:
    """Request for features with point-in-time constraint"""
    symbol: str
    feature_name: str
    as_of: Union[datetime, date, str]
    lookback_days: int = 20
    feature_type: FeatureType = FeatureType.TECHNICAL
    params: Optional[Dict] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
        
        # Convert as_of to datetime
        if isinstance(self.as_of, str):
            self.as_of = pd.to_datetime(self.as_of)
        elif isinstance(self.as_of, date):
            self.as_of = pd.to_datetime(self.as_of)


@dataclass
class FeatureResult:
    """Result of feature request"""
    symbol: str
    feature_name: str
    as_of: datetime
    value: float
    computed_at: datetime
    data_timestamp: datetime
    lookback_used: int
    feature_version: str


class PITFeatureStore:
    """
    Point-in-Time Feature Store.
    
    This ensures that when you request features as of a specific date,
    you only get data that would have been available on that date.
    
    This prevents look-ahead bias in backtesting.
    
    Architecture:
    1. All raw data is stored with timestamps
    2. Feature computation respects as_of constraint
    3. No future data can leak into feature values
    4. Feature versioning for reproducibility
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize PIT feature store.
        
        Args:
            data_dir: Directory for storing feature data
            cache_enabled: Whether to cache computed features
            cache_ttl_hours: Cache time-to-live in hours
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent / "pit_data"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours
        
        # Feature version (increment when feature logic changes)
        self.feature_version = "1.0"
    
    def _get_cache_key(self, request: FeatureRequest) -> str:
        """Generate cache key for a feature request."""
        key_data = {
            'symbol': request.symbol,
            'feature': request.feature_name,
            'as_of': str(request.as_of),
            'lookback': request.lookback_days,
            'params': request.params,
            'version': self.feature_version
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.data_dir / f"{cache_key}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not self.cache_enabled or not cache_path.exists():
            return False
        
        import time
        cache_age = time.time() - cache_path.stat().st_mtime
        return cache_age < (self.cache_ttl_hours * 3600)
    
    def _filter_data_by_as_of(
        self,
        data: pd.DataFrame,
        as_of: datetime,
        timestamp_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Filter data to only include rows with timestamp <= as_of.
        
        This is the core PIT logic - no future data allowed.
        
        Args:
            data: DataFrame with timestamp column
            as_of: Point-in-time constraint
            timestamp_col: Name of timestamp column
            
        Returns:
            Filtered DataFrame
        """
        if timestamp_col not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in data")
        
        # Ensure timestamp column is datetime
        data = data.copy()
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        
        # Filter: only data with timestamp <= as_of
        filtered = data[data[timestamp_col] <= as_of]
        
        if len(filtered) == 0:
            logger.warning(
                f"No data available for {as_of} after PIT filter. "
                f"Latest data: {data[timestamp_col].max() if len(data) > 0 else 'None'}"
            )
        
        return filtered
    
    def compute_feature(
        self,
        request: FeatureRequest,
        price_data: pd.DataFrame
    ) -> FeatureResult:
        """
        Compute a feature with point-in-time constraint.
        
        Args:
            request: Feature request with as_of constraint
            price_data: Historical price data
            
        Returns:
            FeatureResult with computed value
        """
        cache_key = self._get_cache_key(request)
        cache_path = self._get_cache_path(cache_key)
        
        # Check cache
        if self._is_cache_valid(cache_path):
            logger.debug(f"Using cached feature for {request.symbol} {request.feature_name}")
            cached = pd.read_parquet(cache_path)
            return FeatureResult(
                symbol=request.symbol,
                feature_name=request.feature_name,
                as_of=request.as_of,
                value=float(cached['value'].iloc[0]),
                computed_at=pd.to_datetime(cached['computed_at'].iloc[0]),
                data_timestamp=pd.to_datetime(cached['data_timestamp'].iloc[0]),
                lookback_used=int(cached['lookback_used'].iloc[0]),
                feature_version=cached['version'].iloc[0]
            )
        
        # Apply PIT filter
        pit_data = self._filter_data_by_as_of(price_data, request.as_of)
        
        if len(pit_data) == 0:
            raise ValueError(f"No data available for {request.symbol} as of {request.as_of}")
        
        # Compute feature based on type
        value = self._compute_feature_value(request, pit_data)
        
        # Get the actual timestamp of the data used
        data_timestamp = pit_data['date'].max()
        
        # Create result
        result = FeatureResult(
            symbol=request.symbol,
            feature_name=request.feature_name,
            as_of=request.as_of,
            value=value,
            computed_at=datetime.now(),
            data_timestamp=data_timestamp,
            lookback_used=min(request.lookback_days, len(pit_data)),
            feature_version=self.feature_version
        )
        
        # Cache the result
        if self.cache_enabled:
            cache_df = pd.DataFrame([{
                'value': result.value,
                'computed_at': result.computed_at,
                'data_timestamp': result.data_timestamp,
                'lookback_used': result.lookback_used,
                'version': result.feature_version
            }])
            cache_df.to_parquet(cache_path)
        
        return result
    
    def _compute_feature_value(
        self,
        request: FeatureRequest,
        data: pd.DataFrame
    ) -> float:
        """
        Compute the actual feature value.
        
        Args:
            request: Feature request
            data: PIT-filtered data
            
        Returns:
            Computed feature value
        """
        feature_name = request.feature_name.lower()
        
        # Get lookback window
        lookback = min(request.lookback_days, len(data))
        window_data = data.tail(lookback)
        
        # Price-based features
        if feature_name in ['close', 'price', 'last_price']:
            return float(window_data['close'].iloc[-1])
        
        elif feature_name == 'returns':
            if len(window_data) < 2:
                return 0.0
            return float(window_data['close'].pct_change().iloc[-1])
        
        elif feature_name == 'log_returns':
            if len(window_data) < 2:
                return 0.0
            return float(np.log(window_data['close'].iloc[-1] / window_data['close'].iloc[-2]))
        
        # Technical indicators
        elif feature_name == 'sma':
            period = request.params.get('period', 20)
            if len(window_data) < period:
                return float(window_data['close'].mean())
            return float(window_data['close'].tail(period).mean())
        
        elif feature_name == 'ema':
            period = request.params.get('period', 20)
            if len(window_data) < period:
                return float(window_data['close'].mean())
            return float(window_data['close'].tail(period).ewm(span=period, adjust=False).mean().iloc[-1])
        
        elif feature_name == 'rsi':
            period = request.params.get('period', 14)
            if len(window_data) < period + 1:
                return 50.0  # Neutral
            delta = window_data['close'].diff()
            gain = delta.clip(lower=0).rolling(window=period).mean()
            loss = (-delta.clip(upper=0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        
        elif feature_name == 'volatility':
            period = request.params.get('period', 20)
            if len(window_data) < period:
                return float(window_data['close'].pct_change().std())
            returns = window_data['close'].pct_change().tail(period)
            return float(returns.std())
        
        elif feature_name == 'atr':
            period = request.params.get('period', 14)
            if len(window_data) < period + 1:
                high_low = window_data['high'] - window_data['low']
                return float(high_low.mean())
            high_low = window_data['high'] - window_data['low']
            high_close = (window_data['high'] - window_data['close'].shift()).abs()
            low_close = (window_data['low'] - window_data['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            return float(atr.iloc[-1])
        
        elif feature_name == 'bollinger_upper':
            period = request.params.get('period', 20)
            std_dev = request.params.get('std_dev', 2)
            if len(window_data) < period:
                sma = window_data['close'].mean()
                std = window_data['close'].std()
            else:
                sma = window_data['close'].tail(period).mean()
                std = window_data['close'].tail(period).std()
            return float(sma + std_dev * std)
        
        elif feature_name == 'bollinger_lower':
            period = request.params.get('period', 20)
            std_dev = request.params.get('std_dev', 2)
            if len(window_data) < period:
                sma = window_data['close'].mean()
                std = window_data['close'].std()
            else:
                sma = window_data['close'].tail(period).mean()
                std = window_data['close'].tail(period).std()
            return float(sma - std_dev * std)
        
        elif feature_name == 'volume_sma':
            period = request.params.get('period', 20)
            if 'volume' in window_data.columns:
                if len(window_data) < period:
                    return float(window_data['volume'].mean())
                return float(window_data['volume'].tail(period).mean())
            return 0.0
        
        elif feature_name == 'volume_ratio':
            if 'volume' in window_data.columns:
                current_vol = window_data['volume'].iloc[-1]
                avg_vol = window_data['volume'].mean()
                if avg_vol > 0:
                    return float(current_vol / avg_vol)
            return 1.0
        
        else:
            raise ValueError(f"Unknown feature: {request.feature_name}")
    
    def batch_compute_features(
        self,
        requests: List[FeatureRequest],
        price_data_dict: Dict[str, pd.DataFrame]
    ) -> List[FeatureResult]:
        """
        Compute multiple features in batch.
        
        Args:
            requests: List of feature requests
            price_data_dict: Dictionary of symbol -> price data
            
        Returns:
            List of FeatureResults
        """
        results = []
        
        for request in requests:
            try:
                if request.symbol not in price_data_dict:
                    logger.warning(f"No price data for {request.symbol}")
                    continue
                
                result = self.compute_feature(request, price_data_dict[request.symbol])
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to compute feature {request.feature_name} for {request.symbol}: {e}")
                continue
        
        return results
    
    def get_feature_history(
        self,
        symbol: str,
        feature_name: str,
        start_date: date,
        end_date: date,
        price_data: pd.DataFrame,
        params: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Get feature values over a date range with PIT constraint.
        
        This is useful for backtesting - you get the feature value
        that would have been available on each historical date.
        
        Args:
            symbol: Stock symbol
            feature_name: Feature to compute
            start_date: Start date
            end_date: End date
            price_data: Historical price data
            params: Feature parameters
            
        Returns:
            DataFrame with date and feature value
        """
        if params is None:
            params = {}
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        results = []
        for as_of_date in dates:
            try:
                request = FeatureRequest(
                    symbol=symbol,
                    feature_name=feature_name,
                    as_of=as_of_date,
                    lookback_days=20,
                    params=params
                )
                
                result = self.compute_feature(request, price_data)
                results.append({
                    'date': as_of_date,
                    'value': result.value,
                    'data_timestamp': result.data_timestamp
                })
                
            except Exception as e:
                logger.debug(f"Failed to compute feature for {as_of_date}: {e}")
                continue
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.set_index('date')
        
        return df
    
    def clear_cache(self) -> None:
        """Clear all cached features."""
        if self.data_dir.exists():
            for cache_file in self.data_dir.glob("*.parquet"):
                cache_file.unlink()
        logger.info("Cleared all feature cache")


def get_pit_feature_store(
    data_dir: Optional[str] = None,
    cache_enabled: bool = True
) -> PITFeatureStore:
    """
    Factory function to get a PIT feature store.
    
    Args:
        data_dir: Directory for storing feature data
        cache_enabled: Whether to enable caching
        
    Returns:
        PITFeatureStore instance
    """
    return PITFeatureStore(
        data_dir=data_dir,
        cache_enabled=cache_enabled
    )


if __name__ == "__main__":
    # Test the PIT feature store
    print("Testing PIT Feature Store...")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates
    )
    
    # Create OHLCV DataFrame
    price_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
        'high': prices * (1 + np.random.uniform(0.001, 0.01, 100)),
        'low': prices * (1 - np.random.uniform(0.001, 0.01, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    price_data = price_data.set_index('date')
    
    # Initialize PIT store
    store = get_pit_feature_store(cache_enabled=True)
    
    # Test single feature request
    print("\nTesting single feature request...")
    request = FeatureRequest(
        symbol='RELIANCE',
        feature_name='rsi',
        as_of='2024-02-01',
        lookback_days=20,
        params={'period': 14}
    )
    
    result = store.compute_feature(request, price_data)
    print(f"RSI for RELIANCE as of 2024-02-01: {result.value:.2f}")
    print(f"Data timestamp: {result.data_timestamp}")
    print(f"Lookback used: {result.lookback_used}")
    
    # Test feature history
    print("\nTesting feature history...")
    history = store.get_feature_history(
        symbol='RELIANCE',
        feature_name='rsi',
        start_date=date(2024, 1, 15),
        end_date=date(2024, 1, 31),
        price_data=price_data,
        params={'period': 14}
    )
    
    print(f"Generated {len(history)} historical feature values")
    if len(history) > 0:
        print(f"\nFirst 5 values:")
        print(history.head())
    
    # Test batch requests
    print("\nTesting batch requests...")
    requests = [
        FeatureRequest('RELIANCE', 'rsi', '2024-02-01', 20, params={'period': 14}),
        FeatureRequest('RELIANCE', 'sma', '2024-02-01', 20, params={'period': 20}),
        FeatureRequest('RELIANCE', 'volatility', '2024-02-01', 20),
    ]
    
    results = store.batch_compute_features(requests, {'RELIANCE': price_data})
    print(f"Computed {len(results)} features")
    for r in results:
        print(f"  {r.feature_name}: {r.value:.4f}")
    
    print("\nPIT Feature Store test complete.")

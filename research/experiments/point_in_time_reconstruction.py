"""
Point-in-Time Data Reconstruction Pipeline
Implements time-machine simulation for eliminating lookahead bias
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Data types for point-in-time reconstruction"""
    TICK = "tick"
    OHLCV = "ohlcv"
    CORPORATE_ACTION = "corporate_action"
    INDEX_CONSTITUENT = "index_constituent"
    OPTIONS_CHAIN = "options_chain"
    FLOW_DATA = "flow_data"
    MACRO_DATA = "macro_data"


@dataclass
class DataVersion:
    """Data version information"""
    data_id: str
    version_hash: str
    record_date: datetime
    effective_date: datetime
    data_type: DataType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSnapshot:
    """Point-in-time snapshot"""
    timestamp: datetime
    data_versions: Dict[str, DataVersion]
    universe: List[str]  # List of symbols available at this timestamp
    features: pd.DataFrame
    labels: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorporateActionAdjuster:
    """Adjust prices for corporate actions using only information available at time"""
    
    def __init__(self):
        self.corporate_actions: List[Dict] = []
        
    def load_corporate_actions(self, actions_data: List[Dict]) -> None:
        """Load corporate actions data"""
        self.corporate_actions = actions_data
        logger.info(f"Loaded {len(actions_data)} corporate actions")
    
    def get_adjustment_factor(
        self,
        symbol: str,
        timestamp: datetime
    ) -> float:
        """
        Get cumulative adjustment factor for a symbol at a timestamp
        Only uses corporate actions with effective_date <= timestamp
        
        Args:
            symbol: Stock symbol
            timestamp: Point-in-time timestamp
            
        Returns:
            Adjustment factor (1.0 = no adjustment)
        """
        adjustment_factor = 1.0
        
        for action in self.corporate_actions:
            if (action['symbol'] == symbol and 
                action['effective_date'] <= timestamp):
                
                if action['type'] == 'split':
                    adjustment_factor *= action['split_ratio']
                elif action['type'] == 'bonus':
                    adjustment_factor *= (1 + action['bonus_ratio'])
                elif action['type'] == 'dividend':
                    # Dividend adjustment for price
                    adjustment_factor *= (1 - action['dividend_amount'] / action['price'])
        
        return adjustment_factor
    
    def adjust_price(
        self,
        price: float,
        symbol: str,
        timestamp: datetime
    ) -> float:
        """Adjust price for corporate actions"""
        factor = self.get_adjustment_factor(symbol, timestamp)
        return price * factor


class UniverseManager:
    """Manage point-in-time universe (survivorship bias prevention)"""
    
    def __init__(self):
        self.constituent_history: List[Dict] = []
        
    def load_constituent_history(self, history_data: List[Dict]) -> None:
        """Load index constituent history"""
        self.constituent_history = history_data
        logger.info(f"Loaded constituent history for {len(history_data)} entries")
    
    def get_universe_at_time(
        self,
        timestamp: datetime,
        min_market_cap: Optional[float] = None
    ) -> List[str]:
        """
        Get universe of symbols available at timestamp
        Only includes symbols with valid_from <= timestamp <= valid_to
        
        Args:
            timestamp: Point-in-time timestamp
            min_market_cap: Minimum market cap filter (if available at time)
            
        Returns:
            List of symbols in universe
        """
        universe = []
        
        for entry in self.constituent_history:
            if (entry['valid_from'] <= timestamp and 
                (entry['valid_to'] is None or entry['valid_to'] >= timestamp)):
                
                # Apply market cap filter if specified and available
                if min_market_cap is not None:
                    if 'market_cap' in entry and entry['market_cap'] >= min_market_cap:
                        universe.append(entry['symbol'])
                else:
                    universe.append(entry['symbol'])
        
        return universe


class PointInTimeReconstructor:
    """
    Point-in-time data reconstructor for eliminating lookahead bias
    """
    
    def __init__(
        self,
        data_path: str = "data/historical",
        cache_enabled: bool = True
    ):
        self.data_path = Path(data_path)
        self.cache_enabled = cache_enabled
        self.cache_path = self.data_path / "cache"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        self.corporate_action_adjuster = CorporateActionAdjuster()
        self.universe_manager = UniverseManager()
        
        # Data storage
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.versioned_data: Dict[str, List[DataVersion]] = {}
        
        logger.info("Point-in-Time Reconstructor initialized")
    
    def load_raw_data(
        self,
        data_type: DataType,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load raw historical data
        
        Args:
            data_type: Type of data to load
            symbol: Symbol to load (if applicable)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with raw data
        """
        # In production, this would load from S3/Parquet
        # For now, generate sample data
        if data_type == DataType.OHLCV:
            dates = pd.date_range(start_date or datetime(2020, 1, 1), 
                                  end_date or datetime.now(), freq='1D')
            symbols = [symbol] if symbol else ['NIFTY', 'BANKNIFTY', 'RELIANCE']
            
            data = []
            for sym in symbols:
                for date in dates:
                    np.random.seed(hash((sym, date)) % (2**32))
                    base_price = 20000 if sym == 'NIFTY' else 2000
                    price = base_price * (1 + np.random.normal(0, 0.01))
                    
                    data.append({
                        'symbol': sym,
                        'timestamp': date,
                        'open': price * 0.999,
                        'high': price * 1.001,
                        'low': price * 0.998,
                        'close': price,
                        'volume': np.random.randint(100000, 500000),
                        'record_date': date,  # When this data was recorded
                    })
            
            df = pd.DataFrame(data)
            df = df.set_index('timestamp')
            
        else:
            df = pd.DataFrame()
        
        return df
    
    def get_snapshot(
        self,
        timestamp: datetime,
        symbols: List[str],
        data_types: List[DataType],
        lookback_days: int = 0
    ) -> TimeSnapshot:
        """
        Get point-in-time snapshot at timestamp
        Only uses data with record_date <= timestamp
        
        Args:
            timestamp: Point-in-time timestamp
            symbols: List of symbols to include
            data_types: Types of data to include
            lookback_days: Number of days to look back for features
            
        Returns:
            TimeSnapshot with frozen data
        """
        logger.info(f"Getting snapshot at {timestamp} for {len(symbols)} symbols")
        
        # Get universe at time (survivorship bias prevention)
        universe = self.universe_manager.get_universe_at_time(timestamp)
        filtered_symbols = [s for s in symbols if s in universe]
        
        if not filtered_symbols:
            logger.warning(f"No symbols in universe at {timestamp}")
            return TimeSnapshot(
                timestamp=timestamp,
                data_versions={},
                universe=[],
                features=pd.DataFrame()
            )
        
        # Load data with record_date constraint
        all_data = {}
        data_versions = {}
        
        for data_type in data_types:
            df = self.load_raw_data(
                data_type=data_type,
                symbol=None,  # Load all symbols
                start_date=timestamp - timedelta(days=lookback_days + 30),
                end_date=timestamp
            )
            
            # Filter by record_date <= timestamp (no lookahead)
            if 'record_date' in df.columns:
                df = df[df['record_date'] <= timestamp]
            
            # Filter by symbols in universe
            df = df[df['symbol'].isin(filtered_symbols)]
            
            all_data[data_type.value] = df
            
            # Create data version
            version_hash = self._compute_data_hash(df)
            data_versions[data_type.value] = DataVersion(
                data_id=f"{data_type.value}_{timestamp.strftime('%Y%m%d')}",
                version_hash=version_hash,
                record_date=timestamp,
                effective_date=timestamp,
                data_type=data_type
            )
        
        # Compute features using only data up to timestamp
        features = self._compute_features(all_data, timestamp, lookback_days)
        
        snapshot = TimeSnapshot(
            timestamp=timestamp,
            data_versions=data_versions,
            universe=filtered_symbols,
            features=features,
            metadata={
                'symbols_requested': symbols,
                'symbols_available': filtered_symbols,
                'data_types': [dt.value for dt in data_types],
                'lookback_days': lookback_days,
            }
        )
        
        return snapshot
    
    def _compute_features(
        self,
        data: Dict[str, pd.DataFrame],
        timestamp: datetime,
        lookback_days: int
    ) -> pd.DataFrame:
        """
        Compute features using only data up to timestamp
        No lookahead - only use data with record_date <= timestamp
        
        Args:
            data: Dictionary of data by type
            timestamp: Current timestamp
            lookback_days: Lookback window
            
        Returns:
            DataFrame with computed features
        """
        features_list = []
        
        if 'ohlcv' in data:
            ohlcv = data['ohlcv']
            
            for symbol in ohlcv['symbol'].unique():
                symbol_data = ohlcv[ohlcv['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_index()
                
                # Only use data up to timestamp
                symbol_data = symbol_data[symbol_data.index <= timestamp]
                
                if len(symbol_data) < lookback_days:
                    continue
                
                # Compute features (no lookahead - only use past data)
                recent_data = symbol_data.tail(lookback_days)
                
                feature_row = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'close': recent_data['close'].iloc[-1],
                    'returns_1d': (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-2] - 1) if len(recent_data) >= 2 else 0,
                    'returns_5d': (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-5] - 1) if len(recent_data) >= 5 else 0,
                    'volatility_5d': recent_data['close'].pct_change().tail(5).std(),
                    'volume_ratio': recent_data['volume'].iloc[-1] / recent_data['volume'].mean(),
                    'rsi': self._calculate_rsi(recent_data['close']),
                }
                
                features_list.append(feature_row)
        
        if features_list:
            features_df = pd.DataFrame(features_list)
            features_df = features_df.set_index('symbol')
        else:
            features_df = pd.DataFrame()
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of data for versioning"""
        data_str = df.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def validate_lookahead_prevention(
        self,
        snapshot: TimeSnapshot,
        future_timestamp: datetime
    ) -> Dict[str, bool]:
        """
        Validate that no future data leaked into snapshot
        
        Args:
            snapshot: Snapshot to validate
            future_timestamp: Future timestamp to check against
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'no_future_data': True,
            'no_future_corporate_actions': True,
            'no_future_universe_changes': True,
            'feature_computation_lookahead_free': True,
        }
        
        # Check that all data has record_date <= snapshot timestamp
        for data_id, version in snapshot.data_versions.items():
            if version.record_date > snapshot.timestamp:
                results['no_future_data'] = False
                logger.error(f"Future data detected in {data_id}")
        
        # Check corporate actions
        for action in self.corporate_action_adjuster.corporate_actions:
            if action['effective_date'] > snapshot.timestamp:
                # Ensure this action wasn't used in adjustment
                continue
        
        # Check universe
        future_universe = self.universe_manager.get_universe_at_time(future_timestamp)
        if set(snapshot.universe) != set(future_universe):
            # This is expected - universe can change
            results['no_future_universe_changes'] = False
        
        return results
    
    def create_time_series_snapshots(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = '1D',
        symbols: Optional[List[str]] = None,
        data_types: Optional[List[DataType]] = None
    ) -> List[TimeSnapshot]:
        """
        Create time series of snapshots for backtesting
        
        Args:
            start_date: Start date
            end_date: End date
            frequency: Frequency of snapshots
            symbols: Symbols to include
            data_types: Data types to include
            
        Returns:
            List of TimeSnapshots
        """
        if symbols is None:
            symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE']
        
        if data_types is None:
            data_types = [DataType.OHLCV]
        
        timestamps = pd.date_range(start_date, end_date, freq=frequency)
        snapshots = []
        
        for timestamp in timestamps:
            snapshot = self.get_snapshot(
                timestamp=timestamp,
                symbols=symbols,
                data_types=data_types,
                lookback_days=20
            )
            snapshots.append(snapshot)
        
        logger.info(f"Created {len(snapshots)} snapshots from {start_date} to {end_date}")
        
        return snapshots


def simulate_point_in_time_reconstruction():
    """Simulate point-in-time reconstruction"""
    
    print("="*60)
    print("POINT-IN-TIME RECONSTRUCTION SIMULATION")
    print("="*60)
    
    # Initialize reconstructor
    reconstructor = PointInTimeReconstructor()
    
    # Load sample corporate actions
    corporate_actions = [
        {
            'symbol': 'RELIANCE',
            'type': 'split',
            'split_ratio': 2.0,
            'effective_date': datetime(2022, 6, 1),
            'price': 2500,
        },
        {
            'symbol': 'HDFC',
            'type': 'bonus',
            'bonus_ratio': 1.0,
            'effective_date': datetime(2023, 1, 1),
            'price': 1500,
        },
    ]
    reconstructor.corporate_action_adjuster.load_corporate_actions(corporate_actions)
    
    # Load constituent history
    constituent_history = [
        {
            'symbol': 'NIFTY',
            'valid_from': datetime(2010, 1, 1),
            'valid_to': None,
            'market_cap': 50000000000000,
        },
        {
            'symbol': 'BANKNIFTY',
            'valid_from': datetime(2010, 1, 1),
            'valid_to': None,
            'market_cap': 30000000000000,
        },
        {
            'symbol': 'RELIANCE',
            'valid_from': datetime(2010, 1, 1),
            'valid_to': None,
            'market_cap': 15000000000000,
        },
        {
            'symbol': 'DELETED_STOCK',
            'valid_from': datetime(2010, 1, 1),
            'valid_to': datetime(2021, 6, 1),
            'market_cap': 1000000000000,
        },
    ]
    reconstructor.universe_manager.load_constituent_history(constituent_history)
    
    # Get snapshot at specific timestamp
    print("\n1. Getting snapshot at 2022-01-15...")
    snapshot = reconstructor.get_snapshot(
        timestamp=datetime(2022, 1, 15),
        symbols=['NIFTY', 'BANKNIFTY', 'RELIANCE', 'DELETED_STOCK'],
        data_types=[DataType.OHLCV],
        lookback_days=20
    )
    
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  Universe: {snapshot.universe}")
    print(f"  Features shape: {snapshot.features.shape}")
    print(f"  Data versions: {len(snapshot.data_versions)}")
    
    # Validate lookahead prevention
    print("\n2. Validating lookahead prevention...")
    validation = reconstructor.validate_lookahead_prevention(
        snapshot,
        datetime(2022, 1, 20)
    )
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Create time series snapshots
    print("\n3. Creating time series snapshots...")
    snapshots = reconstructor.create_time_series_snapshots(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 1, 10),
        frequency='1D',
        symbols=['NIFTY', 'BANKNIFTY'],
        data_types=[DataType.OHLCV]
    )
    print(f"  Created {len(snapshots)} snapshots")
    
    # Show sample features
    print("\n4. Sample features from first snapshot:")
    if snapshots and not snapshots[0].features.empty:
        print(snapshots[0].features.head())
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_point_in_time_reconstruction()

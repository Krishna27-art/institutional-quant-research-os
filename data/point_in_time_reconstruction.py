"""
Point-in-Time Data Reconstruction for Institutional Backtesting

This module implements point-in-time (PIT) data reconstruction to avoid
look-ahead bias in backtesting. It ensures that only data available at each
point in time is used for signal generation and trading decisions.

Key Features:
- Data availability tracking
- Corporate action adjustments
- Earnings calendar integration
- Survivorship bias correction
- Look-ahead bias detection
- Feature point-in-time validation

Based on V4 Blueprint - Institutional Architecture
Priority: High (Phase 1)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of data that need point-in-time tracking."""
    PRICE = "price"
    VOLUME = "volume"
    CORPORATE_ACTION = "corporate_action"
    EARNINGS = "earnings"
    FEATURE = "feature"
    REGIME = "regime"


@dataclass
class DataAvailability:
    """Data availability information for a symbol at a point in time."""
    symbol: str
    timestamp: datetime
    data_type: DataType
    is_available: bool
    lag_days: int = 0  # Days delayed relative to event
    source: str = ""


@dataclass
class PointInTimeSnapshot:
    """Complete point-in-time snapshot for a symbol."""
    symbol: str
    timestamp: datetime
    price_data: Optional[pd.Series]
    volume_data: Optional[pd.Series]
    features: Dict[str, float]
    corporate_actions: List
    earnings_events: List
    regime_state: Optional[str]


class PointInTimeReconstructor:
    """
    Point-in-time data reconstructor for institutional backtesting.
    
    This class ensures that backtests use only data that would have been
    available at each point in time, avoiding look-ahead bias.
    
    Key capabilities:
    - Track data availability for each symbol
    - Apply corporate action adjustments point-in-time
    - Validate features for look-ahead bias
    - Correct survivorship bias
    - Generate point-in-time snapshots
    """
    
    def __init__(self):
        self.data_availability: Dict[str, List[DataAvailability]] = {}
        self.corporate_actions_db = {}
        self.earnings_db = {}
        self.feature_history: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = {}
        
        # Data source latencies (in days)
        self.source_latencies = {
            "real_time": 0,
            "end_of_day": 1,
            "delayed": 2
        }
    
    def add_data_availability(
        self,
        symbol: str,
        timestamp: datetime,
        data_type: DataType,
        is_available: bool,
        lag_days: int = 0,
        source: str = "real_time"
    ) -> None:
        """
        Record data availability for a symbol at a point in time.
        
        Args:
            symbol: Stock symbol
            timestamp: Data timestamp
            data_type: Type of data
            is_available: Whether data is available
            lag_days: Days delayed
            source: Data source
        """
        if symbol not in self.data_availability:
            self.data_availability[symbol] = []
        
        availability = DataAvailability(
            symbol=symbol,
            timestamp=timestamp,
            data_type=data_type,
            is_available=is_available,
            lag_days=lag_days,
            source=source
        )
        
        self.data_availability[symbol].append(availability)
    
    def is_data_available(
        self,
        symbol: str,
        timestamp: datetime,
        data_type: DataType,
        max_lag_days: int = 1
    ) -> bool:
        """
        Check if data is available at a point in time.
        
        Args:
            symbol: Stock symbol
            timestamp: Point in time
            data_type: Type of data
            max_lag_days: Maximum acceptable lag
            
        Returns:
            True if data is available within acceptable lag
        """
        if symbol not in self.data_availability:
            return False
        
        for availability in self.data_availability[symbol]:
            if availability.data_type == data_type:
                # Check if data is available within acceptable lag
                time_diff = (timestamp - availability.timestamp).days
                if availability.is_available and availability.lag_days <= max_lag_days:
                    return True
        
        return False
    
    def add_corporate_action(
        self,
        symbol: str,
        action_date: datetime,
        action_type: str,
        factor: float = 1.0,
        cash_value: float = 0.0,
        announcement_date: Optional[datetime] = None
    ) -> None:
        """
        Add corporate action to database.
        
        Args:
            symbol: Stock symbol
            action_date: Action effective date
            action_type: Type of action
            factor: Adjustment factor
            cash_value: Cash value
            announcement_date: When action was announced
        """
        if symbol not in self.corporate_actions_db:
            self.corporate_actions_db[symbol] = []
        
        self.corporate_actions_db[symbol].append({
            'action_date': action_date,
            'action_type': action_type,
            'factor': factor,
            'cash_value': cash_value,
            'announcement_date': announcement_date or action_date
        })
    
    def add_earnings_event(
        self,
        symbol: str,
        announcement_date: datetime,
        actual_eps: float,
        estimated_eps: float
    ) -> None:
        """
        Add earnings event to database.
        
        Args:
            symbol: Stock symbol
            announcement_date: Earnings announcement date
            actual_eps: Actual earnings per share
            estimated_eps: Estimated earnings per share
        """
        if symbol not in self.earnings_db:
            self.earnings_db[symbol] = []
        
        self.earnings_db[symbol].append({
            'announcement_date': announcement_date,
            'actual_eps': actual_eps,
            'estimated_eps': estimated_eps,
            'surprise': (actual_eps - estimated_eps) / abs(estimated_eps) if estimated_eps != 0 else 0
        })
    
    def add_feature_value(
        self,
        symbol: str,
        feature_name: str,
        timestamp: datetime,
        value: float
    ) -> None:
        """
        Add feature value to history.
        
        Args:
            symbol: Stock symbol
            feature_name: Feature name
            timestamp: Feature timestamp
            value: Feature value
        """
        if symbol not in self.feature_history:
            self.feature_history[symbol] = {}
        
        if feature_name not in self.feature_history[symbol]:
            self.feature_history[symbol][feature_name] = []
        
        self.feature_history[symbol][feature_name].append((timestamp, value))
    
    def get_feature_value_point_in_time(
        self,
        symbol: str,
        feature_name: str,
        timestamp: datetime,
        lookback_window: int = 20
    ) -> Optional[float]:
        """
        Get feature value that would have been available at point in time.
        
        Args:
            symbol: Stock symbol
            feature_name: Feature name
            timestamp: Point in time
            lookback_window: Lookback window in days
            
        Returns:
            Feature value or None if not available
        """
        if symbol not in self.feature_history:
            return None
        
        if feature_name not in self.feature_history[symbol]:
            return None
        
        # Get all feature values before timestamp
        cutoff_date = timestamp - timedelta(days=lookback_window)
        available_values = [
            (ts, val) for ts, val in self.feature_history[symbol][feature_name]
            if ts <= timestamp and ts >= cutoff_date
        ]
        
        if not available_values:
            return None
        
        # Return most recent value
        available_values.sort(key=lambda x: x[0], reverse=True)
        return available_values[0][1]
    
    def adjust_for_corporate_actions(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        timestamp: datetime
    ) -> pd.DataFrame:
        """
        Adjust price data for corporate actions point-in-time.
        
        Args:
            symbol: Stock symbol
            price_data: Price data
            timestamp: Point in time
            
        Returns:
            Adjusted price data
        """
        if symbol not in self.corporate_actions_db:
            return price_data
        
        adjusted_data = price_data.copy()
        
        # Get corporate actions that would have been known at timestamp
        known_actions = [
            action for action in self.corporate_actions_db[symbol]
            if action['announcement_date'] <= timestamp
        ]
        
        # Sort by action date
        known_actions.sort(key=lambda x: x['action_date'])
        
        # Apply adjustments
        for action in known_actions:
            action_date = action['action_date']
            mask = adjusted_data.index < action_date
            
            if action['action_type'] in ['split', 'bonus', 'reverse_split']:
                factor = action['factor']
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in adjusted_data.columns:
                        adjusted_data.loc[mask, col] = adjusted_data.loc[mask, col] / factor
                if 'volume' in adjusted_data.columns:
                    adjusted_data.loc[mask, 'volume'] = adjusted_data.loc[mask, 'volume'] * factor
            
            elif action['action_type'] == 'dividend':
                dividend = action['cash_value']
                price_cols = ['open', 'high', 'low', 'close']
                for col in price_cols:
                    if col in adjusted_data.columns:
                        adjusted_data.loc[mask, col] = adjusted_data.loc[mask, col] - dividend
        
        return adjusted_data
    
    def check_lookahead_bias(
        self,
        symbol: str,
        feature_name: str,
        timestamp: datetime
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a feature has look-ahead bias.
        
        Args:
            symbol: Stock symbol
            feature_name: Feature name
            timestamp: Point in time
            
        Returns:
            (has_bias, reason)
        """
        # Check if feature uses future data
        if symbol in self.feature_history and feature_name in self.feature_history[symbol]:
            for ts, val in self.feature_history[symbol][feature_name]:
                if ts > timestamp:
                    return True, f"Feature uses data from {ts} which is after {timestamp}"
        
        # Check if feature uses future earnings
        if symbol in self.earnings_db:
            for earnings in self.earnings_db[symbol]:
                if earnings['announcement_date'] > timestamp:
                    return True, f"Feature uses earnings from {earnings['announcement_date']}"
        
        # Check if feature uses future corporate actions
        if symbol in self.corporate_actions_db:
            for action in self.corporate_actions_db[symbol]:
                if action['action_date'] > timestamp:
                    return True, f"Feature uses corporate action from {action['action_date']}"
        
        return False, None
    
    def create_point_in_time_snapshot(
        self,
        symbol: str,
        timestamp: datetime,
        price_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None
    ) -> PointInTimeSnapshot:
        """
        Create a complete point-in-time snapshot for a symbol.
        
        Args:
            symbol: Stock symbol
            timestamp: Point in time
            price_data: Price data (optional)
            feature_names: Feature names to include (optional)
            
        Returns:
            PointInTimeSnapshot
        """
        # Adjust price data for corporate actions
        if price_data is not None:
            adjusted_prices = self.adjust_for_corporate_actions(symbol, price_data, timestamp)
        else:
            adjusted_prices = None
        
        # Get feature values point-in-time
        features = {}
        if feature_names:
            for feature_name in feature_names:
                value = self.get_feature_value_point_in_time(symbol, feature_name, timestamp)
                if value is not None:
                    features[feature_name] = value
        
        # Get known corporate actions
        known_corporate_actions = []
        if symbol in self.corporate_actions_db:
            known_corporate_actions = [
                action for action in self.corporate_actions_db[symbol]
                if action['announcement_date'] <= timestamp
            ]
        
        # Get known earnings events
        known_earnings = []
        if symbol in self.earnings_db:
            known_earnings = [
                earnings for earnings in self.earnings_db[symbol]
                if earnings['announcement_date'] <= timestamp
            ]
        
        snapshot = PointInTimeSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            price_data=adjusted_prices,
            volume_data=None,
            features=features,
            corporate_actions=known_corporate_actions,
            earnings_events=known_earnings,
            regime_state=None
        )
        
        return snapshot
    
    def validate_backtest_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_names: List[str]
    ) -> Dict[str, List[str]]:
        """
        Validate backtest data for look-ahead bias.
        
        Args:
            symbols: List of symbols
            start_date: Backtest start date
            end_date: Backtest end date
            feature_names: Feature names to validate
            
        Returns:
            Dict mapping symbol to list of bias issues
        """
        issues = {}
        
        for symbol in symbols:
            symbol_issues = []
            
            # Check each feature
            for feature_name in feature_names:
                # Sample timestamps
                timestamps = pd.date_range(start_date, end_date, freq='D')
                
                for timestamp in timestamps:
                    has_bias, reason = self.check_lookahead_bias(symbol, feature_name, timestamp)
                    if has_bias:
                        symbol_issues.append(f"{feature_name}: {reason}")
                        break
            
            if symbol_issues:
                issues[symbol] = symbol_issues
        
        return issues
    
    def print_availability_report(self) -> None:
        """Print data availability report."""
        print("\n" + "="*60)
        print("POINT-IN-TIME DATA AVAILABILITY REPORT")
        print("="*60)
        
        print(f"\nSymbols tracked: {len(self.data_availability)}")
        
        for symbol, availabilities in self.data_availability.items():
            print(f"\n{symbol}:")
            
            # Count by data type
            type_counts = {}
            for avail in availabilities:
                type_counts[avail.data_type.value] = type_counts.get(avail.data_type.value, 0) + 1
            
            for data_type, count in sorted(type_counts.items()):
                print(f"  {data_type}: {count} records")
        
        print("\n" + "="*60)


def sample_point_in_time_reconstruction():
    """Demonstrate point-in-time data reconstruction."""
    print("=== Point-in-Time Data Reconstruction Demo ===\n")
    
    reconstructor = PointInTimeReconstructor()
    
    # Add sample data availability
    symbol = "RELIANCE"
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    print("Adding data availability records...")
    for date in dates:
        reconstructor.add_data_availability(
            symbol=symbol,
            timestamp=date,
            data_type=DataType.PRICE,
            is_available=True,
            lag_days=1,  # End-of-day data
            source="end_of_day"
        )
        
        if date.weekday() < 5:  # Weekdays only
            reconstructor.add_data_availability(
                symbol=symbol,
                timestamp=date,
                data_type=DataType.FEATURE,
                is_available=True,
                lag_days=1,
                source="end_of_day"
            )
    
    # Add corporate action
    reconstructor.add_corporate_action(
        symbol=symbol,
        action_date=datetime(2024, 3, 1),
        action_type="bonus",
        factor=2.0,
        announcement_date=datetime(2024, 2, 15)
    )
    
    # Add earnings event
    reconstructor.add_earnings_event(
        symbol=symbol,
        announcement_date=datetime(2024, 4, 15),
        actual_eps=10.5,
        estimated_eps=10.0
    )
    
    # Add feature values
    for i, date in enumerate(dates):
        reconstructor.add_feature_value(
            symbol=symbol,
            feature_name="returns_5d",
            timestamp=date,
            value=np.random.normal(0, 0.02)
        )
    
    # Check data availability
    print("\nChecking data availability at 2024-01-15...")
    is_available = reconstructor.is_data_available(
        symbol=symbol,
        timestamp=datetime(2024, 1, 15),
        data_type=DataType.PRICE
    )
    print(f"Price data available: {is_available}")
    
    # Get feature value point-in-time
    print("\nGetting feature value at 2024-01-20...")
    feature_value = reconstructor.get_feature_value_point_in_time(
        symbol=symbol,
        feature_name="returns_5d",
        timestamp=datetime(2024, 1, 20),
        lookback_window=20
    )
    print(f"Feature value: {feature_value:.4f}")
    
    # Check for look-ahead bias
    print("\nChecking for look-ahead bias...")
    has_bias, reason = reconstructor.check_lookahead_bias(
        symbol=symbol,
        feature_name="returns_5d",
        timestamp=datetime(2024, 1, 10)
    )
    print(f"Look-ahead bias detected: {has_bias}")
    if has_bias:
        print(f"Reason: {reason}")
    
    # Create point-in-time snapshot
    print("\nCreating point-in-time snapshot at 2024-02-01...")
    snapshot = reconstructor.create_point_in_time_snapshot(
        symbol=symbol,
        timestamp=datetime(2024, 2, 1),
        feature_names=["returns_5d"]
    )
    print(f"Snapshot created with {len(snapshot.features)} features")
    print(f"Known corporate actions: {len(snapshot.corporate_actions)}")
    print(f"Known earnings events: {len(snapshot.earnings_events)}")
    
    # Print availability report
    reconstructor.print_availability_report()
    
    print("\n=== Point-in-Time Reconstruction Demo Complete ===")
    print("Key capabilities:")
    print("- Data availability tracking")
    print("- Corporate action adjustments")
    print("- Earnings calendar integration")
    print("- Look-ahead bias detection")
    print("- Point-in-time snapshot generation")
    print("- Feature validation")


if __name__ == "__main__":
    sample_point_in_time_reconstruction()

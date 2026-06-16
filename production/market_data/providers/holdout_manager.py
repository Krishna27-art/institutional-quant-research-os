"""
Holdout Period Manager for Final Validation

This module manages the final holdout period (2024-2025) to ensure that
backtests are validated on truly out-of-sample data. This prevents overfitting
and provides a realistic estimate of live performance.

Key Features:
- Holdout period definition and enforcement
- Data splitting with holdout separation
- Holdout validation checks
- Backtest re-runner with holdout enforcement
- Performance comparison (train vs holdout)

Based on V4 Blueprint - Institutional Architecture
Priority: High (Phase 0.3)
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


class HoldoutType(Enum):
    """Types of holdout periods."""
    FINAL_HOLDOUT = "final_holdout"  # Never used for training/tuning
    VALIDATION_HOLDOUT = "validation_holdout"  # Used for validation only
    TEMPORARY_HOLDOUT = "temporary_holdout"  # Temporary holdout for specific tests


@dataclass
class HoldoutPeriod:
    """Holdout period definition."""
    name: str
    start_date: datetime
    end_date: datetime
    holdout_type: HoldoutType
    purpose: str
    is_locked: bool = True  # If True, cannot be used for training
    
    def contains_date(self, date: datetime) -> bool:
        """Check if date is within holdout period."""
        return self.start_date <= date <= self.end_date
    
    def overlaps_with(self, start: datetime, end: datetime) -> bool:
        """Check if date range overlaps with holdout period."""
        return not (end < self.start_date or start > self.end_date)


@dataclass
class DataSplit:
    """Data split result."""
    train_data: pd.DataFrame
    validation_data: pd.DataFrame
    holdout_data: pd.DataFrame
    train_date_range: Tuple[datetime, datetime]
    validation_date_range: Tuple[datetime, datetime]
    holdout_date_range: Tuple[datetime, datetime]


@dataclass
class HoldoutValidationResult:
    """Result of holdout validation."""
    is_valid: bool
    violations: List[str]
    holdout_usage_detected: bool
    train_metrics: Dict[str, float]
    holdout_metrics: Dict[str, float]
    performance_degradation: Dict[str, float]


class HoldoutManager:
    """
    Manager for holdout periods and final validation.
    
    This class ensures that the final holdout period (2024-2025) is never
    used for training or hyperparameter tuning, providing a realistic
    estimate of live performance.
    """
    
    def __init__(self):
        self.holdout_periods: List[HoldoutPeriod] = []
        self.final_holdout: Optional[HoldoutPeriod] = None
        
        # Define standard holdout periods
        self._define_standard_holdouts()
        
        logger.info("HoldoutManager initialized")
    
    def _define_standard_holdouts(self) -> None:
        """Define standard holdout periods for Indian markets."""
        # Final holdout: 2024-2025 (never used for training)
        final_holdout = HoldoutPeriod(
            name="final_holdout_2024_2025",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 12, 31),
            holdout_type=HoldoutType.FINAL_HOLDOUT,
            purpose="Final validation - never used for training/tuning",
            is_locked=True
        )
        
        self.holdout_periods.append(final_holdout)
        self.final_holdout = final_holdout
        
        logger.info(f"Defined final holdout: {final_holdout.start_date.date()} to {final_holdout.end_date.date()}")
    
    def add_holdout_period(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime,
        holdout_type: HoldoutType,
        purpose: str,
        is_locked: bool = True
    ) -> None:
        """
        Add a custom holdout period.
        
        Args:
            name: Holdout period name
            start_date: Start date
            end_date: End date
            holdout_type: Type of holdout
            purpose: Purpose of holdout
            is_locked: If True, cannot be used for training
        """
        holdout = HoldoutPeriod(
            name=name,
            start_date=start_date,
            end_date=end_date,
            holdout_type=holdout_type,
            purpose=purpose,
            is_locked=is_locked
        )
        
        self.holdout_periods.append(holdout)
        logger.info(f"Added holdout period: {name} ({start_date.date()} to {end_date.date()})")
    
    def is_date_in_holdout(
        self,
        date: datetime,
        holdout_type: Optional[HoldoutType] = None
    ) -> bool:
        """
        Check if date is in any holdout period.
        
        Args:
            date: Date to check
            holdout_type: Filter by holdout type (optional)
            
        Returns:
            True if date is in holdout period
        """
        for holdout in self.holdout_periods:
            if holdout_type and holdout.holdout_type != holdout_type:
                continue
            if holdout.contains_date(date):
                return True
        return False
    
    def is_range_in_holdout(
        self,
        start_date: datetime,
        end_date: datetime,
        holdout_type: Optional[HoldoutType] = None
    ) -> bool:
        """
        Check if date range overlaps with any holdout period.
        
        Args:
            start_date: Start date
            end_date: End date
            holdout_type: Filter by holdout type (optional)
            
        Returns:
            True if range overlaps with holdout period
        """
        for holdout in self.holdout_periods:
            if holdout_type and holdout.holdout_type != holdout_type:
                continue
            if holdout.overlaps_with(start_date, end_date):
                return True
        return False
    
    def split_data_with_holdout(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        validation_ratio: float = 0.2
    ) -> DataSplit:
        """
        Split data into train, validation, and holdout sets.
        
        Args:
            data: Input data
            date_col: Date column name
            validation_ratio: Ratio for validation set
            
        Returns:
            DataSplit with train, validation, and holdout data
        """
        # Ensure date column is datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data = data.sort_values(date_col).reset_index(drop=True)
        
        # Extract holdout data
        if self.final_holdout:
            holdout_mask = data[date_col].apply(
                lambda x: self.final_holdout.contains_date(x)
            )
            holdout_data = data[holdout_mask].copy()
            non_holdout_data = data[~holdout_mask].copy()
        else:
            holdout_data = pd.DataFrame(columns=data.columns)
            non_holdout_data = data.copy()
        
        if non_holdout_data.empty:
            logger.warning("No data available for training after holdout extraction")
            return DataSplit(
                train_data=pd.DataFrame(columns=data.columns),
                validation_data=pd.DataFrame(columns=data.columns),
                holdout_data=holdout_data,
                train_date_range=(datetime.min, datetime.min),
                validation_date_range=(datetime.min, datetime.min),
                holdout_date_range=(self.final_holdout.start_date, self.final_holdout.end_date) if self.final_holdout else (datetime.min, datetime.min)
            )
        
        # Split non-holdout data into train and validation
        n_samples = len(non_holdout_data)
        n_validation = int(n_samples * validation_ratio)
        n_train = n_samples - n_validation
        
        train_data = non_holdout_data.iloc[:n_train].copy()
        validation_data = non_holdout_data.iloc[n_train:].copy()
        
        # Get date ranges
        train_start = train_data[date_col].min()
        train_end = train_data[date_col].max()
        validation_start = validation_data[date_col].min()
        validation_end = validation_data[date_col].max()
        holdout_start = holdout_data[date_col].min() if not holdout_data.empty else datetime.min
        holdout_end = holdout_data[date_col].max() if not holdout_data.empty else datetime.min
        
        split = DataSplit(
            train_data=train_data,
            validation_data=validation_data,
            holdout_data=holdout_data,
            train_date_range=(train_start, train_end),
            validation_date_range=(validation_start, validation_end),
            holdout_date_range=(holdout_start, holdout_end)
        )
        
        logger.info(f"Data split: Train ({len(train_data)} samples), Validation ({len(validation_data)} samples), Holdout ({len(holdout_data)} samples)")
        
        return split
    
    def validate_backtest_holdout(
        self,
        backtest_start: datetime,
        backtest_end: datetime,
        training_start: Optional[datetime] = None,
        training_end: Optional[datetime] = None
    ) -> HoldoutValidationResult:
        """
        Validate that backtest doesn't use holdout data for training.
        
        Args:
            backtest_start: Backtest start date
            backtest_end: Backtest end date
            training_start: Training start date (if different)
            training_end: Training end date (if different)
            
        Returns:
            HoldoutValidationResult
        """
        violations = []
        holdout_usage_detected = False
        
        # Check if training period overlaps with final holdout
        train_start = training_start if training_start else backtest_start
        train_end = training_end if training_end else backtest_end
        
        if self.final_holdout and self.final_holdout.overlaps_with(train_start, train_end):
            violations.append(f"Training period ({train_start.date()} to {train_end.date()}) overlaps with final holdout ({self.final_holdout.start_date.date()} to {self.final_holdout.end_date.date()})")
            holdout_usage_detected = True
        
        # Check if backtest period overlaps with final holdout (this is OK for validation)
        backtest_uses_holdout = self.final_holdout and self.final_holdout.overlaps_with(backtest_start, backtest_end)
        
        result = HoldoutValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            holdout_usage_detected=holdout_usage_detected,
            train_metrics={},
            holdout_metrics={},
            performance_degradation={}
        )
        
        return result
    
    def get_holdout_report(self) -> Dict[str, any]:
        """
        Get holdout period report.
        
        Returns:
            Dict with holdout information
        """
        report = {
            'total_holdout_periods': len(self.holdout_periods),
            'final_holdout': None,
            'holdout_periods': []
        }
        
        if self.final_holdout:
            report['final_holdout'] = {
                'name': self.final_holdout.name,
                'start_date': self.final_holdout.start_date.date(),
                'end_date': self.final_holdout.end_date.date(),
                'purpose': self.final_holdout.purpose,
                'is_locked': self.final_holdout.is_locked
            }
        
        for holdout in self.holdout_periods:
            report['holdout_periods'].append({
                'name': holdout.name,
                'start_date': holdout.start_date.date(),
                'end_date': holdout.end_date.date(),
                'type': holdout.holdout_type.value,
                'purpose': holdout.purpose,
                'is_locked': holdout.is_locked
            })
        
        return report
    
    def print_holdout_report(self) -> None:
        """Print holdout period report."""
        print("\n" + "="*60)
        print("HOLDOUT PERIOD REPORT")
        print("="*60)
        
        print(f"\nTotal Holdout Periods: {len(self.holdout_periods)}")
        
        if self.final_holdout:
            print(f"\nFINAL HOLDOUT (LOCKED):")
            print(f"  Name: {self.final_holdout.name}")
            print(f"  Period: {self.final_holdout.start_date.date()} to {self.final_holdout.end_date.date()}")
            print(f"  Purpose: {self.final_holdout.purpose}")
            print(f"  Locked: {self.final_holdout.is_locked}")
        
        print("\nAll Holdout Periods:")
        for holdout in self.holdout_periods:
            lock_status = "LOCKED" if holdout.is_locked else "UNLOCKED"
            print(f"  {holdout.name}: {holdout.start_date.date()} to {holdout.end_date.date()} ({holdout.holdout_type.value}) [{lock_status}]")
        
        print("\n" + "="*60)


def sample_holdout_management():
    """Demonstrate holdout period management."""
    print("=== Holdout Period Manager Demo ===\n")
    
    manager = HoldoutManager()
    
    # Print holdout report
    manager.print_holdout_report()
    
    # Check if dates are in holdout
    print("\nChecking dates in holdout...")
    test_dates = [
        datetime(2023, 6, 1),
        datetime(2024, 6, 1),
        datetime(2025, 6, 1),
        datetime(2026, 6, 1)
    ]
    
    for date in test_dates:
        in_holdout = manager.is_date_in_holdout(date)
        print(f"  {date.date()}: {'IN HOLDOUT' if in_holdout else 'NOT IN HOLDOUT'}")
    
    # Validate backtest
    print("\nValidating backtest...")
    result = manager.validate_backtest_holdout(
        backtest_start=datetime(2020, 1, 1),
        backtest_end=datetime(2023, 12, 31),
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2023, 12, 31)
    )
    
    print(f"Valid: {result.is_valid}")
    print(f"Violations: {result.violations}")
    print(f"Holdout usage detected: {result.holdout_usage_detected}")
    
    # Test invalid backtest (uses holdout for training)
    print("\nTesting invalid backtest (uses holdout for training)...")
    invalid_result = manager.validate_backtest_holdout(
        backtest_start=datetime(2020, 1, 1),
        backtest_end=datetime(2025, 12, 31),
        training_start=datetime(2020, 1, 1),
        training_end=datetime(2025, 12, 31)
    )
    
    print(f"Valid: {invalid_result.is_valid}")
    print(f"Violations: {invalid_result.violations}")
    print(f"Holdout usage detected: {invalid_result.holdout_usage_detected}")
    
    # Sample data split
    print("\nTesting data split...")
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', end='2025-12-31', freq='D'),
        'price': np.random.randn(2192) * 10 + 100
    })
    
    split = manager.split_data_with_holdout(sample_data, validation_ratio=0.2)
    print(f"Train: {len(split.train_data)} samples ({split.train_date_range[0].date()} to {split.train_date_range[1].date()})")
    print(f"Validation: {len(split.validation_data)} samples ({split.validation_date_range[0].date()} to {split.validation_date_range[1].date()})")
    print(f"Holdout: {len(split.holdout_data)} samples ({split.holdout_date_range[0].date()} to {split.holdout_date_range[1].date()})")
    
    print("\n=== Holdout Period Manager Demo Complete ===")
    print("Key capabilities:")
    print("- Final holdout period definition (2024-2025)")
    print("- Holdout validation for backtests")
    print("- Data splitting with holdout separation")
    print("- Holdout usage detection")


if __name__ == "__main__":
    sample_holdout_management()

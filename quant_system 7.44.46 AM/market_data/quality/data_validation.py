"""
Data Validation Pipeline
Based on Institutional Audit Recommendations

Key findings from audit:
- No checks for stuck prices, outliers, missing ticks
- Garbage data → garbage signals
- Need: Data quality DAG (great_expectations)

Architecture V2 Upgrade - 90-Day Plan Item #2
Priority: P0 (Critical)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    severity: str  # "critical", "warning", "info"
    message: str
    details: Dict
    timestamp: datetime


@dataclass
class DataQualityReport:
    """Overall data quality report"""
    symbol: str
    date: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_failures: int
    validation_results: List[ValidationResult]
    overall_status: str  # "pass", "warning", "fail"


class DataValidator:
    """
    Data Validator for market data quality checks.
    
    Validation Rules:
    - Price must be > 0
    - Volume must be >= 0
    - No stuck prices (same price for > 100 consecutive ticks)
    - No outliers (price change > 10σ)
    - No missing ticks (> 5% of expected ticks)
    - Timestamps must be monotonically increasing
    - Bid must be <= Ask
    - High/Low must contain Close
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_price_positive(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that all prices are positive.
        
        Args:
            data: DataFrame with price columns
            
        Returns:
            ValidationResult
        """
        price_cols = ['open', 'high', 'low', 'close', 'bid', 'ask']
        price_cols = [col for col in price_cols if col in data.columns]
        
        violations = []
        for col in price_cols:
            if (data[col] <= 0).any():
                count = (data[col] <= 0).sum()
                violations.append(f"{col}: {count} non-positive values")
        
        passed = len(violations) == 0
        severity = "critical" if not passed else "info"
        message = "All prices positive" if passed else f"Non-positive prices: {', '.join(violations)}"
        
        return ValidationResult(
            check_name="price_positive",
            passed=passed,
            severity=severity,
            message=message,
            details={"violations": violations},
            timestamp=datetime.now()
        )
    
    def validate_volume_non_negative(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that volume is non-negative.
        
        Args:
            data: DataFrame with volume column
            
        Returns:
            ValidationResult
        """
        if 'volume' not in data.columns:
            return ValidationResult(
                check_name="volume_non_negative",
                passed=True,
                severity="info",
                message="Volume column not found",
                details={},
                timestamp=datetime.now()
            )
        
        violations = (data['volume'] < 0).sum()
        passed = violations == 0
        severity = "critical" if not passed else "info"
        message = "All volumes non-negative" if passed else f"{violations} negative volume values"
        
        return ValidationResult(
            check_name="volume_non_negative",
            passed=passed,
            severity=severity,
            message=message,
            details={"negative_volume_count": violations},
            timestamp=datetime.now()
        )
    
    def validate_stuck_prices(self, data: pd.DataFrame, max_consecutive: int = 100) -> ValidationResult:
        """
        Validate no stuck prices (same price for too many consecutive ticks).
        
        Args:
            data: DataFrame with price column
            max_consecutive: Maximum allowed consecutive same prices
            
        Returns:
            ValidationResult
        """
        if 'close' not in data.columns:
            return ValidationResult(
                check_name="stuck_prices",
                passed=True,
                severity="info",
                message="Close column not found",
                details={},
                timestamp=datetime.now()
            )
        
        # Find consecutive same prices
        prices = data['close'].values
        consecutive = 1
        max_streak = 1
        stuck_periods = []
        
        for i in range(1, len(prices)):
            if prices[i] == prices[i-1]:
                consecutive += 1
                max_streak = max(max_streak, consecutive)
            else:
                if consecutive > max_consecutive:
                    stuck_periods.append((i - consecutive, i))
                consecutive = 1
        
        if consecutive > max_consecutive:
            stuck_periods.append((len(prices) - consecutive, len(prices)))
        
        passed = len(stuck_periods) == 0
        severity = "warning" if not passed else "info"
        message = "No stuck prices" if passed else f"Found {len(stuck_periods)} stuck price periods (max streak: {max_streak})"
        
        return ValidationResult(
            check_name="stuck_prices",
            passed=passed,
            severity=severity,
            message=message,
            details={"stuck_periods": stuck_periods, "max_streak": max_streak},
            timestamp=datetime.now()
        )
    
    def validate_price_outliers(self, data: pd.DataFrame, sigma_threshold: float = 10.0) -> ValidationResult:
        """
        Validate no extreme price outliers (price change > Nσ).
        
        Args:
            data: DataFrame with close column
            sigma_threshold: Standard deviation threshold
            
        Returns:
            ValidationResult
        """
        if 'close' not in data.columns:
            return ValidationResult(
                check_name="price_outliers",
                passed=True,
                severity="info",
                message="Close column not found",
                details={},
                timestamp=datetime.now()
            )
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 2:
            return ValidationResult(
                check_name="price_outliers",
                passed=True,
                severity="info",
                message="Insufficient data for outlier detection",
                details={},
                timestamp=datetime.now()
            )
        
        # Calculate z-scores
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return ValidationResult(
                check_name="price_outliers",
                passed=True,
                severity="info",
                message="Zero volatility - cannot detect outliers",
                details={},
                timestamp=datetime.now()
            )
        
        z_scores = np.abs((returns - mean_return) / std_return)
        outliers = (z_scores > sigma_threshold).sum()
        
        passed = outliers == 0
        severity = "warning" if not passed else "info"
        message = "No price outliers" if passed else f"Found {outliers} price outliers (> {sigma_threshold}σ)"
        
        return ValidationResult(
            check_name="price_outliers",
            passed=passed,
            severity=severity,
            message=message,
            details={"outlier_count": outliers, "max_z_score": z_scores.max()},
            timestamp=datetime.now()
        )
    
    def validate_missing_ticks(self, data: pd.DataFrame, expected_ticks_per_day: int = 375) -> ValidationResult:
        """
        Validate no excessive missing ticks.
        
        Args:
            data: DataFrame with datetime index
            expected_ticks_per_day: Expected number of ticks per day
            
        Returns:
            ValidationResult
        """
        if len(data) == 0:
            return ValidationResult(
                check_name="missing_ticks",
                passed=False,
                severity="critical",
                message="No data provided",
                details={},
                timestamp=datetime.now()
            )
        
        # Group by date
        data = data.copy()
        data['date'] = pd.to_datetime(data.index).date
        ticks_per_day = data.groupby('date').size()
        
        # Calculate missing percentage
        total_expected = len(ticks_per_day) * expected_ticks_per_day
        total_actual = len(data)
        missing_pct = (total_expected - total_actual) / total_expected * 100 if total_expected > 0 else 0
        
        passed = missing_pct < 5.0
        severity = "critical" if not passed else "info"
        message = f"Missing ticks: {missing_pct:.2f}%"
        
        return ValidationResult(
            check_name="missing_ticks",
            passed=passed,
            severity=severity,
            message=message,
            details={
                "missing_percentage": missing_pct,
                "total_expected": total_expected,
                "total_actual": total_actual,
                "ticks_per_day": ticks_per_day.to_dict()
            },
            timestamp=datetime.now()
        )
    
    def validate_timestamp_monotonic(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate timestamps are monotonically increasing.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            ValidationResult
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            return ValidationResult(
                check_name="timestamp_monotonic",
                passed=False,
                severity="critical",
                message="Index is not DatetimeIndex",
                details={},
                timestamp=datetime.now()
            )
        
        is_monotonic = data.index.is_monotonic_increasing
        passed = is_monotonic
        severity = "critical" if not passed else "info"
        message = "Timestamps are monotonic" if passed else "Timestamps are not monotonically increasing"
        
        return ValidationResult(
            check_name="timestamp_monotonic",
            passed=passed,
            severity=severity,
            message=message,
            details={"is_monotonic": is_monotonic},
            timestamp=datetime.now()
        )
    
    def validate_bid_ask_spread(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate bid <= ask.
        
        Args:
            data: DataFrame with bid and ask columns
            
        Returns:
            ValidationResult
        """
        if 'bid' not in data.columns or 'ask' not in data.columns:
            return ValidationResult(
                check_name="bid_ask_spread",
                passed=True,
                severity="info",
                message="Bid/Ask columns not found",
                details={},
                timestamp=datetime.now()
            )
        
        violations = (data['bid'] > data['ask']).sum()
        passed = violations == 0
        severity = "critical" if not passed else "info"
        message = "Bid <= Ask for all rows" if passed else f"{violations} rows with bid > ask"
        
        return ValidationResult(
            check_name="bid_ask_spread",
            passed=passed,
            severity=severity,
            message=message,
            details={"violation_count": violations},
            timestamp=datetime.now()
        )
    
    def validate_ohlc_consistency(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate High >= Close >= Low.
        
        Args:
            data: DataFrame with OHLC columns
            
        Returns:
            ValidationResult
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return ValidationResult(
                check_name="ohlc_consistency",
                passed=True,
                severity="info",
                message="OHLC columns not found",
                details={},
                timestamp=datetime.now()
            )
        
        # Check high >= close >= low
        high_low_violations = (data['high'] < data['low']).sum()
        high_close_violations = (data['high'] < data['close']).sum()
        close_low_violations = (data['close'] < data['low']).sum()
        
        total_violations = high_low_violations + high_close_violations + close_low_violations
        passed = total_violations == 0
        severity = "critical" if not passed else "info"
        
        details = {
            "high_low_violations": high_low_violations,
            "high_close_violations": high_close_violations,
            "close_low_violations": close_low_violations
        }
        message = "OHLC data consistent" if passed else f"OHLC violations: {details}"
        
        return ValidationResult(
            check_name="ohlc_consistency",
            passed=passed,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def validate_all(self, data: pd.DataFrame, symbol: str, date: str) -> DataQualityReport:
        """
        Run all validation checks.
        
        Args:
            data: DataFrame with market data
            symbol: Stock symbol
            date: Date string
            
        Returns:
            DataQualityReport with all validation results
        """
        self.validation_results = []
        
        # Run all checks
        self.validation_results.append(self.validate_price_positive(data))
        self.validation_results.append(self.validate_volume_non_negative(data))
        self.validation_results.append(self.validate_stuck_prices(data))
        self.validation_results.append(self.validate_price_outliers(data))
        self.validation_results.append(self.validate_missing_ticks(data))
        self.validation_results.append(self.validate_timestamp_monotonic(data))
        self.validation_results.append(self.validate_bid_ask_spread(data))
        self.validation_results.append(self.validate_ohlc_consistency(data))
        
        # Calculate summary
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        failed_checks = total_checks - passed_checks
        critical_failures = sum(1 for r in self.validation_results if not r.passed and r.severity == "critical")
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = "fail"
        elif failed_checks > 0:
            overall_status = "warning"
        else:
            overall_status = "pass"
        
        return DataQualityReport(
            symbol=symbol,
            date=date,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_failures=critical_failures,
            validation_results=self.validation_results,
            overall_status=overall_status
        )
    
    def print_report(self, report: DataQualityReport) -> None:
        """Print validation report."""
        print("\n" + "="*60)
        print(f"DATA QUALITY REPORT: {report.symbol} on {report.date}")
        print("="*60)
        print(f"Overall Status: {report.overall_status.upper()}")
        print(f"Total Checks: {report.total_checks}")
        print(f"Passed: {report.passed_checks}")
        print(f"Failed: {report.failed_checks}")
        print(f"Critical Failures: {report.critical_failures}")
        
        print("\nValidation Results:")
        for result in report.validation_results:
            status_icon = "✅" if result.passed else "❌"
            severity_icon = "🔴" if result.severity == "critical" else "⚠️" if result.severity == "warning" else "ℹ️"
            print(f"  {status_icon} {severity_icon} {result.check_name:<25}: {result.message}")
        
        print("="*60)
    
    def to_json(self, report: DataQualityReport) -> str:
        """Convert report to JSON."""
        report_dict = {
            "symbol": report.symbol,
            "date": report.date,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "critical_failures": report.critical_failures,
            "overall_status": report.overall_status,
            "validation_results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in report.validation_results
            ]
        }
        return json.dumps(report_dict, indent=2)


def run_sample_validation():
    """Run sample data validation."""
    # Create sample market data
    dates = pd.date_range("2024-01-01 09:15:00", periods=375, freq="1min")
    np.random.seed(42)
    
    prices = 20000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, len(dates)))
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(dates)),
        'bid': prices - 1,
        'ask': prices + 1
    }, index=dates)
    
    # Add some violations for testing
    data.loc[10, 'close'] = -100  # Negative price
    data.loc[20, 'volume'] = -50  # Negative volume
    data.loc[30:40, 'close'] = data.loc[30, 'close']  # Stuck prices
    
    # Run validation
    validator = DataValidator()
    report = validator.validate_all(data, "NIFTY", "2024-01-01")
    
    # Print report
    validator.print_report(report)
    
    # Export JSON
    print("\nJSON Report:")
    print(validator.to_json(report))
    
    return report


if __name__ == "__main__":
    run_sample_validation()

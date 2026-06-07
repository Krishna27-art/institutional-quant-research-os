"""
Data Quality Engine for Market Data Validation

This module provides comprehensive data quality monitoring and validation
to prevent trading on stale, incorrect, or incomplete market data.

Key Features:
- Staleness detection with configurable thresholds
- Data completeness validation
- Price consistency checks
- Automatic alerts for data quality issues
- Blocking of predictions on stale data
- Data source health monitoring
- Statistical validation using theoretical foundation
- Market calendar integration for trading hours

Based on Audit Report Priority 0: Critical - Week 1-2
Enhanced with institutional-grade specifications from blueprint
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import pytz

# Import market calendar
from core.market_hours import NSE_HOLIDAYS, MARKET_OPEN, MARKET_CLOSE

# Import theoretical foundation modules
try:
    from foundation.math_toolkit import ProbabilityDistributions, StochasticProcesses
    FOUNDATION_AVAILABLE = True
except ImportError:
    FOUNDATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Foundation modules not available - using basic validation")

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


class DataQualityStatus(Enum):
    """Data quality status levels."""
    GOOD = "good"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    CORRUPT = "corrupt"
    MISSING = "missing"


@dataclass
class DataQualityCheck:
    """Result of a data quality check."""
    symbol: str
    data_source: str
    status: DataQualityStatus
    last_update: datetime
    staleness_seconds: float
    is_acceptable: bool
    issues: List[str]
    recommendations: List[str]


@dataclass
class DataQualityConfig:
    """Configuration for data quality engine."""
    
    # Staleness thresholds (in seconds)
    index_staleness_threshold: int = 300  # 5 minutes for indices
    stock_staleness_threshold: int = 600  # 10 minutes for stocks
    options_staleness_threshold: int = 1800  # 30 minutes for options
    
    # Expected update intervals (from blueprint)
    expected_update_intervals: Dict[str, timedelta] = None
    
    # Completeness thresholds
    min_required_bars: int = 100  # Minimum bars for analysis
    max_missing_pct: float = 0.05  # Max 5% missing data allowed
    
    # Price consistency thresholds
    max_price_change_pct: float = 0.50  # Max 50% daily change
    min_volume: int = 0  # Volume must be non-negative
    max_tick_move_pct: float = 0.10  # Max 10% move in single tick
    
    # Alert thresholds
    alert_on_stale: bool = True
    alert_on_incomplete: bool = True
    alert_on_corrupt: bool = True
    
    # Blocking behavior
    block_predictions_on_stale: bool = True
    block_predictions_on_incomplete: bool = True
    block_predictions_on_corrupt: bool = True
    
    def __post_init__(self):
        if self.expected_update_intervals is None:
            self.expected_update_intervals = {
                'tick': timedelta(seconds=1),
                '1min': timedelta(minutes=1),
                '5min': timedelta(minutes=5),
                '15min': timedelta(minutes=15),
                '1hour': timedelta(hours=1),
                '1day': timedelta(days=1),
            }


class DataQualityEngine:
    """
    Data Quality Engine for market data validation.
    
    This engine monitors data quality across all data sources and
    prevents trading on poor quality data.
    Enhanced with theoretical foundation statistical validation
    and market calendar integration.
    """
    
    def __init__(self, config: DataQualityConfig = None):
        """
        Initialize data quality engine.
        
        Args:
            config: Configuration for quality checks
        """
        self.config = config or DataQualityConfig()
        
        # Track last update times for each symbol
        self.last_updates: Dict[str, datetime] = {}
        
        # Track quality check history
        self.check_history: List[DataQualityCheck] = []
        
        # Track blocked symbols
        self.blocked_symbols: Dict[str, Tuple[DataQualityStatus, str]] = {}
        
        # Track previous prices for tick validation
        self.prev_prices: Dict[str, float] = {}
        
        # Initialize theoretical foundation modules if available
        if FOUNDATION_AVAILABLE:
            self.prob_dist = ProbabilityDistributions()
            self.stoch_proc = StochasticProcesses()
            logger.info("DataQualityEngine initialized with theoretical foundation support")
        else:
            self.prob_dist = None
            self.stoch_proc = None
            logger.info("DataQualityEngine initialized (basic mode)")
    
    def is_trading_hour(self, dt: datetime) -> bool:
        """
        Check if given datetime is during trading hours.
        
        Args:
            dt: Datetime to check
            
        Returns:
            True if during trading hours, False otherwise
        """
        # Convert to IST if needed
        if dt.tzinfo is None:
            dt = IST.localize(dt)
        elif dt.tzinfo != IST:
            dt = dt.astimezone(IST)
        
        # Check if weekend
        if dt.weekday() >= 5:
            return False
        
        # Check if holiday
        if dt.date() in NSE_HOLIDAYS:
            return False
        
        # Check if within trading hours
        current_time = dt.time()
        return MARKET_OPEN <= current_time <= MARKET_CLOSE
    
    def is_stale(self, symbol: str, data_type: str = '1min') -> bool:
        """
        Check if data for symbol is stale.
        
        Args:
            symbol: Stock/index symbol
            data_type: Type of data ('tick', '1min', '5min', '15min', '1hour', '1day')
            
        Returns:
            True if data is stale, False otherwise
        """
        last_update = self.get_last_update(symbol, data_type)
        if last_update is None:
            return True
        
        now = datetime.now(IST)
        
        # If market is closed, don't mark as stale
        if not self.is_trading_hour(now):
            return False
        
        age = now - last_update
        threshold = self.config.expected_update_intervals.get(data_type, timedelta(minutes=1)) * 3
        return age > threshold
    
    def validate_tick(self, tick: Dict) -> bool:
        """
        Validate a single tick of data.
        
        Args:
            tick: Dictionary with 'symbol', 'price', 'volume', 'timestamp'
            
        Returns:
            True if tick is valid, False otherwise
        """
        # Basic sanity
        if tick.get('price', 0) <= 0 or tick.get('volume', 0) < 0:
            self.alert("invalid_tick_values", tick)
            return False
        
        # Check for extreme moves (e.g., > 10% in a single tick)
        prev = self.prev_prices.get(tick['symbol'])
        if prev and abs(tick['price'] / prev - 1) > self.config.max_tick_move_pct:
            self.alert("suspicious_tick", tick)
            return False
        
        # Update previous price
        self.prev_prices[tick['symbol']] = tick['price']
        return True
    
    def get_last_update(self, symbol: str, data_type: str = '1min') -> Optional[datetime]:
        """
        Get last update time for symbol and data type.
        
        Args:
            symbol: Stock/index symbol
            data_type: Type of data
            
        Returns:
            Last update datetime or None if not available
        """
        key = f"{symbol}_{data_type}"
        return self.last_updates.get(key)
    
    def alert(self, alert_type: str, data: Dict) -> None:
        """
        Send an alert for data quality issue.
        
        Args:
            alert_type: Type of alert
            data: Data associated with the alert
        """
        logger.warning(f"Data quality alert [{alert_type}]: {data}")
        # In production, this would send to monitoring system (PagerDuty, etc.)
    
    def check_data_quality(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_source: str = "unknown"
    ) -> DataQualityCheck:
        """
        Perform comprehensive data quality check.
        
        Args:
            symbol: Stock/index symbol
            data: DataFrame with OHLCV data
            data_source: Source of the data (e.g., "NSE", "Yahoo Finance")
            
        Returns:
            DataQualityCheck with results
        """
        issues = []
        recommendations = []
        status = DataQualityStatus.GOOD
        
        # Check if data exists
        if data.empty:
            status = DataQualityStatus.MISSING
            issues.append("No data available")
            recommendations.append("Check data source connection")
            return self._create_check_result(
                symbol, data_source, status, None, float('inf'),
                False, issues, recommendations
            )
        
        # Check staleness
        last_update = self._get_last_update_time(data)
        staleness = (datetime.now() - last_update).total_seconds()
        
        # Determine appropriate threshold based on symbol type
        threshold = self._get_staleness_threshold(symbol)
        
        if staleness > threshold:
            status = DataQualityStatus.STALE
            issues.append(f"Data is stale: {staleness:.0f}s old (threshold: {threshold}s)")
            recommendations.append("Refresh data from source")
            
            if self.config.block_predictions_on_stale:
                self._block_symbol(symbol, status, "Stale data")
        
        # Check completeness
        if len(data) < self.config.min_required_bars:
            status = DataQualityStatus.INCOMPLETE
            issues.append(f"Insufficient data: {len(data)} bars (minimum: {self.config.min_required_bars})")
            recommendations.append("Wait for more data to accumulate")
            
            if self.config.block_predictions_on_incomplete:
                self._block_symbol(symbol, status, "Insufficient data")
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > self.config.max_missing_pct:
            status = DataQualityStatus.INCOMPLETE
            issues.append(f"Too many missing values: {missing_pct:.1%} (max: {self.config.max_missing_pct:.1%})")
            recommendations.append("Check data source for gaps")
        
        # Check price consistency
        price_issues = self._check_price_consistency(data)
        if price_issues:
            status = DataQualityStatus.CORRUPT
            issues.extend(price_issues)
            recommendations.append("Review data for anomalies")
            
            if self.config.block_predictions_on_corrupt:
                self._block_symbol(symbol, status, "Corrupt data")
        
        # Statistical validation using theoretical foundation
        if FOUNDATION_AVAILABLE and self.prob_dist is not None:
            stat_issues = self._statistical_validation(data, symbol)
            if stat_issues:
                if status == DataQualityStatus.GOOD:
                    status = DataQualityStatus.CORRUPT
                issues.extend(stat_issues)
                recommendations.append("Statistical anomalies detected")
        
        # Update last update time
        self.last_updates[symbol] = last_update
        
        # Determine if data is acceptable
        is_acceptable = status == DataQualityStatus.GOOD
        
        # Create check result
        check_result = self._create_check_result(
            symbol, data_source, status, last_update, staleness,
            is_acceptable, issues, recommendations
        )
        
        # Add to history
        self.check_history.append(check_result)
        
        # Keep only last 1000 checks
        if len(self.check_history) > 1000:
            self.check_history = self.check_history[-1000:]
        
        # Log results
        if not is_acceptable:
            logger.warning(f"Data quality check failed for {symbol}: {status.value}")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info(f"Data quality check passed for {symbol}")
        
        return check_result
    
    def _get_last_update_time(self, data: pd.DataFrame) -> datetime:
        """Get the last update time from data."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index[-1].to_pydatetime()
        elif 'timestamp' in data.columns:
            return pd.to_datetime(data['timestamp'].iloc[-1]).to_pydatetime()
        else:
            return datetime.now()
    
    def _get_staleness_threshold(self, symbol: str) -> int:
        """Get appropriate staleness threshold for symbol type."""
        symbol_upper = symbol.upper()
        
        # Indices have stricter thresholds
        if any(idx in symbol_upper for idx in ['NIFTY', 'BANKNIFTY', 'SENSEX', 'FINNIFTY']):
            return self.config.index_staleness_threshold
        
        # Options have looser thresholds
        if any(opt in symbol_upper for opt in ['CE', 'PE', 'OPT']):
            return self.config.options_staleness_threshold
        
        # Default to stock threshold
        return self.config.stock_staleness_threshold
    
    def _check_price_consistency(self, data: pd.DataFrame) -> List[str]:
        """Check price consistency and identify anomalies."""
        issues = []
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            issues.append("Missing required OHLCV columns")
            return issues
        
        # Check for non-positive prices
        for col in ['open', 'high', 'low', 'close']:
            if (data[col] <= 0).any():
                issues.append(f"Non-positive values in {col}")
        
        # Check OHLC consistency
        if not (data['high'] >= data['low']).all():
            issues.append("High < Low violation")
        if not (data['high'] >= data['open']).all():
            issues.append("High < Open violation")
        if not (data['high'] >= data['close']).all():
            issues.append("High < Close violation")
        if not (data['low'] <= data['open']).all():
            issues.append("Low > Open violation")
        if not (data['low'] <= data['close']).all():
            issues.append("Low > Close violation")
        
        # Check for extreme price changes
        data['daily_change'] = (data['close'] - data['open']) / data['open']
        extreme_changes = data[abs(data['daily_change']) > self.config.max_price_change_pct]
        if len(extreme_changes) > 0:
            issues.append(f"Extreme price changes detected: {len(extreme_changes)} bars")
        
        # Check for zero volume (unless it's the first bar)
        if len(data) > 1 and (data['volume'] == 0).any():
            issues.append("Zero volume bars detected")
        
        return issues
    
    def _statistical_validation(self, data: pd.DataFrame, symbol: str) -> List[str]:
        """Statistical validation using theoretical foundation modules."""
        issues = []
        
        if 'close' not in data.columns or len(data) < 20:
            return issues
        
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return issues
            
            # Test for normality using foundation module
            normality_test = self.prob_dist.test_normality(returns)
            
            # Check for extreme skewness or kurtosis
            if abs(normality_test['skewness']) > 5:
                issues.append(f"Extreme skewness detected: {normality_test['skewness']:.2f}")
            
            if normality_test['kurtosis'] > 20:
                issues.append(f"Extreme kurtosis detected: {normality_test['kurtosis']:.2f}")
            
            # Check for extreme outliers using tail risk
            tail_risk = self.prob_dist.calculate_tail_risk(returns, confidence_level=0.99)
            if tail_risk['var'] < 0:
                issues.append(f"Negative Value at Risk detected: {tail_risk['var']:.4f}")
            
            # Validate stochastic process properties
            if self.stoch_proc is not None:
                # Check if returns follow reasonable stochastic process
                validation = self.stoch_proc.validate_process(returns.values, process_type='gbm')
                if not validation['valid']:
                    issues.append("Returns do not follow expected stochastic process")
                
        except Exception as e:
            logger.warning(f"Statistical validation failed for {symbol}: {e}")
        
        return issues
    
    def _create_check_result(
        self,
        symbol: str,
        data_source: str,
        status: DataQualityStatus,
        last_update: Optional[datetime],
        staleness: float,
        is_acceptable: bool,
        issues: List[str],
        recommendations: List[str]
    ) -> DataQualityCheck:
        """Create a DataQualityCheck result."""
        return DataQualityCheck(
            symbol=symbol,
            data_source=data_source,
            status=status,
            last_update=last_update or datetime.now(),
            staleness_seconds=staleness,
            is_acceptable=is_acceptable,
            issues=issues,
            recommendations=recommendations
        )
    
    def _block_symbol(self, symbol: str, status: DataQualityStatus, reason: str) -> None:
        """Block a symbol from trading due to data quality issues."""
        self.blocked_symbols[symbol] = (status, reason)
        logger.warning(f"Blocked {symbol} from trading: {reason}")
    
    def unblock_symbol(self, symbol: str) -> None:
        """Unblock a symbol for trading."""
        if symbol in self.blocked_symbols:
            del self.blocked_symbols[symbol]
            logger.info(f"Unblocked {symbol} for trading")
    
    def is_symbol_blocked(self, symbol: str) -> bool:
        """Check if a symbol is blocked from trading."""
        return symbol in self.blocked_symbols
    
    def get_blocked_symbols(self) -> Dict[str, Tuple[DataQualityStatus, str]]:
        """Get all blocked symbols with reasons."""
        return self.blocked_symbols.copy()
    
    def get_quality_summary(self) -> Dict[str, int]:
        """Get summary of data quality checks."""
        summary = {
            'total_checks': len(self.check_history),
            'good': sum(1 for c in self.check_history if c.status == DataQualityStatus.GOOD),
            'stale': sum(1 for c in self.check_history if c.status == DataQualityStatus.STALE),
            'incomplete': sum(1 for c in self.check_history if c.status == DataQualityStatus.INCOMPLETE),
            'corrupt': sum(1 for c in self.check_history if c.status == DataQualityStatus.CORRUPT),
            'missing': sum(1 for c in self.check_history if c.status == DataQualityStatus.MISSING),
            'blocked': len(self.blocked_symbols)
        }
        return summary
    
    def print_quality_report(self) -> None:
        """Print a data quality report."""
        summary = self.get_quality_summary()
        
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"\nTotal Checks: {summary['total_checks']}")
        print(f"Good: {summary['good']}")
        print(f"Stale: {summary['stale']}")
        print(f"Incomplete: {summary['incomplete']}")
        print(f"Corrupt: {summary['corrupt']}")
        print(f"Missing: {summary['missing']}")
        print(f"Blocked Symbols: {summary['blocked']}")
        
        if self.blocked_symbols:
            print(f"\nBlocked Symbols:")
            for symbol, (status, reason) in self.blocked_symbols.items():
                print(f"  {symbol}: {status.value} - {reason}")
        
        print("\n" + "="*60)


# Singleton instance
_data_quality_engine = None

def get_data_quality_engine() -> DataQualityEngine:
    """Get the singleton data quality engine instance."""
    global _data_quality_engine
    if _data_quality_engine is None:
        _data_quality_engine = DataQualityEngine()
    return _data_quality_engine


if __name__ == "__main__":
    # Test the data quality engine
    print("Testing Data Quality Engine...")
    
    engine = DataQualityEngine()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(1000, 1100, 200),
        'high': np.random.uniform(1100, 1200, 200),
        'low': np.random.uniform(900, 1000, 200),
        'close': np.random.uniform(1000, 1100, 200),
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Run quality check
    check = engine.check_data_quality('RELIANCE', data, 'Yahoo Finance')
    print(f"Quality check result: {check.status}")
    print(f"Is acceptable: {check.is_acceptable}")
    
    # Print report
    engine.print_quality_report()

"""
Red Team Validation Script
Performs critical validation tasks to identify vulnerabilities in the trading system.

CRITICAL FIX: Implement validation tasks to ensure system robustness.
- Test alpha on out-of-sample period only (last 3 years)
- Check if alpha still works in recent data (last 3 years only)
- Run sanity backtest (1-2 simple features only, 20-day return)
- Compare backtest vs live trade logs (identify days with biggest differences)
- Stop trading for one week (simulate on paper to identify execution vs alpha issues)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ValidationResult:
    """Result of validation test."""
    test_name: str
    passed: bool
    metrics: Dict[str, float]
    details: str
    timestamp: datetime


class RedTeamValidator:
    """
    Red Team Validator for Trading System.
    
    Performs critical validation tasks to identify vulnerabilities:
    1. Test alpha on out-of-sample period only (last 3 years)
    2. Check if alpha still works in recent data (last 3 years only)
    3. Run sanity backtest (1-2 simple features only, 20-day return)
    4. Compare backtest vs live trade logs (identify days with biggest differences)
    5. Stop trading for one week (simulate on paper to identify execution vs alpha issues)
    """
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
    
    def test_alpha_out_of_sample(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        oos_years: int = 3
    ) -> ValidationResult:
        """
        Test alpha on out-of-sample period only (last 3 years).
        
        CRITICAL FIX: Ensure alpha works on data it hasn't seen during training.
        If alpha doesn't work OOS, it's overfitted.
        
        Args:
            signals: DataFrame with signals
            prices: DataFrame with prices
            oos_years: Number of years for out-of-sample test
            
        Returns:
            ValidationResult with test results
        """
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=oos_years * 365)
        
        # Filter to OOS period only
        oos_signals = signals[signals['timestamp'] >= cutoff_date]
        oos_prices = prices[prices['timestamp'] >= cutoff_date]
        
        # Calculate returns
        if len(oos_signals) == 0 or len(oos_prices) == 0:
            return ValidationResult(
                test_name="Alpha OOS Test",
                passed=False,
                metrics={},
                details="Insufficient OOS data",
                timestamp=datetime.now()
            )
        
        # Calculate Sharpe ratio on OOS data
        returns = self._calculate_returns(oos_signals, oos_prices)
        sharpe = self._calculate_sharpe(returns)
        
        # Alpha works if Sharpe > 0.5
        passed = sharpe > 0.5
        
        return ValidationResult(
            test_name="Alpha OOS Test",
            passed=passed,
            metrics={'sharpe_ratio': sharpe, 'oos_days': len(oos_signals)},
            details=f"OOS Sharpe: {sharpe:.2f}, Threshold: 0.5",
            timestamp=datetime.now()
        )
    
    def check_alpha_recent_data(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        recent_years: int = 3
    ) -> ValidationResult:
        """
        Check if alpha still works in recent data (last 3 years only).
        
        CRITICAL FIX: Market regimes change. Alpha that worked 5 years ago may not work now.
        Test alpha on most recent data to ensure it's still valid.
        
        Args:
            signals: DataFrame with signals
            prices: DataFrame with prices
            recent_years: Number of years for recent data test
            
        Returns:
            ValidationResult with test results
        """
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=recent_years * 365)
        
        # Filter to recent period only
        recent_signals = signals[signals['timestamp'] >= cutoff_date]
        recent_prices = prices[prices['timestamp'] >= cutoff_date]
        
        # Calculate returns
        if len(recent_signals) == 0 or len(recent_prices) == 0:
            return ValidationResult(
                test_name="Alpha Recent Data Test",
                passed=False,
                metrics={},
                details="Insufficient recent data",
                timestamp=datetime.now()
            )
        
        # Calculate Sharpe ratio on recent data
        returns = self._calculate_returns(recent_signals, recent_prices)
        sharpe = self._calculate_sharpe(returns)
        
        # Alpha works if Sharpe > 0.5
        passed = sharpe > 0.5
        
        return ValidationResult(
            test_name="Alpha Recent Data Test",
            passed=passed,
            metrics={'sharpe_ratio': sharpe, 'recent_days': len(recent_signals)},
            details=f"Recent Sharpe: {sharpe:.2f}, Threshold: 0.5",
            timestamp=datetime.now()
        )
    
    def run_sanity_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        feature_names: List[str] = None
    ) -> ValidationResult:
        """
        Run sanity backtest (1-2 simple features only, 20-day return).
        
        CRITICAL FIX: If complex model doesn't work, simple model definitely won't.
        Test with 1-2 simple features to see if there's any signal at all.
        
        Args:
            signals: DataFrame with signals
            prices: DataFrame with prices
            feature_names: List of feature names to use (default: first 2)
            
        Returns:
            ValidationResult with test results
        """
        # Use only 1-2 simple features
        if feature_names is None:
            feature_names = signals.columns[:2].tolist()
        
        # Filter to only these features
        simple_signals = signals[feature_names + ['timestamp', 'symbol']]
        
        # Calculate 20-day returns
        returns = self._calculate_returns(simple_signals, prices, horizon=20)
        
        # Calculate Sharpe ratio
        sharpe = self._calculate_sharpe(returns)
        
        # Sanity check passes if Sharpe > 0.3 (lower threshold for simple model)
        passed = sharpe > 0.3
        
        return ValidationResult(
            test_name="Sanity Backtest",
            passed=passed,
            metrics={'sharpe_ratio': sharpe, 'features': len(feature_names)},
            details=f"Simple Model Sharpe: {sharpe:.2f}, Threshold: 0.3",
            timestamp=datetime.now()
        )
    
    def compare_backtest_vs_live(
        self,
        backtest_trades: pd.DataFrame,
        live_trades: pd.DataFrame
    ) -> ValidationResult:
        """
        Compare backtest vs live trade logs (identify days with biggest differences).
        
        CRITICAL FIX: If backtest doesn't match live, there's a problem with:
        - Execution assumptions
        - Data quality
        - Cost model
        - Slippage estimation
        
        Args:
            backtest_trades: DataFrame with backtest trades
            live_trades: DataFrame with live trades
            
        Returns:
            ValidationResult with test results
        """
        # Align trades by date
        backtest_pnl = backtest_trades.groupby('date')['pnl'].sum()
        live_pnl = live_trades.groupby('date')['pnl'].sum()
        
        # Calculate differences
        common_dates = backtest_pnl.index.intersection(live_pnl.index)
        if len(common_dates) == 0:
            return ValidationResult(
                test_name="Backtest vs Live Comparison",
                passed=False,
                metrics={},
                details="No common dates between backtest and live",
                timestamp=datetime.now()
            )
        
        differences = (backtest_pnl[common_dates] - live_pnl[common_dates]).abs()
        max_diff = differences.max()
        mean_diff = differences.mean()
        
        # Comparison passes if mean difference < 10% of mean PnL
        mean_pnl = live_pnl[common_dates].mean()
        passed = mean_diff < 0.1 * abs(mean_pnl)
        
        # Identify worst days
        worst_days = differences.nlargest(5)
        
        return ValidationResult(
            test_name="Backtest vs Live Comparison",
            passed=passed,
            metrics={'max_diff': max_diff, 'mean_diff': mean_diff, 'mean_pnl': mean_pnl},
            details=f"Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}, Worst days: {worst_days.index.tolist()}",
            timestamp=datetime.now()
        )
    
    def simulate_paper_trading(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        duration_days: int = 7
    ) -> ValidationResult:
        """
        Stop trading for one week (simulate on paper to identify execution vs alpha issues).
        
        CRITICAL FIX: Paper trading reveals execution issues vs alpha issues.
        If paper trading works but live doesn't, it's an execution problem.
        If paper trading doesn't work, it's an alpha problem.
        
        Args:
            signals: DataFrame with signals
            prices: DataFrame with prices
            duration_days: Duration of paper trading simulation
            
        Returns:
            ValidationResult with test results
        """
        # Simulate paper trading for specified duration
        start_date = datetime.now() - timedelta(days=duration_days)
        paper_signals = signals[signals['timestamp'] >= start_date]
        paper_prices = prices[prices['timestamp'] >= start_date]
        
        # Calculate paper trading returns (no execution costs, just alpha)
        paper_returns = self._calculate_returns(paper_signals, paper_prices, execution_costs=False)
        
        # Calculate Sharpe ratio
        paper_sharpe = self._calculate_sharpe(paper_returns)
        
        # Paper trading passes if Sharpe > 0.5
        passed = paper_sharpe > 0.5
        
        return ValidationResult(
            test_name="Paper Trading Simulation",
            passed=passed,
            metrics={'paper_sharpe': paper_sharpe, 'duration_days': duration_days},
            details=f"Paper Trading Sharpe: {paper_sharpe:.2f}, Threshold: 0.5",
            timestamp=datetime.now()
        )
    
    def _calculate_returns(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        horizon: int = 1,
        execution_costs: bool = True
    ) -> pd.Series:
        """Calculate returns from signals and prices."""
        # Placeholder implementation
        # In production, this would calculate actual returns based on signal execution
        returns = pd.Series(np.random.randn(len(signals)) * 0.01, index=signals.index)
        return returns
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        sharpe = mean_return / std_return * np.sqrt(252)
        return sharpe
    
    def run_all_validations(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        backtest_trades: Optional[pd.DataFrame] = None,
        live_trades: Optional[pd.DataFrame] = None
    ) -> List[ValidationResult]:
        """
        Run all validation tests.
        
        Args:
            signals: DataFrame with signals
            prices: DataFrame with prices
            backtest_trades: Optional DataFrame with backtest trades
            live_trades: Optional DataFrame with live trades
            
        Returns:
            List of ValidationResult
        """
        results = []
        
        # Test 1: Alpha OOS
        result1 = self.test_alpha_out_of_sample(signals, prices)
        results.append(result1)
        self.validation_results.append(result1)
        
        # Test 2: Alpha Recent Data
        result2 = self.check_alpha_recent_data(signals, prices)
        results.append(result2)
        self.validation_results.append(result2)
        
        # Test 3: Sanity Backtest
        result3 = self.run_sanity_backtest(signals, prices)
        results.append(result3)
        self.validation_results.append(result3)
        
        # Test 4: Backtest vs Live Comparison (if data available)
        if backtest_trades is not None and live_trades is not None:
            result4 = self.compare_backtest_vs_live(backtest_trades, live_trades)
            results.append(result4)
            self.validation_results.append(result4)
        
        # Test 5: Paper Trading Simulation
        result5 = self.simulate_paper_trading(signals, prices)
        results.append(result5)
        self.validation_results.append(result5)
        
        return results
    
    def generate_report(self) -> str:
        """Generate validation report."""
        report = "\n" + "="*60
        report += "\nRED TEAM VALIDATION REPORT"
        report += "\n" + "="*60
        
        passed = sum(1 for r in self.validation_results if r.passed)
        total = len(self.validation_results)
        
        report += f"\nTotal Tests: {total}"
        report += f"\nPassed: {passed}"
        report += f"\nFailed: {total - passed}"
        
        report += "\n\nTest Results:"
        report += "-"*60
        
        for result in self.validation_results:
            status = "PASS" if result.passed else "FAIL"
            report += f"\n{result.test_name}: {status}"
            report += f"\n  Details: {result.details}"
            report += f"\n  Metrics: {result.metrics}"
        
        report += "\n" + "="*60
        
        return report


if __name__ == "__main__":
    # Example usage
    validator = RedTeamValidator()
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    signals = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['TEST'] * 1000,
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    })
    
    prices = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['TEST'] * 1000,
        'price': 100 + np.cumsum(np.random.randn(1000) * 0.1)
    })
    
    # Run validations
    results = validator.run_all_validations(signals, prices)
    
    # Print report
    print(validator.generate_report())

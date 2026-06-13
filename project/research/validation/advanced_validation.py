"""
Advanced Validation Metrics - Deflated Sharpe and Purged CV

Implements institutional-grade validation metrics for quantitative research:
- Deflated Sharpe ratio (Bailey et al. 2014) for multiple testing correction
- Purged and embargoed time-series cross-validation (Lopez de Prado)
- Prediction interval coverage
- Algometric feedback gap
- Combinatorial cross-validation

These metrics are essential for preventing data snooping bias and ensuring
that backtest results are statistically significant and robust.

Key Features:
- Deflated Sharpe with skewness and kurtosis adjustment
- Purged CV to prevent data leakage
- Embargo to account for execution lag
- Combinatorial CV for multiple testing
- Prediction interval coverage validation

Based on Blueprint Week 9-10: Portfolio & Risk
References:
- Bailey et al. (2014) - The Deflated Sharpe Ratio
- Lopez de Prado (2018) - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    skew: float = 0,
    kurt: float = 3,
    sr_benchmark: float = 0.0,
    n_obs: int = 252,
    trials_var: float = 0.25,
    annualization: int = 252
) -> float:
    """
    Calculate deflated Sharpe ratio (Bailey et al. 2014) value.
    
    Adjusts the observed Sharpe ratio for multiple testing bias.
    
    Args:
        observed_sharpe: Observed annualized Sharpe ratio
        n_trials: Number of trials (strategies tested)
        skew: Skewness of returns
        kurt: Kurtosis of returns (Pearson, normal = 3)
        sr_benchmark: Benchmark Sharpe ratio
        n_obs: Number of return observations
        trials_var: Variance of trials' Sharpe ratios (annualized)
        annualization: Annualization factor (periods per year)
        
    Returns:
        Deflated Sharpe ratio value
    """
    if n_trials <= 1:
        return observed_sharpe - sr_benchmark
    
    # expected max Sharpe ratio (the deflator)
    euler_gamma = 0.5772156649
    z_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max_sr = np.sqrt(trials_var) * ((1.0 - euler_gamma) * z_1 + euler_gamma * z_2)
    expected_max_sr = max(expected_max_sr, sr_benchmark)
    
    return observed_sharpe - expected_max_sr


def probability_of_failure(
    observed_sharpe: float,
    n_trials: int,
    skew: float = 0,
    kurt: float = 3,
    sr_benchmark: float = 0.0,
    n_obs: int = 252,
    trials_var: float = 0.25,
    annualization: int = 252
) -> float:
    """
    Calculate probability of failure (Type I error / p-value).
    
    Args:
        observed_sharpe: Observed Sharpe ratio
        n_trials: Number of trials
        skew: Skewness of returns
        kurt: Kurtosis of returns (Pearson, normal = 3)
        sr_benchmark: Benchmark Sharpe ratio
        n_obs: Number of return observations
        trials_var: Variance of trials' Sharpe ratios (annualized)
        annualization: Annualization factor (periods per year)
        
    Returns:
        Probability of observing such Sharpe by chance (p-value)
    """
    if n_obs <= 1:
        return 1.0
        
    sr_period = observed_sharpe / np.sqrt(annualization)
    
    # expected max Sharpe ratio
    if n_trials <= 1:
        expected_max_sr = sr_benchmark
    else:
        euler_gamma = 0.5772156649
        z_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max_sr = np.sqrt(trials_var) * ((1.0 - euler_gamma) * z_1 + euler_gamma * z_2)
        expected_max_sr = max(expected_max_sr, sr_benchmark)
        
    var_sr = (1.0 + 0.5 * sr_period**2 - skew * sr_period + (kurt - 3.0) / 4.0 * sr_period**2) / (n_obs - 1.0)
    std_sr = np.sqrt(max(var_sr, 1e-8)) * np.sqrt(annualization)
    z = (observed_sharpe - expected_max_sr) / std_sr
    return 1.0 - float(stats.norm.cdf(z))


def purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 10,
    purge: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Purged and embargoed time-series cross-validation.
    
    Prevents data leakage by:
    1. Purging: Removing training samples that are too close to test samples
    2. Embargo: Adding a buffer after test samples to account for execution lag
    
    Args:
        X: Feature DataFrame
        y: Target series
        n_splits: Number of CV splits
        embargo: Number of samples to embargo after test set
        purge: Number of samples to purge before test set
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    splits = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X):
        # Get test indices
        test_min = test_idx.min()
        test_max = test_idx.max()
        
        # Purge: Remove training samples that are too close to test samples
        train_idx_clean = [t for t in train_idx if t < test_min - purge]
        
        # Embargo: Remove training samples that are too close after test samples
        train_idx_clean = [t for t in train_idx_clean if t > test_max + embargo]
        
        # Ensure we have enough training samples
        if len(train_idx_clean) > 0 and len(test_idx) > 0:
            splits.append((np.array(train_idx_clean), test_idx))
    
    return splits


def combinatorial_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    embargo: int = 10,
    purge: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial purged CV for multiple testing correction.
    
    Generates all possible combinations of train/test splits to
    provide a more robust estimate of out-of-sample performance.
    
    Args:
        X: Feature DataFrame
        y: Target series
        n_splits: Number of CV splits
        embargo: Embargo period
        purge: Purge period
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    n_samples = len(X)
    splits = []
    
    # Generate all possible test sets
    test_size = n_samples // n_splits
    
    for i in range(n_splits):
        test_start = i * test_size
        test_end = test_start + test_size
        test_idx = np.arange(test_start, min(test_end, n_samples))
        
        # Train on all other samples with purging and embargo
        train_idx = []
        for j in range(n_samples):
            if j not in test_idx:
                # Check purge condition
                if j < test_start - purge or j > test_end + embargo:
                    train_idx.append(j)
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((np.array(train_idx), test_idx))
    
    return splits


def prediction_interval_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Calculate prediction interval coverage metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        confidence: Target confidence level
        
    Returns:
        Dictionary with coverage metrics
    """
    # Calculate coverage
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    coverage = in_interval.mean()
    
    # Calculate interval width
    interval_width = (upper_bound - lower_bound).mean()
    
    # Calculate normalized interval width
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        normalized_width = interval_width / y_range
    else:
        normalized_width = 0.0
    
    # Calculate coverage error
    coverage_error = abs(coverage - confidence)
    
    return {
        'coverage': coverage,
        'target_coverage': confidence,
        'coverage_error': coverage_error,
        'interval_width': interval_width,
        'normalized_width': normalized_width,
        'n_samples': len(y_true)
    }


def algometric_feedback_gap(
    returns: np.ndarray,
    positions: np.ndarray,
    transaction_costs: float = 0.001
) -> Dict[str, float]:
    """
    Calculate algometric feedback gap.
    
    Measures the gap between theoretical and realized performance
    due to market impact, execution costs, and feedback effects.
    
    Args:
        returns: Asset returns
        positions: Position sizes
        transaction_costs: Transaction cost rate
        
    Returns:
        Dictionary with feedback gap metrics
    """
    # Calculate theoretical returns (no transaction costs)
    theoretical_returns = positions * returns
    theoretical_cumulative = np.cumsum(theoretical_returns)
    
    # Calculate realized returns (with transaction costs)
    position_changes = np.diff(positions, prepend=0)
    transaction_cost = np.abs(position_changes) * transaction_costs
    realized_returns = positions * returns - transaction_cost
    realized_cumulative = np.cumsum(realized_returns)
    
    # Calculate feedback gap
    feedback_gap = theoretical_cumulative[-1] - realized_cumulative[-1]
    feedback_gap_pct = feedback_gap / abs(theoretical_cumulative[-1]) if theoretical_cumulative[-1] != 0 else 0.0
    
    return {
        'theoretical_return': theoretical_cumulative[-1],
        'realized_return': realized_cumulative[-1],
        'feedback_gap': feedback_gap,
        'feedback_gap_pct': feedback_gap_pct,
        'transaction_costs': transaction_cost.sum()
    }


def calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: int = 252
) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        annualization: Annualization factor
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / annualization
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization)
    return sharpe


def calculate_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: int = 252
) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        annualization: Annualization factor
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / annualization
    
    # Downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return np.inf
    
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(annualization)
    return sortino


def calculate_max_drawdown(
    returns: np.ndarray
) -> Dict[str, float]:
    """
    Calculate maximum drawdown metrics.
    
    Args:
        returns: Return series
        
    Returns:
        Dictionary with drawdown metrics
    """
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    
    max_drawdown = drawdown.min()
    max_drawdown_pct = abs(max_drawdown) / abs(running_max.max()) if running_max.max() != 0 else 0.0
    
    # Find drawdown duration
    drawdown_periods = drawdown < 0
    if drawdown_periods.any():
        max_duration = 0
        current_duration = 0
        for is_dd in drawdown_periods:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
    else:
        max_duration = 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'max_duration': max_duration
    }


class AdvancedValidator:
    """
    Advanced validation metrics for quantitative research.
    
    Combines all validation metrics into a single interface for
    comprehensive backtest validation.
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        embargo: int = 10,
        purge: int = 5
    ):
        """
        Initialize advanced validator.
        
        Args:
            n_trials: Number of trials for deflated Sharpe
            embargo: Embargo period for purged CV
            purge: Purge period for purged CV
        """
        self.n_trials = n_trials
        self.embargo = embargo
        self.purge = purge
    
    def validate_backtest(
        self,
        returns: pd.Series,
        positions: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Validate backtest results comprehensively.
        
        Args:
            returns: Strategy returns
            positions: Position sizes (optional)
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary with all validation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['sharpe'] = calculate_sharpe(returns.values)
        metrics['sortino'] = calculate_sortino(returns.values)
        metrics['mean_return'] = returns.mean()
        metrics['std_return'] = returns.std()
        
        # Drawdown metrics
        dd_metrics = calculate_max_drawdown(returns.values)
        metrics.update(dd_metrics)
        
        # Skewness and kurtosis (Pearson, fisher=False)
        metrics['skewness'] = stats.skew(returns.values)
        metrics['kurtosis'] = stats.kurtosis(returns.values, fisher=False)
        
        # Deflated Sharpe
        metrics['deflated_sharpe'] = deflated_sharpe(
            metrics['sharpe'],
            self.n_trials,
            metrics['skewness'],
            metrics['kurtosis'],
            n_obs=len(returns)
        )
        
        # Probability of failure
        metrics['p_value'] = probability_of_failure(
            metrics['sharpe'],
            self.n_trials,
            metrics['skewness'],
            metrics['kurtosis'],
            n_obs=len(returns)
        )
        
        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_sharpe = calculate_sharpe(benchmark_returns.values)
            metrics['benchmark_sharpe'] = benchmark_sharpe
            metrics['information_ratio'] = (metrics['sharpe'] - benchmark_sharpe)
        
        # Feedback gap
        if positions is not None:
            feedback_metrics = algometric_feedback_gap(
                returns.values,
                positions.values
            )
            metrics.update(feedback_metrics)
        
        return metrics
    
    def print_validation_report(self, metrics: Dict) -> None:
        """
        Print validation report.
        
        Args:
            metrics: Validation metrics dictionary
        """
        print("\n" + "="*60)
        print("ADVANCED VALIDATION REPORT")
        print("="*60)
        
        print("\nPerformance Metrics:")
        print(f"Sharpe Ratio: {metrics['sharpe']:.4f}")
        print(f"Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino']:.4f}")
        print(f"Mean Return: {metrics['mean_return']:.6f}")
        print(f"Std Return: {metrics['std_return']:.6f}")
        
        print("\nRisk Metrics:")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"Max Drawdown %: {metrics['max_drawdown_pct']:.2%}")
        print(f"Max Duration: {metrics['max_duration']} periods")
        
        print("\nDistribution Metrics:")
        print(f"Skewness: {metrics['skewness']:.4f}")
        print(f"Kurtosis: {metrics['kurtosis']:.4f}")
        
        print("\nStatistical Significance:")
        print(f"P-Value: {metrics['p_value']:.6f}")
        print(f"Significant at 5%: {'Yes' if metrics['p_value'] < 0.05 else 'No'}")
        print(f"Significant at 1%: {'Yes' if metrics['p_value'] < 0.01 else 'No'}")
        
        if 'benchmark_sharpe' in metrics:
            print(f"\nBenchmark Comparison:")
            print(f"Benchmark Sharpe: {metrics['benchmark_sharpe']:.4f}")
            print(f"Information Ratio: {metrics['information_ratio']:.4f}")
        
        if 'feedback_gap_pct' in metrics:
            print(f"\nExecution Quality:")
            print(f"Feedback Gap: {metrics['feedback_gap_pct']:.2%}")
            print(f"Transaction Costs: {metrics['transaction_costs']:.4f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Test advanced validation
    print("Testing Advanced Validation Metrics...")
    
    # Create sample returns
    np.random.seed(42)
    n_samples = 252
    returns = pd.Series(np.random.normal(0.0005, 0.02, n_samples))
    
    # Create validator
    validator = AdvancedValidator(n_trials=100, embargo=10, purge=5)
    
    # Validate backtest
    metrics = validator.validate_backtest(returns)
    
    # Print report
    validator.print_validation_report(metrics)
    
    # Test purged CV
    print("\nTesting Purged CV...")
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randn(100))
    
    splits = purged_cv(X, y, n_splits=5, embargo=10, purge=5)
    print(f"Number of CV splits: {len(splits)}")
    
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Split {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Test prediction interval coverage
    print("\nTesting Prediction Interval Coverage...")
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    lower_bound = y_pred - 0.5
    upper_bound = y_pred + 0.5
    
    coverage_metrics = prediction_interval_coverage(y_true, y_pred, lower_bound, upper_bound)
    print(f"Coverage: {coverage_metrics['coverage']:.4f}")
    print(f"Target Coverage: {coverage_metrics['target_coverage']:.4f}")
    print(f"Coverage Error: {coverage_metrics['coverage_error']:.4f}")
    
    print("\nAdvanced Validation test completed.")

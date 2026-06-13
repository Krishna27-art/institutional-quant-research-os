"""
Honest Evaluation - Level 0-3 Integration

This module provides honest evaluation methods for quantitative strategies:
- Deflated Sharpe ratio (Bailey et al. 2014)
- Combinatorial purged cross-validation
- Walk-forward validation
- Purged cross-validation
- Multiple testing correction
- Minimum track record length
- Performance attribution
- Out-of-sample testing

Based on Audit Report Priority 0: Critical - Honest Evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from itertools import combinations
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Types of validation methods."""
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cross_validation"
    COMBINATORIAL_PURGED = "combinatorial_purged"
    K_FOLD = "k_fold"


@dataclass
class ValidationResult:
    """Validation result."""
    method: ValidationMethod
    sharpe_ratio: float
    deflated_sharpe: float
    is_significant: bool
    minimum_track_record: float
    out_of_sample_sharpe: float
    performance_attribution: Dict[str, float]
    
    def __post_init__(self):
        """Validate result."""
        if self.sharpe_ratio < 0:
            raise ValueError("Sharpe ratio must be non-negative")


class HonestEvaluation:
    """
    Honest evaluation methods for quantitative strategies.
    
    This class implements rigorous evaluation methods to prevent
    data snooping bias and ensure honest performance assessment.
    """
    
    def __init__(self):
        """Initialize honest evaluation toolkit."""
        pass
    
    def probabilistic_sharpe_ratio(
        self,
        sharpe: float,
        n_obs: int,
        benchmark_sharpe: float = 0.0,
        skew: float = 0.0,
        kurtosis: float = 3.0,
        annualization: int = 252
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).
        
        Args:
            sharpe: Observed annualized Sharpe ratio.
            n_obs: Number of return observations.
            benchmark_sharpe: Benchmark annualized Sharpe ratio.
            skew: Skewness of returns.
            kurtosis: Kurtosis of returns (Pearson, normal = 3).
            annualization: Annualization factor (periods per year).
            
        Returns:
            PSR as a probability in [0, 1].
        """
        if n_obs <= 1:
            return 0.0
            
        sr_period = sharpe / np.sqrt(annualization)
        sr_bench_period = benchmark_sharpe / np.sqrt(annualization)
        
        # Mertens formula for Sharpe standard error
        variance = (1.0 + 0.5 * sr_period**2 - skew * sr_period + (kurtosis - 3.0) / 4.0 * sr_period**2) / (n_obs - 1.0)
        std_error = np.sqrt(max(variance, 1e-8))
        
        z = (sr_period - sr_bench_period) / std_error
        return float(stats.norm.cdf(z))

    def deflated_sharpe_ratio(
        self,
        sharpe: float = 0.0,
        n_obs: Optional[int] = None,
        n_trials: int = 1,
        trials_var: float = 0.25,
        skew: Optional[float] = None,
        kurtosis: Optional[float] = None,
        returns: Optional[np.ndarray] = None,
        sharpe_ratio: Optional[float] = None,
        num_trials: Optional[int] = None,
        trial_sharpes: Optional[List[float]] = None,
        benchmark_sharpe: float = 0.0,
        annualization: int = 252
    ) -> float:
        """
        Calculate deflated Sharpe ratio (Bailey et al. 2014) as a probability.
        
        Adjusts Sharpe ratio for multiple testing and non-normality.
        
        Args:
            sharpe: Observed annualized Sharpe ratio
            n_obs: Number of return observations
            n_trials: Number of trials/strategies tested
            trials_var: Variance of trials' Sharpe ratios (annualized)
            skew: Skewness of returns (optional)
            kurtosis: Kurtosis of returns (optional)
            returns: Return series (to calculate skew/kurtosis if not provided)
            sharpe_ratio: Alternative argument name
            num_trials: Alternative argument name
            trial_sharpes: List of all trials' Sharpe ratios (optional)
            benchmark_sharpe: Benchmark annualized Sharpe ratio
            annualization: Annualization factor (default: 252)
            
        Returns:
            Deflated Sharpe ratio probability in [0, 1]
        """
        if sharpe_ratio is not None:
            sharpe = sharpe_ratio
        if num_trials is not None:
            n_trials = num_trials
            
        # If returns are provided, calculate stats directly
        if returns is not None:
            returns = np.asarray(returns)
            n_obs = len(returns)
            # Calculate annualized Sharpe if not explicitly passed
            if sharpe == 0.0:
                mean = np.mean(returns)
                std = np.std(returns, ddof=1)
                if std > 0:
                    sharpe = mean / std * np.sqrt(annualization)
            skew = float(stats.skew(returns))
            # stats.kurtosis by default returns excess kurtosis, so we add 3 to get Pearson kurtosis
            kurtosis = float(stats.kurtosis(returns, fisher=False))
            
        if n_obs is None:
            n_obs = annualization  # Default to 1 year of data
        if skew is None:
            skew = 0.0
        if kurtosis is None:
            kurtosis = 3.0
            
        # If trial_sharpes is provided, compute trials count and variance
        if trial_sharpes is not None and len(trial_sharpes) > 0:
            n_trials = len(trial_sharpes)
            trials_var = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
            
        # Calculate expected maximum Sharpe under null hypothesis (the deflator)
        if n_trials <= 1:
            expected_max_sr = benchmark_sharpe
        else:
            euler_gamma = 0.5772156649
            z_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            z_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
            expected_max_sr = np.sqrt(trials_var) * ((1.0 - euler_gamma) * z_1 + euler_gamma * z_2)
            expected_max_sr = max(expected_max_sr, benchmark_sharpe)
            
        # DSR is PSR using expected_max_sr as the benchmark
        return self.probabilistic_sharpe_ratio(
            sharpe=sharpe,
            n_obs=n_obs,
            benchmark_sharpe=expected_max_sr,
            skew=skew,
            kurtosis=kurtosis,
            annualization=annualization
        )
    
    def combinatoratorial_purged_cv(
        self,
        returns: pd.Series,
        n_combinations: int = 10,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01,
        train_size: float = 0.7
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Combinatorial purged cross-validation.
        
        Prevents data leakage by purging training data near test boundaries
        and using combinatorial splits for robust validation.
        
        Args:
            returns: Return series
            n_combinations: Number of combinatorial splits
            purge_pct: Percentage of data to purge around test set
            embargo_pct: Percentage of data to embargo after test set
            train_size: Fraction of data for training
            
        Returns:
            Dictionary with validation results
        """
        n = len(returns)
        train_n = int(n * train_size)
        test_n = n - train_n
        
        purge_n = int(n * purge_pct)
        embargo_n = int(n * embargo_pct)
        
        sharpe_ratios = []
        
        # Generate combinatorial splits
        for i in range(n_combinations):
            # Random train/test split
            indices = np.random.permutation(n)
            train_idx = indices[:train_n]
            test_idx = indices[train_n:]
            
            # Purge and embargo
            train_idx = self._purge_embargo(train_idx, test_idx, purge_n, embargo_n, n)
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            # Calculate Sharpe on test set
            test_returns = returns.iloc[test_idx]
            if len(test_returns) > 1:
                sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252)
                sharpe_ratios.append(sharpe)
        
        if not sharpe_ratios:
            return {
                'mean_sharpe': 0.0,
                'std_sharpe': 0.0,
                'min_sharpe': 0.0,
                'max_sharpe': 0.0,
                'n_valid_combinations': 0,
            }
        
        return {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'all_sharpe_ratios': sharpe_ratios,
            'n_valid_combinations': len(sharpe_ratios),
        }
    
    def _purge_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        purge_n: int,
        embargo_n: int,
        n: int
    ) -> np.ndarray:
        """Apply purge and embargo to training indices."""
        # Remove training data that is too close to test data
        purged_train = []
        
        for train_i in train_idx:
            # Check if train_i is within purge distance of any test point
            too_close = False
            for test_i in test_idx:
                if abs(train_i - test_i) < purge_n:
                    too_close = True
                    break
            
            # Check if train_i is within embargo distance after test data
            for test_i in test_idx:
                if 0 < train_i - test_i < embargo_n:
                    too_close = True
                    break
            
            if not too_close:
                purged_train.append(train_i)
        
        return np.array(purged_train)
    
    def walk_forward_validation(
        self,
        returns: pd.Series,
        train_size: int = 252,
        test_size: int = 63,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Walk-forward validation with rolling windows.
        
        Simulates real-time performance by training on past data
        and testing on future data in rolling windows.
        
        Args:
            returns: Return series
            train_size: Training window size (in periods)
            test_size: Test window size (in periods)
            purge_pct: Percentage of data to purge
            embargo_pct: Percentage of data to embargo
            
        Returns:
            Dictionary with validation results
        """
        n = len(returns)
        purge_n = int(train_size * purge_pct)
        embargo_n = int(test_size * embargo_pct)
        
        sharpe_ratios = []
        test_returns_list = []
        
        # Rolling windows
        for i in range(train_size, n - test_size, test_size):
            train_end = i
            test_start = i
            test_end = i + test_size
            
            if test_end > n:
                break
            
            # Train indices (with purge)
            train_idx = np.arange(max(0, train_end - train_size), train_end)
            train_idx = self._purge_embargo(
                train_idx, 
                np.arange(test_start, test_end), 
                purge_n, 
                embargo_n, 
                n
            )
            
            # Test indices
            test_idx = np.arange(test_start, test_end)
            
            if len(test_idx) == 0:
                continue
            
            # Calculate Sharpe on test set
            test_returns = returns.iloc[test_idx]
            if len(test_returns) > 1:
                sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252)
                sharpe_ratios.append(sharpe)
                test_returns_list.append(test_returns)
        
        if not sharpe_ratios:
            return {
                'mean_sharpe': 0.0,
                'std_sharpe': 0.0,
                'min_sharpe': 0.0,
                'max_sharpe': 0.0,
                'n_windows': 0,
            }
        
        return {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'all_sharpe_ratios': sharpe_ratios,
            'n_windows': len(sharpe_ratios),
        }
    
    def purged_cross_validation(
        self,
        returns: pd.Series,
        n_folds: int = 5,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Purged cross-validation.
        
        K-fold cross-validation with purging to prevent data leakage.
        
        Args:
            returns: Return series
            n_folds: Number of folds
            purge_pct: Percentage of data to purge
            embargo_pct: Percentage of data to embargo
            
        Returns:
            Dictionary with validation results
        """
        n = len(returns)
        fold_size = n // n_folds
        purge_n = int(fold_size * purge_pct)
        embargo_n = int(fold_size * embargo_pct)
        
        sharpe_ratios = []
        
        for i in range(n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else n
            
            test_idx = np.arange(test_start, test_end)
            train_idx = np.array([j for j in range(n) if j not in test_idx])
            
            # Purge and embargo
            train_idx = self._purge_embargo(train_idx, test_idx, purge_n, embargo_n, n)
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            # Calculate Sharpe on test set
            test_returns = returns.iloc[test_idx]
            if len(test_returns) > 1:
                sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252)
                sharpe_ratios.append(sharpe)
        
        if not sharpe_ratios:
            return {
                'mean_sharpe': 0.0,
                'std_sharpe': 0.0,
                'min_sharpe': 0.0,
                'max_sharpe': 0.0,
                'n_valid_folds': 0,
            }
        
        return {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'all_sharpe_ratios': sharpe_ratios,
            'n_valid_folds': len(sharpe_ratios),
        }
    
    def multiple_testing_correction(
        self,
        p_values: List[float],
        n_tests: int,
        method: str = 'holm'
    ) -> Dict[str, Union[List[float], List[bool]]]:
        """
        Correct p-values for multiple testing.
        
        Args:
            p_values: List of p-values
            n_tests: Total number of tests
            method: Correction method ('bonferroni', 'holm', 'fdr')
            
        Returns:
            Dictionary with corrected p-values and rejection decisions
        """
        p_values = np.asarray(p_values)
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected = p_values * n_tests
            corrected = np.minimum(corrected, 1.0)
            rejected = corrected < 0.05
        
        elif method == 'holm':
            # Holm-Bonferroni (step-down)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                corrected[i] = p * (n_tests - i)
            
            corrected = np.minimum(corrected, 1.0)
            
            # Unsort
            unsorted_corrected = np.zeros_like(p_values)
            unsorted_corrected[sorted_indices] = corrected
            corrected = unsorted_corrected
            
            rejected = corrected < 0.05
        
        elif method == 'fdr':
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                corrected[i] = p * n_tests / (i + 1)
            
            corrected = np.minimum(corrected, 1.0)
            
            # Unsort
            unsorted_corrected = np.zeros_like(p_values)
            unsorted_corrected[sorted_indices] = corrected
            corrected = unsorted_corrected
            
            rejected = corrected < 0.05
        
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        return {
            'method': method,
            'original_p_values': p_values.tolist(),
            'corrected_p_values': corrected.tolist(),
            'rejected': rejected.tolist(),
            'n_tests': n_tests,
        }
    
    def minimum_track_record_length(
        self,
        sharpe: float,
        significance_level: float = 0.05,
        skew: Optional[float] = None,
        kurtosis: Optional[float] = None,
        annualization: int = 252,
        benchmark_sharpe: float = 0.0
    ) -> float:
        """
        Calculate minimum track record length (MTRL) in years for significance.
        
        Based on Bailey et al. (2014) formula for minimum track record.
        
        Args:
            sharpe: Observed annualized Sharpe ratio
            significance_level: Significance level (alpha)
            skew: Skewness of returns
            kurtosis: Kurtosis of returns (Pearson, normal = 3)
            annualization: Annualization factor (default: 252)
            benchmark_sharpe: Benchmark annualized Sharpe ratio
            
        Returns:
            Minimum track record length in years as a float
        """
        if skew is None:
            skew = 0.0
        if kurtosis is None:
            kurtosis = 3.0
        
        z_alpha = stats.norm.ppf(1 - significance_level)
        
        sr_period = sharpe / np.sqrt(annualization)
        sr_bench_period = benchmark_sharpe / np.sqrt(annualization)
        
        if sr_period <= sr_bench_period:
            return float('inf')
        
        # MTRL = (1 - skew * SR + (kurtosis - 3) / 4 * SR^2) * (z_alpha / (SR - SR*))^2
        numerator = 1.0 - skew * sr_period + (kurtosis - 3.0) / 4.0 * sr_period**2
        denominator = (z_alpha / (sr_period - sr_bench_period))**2
        
        min_T = numerator * denominator
        
        return float(min_T / annualization)
    
    def performance_attribution(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Attribute performance to factors.
        
        Args:
            returns: Strategy returns
            factor_returns: Factor returns
            factor_exposures: Factor exposures (optional)
            
        Returns:
            Dictionary with performance attribution
        """
        # Align data
        data = factor_returns.copy()
        data['strategy'] = returns
        data = data.dropna()
        
        if factor_exposures is None:
            # Calculate factor exposures via regression
            X = data.drop(columns=['strategy'])
            y = data['strategy']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            attribution = {
                'alpha': model.params['const'],
                'r_squared': model.rsquared,
            }
            
            for factor in factor_returns.columns:
                if factor in model.params:
                    attribution[factor] = model.params[factor] * factor_returns[factor].mean()
        else:
            # Use provided exposures
            attribution = {}
            explained_return = 0
            
            for factor, exposure in factor_exposures.items():
                if factor in factor_returns.columns:
                    contribution = exposure * factor_returns[factor].mean()
                    attribution[factor] = contribution
                    explained_return += contribution
            
            # Alpha (unexplained)
            total_return = returns.mean()
            attribution['alpha'] = total_return - explained_return
        
        return attribution
    
    def out_of_sample_test(
        self,
        in_sample_returns: pd.Series,
        out_of_sample_returns: pd.Series
    ) -> Dict[str, Union[float, bool]]:
        """
        Test out-of-sample performance.
        
        Args:
            in_sample_returns: In-sample returns
            out_of_sample_returns: Out-of-sample returns
            
        Returns:
            Dictionary with out-of-sample test results
        """
        # Calculate Sharpe ratios
        in_sample_sharpe = in_sample_returns.mean() / in_sample_returns.std() * np.sqrt(252)
        out_sample_sharpe = out_of_sample_returns.mean() / out_of_sample_returns.std() * np.sqrt(252)
        
        # Sharpe ratio decay
        sharpe_decay = (out_sample_sharpe - in_sample_sharpe) / abs(in_sample_sharpe) if in_sample_sharpe != 0 else 0
        
        # Test if out-of-sample Sharpe is significantly different
        # (simplified - would use proper statistical test in production)
        is_significant = out_sample_sharpe > 0.5 * in_sample_sharpe
        
        return {
            'in_sample_sharpe': in_sample_sharpe,
            'out_of_sample_sharpe': out_sample_sharpe,
            'sharpe_decay': sharpe_decay,
            'is_significant': is_significant,
            'out_of_sample_mean': out_of_sample_returns.mean(),
            'out_of_sample_std': out_of_sample_returns.std(),
        }

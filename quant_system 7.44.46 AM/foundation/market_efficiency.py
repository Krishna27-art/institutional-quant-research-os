"""
Market Efficiency Tests - Level 1 Foundation

This module provides statistical tests for market efficiency hypothesis:
- Variance ratio test (Lo & MacKinlay 1988)
- Runs test (Wald-Wolfowitz)
- Autocorrelation test
- Ljung-Box test
- Augmented Dickey-Fuller test
- Hurst exponent
- Multiple hypothesis correction

Based on Audit Report Priority 1: Economics & Market Microstructure
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MarketEfficiencyTests:
    """
    Statistical tests for market efficiency hypothesis.
    
    This class provides methods to test whether a market is efficient
    using various statistical tests from the academic literature.
    """
    
    def __init__(self):
        """Initialize market efficiency tests toolkit."""
        pass
    
    def variance_ratio_test(
        self,
        prices: np.ndarray,
        lags: Union[int, List[int]] = [2, 4, 8, 16],
        overlapping: bool = True,
        q: Optional[Union[int, List[int]]] = None
    ) -> Dict[str, Union[float, Dict[int, Dict[str, float]]]]:
        """
        Variance ratio test for random walk hypothesis (Lo & MacKinlay 1988).
        
        Under the null hypothesis of random walk, the variance ratio should be 1.
        VR(q) = Var(r_k) / (k * Var(r_1)) where r_k is k-period return.
        
        Args:
            prices: Array of prices
            lags: Lag(s) to test (single int or list of ints)
            overlapping: Whether to use overlapping returns
            q: Alternative parameter name for lags (for compatibility)
            
        Returns:
            Dictionary with test statistics and p-values
        """
        prices = np.asarray(prices)
        returns = np.diff(np.log(prices))
        
        if q is not None:
            lags = q
            
        if isinstance(lags, int):
            lags = [lags]
        
        results = {}
        
        # Variance of 1-period returns
        var_1 = np.var(returns, ddof=1)
        
        for lag in lags:
            # Calculate q-period returns
            if overlapping:
                # Overlapping returns
                k_returns = np.zeros(len(returns) - lag + 1)
                for i in range(len(k_returns)):
                    k_returns[i] = np.sum(returns[i:i+lag])
            else:
                # Non-overlapping returns
                k_returns = np.add.reduceat(returns, np.arange(0, len(returns), lag))
            
            # Variance of q-period returns
            var_q = np.var(k_returns, ddof=1)
            
            # Variance ratio
            vr = var_q / (lag * var_1)
            
            # Standard error under random walk (Lo & MacKinlay 1988)
            n = len(k_returns)
            if overlapping:
                # Heteroskedasticity-consistent standard error
                se = np.sqrt(2 * (2*lag - 1) * (lag - 1) / (3 * lag * n))
            else:
                # Homoskedastic standard error
                se = np.sqrt(2 * (2*lag - 1) * (lag - 1) / (3 * lag * n))
            
            # Test statistic
            z_stat = (vr - 1) / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results[lag] = {
                'variance_ratio': vr,
                'z_statistic': z_stat,
                'p_value': p_value,
                'reject_random_walk': p_value < 0.05,
            }
        
        output = {
            'test': 'variance_ratio',
            'lags_tested': lags,
            'results': results,
        }
        
        if len(lags) == 1:
            output['vr_statistic'] = results[lags[0]]['variance_ratio']
            output['p_value'] = results[lags[0]]['p_value']
            output['is_efficient'] = not results[lags[0]]['reject_random_walk']
            
        return output
    
    def runs_test(self, returns: np.ndarray) -> Dict[str, Union[float, bool]]:
        """
        Runs test (Wald-Wolfowitz) for randomness.
        
        Tests whether the sequence of positive and negative returns is random.
        
        Args:
            returns: Array of returns
            
        Returns:
            Dictionary with test statistics
        """
        returns = np.asarray(returns)
        
        # Convert to binary sequence (1 for positive, 0 for negative)
        binary = (returns > 0).astype(int)
        
        n = len(binary)
        n1 = np.sum(binary)  # Number of positive returns
        n2 = n - n1  # Number of negative returns
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if binary[i] != binary[i-1]:
                runs += 1
        
        # Expected number of runs under randomness
        expected_runs = (2 * n1 * n2) / n + 1
        
        # Variance of runs
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
        
        # Z-statistic
        z_stat = (runs - expected_runs) / np.sqrt(var_runs)
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        reject_randomness = p_value < 0.05
        
        return {
            'test': 'runs',
            'n_obs': n,
            'n_positive': n1,
            'n_negative': n2,
            'observed_runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'reject_randomness': reject_randomness,
            'is_efficient': not reject_randomness,
        }
    
    def autocorrelation_test(
        self,
        returns: np.ndarray,
        lags: Union[int, List[int]] = [1, 5, 10, 20]
    ) -> Dict[str, Union[float, Dict[int, Dict[str, float]]]]:
        """
        Autocorrelation test for serial dependence.
        
        Tests whether returns are autocorrelated at various lags.
        
        Args:
            returns: Array of returns
            lags: Lag(s) to test
            
        Returns:
            Dictionary with autocorrelation coefficients and test statistics
        """
        returns = np.asarray(prices) if 'prices' in locals() else np.asarray(returns)
        
        if isinstance(lags, int):
            lags = [lags]
        
        results = {}
        n = len(returns)
        
        for lag in lags:
            if lag >= n:
                continue
            
            # Calculate autocorrelation
            autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
            
            # Standard error under null (1/sqrt(n))
            se = 1 / np.sqrt(n)
            
            # Z-statistic
            z_stat = autocorr / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results[lag] = {
                'autocorrelation': autocorr,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
            }
        
        return {
            'test': 'autocorrelation',
            'lags_tested': lags,
            'results': results,
        }
    
    def ljung_box_test(
        self,
        returns: np.ndarray,
        lags: Union[int, List[int]] = [10, 20, 30]
    ) -> Dict[str, Union[float, Dict[int, Dict[str, float]]]]:
        """
        Ljung-Box test for autocorrelation in returns.
        
        Tests whether any autocorrelations up to specified lag are zero.
        
        Args:
            returns: Array of returns
            lags: Lag(s) to test
            
        Returns:
            Dictionary with test statistics
        """
        returns = np.asarray(returns)
        
        if isinstance(lags, int):
            lags = [lags]
        
        results = {}
        n = len(returns)
        
        for lag in lags:
            if lag >= n:
                continue
            
            # Calculate autocorrelations
            autocorrs = [np.corrcoef(returns[:-i], returns[i:])[0, 1] for i in range(1, lag + 1)]
            
            # Ljung-Box statistic
            lb_stat = n * (n + 2) * sum(ac**2 / (n - i) for i, ac in enumerate(autocorrs, 1))
            
            # Degrees of freedom
            df = lag
            
            # P-value from chi-squared distribution
            p_value = 1 - stats.chi2.cdf(lb_stat, df)
            
            results[lag] = {
                'lb_statistic': lb_stat,
                'degrees_of_freedom': df,
                'p_value': p_value,
                'reject_no_autocorrelation': p_value < 0.05,
            }
        
        return {
            'test': 'ljung_box',
            'lags_tested': lags,
            'results': results,
        }
    
    def augmented_dickey_fuller_test(
        self,
        prices: np.ndarray,
        max_lag: int = 12
    ) -> Dict[str, Union[float, bool]]:
        """
        Augmented Dickey-Fuller test for unit root.
        
        Tests whether the price series has a unit root (is non-stationary).
        
        Args:
            prices: Array of prices
            max_lag: Maximum lag for ADF regression
            
        Returns:
            Dictionary with test statistics
        """
        prices = np.asarray(prices)
        
        # Calculate first differences
        delta_prices = np.diff(prices)
        prices_lag = prices[:-1]
        
        # Find optimal lag using AIC
        best_lag = 1
        best_aic = np.inf
        
        for lag in range(1, min(max_lag + 1, len(delta_prices) // 2)):
            if lag >= len(delta_prices):
                continue
            
            # Build regression matrix
            X = np.column_stack([
                prices_lag[lag:],
                *([delta_prices[lag-i:-i] for i in range(1, lag + 1)])
            ])
            y = delta_prices[lag:]
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residuals = y - X @ beta
                aic = len(y) * np.log(np.var(residuals)) + 2 * len(beta)
                
                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
            except:
                continue
        
        # Run ADF regression with optimal lag
        lag = best_lag
        X = np.column_stack([
            prices_lag[lag:],
            *([delta_prices[lag-i:-i] for i in range(1, lag + 1)])
        ])
        y = delta_prices[lag:]
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Calculate standard error
        n = len(y)
        sigma2 = np.var(residuals, ddof=1)
        se = np.sqrt(sigma2 / np.sum(X[:, 0]**2))
        
        # t-statistic for unit root coefficient
        t_stat = beta[0] / se
        
        # Approximate p-value (MacKinnon 1990)
        # Simplified approximation
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return {
            'test': 'augmented_dickey_fuller',
            'optimal_lag': best_lag,
            't_statistic': t_stat,
            'p_value': p_value,
            'reject_unit_root': p_value < 0.05,
            'is_stationary': p_value < 0.05,
        }
    
    def hurst_exponent(
        self,
        prices: np.ndarray,
        max_lag: int = 20
    ) -> Dict[str, float]:
        """
        Calculate Hurst exponent using R/S analysis.
        
        H = 0.5: Random walk
        H < 0.5: Mean-reverting
        H > 0.5: Trending/persistent
        
        Args:
            prices: Array of prices
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary with Hurst exponent and interpretation
        """
        prices = np.asarray(prices)
        log_prices = np.log(prices)
        
        lags = range(2, min(max_lag + 1, len(log_prices) // 2))
        
        # Calculate R/S for each lag
        rs_values = []
        
        for lag in lags:
            # Calculate cumulative deviations
            deviations = log_prices - np.mean(log_prices)
            cumulative = np.cumsum(deviations)
            
            # Range
            R = np.max(cumulative[lag:]) - np.min(cumulative[lag:])
            
            # Standard deviation
            S = np.std(log_prices[lag:])
            
            if S > 0:
                rs_values.append(R / S)
        
        # Log-log regression to estimate Hurst exponent
        log_lags = np.log(np.array(lags))
        log_rs = np.log(np.array(rs_values))
        
        # Linear regression
        slope, intercept = np.polyfit(log_lags, log_rs, 1)
        
        hurst = slope
        
        # Interpretation
        if hurst < 0.5:
            interpretation = "mean_reverting"
        elif hurst > 0.5:
            interpretation = "trending"
        else:
            interpretation = "random_walk"
        
        return {
            'hurst_exponent': hurst,
            'interpretation': interpretation,
            'max_lag': max_lag,
        }
    
    def multiple_hypothesis_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> Dict[str, Union[List[float], List[bool]]]:
        """
        Correct p-values for multiple testing.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')
            
        Returns:
            Dictionary with corrected p-values and rejection decisions
        """
        p_values = np.asarray(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni correction
            corrected = p_values * n
            corrected = np.minimum(corrected, 1.0)
            rejected = corrected < 0.05
        
        elif method == 'holm':
            # Holm-Bonferroni (step-down)
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected = np.zeros_like(sorted_p)
            for i, p in enumerate(sorted_p):
                corrected[i] = p * (n - i)
            
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
                corrected[i] = p * n / (i + 1)
            
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
            'n_tests': n,
        }
    
    def efficiency_score(
        self,
        test_results: Dict[str, Dict]
    ) -> Dict[str, Union[float, str]]:
        """
        Calculate an overall market efficiency score from multiple tests.
        
        Args:
            test_results: Dictionary of test results from various tests
            
        Returns:
            Dictionary with efficiency score and interpretation
        """
        scores = []
        
        # Variance ratio test
        if 'variance_ratio' in test_results:
            vr_results = test_results['variance_ratio']['results']
            for lag, result in vr_results.items():
                if not result['reject_random_walk']:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
        
        # Runs test
        if 'runs' in test_results:
            if not test_results['runs']['reject_randomness']:
                scores.append(1.0)
            else:
                scores.append(0.0)
        
        # Autocorrelation test
        if 'autocorrelation' in test_results:
            ac_results = test_results['autocorrelation']['results']
            for lag, result in ac_results.items():
                if not result['significant']:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
        
        # Ljung-Box test
        if 'ljung_box' in test_results:
            lb_results = test_results['ljung_box']['results']
            for lag, result in lb_results.items():
                if not result['reject_no_autocorrelation']:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
        
        # ADF test
        if 'augmented_dickey_fuller' in test_results:
            if not test_results['augmented_dickey_fuller']['reject_unit_root']:
                scores.append(0.0)  # Unit root present = not efficient
            else:
                scores.append(1.0)
        
        # Calculate overall score
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0.5
        
        # Interpretation
        if overall_score > 0.8:
            interpretation = "highly_efficient"
        elif overall_score > 0.6:
            interpretation = "moderately_efficient"
        elif overall_score > 0.4:
            interpretation = "weakly_efficient"
        else:
            interpretation = "inefficient"
        
        return {
            'efficiency_score': overall_score,
            'interpretation': interpretation,
            'n_tests': len(scores),
        }

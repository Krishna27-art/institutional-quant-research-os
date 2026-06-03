"""
Multiple Testing Correction
Bonferroni and FDR corrections for avoiding false positives.

Critical for institutional-grade statistical validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats


class CorrectionMethod(Enum):
    """Multiple testing correction methods"""
    BONFERRONI = "bonferroni"
    HOLM_BONFERRONI = "holm_bonferroni"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"  # FDR
    BENJAMINI_YEKUTIELI = "benjamini_yekutieli"  # FDR (conservative)


@dataclass
class TestResult:
    """Result of a single statistical test"""
    test_id: str
    test_name: str
    p_value: float
    statistic: float
    is_significant: bool
    corrected_p_value: float
    correction_method: str


class MultipleTestingCorrector:
    """
    Multiple Testing Correction
    
    Corrects p-values for multiple comparisons to avoid false positives.
    
    Methods:
    1. Bonferroni: Simple but conservative
    2. Holm-Bonferroni: Less conservative, step-down
    3. Benjamini-Hochberg (FDR): Controls false discovery rate
    4. Benjamini-Yekutieli: Conservative FDR for dependent tests
    """
    
    def __init__(self, method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
                 alpha: float = 0.05):
        self.method = method
        self.alpha = alpha
        self.results: List[TestResult] = []
    
    def correct(self, test_results: List[Tuple[str, str, float, float]]) -> List[TestResult]:
        """
        Apply multiple testing correction to test results.
        
        Args:
            test_results: List of (test_id, test_name, p_value, statistic)
        
        Returns:
            List of TestResult with corrected p-values
        """
        n_tests = len(test_results)
        
        if n_tests == 0:
            return []
        
        # Extract p-values
        p_values = np.array([r[2] for r in test_results])
        
        # Apply correction
        if self.method == CorrectionMethod.BONFERRONI:
            corrected_p = self._bonferroni(p_values)
        elif self.method == CorrectionMethod.HOLM_BONFERRONI:
            corrected_p = self._holm_bonferroni(p_values)
        elif self.method == CorrectionMethod.BENJAMINI_HOCHBERG:
            corrected_p = self._benjamini_hochberg(p_values)
        elif self.method == CorrectionMethod.BENJAMINI_YEKUTIELI:
            corrected_p = self._benjamini_yekutieli(p_values)
        else:
            corrected_p = p_values
        
        # Create result objects
        self.results = []
        for i, (test_id, test_name, p_value, statistic) in enumerate(test_results):
            is_significant = corrected_p[i] < self.alpha
            
            result = TestResult(
                test_id=test_id,
                test_name=test_name,
                p_value=p_value,
                statistic=statistic,
                is_significant=is_significant,
                corrected_p_value=corrected_p[i],
                correction_method=self.method.value
            )
            self.results.append(result)
        
        return self.results
    
    def _bonferroni(self, p_values: np.ndarray) -> np.ndarray:
        """Bonferroni correction"""
        n = len(p_values)
        return np.minimum(p_values * n, 1.0)
    
    def _holm_bonferroni(self, p_values: np.ndarray) -> np.ndarray:
        """Holm-Bonferroni step-down correction"""
        n = len(p_values)
        
        # Sort p-values with indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Step-down correction
        corrected_p = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected_p[i] = min(p * (n - i), 1.0)
        
        # Ensure monotonicity
        for i in range(1, len(corrected_p)):
            corrected_p[i] = max(corrected_p[i], corrected_p[i-1])
        
        # Unsort
        unsorted_corrected = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            unsorted_corrected[idx] = corrected_p[i]
        
        return unsorted_corrected
    
    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction"""
        n = len(p_values)
        
        # Sort p-values with indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # BH correction
        corrected_p = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected_p[i] = min(p * n / (i + 1), 1.0)
        
        # Ensure monotonicity
        for i in range(len(corrected_p)-2, -1, -1):
            corrected_p[i] = min(corrected_p[i], corrected_p[i+1])
        
        # Unsort
        unsorted_corrected = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            unsorted_corrected[idx] = corrected_p[i]
        
        return unsorted_corrected
    
    def _benjamini_yekutieli(self, p_values: np.ndarray) -> np.ndarray:
        """Benjamini-Yekutieli FDR correction (conservative)"""
        n = len(p_values)
        
        # Calculate harmonic number
        harmonic = sum(1.0 / i for i in range(1, n + 1))
        
        # Sort p-values with indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # BY correction
        corrected_p = np.zeros_like(sorted_p)
        for i, p in enumerate(sorted_p):
            corrected_p[i] = min(p * n * harmonic / (i + 1), 1.0)
        
        # Ensure monotonicity
        for i in range(len(corrected_p)-2, -1, -1):
            corrected_p[i] = min(corrected_p[i], corrected_p[i+1])
        
        # Unsort
        unsorted_corrected = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            unsorted_corrected[idx] = corrected_p[i]
        
        return unsorted_corrected
    
    def get_significant_tests(self) -> List[TestResult]:
        """Get tests that are significant after correction"""
        return [r for r in self.results if r.is_significant]
    
    def get_false_discovery_rate(self) -> float:
        """Estimate false discovery rate"""
        if not self.results:
            return 0.0
        
        significant = self.get_significant_tests()
        if not significant:
            return 0.0
        
        # Estimate FDR as mean of significant p-values
        return np.mean([r.corrected_p_value for r in significant])
    
    def generate_report(self) -> str:
        """Generate correction report"""
        if not self.results:
            return "No test results available"
        
        total = len(self.results)
        significant = len(self.get_significant_tests())
        fdr = self.get_false_discovery_rate()
        
        report = f"""
Multiple Testing Correction Report
{'=' * 50}
Correction Method: {self.method.value}
Alpha: {self.alpha}
Total Tests: {total}
Significant (after correction): {significant} ({significant/total*100:.1f}%)
Estimated FDR: {fdr:.3f}

Significant Tests:
{'-' * 50}
"""
        
        for result in self.get_significant_tests():
            report += f"- {result.test_name}: p={result.p_value:.4f}, "
            report += f"corrected_p={result.corrected_p_value:.4f}\n"
        
        report += f"\nAll Tests:\n{'-' * 50}\n"
        for result in self.results:
            status = "SIGNIFICANT" if result.is_significant else "not significant"
            report += f"- {result.test_name}: p={result.p_value:.4f}, "
            report += f"corrected_p={result.corrected_p_value:.4f} [{status}]\n"
        
        return report


def correct_strategy_sharpe(sharpe_values: Dict[str, float], 
                            method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
                            alpha: float = 0.05) -> Dict[str, bool]:
    """
    Correct multiple strategy Sharpe ratio tests.
    
    Args:
        sharpe_values: Dictionary of strategy_name -> Sharpe ratio
        method: Correction method
        alpha: Significance level
    
    Returns:
        Dictionary of strategy_name -> is_significant
    """
    # Convert Sharpe to p-values (assuming normal distribution)
    p_values = {}
    for name, sharpe in sharpe_values.items():
        # Approximate p-value from Sharpe (simplified)
        p_values[name] = 1 - stats.norm.cdf(sharpe)
    
    # Create test results
    test_results = [(name, name, p_values[name], sharpe) for name, sharpe in sharpe_values.items()]
    
    # Apply correction
    corrector = MultipleTestingCorrector(method=method, alpha=alpha)
    results = corrector.correct(test_results)
    
    # Return significance
    return {r.test_name: r.is_significant for r in results}


if __name__ == "__main__":
    # Example usage
    # Simulate 100 strategy tests
    np.random.seed(42)
    n_tests = 100
    
    # 5 truly significant strategies (Sharpe > 2)
    sharpe_values = {}
    for i in range(n_tests):
        if i < 5:
            sharpe_values[f"strategy_{i}"] = np.random.uniform(2.0, 3.0)
        else:
            sharpe_values[f"strategy_{i}"] = np.random.uniform(-0.5, 1.0)
    
    # Apply correction
    print("Applying Benjamini-Hochberg FDR correction...")
    significant = correct_strategy_sharpe(sharpe_values, 
                                          method=CorrectionMethod.BENJAMINI_HOCHBERG)
    
    print(f"\nSignificant strategies: {sum(significant.values())}/{len(significant)}")
    
    # Test with Bonferroni
    print("\nApplying Bonferroni correction...")
    significant_bonf = correct_strategy_sharpe(sharpe_values,
                                              method=CorrectionMethod.BONFERRONI)
    
    print(f"Significant strategies: {sum(significant_bonf.values())}/{len(significant_bonf)}")
    
    # Full report
    corrector = MultipleTestingCorrector(method=CorrectionMethod.BENJAMINI_HOCHBERG)
    test_results = [(name, name, 1 - stats.norm.cdf(sharpe), sharpe) 
                   for name, sharpe in sharpe_values.items()]
    corrector.correct(test_results)
    print(corrector.generate_report())

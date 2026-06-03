"""
Adversarial Validation
Try to break alphas by testing under adverse conditions.

Critical for institutional-grade robustness testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class AdversarialScenario(Enum):
    """Types of adversarial scenarios"""
    REGIME_SHUFFLE = "regime_shuffle"
    NOISE_INJECTION = "noise_injection"
    WORST_PERIOD = "worst_period"
    FEATURE_CORRUPTION = "feature_corruption"
    LABEL_SHUFFLE = "label_shuffle"
    TRANSACTION_COST_SPIKE = "transaction_cost_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class AdversarialTestResult:
    """Result of adversarial test"""
    scenario: AdversarialScenario
    original_sharpe: float
    adversarial_sharpe: float
    sharpe_decline: float
    passed: bool
    threshold: float
    description: str


class AdversarialValidator:
    """
    Adversarial Validator
    
    Tests alpha strategies under adverse conditions to ensure robustness.
    
    Scenarios:
    1. Regime shuffle: Test if alpha works in wrong regime
    2. Noise injection: Add noise to features
    3. Worst period: Test on historically worst period
    4. Feature corruption: Corrupt some features
    5. Label shuffle: Shuffle labels (should fail)
    6. Transaction cost spike: Increase costs 2x
    7. Liquidity crisis: Simulate illiquidity
    """
    
    def __init__(self, sharpe_decline_threshold: float = 0.5):
        self.sharpe_decline_threshold = sharpe_decline_threshold
        self.results: List[AdversarialTestResult] = []
        self.backtest_func: Optional[Callable] = None
    
    def set_backtest_function(self, func: Callable):
        """Set backtesting function"""
        self.backtest_func = func
    
    def validate(self, alpha_name: str, features: pd.DataFrame, 
                returns: pd.Series, original_sharpe: float) -> List[AdversarialTestResult]:
        """
        Run all adversarial scenarios on alpha.
        
        Args:
            alpha_name: Name of alpha strategy
            features: Feature DataFrame
            returns: Returns series
            original_sharpe: Original Sharpe ratio
        
        Returns:
            List of adversarial test results
        """
        self.results = []
        
        # Run all scenarios
        self._test_regime_shuffle(alpha_name, features, returns, original_sharpe)
        self._test_noise_injection(alpha_name, features, returns, original_sharpe)
        self._test_worst_period(alpha_name, features, returns, original_sharpe)
        self._test_feature_corruption(alpha_name, features, returns, original_sharpe)
        self._test_label_shuffle(alpha_name, features, returns, original_sharpe)
        self._test_transaction_cost_spike(alpha_name, features, returns, original_sharpe)
        self._test_liquidity_crisis(alpha_name, features, returns, original_sharpe)
        
        return self.results
    
    def _test_regime_shuffle(self, alpha_name: str, features: pd.DataFrame,
                           returns: pd.Series, original_sharpe: float):
        """Test alpha with shuffled regime labels"""
        # Shuffle returns to simulate wrong regime
        shuffled_returns = returns.sample(frac=1).reset_index(drop=True)
        
        adversarial_sharpe = self._run_backtest(features, shuffled_returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        passed = sharpe_decline < self.sharpe_decline_threshold
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.REGIME_SHUFFLE,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=self.sharpe_decline_threshold,
            description=f"Sharpe declined by {sharpe_decline:.1%} under regime shuffle"
        )
        
        self.results.append(result)
    
    def _test_noise_injection(self, alpha_name: str, features: pd.DataFrame,
                             returns: pd.Series, original_sharpe: float):
        """Test alpha with noisy features"""
        # Add Gaussian noise to features
        noisy_features = features + np.random.normal(0, 0.1, features.shape)
        
        adversarial_sharpe = self._run_backtest(noisy_features, returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        passed = sharpe_decline < self.sharpe_decline_threshold
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.NOISE_INJECTION,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=self.sharpe_decline_threshold,
            description=f"Sharpe declined by {sharpe_decline:.1%} with noisy features"
        )
        
        self.results.append(result)
    
    def _test_worst_period(self, alpha_name: str, features: pd.DataFrame,
                          returns: pd.Series, original_sharpe: float):
        """Test alpha on historically worst period"""
        # Find worst 20% of returns
        worst_threshold = returns.quantile(0.2)
        worst_mask = returns <= worst_threshold
        
        if worst_mask.sum() < 50:  # Need at least 50 points
            return
        
        worst_features = features[worst_mask]
        worst_returns = returns[worst_mask]
        
        adversarial_sharpe = self._run_backtest(worst_features, worst_returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        # Allow more decline for worst period
        passed = sharpe_decline < self.sharpe_decline_threshold * 1.5
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.WORST_PERIOD,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=self.sharpe_decline_threshold * 1.5,
            description=f"Sharpe declined by {sharpe_decline:.1%} on worst period"
        )
        
        self.results.append(result)
    
    def _test_feature_corruption(self, alpha_name: str, features: pd.DataFrame,
                                returns: pd.Series, original_sharpe: float):
        """Test alpha with corrupted features"""
        # Corrupt 10% of features
        corrupted_features = features.copy()
        n_corrupt = int(len(features.columns) * 0.1)
        
        for col in np.random.choice(features.columns, n_corrupt, replace=False):
            corrupted_features[col] = np.random.randn(len(features))
        
        adversarial_sharpe = self._run_backtest(corrupted_features, returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        passed = sharpe_decline < self.sharpe_decline_threshold
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.FEATURE_CORRUPTION,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=self.sharpe_decline_threshold,
            description=f"Sharpe declined by {sharpe_decline:.1%} with corrupted features"
        )
        
        self.results.append(result)
    
    def _test_label_shuffle(self, alpha_name: str, features: pd.DataFrame,
                           returns: pd.Series, original_sharpe: float):
        """Test alpha with shuffled labels (should fail)"""
        # Shuffle returns - alpha should fail
        shuffled_returns = returns.sample(frac=1).reset_index(drop=True)
        
        adversarial_sharpe = self._run_backtest(features, shuffled_returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        # This SHOULD fail (high decline expected)
        passed = sharpe_decline > 0.5  # Should decline significantly
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.LABEL_SHUFFLE,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=0.5,
            description=f"Sharpe declined by {sharpe_decline:.1%} with shuffled labels (expected to fail)"
        )
        
        self.results.append(result)
    
    def _test_transaction_cost_spike(self, alpha_name: str, features: pd.DataFrame,
                                     returns: pd.Series, original_sharpe: float):
        """Test alpha with 2x transaction costs"""
        # Simulate 2x costs by reducing returns
        cost_adjusted_returns = returns * 0.98  # 2% additional cost
        
        adversarial_sharpe = self._run_backtest(features, cost_adjusted_returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        passed = sharpe_decline < self.sharpe_decline_threshold
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.TRANSACTION_COST_SPIKE,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=self.sharpe_decline_threshold,
            description=f"Sharpe declined by {sharpe_decline:.1%} with 2x transaction costs"
        )
        
        self.results.append(result)
    
    def _test_liquidity_crisis(self, alpha_name: str, features: pd.DataFrame,
                              returns: pd.Series, original_sharpe: float):
        """Test alpha under liquidity crisis"""
        # Simulate liquidity crisis by reducing returns (wider spreads)
        crisis_returns = returns * 0.95  # 5% additional slippage
        
        adversarial_sharpe = self._run_backtest(features, crisis_returns)
        sharpe_decline = (original_sharpe - adversarial_sharpe) / original_sharpe
        
        passed = sharpe_decline < self.sharpe_decline_threshold
        
        result = AdversarialTestResult(
            scenario=AdversarialScenario.LIQUIDITY_CRISIS,
            original_sharpe=original_sharpe,
            adversarial_sharpe=adversarial_sharpe,
            sharpe_decline=sharpe_decline,
            passed=passed,
            threshold=self.sharpe_decline_threshold,
            description=f"Sharpe declined by {sharpe_decline:.1%} under liquidity crisis"
        )
        
        self.results.append(result)
    
    def _run_backtest(self, features: pd.DataFrame, returns: pd.Series) -> float:
        """Run backtest and return Sharpe"""
        if self.backtest_func:
            return self.backtest_func(features, returns)
        
        # Placeholder: calculate Sharpe from returns
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe
    
    def is_alpha_robust(self) -> bool:
        """Check if alpha passes adversarial validation"""
        # Must pass at least 5 out of 7 scenarios
        passed_count = sum(1 for r in self.results if r.passed)
        return passed_count >= 5
    
    def get_failed_scenarios(self) -> List[AdversarialTestResult]:
        """Get scenarios that failed"""
        return [r for r in self.results if not r.passed]
    
    def generate_report(self) -> str:
        """Generate adversarial validation report"""
        if not self.results:
            return "No adversarial test results available"
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        robust = self.is_alpha_robust()
        
        report = f"""
Adversarial Validation Report
{'=' * 50}
Total Scenarios: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Robust: {robust}

Scenario Results:
{'-' * 50}
"""
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            report += f"{result.scenario.value}: {status}\n"
            report += f"  Original Sharpe: {result.original_sharpe:.3f}\n"
            report += f"  Adversarial Sharpe: {result.adversarial_sharpe:.3f}\n"
            report += f"  Decline: {result.sharpe_decline:.1%}\n"
            report += f"  {result.description}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    features = pd.DataFrame(np.random.randn(n, 20), columns=[f"feature_{i}" for i in range(20)])
    returns = pd.Series(np.random.randn(n) * 0.01)
    
    validator = AdversarialValidator(sharpe_decline_threshold=0.5)
    
    # Calculate original Sharpe
    original_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    # Run adversarial validation
    results = validator.validate("test_alpha", features, returns, original_sharpe)
    
    print(validator.generate_report())
    
    print(f"\nAlpha Robust: {validator.is_alpha_robust()}")
    print(f"Failed Scenarios: {len(validator.get_failed_scenarios())}")

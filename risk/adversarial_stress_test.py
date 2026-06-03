"""
Adversarial Stress Testing
Run crash scenarios (2008, COVID, 2022 rate hike) on live portfolio.

Critical for institutional risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class StressScenario(Enum):
    """Stress test scenarios"""
    CRASH_2008 = "crash_2008"
    COVID_2020 = "covid_2020"
    RATE_HIKE_2022 = "rate_hike_2022"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FACTOR_CROWDING = "factor_crowding"
    CUSTOM = "custom"


@dataclass
class StressTestResult:
    """Result of stress test"""
    scenario: StressScenario
    initial_value: float
    final_value: float
    drawdown: float
    sharpe: float
    max_drawdown: float
    recovery_days: int
    passed: bool
    threshold: float


class AdversarialStressTester:
    """
    Adversarial Stress Tester
    
    Runs adversarial stress scenarios on portfolio:
    - 2008 crash: -50% market drop
    - COVID 2020: -30% drop, high volatility
    - 2022 rate hike: -20% drop, rising rates
    - Liquidity crisis: 10x spread widening
    - Factor crowding: Factor reversal
    
    Rules:
    - Portfolio must survive all scenarios
    - Max drawdown < 50%
    - Recovery within 90 days
    """
    
    def __init__(self, max_drawdown_threshold: float = 0.50,
                 recovery_days_threshold: int = 90):
        self.max_drawdown_threshold = max_drawdown_threshold
        self.recovery_days_threshold = recovery_days_threshold
        
        self.scenario_definitions = {
            StressScenario.CRASH_2008: {
                "market_drop": -0.50,
                "volatility_spike": 3.0,
                "duration_days": 126,
                "spread_widening": 5.0
            },
            StressScenario.COVID_2020: {
                "market_drop": -0.30,
                "volatility_spike": 4.0,
                "duration_days": 33,
                "spread_widening": 10.0
            },
            StressScenario.RATE_HIKE_2022: {
                "market_drop": -0.20,
                "volatility_spike": 2.0,
                "duration_days": 180,
                "spread_widening": 3.0
            },
            StressScenario.LIQUIDITY_CRISIS: {
                "market_drop": -0.15,
                "volatility_spike": 2.5,
                "duration_days": 60,
                "spread_widening": 10.0
            },
            StressScenario.FACTOR_CROWDING: {
                "market_drop": -0.10,
                "volatility_spike": 1.5,
                "duration_days": 45,
                "spread_widening": 2.0
            }
        }
        
        self.test_results: List[StressTestResult] = []
    
    def run_scenario(self, scenario: StressScenario, portfolio_value: float,
                    portfolio_beta: float = 1.0, portfolio_volatility: float = 0.15,
                    custom_params: Optional[Dict] = None) -> StressTestResult:
        """
        Run stress test scenario.
        
        Args:
            scenario: Stress scenario to run
            portfolio_value: Initial portfolio value
            portfolio_beta: Portfolio beta to market
            portfolio_volatility: Portfolio volatility
            custom_params: Custom parameters for custom scenario
        
        Returns:
            StressTestResult
        """
        # Get scenario parameters
        if scenario == StressScenario.CUSTOM:
            params = custom_params if custom_params else {}
        else:
            params = self.scenario_definitions.get(scenario, {})
        
        market_drop = params.get("market_drop", -0.20)
        vol_spike = params.get("volatility_spike", 2.0)
        duration_days = params.get("duration_days", 60)
        spread_widening = params.get("spread_widening", 3.0)
        
        # Simulate scenario
        initial_value = portfolio_value
        
        # Market impact
        market_impact = market_drop * portfolio_beta
        
        # Volatility impact (additional drawdown from high vol)
        vol_impact = -portfolio_volatility * vol_spike * np.sqrt(duration_days / 252)
        
        # Spread impact (execution costs)
        spread_impact = -0.01 * spread_widening  # 1% per spread widening
        
        # Total drawdown
        total_drawdown = market_impact + vol_impact + spread_impact
        final_value = initial_value * (1 + total_drawdown)
        
        # Calculate metrics
        drawdown = abs(total_drawdown)
        sharpe = total_drawdown / (np.sqrt(duration_days / 252) * portfolio_volatility) if portfolio_volatility > 0 else 0
        max_drawdown = drawdown  # Simplified
        
        # Recovery days (simplified: proportional to drawdown)
        recovery_days = int(duration_days * (drawdown / 0.3)) if drawdown > 0 else 0
        
        # Check if passed
        passed = (max_drawdown < self.max_drawdown_threshold and
                 recovery_days < self.recovery_days_threshold)
        
        result = StressTestResult(
            scenario=scenario,
            initial_value=initial_value,
            final_value=final_value,
            drawdown=drawdown,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            recovery_days=recovery_days,
            passed=passed,
            threshold=self.max_drawdown_threshold
        )
        
        self.test_results.append(result)
        
        return result
    
    def run_all_scenarios(self, portfolio_value: float, portfolio_beta: float = 1.0,
                         portfolio_volatility: float = 0.15) -> List[StressTestResult]:
        """
        Run all stress test scenarios.
        
        Args:
            portfolio_value: Initial portfolio value
            portfolio_beta: Portfolio beta
            portfolio_volatility: Portfolio volatility
        
        Returns:
            List of stress test results
        """
        results = []
        
        for scenario in StressScenario:
            if scenario == StressScenario.CUSTOM:
                continue
            
            result = self.run_scenario(scenario, portfolio_value, portfolio_beta, portfolio_volatility)
            results.append(result)
        
        return results
    
    def get_worst_scenario(self) -> Optional[StressTestResult]:
        """Get worst performing scenario"""
        if not self.test_results:
            return None
        
        return max(self.test_results, key=lambda r: r.drawdown)
    
    def get_failed_scenarios(self) -> List[StressTestResult]:
        """Get scenarios that failed stress test"""
        return [r for r in self.test_results if not r.passed]
    
    def is_portfolio_resilient(self) -> bool:
        """Check if portfolio passes all stress tests"""
        return all(r.passed for r in self.test_results)
    
    def generate_report(self) -> str:
        """Generate stress test report"""
        if not self.test_results:
            return "No stress test results available"
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = total - passed
        resilient = self.is_portfolio_resilient()
        
        report = f"""
Adversarial Stress Test Report
{'=' * 50}
Max Drawdown Threshold: {self.max_drawdown_threshold:.1%}
Recovery Days Threshold: {self.recovery_days_threshold}
Total Scenarios: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Failed: {failed} ({failed/total*100:.1f}%)
Portfolio Resilient: {resilient}

Scenario Results:
{'-' * 50}
"""
        
        for result in self.test_results:
            status = "PASS" if result.passed else "FAIL"
            report += f"{result.scenario.value}: {status}\n"
            report += f"  Initial: {result.initial_value:,.0f}\n"
            report += f"  Final: {result.final_value:,.0f}\n"
            report += f"  Drawdown: {result.drawdown:.1%}\n"
            report += f"  Max Drawdown: {result.max_drawdown:.1%}\n"
            report += f"  Recovery Days: {result.recovery_days}\n"
            report += f"  Sharpe: {result.sharpe:.2f}\n\n"
        
        worst = self.get_worst_scenario()
        if worst:
            report += f"Worst Scenario: {worst.scenario.value} ({worst.drawdown:.1%} drawdown)\n"
        
        failed = self.get_failed_scenarios()
        if failed:
            report += f"\nFailed Scenarios:\n{'-' * 50}\n"
            for f in failed:
                report += f"- {f.scenario.value}: {f.drawdown:.1%} drawdown (threshold: {f.threshold:.1%})\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    tester = AdversarialStressTester(max_drawdown_threshold=0.50, recovery_days_threshold=90)
    
    # Run all scenarios
    print("Running adversarial stress tests...")
    results = tester.run_all_scenarios(portfolio_value=1000000000, portfolio_beta=1.0, portfolio_volatility=0.15)
    
    print(f"\nRan {len(results)} stress scenarios")
    print(tester.generate_report())
    
    # Test custom scenario
    print("\nTesting custom scenario...")
    custom_result = tester.run_scenario(
        scenario=StressScenario.CUSTOM,
        portfolio_value=1000000000,
        portfolio_beta=1.2,
        portfolio_volatility=0.20,
        custom_params={
            "market_drop": -0.40,
            "volatility_spike": 3.5,
            "duration_days": 90,
            "spread_widening": 8.0
        }
    )
    
    print(f"Custom scenario: {custom_result.drawdown:.1%} drawdown, {'PASS' if custom_result.passed else 'FAIL'}")

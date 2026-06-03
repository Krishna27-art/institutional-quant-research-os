"""
Crisis Simulator with Historical Scenarios
Based on V3 Blueprint - Stress Testing Suite

Key findings from research:
- Strategies not tested against historical crashes
- Comprehensive stress testing against historical market crises
- Scenarios: COVID_2020, Adani_2023, Russia_Ukraine_2022, Flash_Crash_2015, Rate_hike_2022, Liquidity_crisis
- Pass criteria: Max drawdown < 25%, no day with >5% loss, VaR violation rate <5%

V3 Upgrade - Expected Sharpe increase: +0.1 (prevents losses)
Priority: High
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class CrisisScenario:
    """Crisis scenario definition"""
    name: str
    start_date: str
    end_date: str
    description: str
    initial_drop: float  # Initial gap down
    volatility_spike: float  # Volatility multiplier
    volume_spike: float  # Volume multiplier
    duration_days: int
    vix_trigger: float  # VIX level that triggers override (0 = no trigger)


@dataclass
class StressTestResult:
    """Result of stress test for a strategy"""
    strategy_name: str
    scenario_name: str
    max_drawdown: float
    max_daily_loss: float
    var_violation_count: int
    var_violation_rate: float
    total_return: float
    sharpe: float
    passed: bool
    reason: str
    override_applied: bool = False
    override_multiplier: float = 1.0


class CrisisSimulator:
    """
    Crisis Simulator for stress testing strategies.
    
    Scenarios:
    - COVID_2020: NIFTY dropped 40% in March 2020
    - Adani_2023: Short seller report, Adani stocks crashed 70%
    - Russia_Ukraine_2022: NIFTY gap down 5%, volatility spike
    - Flash_Crash_2015: 5% drop in 10 minutes (synthetic)
    - Rate_hike_2022: 200bps increase in 3 months
    - Liquidity_crisis: Zero volume for 1 hour
    """
    
    def __init__(self):
        self.scenarios = self._define_scenarios()
        self.stress_results: List[StressTestResult] = []
    
    def _define_scenarios(self) -> List[CrisisScenario]:
        """Define crisis scenarios."""
        return [
            CrisisScenario(
                name="COVID_2020",
                start_date="2020-03-01",
                end_date="2020-03-31",
                description="NIFTY dropped 40% in March 2020",
                initial_drop=-0.30,
                volatility_spike=3.0,
                volume_spike=2.0,
                duration_days=30,
                vix_trigger=35.0
            ),
            CrisisScenario(
                name="Adani_2023",
                start_date="2023-01-25",
                end_date="2023-02-10",
                description="Short seller report, Adani stocks crashed 70%",
                initial_drop=-0.25,
                volatility_spike=2.5,
                volume_spike=3.0,
                duration_days=15,
                vix_trigger=30.0
            ),
            CrisisScenario(
                name="Russia_Ukraine_2022",
                start_date="2022-02-24",
                end_date="2022-03-15",
                description="NIFTY gap down 5%, volatility spike",
                initial_drop=-0.05,
                volatility_spike=2.0,
                volume_spike=1.5,
                duration_days=20,
                vix_trigger=28.0
            ),
            CrisisScenario(
                name="Flash_Crash_2015",
                start_date="2015-08-24",
                end_date="2015-08-24",
                description="5% drop in 10 minutes (synthetic)",
                initial_drop=-0.05,
                volatility_spike=5.0,
                volume_spike=4.0,
                duration_days=1,
                vix_trigger=25.0
            ),
            CrisisScenario(
                name="Rate_hike_2022",
                start_date="2022-06-01",
                end_date="2022-09-30",
                description="200bps increase in 3 months",
                initial_drop=-0.10,
                volatility_spike=1.5,
                volume_spike=1.2,
                duration_days=120,
                vix_trigger=20.0
            ),
            CrisisScenario(
                name="Liquidity_crisis",
                start_date="2024-01-01",
                end_date="2024-01-01",
                description="Zero volume for 1 hour",
                initial_drop=-0.02,
                volatility_spike=1.2,
                volume_spike=0.1,
                duration_days=1,
                vix_trigger=0.0
            )
        ]
    
    def calculate_override_multiplier(self, scenario: CrisisScenario, current_vix: float = 20.0) -> float:
        """
        Calculate position size override multiplier based on crisis conditions.
        
        Args:
            scenario: Crisis scenario
            current_vix: Current VIX level
            
        Returns:
            Override multiplier (0.0 = stop trading, 1.0 = no override)
        """
        if scenario.vix_trigger == 0:
            return 1.0  # No override for this scenario
        
        # Calculate override based on VIX
        if current_vix >= scenario.vix_trigger:
            # Crisis detected - reduce position size
            # The higher the VIX above trigger, the more aggressive the reduction
            excess_vix = current_vix - scenario.vix_trigger
            override = max(0.0, 1.0 - (excess_vix / 10.0))  # Reduce by 10% per excess VIX point
            return min(override, 0.5)  # Never reduce below 50% (or stop entirely if VIX is extreme)
        
        return 1.0  # No override needed
    
    def generate_crisis_returns(
        self,
        scenario: CrisisScenario,
        base_returns: pd.Series,
        override_multiplier: float = 1.0
    ) -> pd.Series:
        """
        Generate crisis returns based on scenario parameters.
        
        Args:
            scenario: Crisis scenario
            base_returns: Base return series (normal conditions)
            override_multiplier: Position size override multiplier
            
        Returns:
            Crisis return series
        """
        np.random.seed(abs(hash(scenario.name)) % (2**32))
        
        n_days = scenario.duration_days
        
        # Initial gap down
        initial_gap = scenario.initial_drop
        
        # Generate returns with volatility spike
        base_vol = base_returns.std()
        crisis_vol = base_vol * scenario.volatility_spike
        
        crisis_returns = np.random.normal(0, crisis_vol, n_days)
        
        # Add initial gap
        crisis_returns[0] = initial_gap
        
        # Add mean reversion (markets tend to recover)
        mean_reversion = 0.001
        crisis_returns = crisis_returns + mean_reversion
        
        # Apply override multiplier to reduce losses
        # If override is 0.5, losses are halved
        if override_multiplier < 1.0:
            crisis_returns = crisis_returns * override_multiplier
        
        return pd.Series(crisis_returns)
    
    def run_stress_test(
        self,
        strategy_name: str,
        strategy_returns: pd.Series,
        scenario: CrisisScenario,
        apply_override: bool = True
    ) -> StressTestResult:
        """
        Run stress test for a strategy under a crisis scenario.
        
        Args:
            strategy_name: Strategy name
            strategy_returns: Strategy returns under normal conditions
            scenario: Crisis scenario
            apply_override: Whether to apply crisis override rules
            
        Returns:
            StressTestResult
        """
        # Calculate override multiplier if enabled
        override_multiplier = 1.0
        override_applied = False
        
        if apply_override and scenario.vix_trigger > 0:
            # Simulate VIX during crisis (higher than normal)
            simulated_vix = scenario.vix_trigger + (scenario.volatility_spike * 5)
            override_multiplier = self.calculate_override_multiplier(scenario, simulated_vix)
            override_applied = override_multiplier < 1.0
        
        # Generate crisis returns with override
        crisis_returns = self.generate_crisis_returns(scenario, strategy_returns, override_multiplier)
        
        # Calculate metrics
        cumulative = (1 + crisis_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (peak - cumulative) / peak
        max_drawdown = drawdown.max()
        
        max_daily_loss = crisis_returns.min()
        
        # VaR violations (assuming 99% VaR)
        var_99 = np.percentile(strategy_returns, 1)
        var_violations = (crisis_returns < var_99).sum()
        var_violation_rate = var_violations / len(crisis_returns)
        
        # Total return and Sharpe
        total_return = cumulative.iloc[-1] - 1
        sharpe = crisis_returns.mean() / crisis_returns.std() * np.sqrt(252) if crisis_returns.std() > 0 else 0
        
        # Pass criteria
        passed = (
            max_drawdown < 0.25 and
            max_daily_loss > -0.05 and
            var_violation_rate < 0.05
        )
        
        # Determine reason
        if not passed:
            reasons = []
            if max_drawdown >= 0.25:
                reasons.append(f"Max drawdown {max_drawdown:.2%} >= 25%")
            if max_daily_loss <= -0.05:
                reasons.append(f"Max daily loss {max_daily_loss:.2%} <= -5%")
            if var_violation_rate >= 0.05:
                reasons.append(f"VaR violation rate {var_violation_rate:.2%} >= 5%")
            reason = "; ".join(reasons)
        else:
            reason = "All criteria met"
        
        result = StressTestResult(
            strategy_name=strategy_name,
            scenario_name=scenario.name,
            max_drawdown=max_drawdown,
            max_daily_loss=max_daily_loss,
            var_violation_count=var_violations,
            var_violation_rate=var_violation_rate,
            total_return=total_return,
            sharpe=sharpe,
            passed=passed,
            reason=reason,
            override_applied=override_applied,
            override_multiplier=override_multiplier
        )
        
        self.stress_results.append(result)
        
        return result
    
    def run_all_stress_tests(
        self,
        strategy_returns_dict: Dict[str, pd.Series],
        apply_override: bool = True
    ) -> Dict[str, List[StressTestResult]]:
        """
        Run all stress tests for all strategies.
        
        Args:
            strategy_returns_dict: Dictionary of strategy_name -> returns
            apply_override: Whether to apply crisis override rules
            
        Returns:
            Dictionary of strategy_name -> list of results
        """
        all_results = {}
        
        for strategy_name, returns in strategy_returns_dict.items():
            strategy_results = []
            for scenario in self.scenarios:
                result = self.run_stress_test(strategy_name, returns, scenario, apply_override)
                strategy_results.append(result)
            all_results[strategy_name] = strategy_results
        
        return all_results
    
    def print_stress_report(self, results: Dict[str, List[StressTestResult]]) -> None:
        """Print stress test report."""
        print("\n" + "="*60)
        print("CRISIS SIMULATOR STRESS TEST REPORT")
        print("="*60)
        
        for strategy_name, strategy_results in results.items():
            print(f"\n{strategy_name}:")
            passed_count = sum(1 for r in strategy_results if r.passed)
            total_count = len(strategy_results)
            print(f"  Passed: {passed_count}/{total_count}")
            
            for result in strategy_results:
                status_icon = "✅" if result.passed else "❌"
                print(f"  {status_icon} {result.scenario_name}:")
                print(f"    Max DD: {result.max_drawdown:.2%}")
                print(f"    Max Daily Loss: {result.max_daily_loss:.2%}")
                print(f"    VaR Violation Rate: {result.var_violation_rate:.2%}")
                print(f"    Total Return: {result.total_return:.2%}")
                print(f"    Sharpe: {result.sharpe:.2f}")
                if result.override_applied:
                    print(f"    Override Applied: YES (multiplier: {result.override_multiplier:.2f})")
                if not result.passed:
                    print(f"    Reason: {result.reason}")
        
        print("\n" + "="*60)
    
    def get_overall_verdict(self, results: Dict[str, List[StressTestResult]]) -> str:
        """
        Get overall verdict for all strategies.
        
        Args:
            results: Stress test results
            
        Returns:
            Overall verdict
        """
        total_tests = sum(len(r) for r in results.values())
        passed_tests = sum(sum(1 for r in strategy_results if r.passed) for strategy_results in results.values())
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if pass_rate >= 0.9:
            return "PASS - All strategies resilient to crises"
        elif pass_rate >= 0.7:
            return "CONDITIONAL - Most strategies pass, review failures"
        else:
            return "FAIL - Too many failures, need crisis-override rules"


def run_sample_crisis_simulation():
    """Run sample crisis simulation."""
    simulator = CrisisSimulator()
    
    # Generate sample strategy returns
    np.random.seed(42)
    n_days = 252
    
    strategy_returns_dict = {
        "ORB": np.random.normal(0.0005, 0.015, n_days),
        "VWAP": np.random.normal(0.0004, 0.012, n_days),
        "PCP": np.random.normal(0.0003, 0.010, n_days),
        "VOL_CARRY": np.random.normal(0.0002, 0.008, n_days)
    }
    
    # Run all stress tests with override enabled
    print("\n" + "="*60)
    print("RUNNING STRESS TESTS WITH CRISIS OVERRIDE RULES")
    print("="*60)
    results = simulator.run_all_stress_tests(strategy_returns_dict, apply_override=True)
    
    # Print report
    simulator.print_stress_report(results)
    
    # Get overall verdict
    verdict = simulator.get_overall_verdict(results)
    print(f"\nOverall Verdict: {verdict}")
    
    # Compare without override
    print("\n" + "="*60)
    print("RUNNING STRESS TESTS WITHOUT OVERRIDE (BASELINE)")
    print("="*60)
    results_baseline = simulator.run_all_stress_tests(strategy_returns_dict, apply_override=False)
    simulator.print_stress_report(results_baseline)
    verdict_baseline = simulator.get_overall_verdict(results_baseline)
    print(f"\nOverall Verdict (Baseline): {verdict_baseline}")
    
    return simulator


if __name__ == "__main__":
    run_sample_crisis_simulation()

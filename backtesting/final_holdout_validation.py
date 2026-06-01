"""
Final Holdout Validation (2024-2025 Data)
Based on Institutional Audit Recommendations

Key findings from audit:
- No truly unseen test set (all data from 2016-2023 is used)
- Performance in paper may not generalize to future
- Need: Reserve last 12 months (2024-2025) as final test

Architecture V2 Upgrade - 90-Day Plan Item #1
Priority: P0 (Critical)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class HoldoutValidationResult:
    """Result of holdout validation for a strategy"""
    strategy_name: str
    training_period: str
    holdout_period: str
    training_sharpe: float
    holdout_sharpe: float
    sharpe_decay: float  # (holdout - training) / training
    training_max_dd: float
    holdout_max_dd: float
    training_win_rate: float
    holdout_win_rate: float
    passed: bool
    reason: str


@dataclass
class ValidationReport:
    """Overall validation report"""
    validation_date: str
    total_strategies: int
    passed_strategies: int
    failed_strategies: int
    results: List[HoldoutValidationResult]
    overall_verdict: str


class FinalHoldoutValidator:
    """
    Final holdout validator for alpha strategies.
    
    Validation Rules:
    - Training data: 2016-2023 (8 years)
    - Holdout data: 2024-2025 (2 years, never used in training)
    - Pass criteria: Holdout Sharpe >= 0.8 * Training Sharpe
    - Pass criteria: Holdout Max DD <= 1.5 * Training Max DD
    - Pass criteria: Holdout Win Rate >= 0.9 * Training Win Rate
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_strategy(
        self,
        strategy_name: str,
        training_returns: pd.Series,
        holdout_returns: pd.Series
    ) -> HoldoutValidationResult:
        """
        Validate a strategy on holdout data.
        
        Args:
            strategy_name: Strategy name
            training_returns: Returns from training period (2016-2023)
            holdout_returns: Returns from holdout period (2024-2025)
            
        Returns:
            HoldoutValidationResult
        """
        # Calculate training metrics
        training_sharpe = self.calculate_sharpe(training_returns)
        training_max_dd = self.calculate_max_drawdown(training_returns)
        training_win_rate = self.calculate_win_rate(training_returns)
        
        # Calculate holdout metrics
        holdout_sharpe = self.calculate_sharpe(holdout_returns)
        holdout_max_dd = self.calculate_max_drawdown(holdout_returns)
        holdout_win_rate = self.calculate_win_rate(holdout_returns)
        
        # Calculate decay
        sharpe_decay = (holdout_sharpe - training_sharpe) / training_sharpe if training_sharpe != 0 else 0
        
        # Pass criteria
        sharpe_pass = holdout_sharpe >= 0.8 * training_sharpe
        dd_pass = holdout_max_dd <= 1.5 * training_max_dd
        win_rate_pass = holdout_win_rate >= 0.9 * training_win_rate
        
        passed = sharpe_pass and dd_pass and win_rate_pass
        
        # Determine reason
        if not passed:
            reasons = []
            if not sharpe_pass:
                reasons.append(f"Sharpe decay: {sharpe_decay:.2%}")
            if not dd_pass:
                reasons.append(f"Drawdown increase: {holdout_max_dd / training_max_dd:.2f}x")
            if not win_rate_pass:
                reasons.append(f"Win rate drop: {holdout_win_rate / training_win_rate:.2f}x")
            reason = "; ".join(reasons)
        else:
            reason = "All criteria met"
        
        result = HoldoutValidationResult(
            strategy_name=strategy_name,
            training_period="2016-2023",
            holdout_period="2024-2025",
            training_sharpe=training_sharpe,
            holdout_sharpe=holdout_sharpe,
            sharpe_decay=sharpe_decay,
            training_max_dd=training_max_dd,
            holdout_max_dd=holdout_max_dd,
            training_win_rate=training_win_rate,
            holdout_win_rate=holdout_win_rate,
            passed=passed,
            reason=reason
        )
        
        self.validation_results.append(result)
        return result
    
    def calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean() * 252  # Annualize
        std_return = returns.std() * np.sqrt(252)  # Annualize
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        return abs(max_dd)
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (positive return days / total days)."""
        if len(returns) == 0:
            return 0.0
        
        wins = (returns > 0).sum()
        total = len(returns)
        
        return wins / total if total > 0 else 0.0
    
    def generate_report(self) -> ValidationReport:
        """Generate overall validation report."""
        total_strategies = len(self.validation_results)
        passed_strategies = sum(1 for r in self.validation_results if r.passed)
        failed_strategies = total_strategies - passed_strategies
        
        # Overall verdict
        if failed_strategies == 0:
            overall_verdict = "PASS - All strategies validated for live trading"
        elif passed_strategies >= total_strategies * 0.8:
            overall_verdict = "CONDITIONAL - Most strategies validated, review failed ones"
        else:
            overall_verdict = "FAIL - Too many strategies failed validation"
        
        return ValidationReport(
            validation_date=datetime.now().isoformat(),
            total_strategies=total_strategies,
            passed_strategies=passed_strategies,
            failed_strategies=failed_strategies,
            results=self.validation_results,
            overall_verdict=overall_verdict
        )
    
    def print_report(self, report: ValidationReport) -> None:
        """Print validation report."""
        print("\n" + "="*60)
        print("FINAL HOLDOUT VALIDATION REPORT (2024-2025)")
        print("="*60)
        print(f"Validation Date: {report.validation_date}")
        print(f"Total Strategies: {report.total_strategies}")
        print(f"Passed: {report.passed_strategies}")
        print(f"Failed: {report.failed_strategies}")
        print(f"\nOverall Verdict: {report.overall_verdict}")
        
        print("\nStrategy Results:")
        for result in report.results:
            status_icon = "✅" if result.passed else "❌"
            print(f"\n{status_icon} {result.strategy_name}")
            print(f"  Training Sharpe: {result.training_sharpe:.2f}")
            print(f"  Holdout Sharpe: {result.holdout_sharpe:.2f}")
            print(f"  Sharpe Decay: {result.sharpe_decay:.2%}")
            print(f"  Training Max DD: {result.training_max_dd:.2%}")
            print(f"  Holdout Max DD: {result.holdout_max_dd:.2%}")
            print(f"  Training Win Rate: {result.training_win_rate:.2%}")
            print(f"  Holdout Win Rate: {result.holdout_win_rate:.2%}")
            print(f"  Reason: {result.reason}")
        
        print("\n" + "="*60)
    
    def to_json(self, report: ValidationReport) -> str:
        """Convert report to JSON."""
        report_dict = {
            "validation_date": report.validation_date,
            "total_strategies": report.total_strategies,
            "passed_strategies": report.passed_strategies,
            "failed_strategies": report.failed_strategies,
            "overall_verdict": report.overall_verdict,
            "results": [
                {
                    "strategy_name": r.strategy_name,
                    "training_period": r.training_period,
                    "holdout_period": r.holdout_period,
                    "training_sharpe": r.training_sharpe,
                    "holdout_sharpe": r.holdout_sharpe,
                    "sharpe_decay": r.sharpe_decay,
                    "training_max_dd": r.training_max_dd,
                    "holdout_max_dd": r.holdout_max_dd,
                    "training_win_rate": r.training_win_rate,
                    "holdout_win_rate": r.holdout_win_rate,
                    "passed": r.passed,
                    "reason": r.reason
                }
                for r in report.results
            ]
        }
        return json.dumps(report_dict, indent=2)


def run_sample_validation():
    """Run sample final holdout validation."""
    validator = FinalHoldoutValidator()
    
    # Generate sample data
    np.random.seed(42)
    
    # Training period (2016-2023) - 8 years
    training_dates = pd.date_range("2016-01-01", "2023-12-31", freq="D")
    training_dates = training_dates[training_dates.dayofweek < 5]  # Weekdays only
    
    # Holdout period (2024-2025) - 2 years
    holdout_dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    holdout_dates = holdout_dates[holdout_dates.dayofweek < 5]  # Weekdays only
    
    # Strategy 1: ORB - Good performance in both periods
    orb_training = pd.Series(np.random.normal(0.0005, 0.015, len(training_dates)), index=training_dates)
    orb_holdout = pd.Series(np.random.normal(0.0004, 0.016, len(holdout_dates)), index=holdout_dates)
    
    # Strategy 2: VWAP - Slight decay in holdout
    vwap_training = pd.Series(np.random.normal(0.0004, 0.012, len(training_dates)), index=training_dates)
    vwap_holdout = pd.Series(np.random.normal(0.0002, 0.013, len(holdout_dates)), index=holdout_dates)
    
    # Strategy 3: PCP - Significant decay in holdout
    pcp_training = pd.Series(np.random.normal(0.0003, 0.010, len(training_dates)), index=training_dates)
    pcp_holdout = pd.Series(np.random.normal(-0.0001, 0.012, len(holdout_dates)), index=holdout_dates)
    
    # Strategy 4: Vol Carry - Good performance
    vol_training = pd.Series(np.random.normal(0.0002, 0.008, len(training_dates)), index=training_dates)
    vol_holdout = pd.Series(np.random.normal(0.00015, 0.009, len(holdout_dates)), index=holdout_dates)
    
    # Strategy 5: Game Theoretic - Failed in holdout
    gt_training = pd.Series(np.random.normal(0.0006, 0.014, len(training_dates)), index=training_dates)
    gt_holdout = pd.Series(np.random.normal(-0.0005, 0.018, len(holdout_dates)), index=holdout_dates)
    
    # Validate each strategy
    validator.validate_strategy("ORB", orb_training, orb_holdout)
    validator.validate_strategy("VWAP", vwap_training, vwap_holdout)
    validator.validate_strategy("PCP", pcp_training, pcp_holdout)
    validator.validate_strategy("VOL_CARRY", vol_training, vol_holdout)
    validator.validate_strategy("GAME_THEORETIC", gt_training, gt_holdout)
    
    # Generate and print report
    report = validator.generate_report()
    validator.print_report(report)
    
    # Export JSON
    print("\nJSON Report:")
    print(validator.to_json(report))
    
    return report


if __name__ == "__main__":
    run_sample_validation()

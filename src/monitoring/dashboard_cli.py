"""
Alpha Performance CLI Dashboard
Prints active signals, rolling ICs, regime breakdowns, and strategy demotion recommendations.
"""

import sys
from datetime import datetime
from typing import Optional

from src.alpha.prediction_registry import get_prediction_registry


class AlphaPerformanceCLIDashboard:
    """CLI Dashboard for monitoring alpha strategies and outcome statistics."""

    def __init__(self, min_ic: float = 0.02) -> None:
        self.registry = get_prediction_registry()
        self.min_ic = min_ic

    def render(self, output_stream=sys.stdout) -> None:
        """Render the performance report to the output stream."""
        summary = self.registry.get_summary()
        reports = self.registry.get_all_reports()

        output_stream.write("\n" + "=" * 70 + "\n")
        output_stream.write("          INSTITUTIONAL QUANT OS: ALPHA PERFORMANCE DASHBOARD\n")
        output_stream.write(f"          Report Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        output_stream.write("=" * 70 + "\n\n")

        # 1. Prediction Statistics Summary
        output_stream.write("── SYSTEM OVERVIEW ───────────────────────────────────────────────────\n")
        output_stream.write(f"  Total Predictions Logged: {summary.get('total_predictions', 0)}\n")
        output_stream.write(f"  Pending Resolution:       {summary.get('pending', 0)}\n")
        output_stream.write(f"  Resolved Outcomes:        {summary.get('resolved', 0)}\n")
        output_stream.write(f"  Active Strategies:        {summary.get('strategies_tracked', 0)}\n\n")

        # 2. Strategy Performance Table
        output_stream.write("── STRATEGY PERFORMANCE ──────────────────────────────────────────────\n")
        output_stream.write(f"{'Strategy':<18} | {'Total':<6} | {'Hit Rate':<8} | {'Rolling IC':<10} | {'Sharpe':<6} | {'Status':<8}\n")
        output_stream.write("-" * 70 + "\n")
        
        for report in reports:
            status = "ACTIVE" if report.is_active else "DEMOTED"
            output_stream.write(
                f"{report.strategy:<18} | "
                f"{report.total_predictions:<6} | "
                f"{report.hit_rate:<8.1%} | "
                f"{report.rolling_ic:<10.3f} | "
                f"{report.sharpe:<6.2f} | "
                f"{status:<8}\n"
            )
        output_stream.write("\n")

        # 3. Strategy Demotion & Actions
        output_stream.write("── RECOMMENDATIONS & ACTIONS ─────────────────────────────────────────\n")
        demoted = summary.get("demoted_strategies", [])
        if demoted:
            output_stream.write("  [WARNING] The following strategies have low Information Coefficient (IC):\n")
            for strat in demoted:
                ic_val = summary.get("strategy_ics", {}).get(strat, 0.0)
                output_stream.write(f"    - Strategy '{strat}' has rolling IC = {ic_val:.4f} (< {self.min_ic}).\n")
                output_stream.write(f"      ACTION: Demote strategy weight / suspend signals.\n")
        else:
            output_stream.write("  [OK] All active strategies maintain healthy Information Coefficients.\n")
            output_stream.write("  ACTION: Maintain default portfolio optimization limits.\n")
        
        output_stream.write("=" * 70 + "\n\n")


def print_dashboard() -> None:
    """Helper function to print dashboard directly to stdout."""
    dashboard = AlphaPerformanceCLIDashboard()
    dashboard.render()

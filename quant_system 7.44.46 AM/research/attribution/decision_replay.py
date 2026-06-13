"""Decision replay system for epistemic debugging.

This system exposes hidden reasoning failures by replaying past decisions
with current data and understanding. It catches:
- Bad assumptions
- Contradictory filters
- Hidden correlations
- Timing artifacts
- Stale features

This is not about backtesting. This is about debugging the reasoning process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DecisionContext:
    """The context in which a decision was made."""
    timestamp: datetime
    symbol: str
    market_data: pd.DataFrame
    features: Dict[str, float]
    filters_applied: List[str]
    signal_generated: bool
    signal_direction: int
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "features": self.features,
            "filters_applied": self.filters_applied,
            "signal_generated": self.signal_generated,
            "signal_direction": self.signal_direction,
            "confidence": self.confidence,
        }


@dataclass
class ReplayResult:
    """Result of replaying a decision."""

    original_decision: DecisionContext
    replayed_decision: Optional[DecisionContext]

    # Comparison metrics
    decision_changed: bool
    signal_changed: bool
    confidence_delta: float

    # Diagnostics
    feature_changes: Dict[str, Tuple[float, float]]  # (original, replayed)
    filter_changes: List[str]
    reasoning_failures: List[str]

    # Assessment
    failure_type: Optional[str]
    severity: str  # "low", "medium", "high"

    def to_dict(self) -> Dict:
        return {
            "original_timestamp": self.original_decision.timestamp.isoformat(),
            "symbol": self.original_decision.symbol,
            "decision_changed": self.decision_changed,
            "signal_changed": self.signal_changed,
            "confidence_delta": self.confidence_delta,
            "feature_changes": self.feature_changes,
            "filter_changes": self.filter_changes,
            "reasoning_failures": self.reasoning_failures,
            "failure_type": self.failure_type,
            "severity": self.severity,
        }


class DecisionReplayEngine:
    """Replays past decisions to expose hidden reasoning failures."""

    def __init__(self):
        self.decision_history: List[DecisionContext] = []
        self.replay_results: List[ReplayResult] = []

    def record_decision(
        self,
        timestamp: datetime,
        symbol: str,
        market_data: pd.DataFrame,
        features: Dict[str, float],
        filters_applied: List[str],
        signal_generated: bool,
        signal_direction: int,
        confidence: float,
    ) -> None:
        """Record a decision for future replay."""
        context = DecisionContext(
            timestamp=timestamp,
            symbol=symbol,
            market_data=market_data.copy(),
            features=features.copy(),
            filters_applied=filters_applied.copy(),
            signal_generated=signal_generated,
            signal_direction=signal_direction,
            confidence=confidence,
        )
        self.decision_history.append(context)

    def replay_decision(
        self,
        original_context: DecisionContext,
        current_market_data: pd.DataFrame,
        current_features: Dict[str, float],
        current_filters: List[str],
    ) -> ReplayResult:
        """Replay a single decision with current data/understanding.

        Args:
            original_context: The original decision context
            current_market_data: Current market data for replay
            current_features: Current feature values
            current_filters: Current filter logic

        Returns:
            ReplayResult with comparison and diagnostics
        """
        # Simulate the decision with current data
        # In a real implementation, this would call the actual strategy logic
        replayed_signal = self._simulate_decision(
            current_market_data, current_features, current_filters
        )

        replayed_context = DecisionContext(
            timestamp=original_context.timestamp,
            symbol=original_context.symbol,
            market_data=current_market_data,
            features=current_features,
            filters_applied=current_filters,
            signal_generated=replayed_signal["signal"],
            signal_direction=replayed_signal["direction"],
            confidence=replayed_signal["confidence"],
        )

        # Compare decisions
        decision_changed = (
            original_context.signal_generated != replayed_context.signal_generated
            or original_context.signal_direction != replayed_context.signal_direction
        )

        signal_changed = (
            original_context.signal_generated != replayed_context.signal_generated
        )

        confidence_delta = replayed_context.confidence - original_context.confidence

        # Detect feature changes
        feature_changes = {}
        for key in set(original_context.features.keys()) | set(current_features.keys()):
            orig_val = original_context.features.get(key, 0.0)
            curr_val = current_features.get(key, 0.0)
            if abs(orig_val - curr_val) > 0.01:  # Significant change
                feature_changes[key] = (orig_val, curr_val)

        # Detect filter changes
        filter_changes = list(
            set(original_context.filters_applied) ^ set(current_filters)
        )

        # Detect reasoning failures
        reasoning_failures = self._detect_reasoning_failures(
            original_context, replayed_context, feature_changes, filter_changes
        )

        # Classify failure type
        failure_type = self._classify_failure(reasoning_failures, feature_changes)

        # Assess severity
        severity = self._assess_severity(
            decision_changed, signal_changed, reasoning_failures
        )

        return ReplayResult(
            original_decision=original_context,
            replayed_decision=replayed_context,
            decision_changed=decision_changed,
            signal_changed=signal_changed,
            confidence_delta=confidence_delta,
            feature_changes=feature_changes,
            filter_changes=filter_changes,
            reasoning_failures=reasoning_failures,
            failure_type=failure_type,
            severity=severity,
        )

    def _simulate_decision(
        self,
        market_data: pd.DataFrame,
        features: Dict[str, float],
        filters: List[str],
    ) -> Dict[str, Any]:
        """Simulate a decision based on current data/understanding.

        This is a placeholder. In a real implementation, this would
        call the actual strategy logic.
        """
        # Simple heuristic: if gap_pct < -1.5 and volume_ratio > 1.5, generate signal
        gap_pct = features.get("gap_pct", 0.0)
        volume_ratio = features.get("volume_ratio", 1.0)

        signal = False
        direction = 1
        confidence = 0.5

        if gap_pct < -1.5 and volume_ratio > 1.5:
            signal = True
            direction = 1  # Long
            confidence = 0.7
        elif gap_pct > 1.5 and volume_ratio > 1.5:
            signal = True
            direction = -1  # Short
            confidence = 0.7

        return {
            "signal": signal,
            "direction": direction,
            "confidence": confidence,
        }

    def _detect_reasoning_failures(
        self,
        original: DecisionContext,
        replayed: DecisionContext,
        feature_changes: Dict[str, Tuple[float, float]],
        filter_changes: List[str],
    ) -> List[str]:
        """Detect reasoning failures in the decision."""
        failures = []

        # Check for contradictory filters
        if "gap_down" in original.filters_applied and "gap_up" in original.filters_applied:
            failures.append("contradictory_filters: both gap_up and gap_down filters active")

        # Check for stale features
        for feature, (orig, curr) in feature_changes.items():
            if abs(orig - curr) > 0.5:  # Large change suggests stale data
                failures.append(f"stale_feature: {feature} changed from {orig:.2f} to {curr:.2f}")

        # Check for hidden correlations
        if "volume_ratio" in feature_changes and "gap_pct" in feature_changes:
            vol_change = abs(feature_changes["volume_ratio"][1] - feature_changes["volume_ratio"][0])
            gap_change = abs(feature_changes["gap_pct"][1] - feature_changes["gap_pct"][0])
            if vol_change > 1.0 and gap_change > 1.0:
                failures.append("hidden_correlation: volume and gap both changed significantly")

        # Check for timing artifacts
        if original.signal_generated and not replayed.signal_generated:
            failures.append("timing_artifact: signal disappeared on replay")

        return failures

    def _classify_failure(
        self,
        reasoning_failures: List[str],
        feature_changes: Dict[str, Tuple[float, float]],
    ) -> Optional[str]:
        """Classify the type of failure."""
        if not reasoning_failures:
            return None

        # Categorize by failure type
        failure_types = {
            "contradictory_filters": [],
            "stale_feature": [],
            "hidden_correlation": [],
            "timing_artifact": [],
        }

        for failure in reasoning_failures:
            for ftype in failure_types:
                if ftype in failure:
                    failure_types[ftype].append(failure)

        # Return the most common failure type
        if failure_types["contradictory_filters"]:
            return "contradictory_filters"
        elif failure_types["stale_feature"]:
            return "stale_feature"
        elif failure_types["hidden_correlation"]:
            return "hidden_correlation"
        elif failure_types["timing_artifact"]:
            return "timing_artifact"
        else:
            return "unknown"

    def _assess_severity(
        self,
        decision_changed: bool,
        signal_changed: bool,
        reasoning_failures: List[str],
    ) -> str:
        """Assess the severity of the failure."""
        if signal_changed:
            return "high"
        elif decision_changed:
            return "medium"
        elif len(reasoning_failures) > 2:
            return "medium"
        elif reasoning_failures:
            return "low"
        else:
            return "low"

    def replay_batch(
        self,
        current_market_data: Dict[str, pd.DataFrame],
        current_features: Dict[str, Dict[str, float]],
        current_filters: List[str],
    ) -> List[ReplayResult]:
        """Replay a batch of decisions.

        Args:
            current_market_data: Dict mapping symbol to current market data
            current_features: Dict mapping symbol to current features
            current_filters: Current filter logic

        Returns:
            List of ReplayResult
        """
        results = []
        for context in self.decision_history:
            symbol = context.symbol
            if symbol not in current_market_data or symbol not in current_features:
                continue

            result = self.replay_decision(
                context,
                current_market_data[symbol],
                current_features[symbol],
                current_filters,
            )
            results.append(result)

        self.replay_results = results
        return results

    def generate_replay_report(self) -> str:
        """Generate a report on replay results."""
        lines = []
        lines.append("=" * 70)
        lines.append("DECISION REPLAY REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Total Decisions Replayed: {len(self.replay_results)}")
        lines.append("")

        # Summary statistics
        changed = sum(1 for r in self.replay_results if r.decision_changed)
        signal_changed = sum(1 for r in self.replay_results if r.signal_changed)
        high_severity = sum(1 for r in self.replay_results if r.severity == "high")

        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Decisions Changed: {changed}/{len(self.replay_results)} ({changed/len(self.replay_results):.1%})")
        lines.append(f"Signals Changed: {signal_changed}/{len(self.replay_results)} ({signal_changed/len(self.replay_results):.1%})")
        lines.append(f"High Severity: {high_severity}/{len(self.replay_results)} ({high_severity/len(self.replay_results):.1%})")
        lines.append("")

        # Failure types
        failure_types = {}
        for r in self.replay_results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        if failure_types:
            lines.append("FAILURE TYPES")
            lines.append("-" * 70)
            for ftype, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {ftype}: {count}")
            lines.append("")

        # High severity failures
        high_severity_results = [r for r in self.replay_results if r.severity == "high"]
        if high_severity_results:
            lines.append("HIGH SEVERITY FAILURES")
            lines.append("-" * 70)
            for r in high_severity_results[:10]:  # Show first 10
                lines.append(f"  {r.original_decision.timestamp.isoformat()} - {r.original_decision.symbol}")
                lines.append(f"    Failure: {r.failure_type}")
                lines.append(f"    Reasoning: {r.reasoning_failures[:3]}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

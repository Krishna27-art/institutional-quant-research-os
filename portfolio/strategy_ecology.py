"""Strategy Ecology Engine.

Strategy ecology models strategies as living organisms in an ecosystem,
not as independent black boxes. It captures:

1. Competitive relationships (strategies eating each other's alpha)
2. Symbiotic relationships (strategies that work better together)
3. Evolutionary pressure (which strategies are thriving vs declining)
4. Ecological balance (diversity, stability, resilience)
5. Opportunity overlap (when strategies compete for the same trades)
6. Resource allocation (capital as ecological resource)

This transforms portfolio construction from "add more strategies" to
"build a healthy ecosystem" where strategies complement rather than compete.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of ecological relationships between strategies."""
    COMPETITIVE = "competitive"  # Strategies compete for same opportunities
    SYMBIOTIC = "symbiotic"  # Strategies enhance each other
    NEUTRAL = "neutral"  # No significant relationship
    PARASITIC = "parasitic"  # One strategy harms another


@dataclass
class StrategyFitness:
    """Fitness metrics for a strategy."""
    strategy_name: str
    fitness_score: float  # Overall fitness (0-1)
    sharpe_ratio: float
    growth_rate: float  # Recent performance trend
    win_rate: float
    max_drawdown: float
    opportunity_count: int  # Number of trading opportunities

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EcologicalRelationship:
    """Relationship between two strategies."""
    strategy_a: str
    strategy_b: str
    relationship_type: RelationshipType
    strength: float  # 0-1, how strong the relationship is
    opportunity_overlap: float  # 0-1, how much they compete for same trades
    correlation: float  # Return correlation

    timestamp: datetime = field(default_factory=datetime.now)


class StrategyEcologyEngine:
    """Engine for modeling strategy ecology."""

    def __init__(self):
        # Strategy fitness tracking
        self.strategy_fitness: Dict[str, StrategyFitness] = {}

        # Ecological relationships
        self.relationships: Dict[Tuple[str, str], EcologicalRelationship] = {}

        # Ecological metrics
        self.diversity_index: float = 0.0
        self.stability_index: float = 0.0
        self.resilience_index: float = 0.0

        logger.info("StrategyEcologyEngine initialized")

    def register_strategy(
        self,
        strategy_name: str,
        sharpe_ratio: float,
        growth_rate: float,
        win_rate: float,
        max_drawdown: float,
        opportunity_count: int,
    ) -> StrategyFitness:
        """Register a strategy for ecological monitoring.

        Args:
            strategy_name: Name of the strategy
            sharpe_ratio: Sharpe ratio
            growth_rate: Recent growth rate
            win_rate: Win rate
            max_drawdown: Maximum drawdown
            opportunity_count: Number of opportunities

        Returns:
            StrategyFitness for the strategy
        """
        # Calculate fitness score
        fitness = self._calculate_fitness(
            sharpe_ratio, growth_rate, win_rate, max_drawdown, opportunity_count
        )

        fitness_record = StrategyFitness(
            strategy_name=strategy_name,
            fitness_score=fitness,
            sharpe_ratio=sharpe_ratio,
            growth_rate=growth_rate,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            opportunity_count=opportunity_count,
        )

        self.strategy_fitness[strategy_name] = fitness_record

        logger.info(f"Registered strategy {strategy_name} with fitness {fitness:.3f}")

        return fitness_record

    def _calculate_fitness(
        self,
        sharpe_ratio: float,
        growth_rate: float,
        win_rate: float,
        max_drawdown: float,
        opportunity_count: int,
    ) -> float:
        """Calculate overall fitness score."""
        # Normalize components
        sharpe_score = min(sharpe_ratio / 2.0, 1.0)  # Sharpe > 2 is excellent
        growth_score = max(min(growth_rate / 0.5, 1.0), -1.0)  # 50% growth is excellent
        win_score = win_rate
        drawdown_score = max(1.0 - abs(max_drawdown) / 0.5, 0.0)  # 50% drawdown is terrible
        opportunity_score = min(opportunity_count / 100.0, 1.0)  # 100+ opportunities is good

        # Weighted average
        fitness = (
            0.3 * sharpe_score +
            0.25 * growth_score +
            0.2 * win_score +
            0.15 * drawdown_score +
            0.1 * opportunity_score
        )

        return max(0.0, min(1.0, fitness))

    def analyze_relationship(
        self,
        strategy_a: str,
        strategy_b: str,
        returns_a: pd.Series,
        returns_b: pd.Series,
        trades_a: pd.DataFrame,
        trades_b: pd.DataFrame,
    ) -> EcologicalRelationship:
        """Analyze the ecological relationship between two strategies.

        Args:
            strategy_a: First strategy name
            strategy_b: Second strategy name
            returns_a: Returns of strategy A
            returns_b: Returns of strategy B
            trades_a: Trades of strategy A
            trades_b: Trades of strategy B

        Returns:
            EcologicalRelationship between the strategies
        """
        # Calculate correlation
        correlation = self._calculate_correlation(returns_a, returns_b)

        # Calculate opportunity overlap
        opportunity_overlap = self._calculate_opportunity_overlap(trades_a, trades_b)

        # Determine relationship type
        relationship_type = self._classify_relationship(correlation, opportunity_overlap)

        # Calculate relationship strength
        strength = self._calculate_relationship_strength(correlation, opportunity_overlap)

        relationship = EcologicalRelationship(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            relationship_type=relationship_type,
            strength=strength,
            opportunity_overlap=opportunity_overlap,
            correlation=correlation,
        )

        # Store relationship (use sorted tuple as key)
        key = tuple(sorted([strategy_a, strategy_b]))
        self.relationships[key] = relationship

        logger.info(
            f"Analyzed relationship {strategy_a} <-> {strategy_b}: "
            f"{relationship_type.value} (strength={strength:.2f})"
        )

        return relationship

    def _calculate_correlation(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
    ) -> float:
        """Calculate correlation between strategy returns."""
        # Align returns
        common_index = returns_a.index.intersection(returns_b.index)
        if len(common_index) < 30:
            return 0.0

        aligned_a = returns_a.loc[common_index]
        aligned_b = returns_b.loc[common_index]

        return aligned_a.corr(aligned_b)

    def _calculate_opportunity_overlap(
        self,
        trades_a: pd.DataFrame,
        trades_b: pd.DataFrame,
    ) -> float:
        """Calculate how much strategies compete for same trades.

        This measures if strategies are taking the same signals.
        """
        if "timestamp" not in trades_a.columns or "timestamp" not in trades_b.columns:
            return 0.0

        # Convert timestamps to dates for comparison
        dates_a = set(pd.to_datetime(trades_a["timestamp"]).dt.date)
        dates_b = set(pd.to_datetime(trades_b["timestamp"]).dt.date)

        if not dates_a or not dates_b:
            return 0.0

        # Calculate overlap
        overlap = len(dates_a & dates_b)
        total = len(dates_a | dates_b)

        return overlap / total if total > 0 else 0.0

    def _classify_relationship(
        self,
        correlation: float,
        opportunity_overlap: float,
    ) -> RelationshipType:
        """Classify the type of ecological relationship."""
        # High correlation + high overlap = competitive
        if abs(correlation) > 0.7 and opportunity_overlap > 0.5:
            return RelationshipType.COMPETITIVE

        # Low/negative correlation + low overlap = symbiotic
        if abs(correlation) < 0.3 and opportunity_overlap < 0.3:
            return RelationshipType.SYMBIOTIC

        # High correlation but one strategy declining = parasitic
        # (This would need additional fitness data)

        # Default to neutral
        return RelationshipType.NEUTRAL

    def _calculate_relationship_strength(
        self,
        correlation: float,
        opportunity_overlap: float,
    ) -> float:
        """Calculate the strength of the relationship."""
        # Strength is combination of correlation magnitude and overlap
        return (abs(correlation) + opportunity_overlap) / 2.0

    def calculate_ecological_metrics(self) -> None:
        """Calculate overall ecological metrics."""
        if not self.strategy_fitness:
            return

        # Diversity: Shannon diversity index based on fitness
        fitness_values = [f.fitness_score for f in self.strategy_fitness.values()]
        total_fitness = sum(fitness_values)

        if total_fitness > 0:
            proportions = [f / total_fitness for f in fitness_values]
            # Shannon diversity
            self.diversity_index = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
        else:
            self.diversity_index = 0.0

        # Stability: Inverse of variance in fitness
        if fitness_values:
            self.stability_index = 1.0 / (1.0 + np.var(fitness_values))
        else:
            self.stability_index = 0.0

        # Resilience: Combination of diversity and competitive relationships
        competitive_count = sum(
            1 for r in self.relationships.values()
            if r.relationship_type == RelationshipType.COMPETITIVE
        )
        total_relationships = len(self.relationships) if self.relationships else 1

        # Fewer competitive relationships = higher resilience
        competitive_ratio = competitive_count / total_relationships
        self.resilience_index = self.diversity_index * (1.0 - competitive_ratio)

    def get_evolutionary_pressure(self) -> Dict[str, str]:
        """Get evolutionary pressure on each strategy.

        Returns:
            Dict mapping strategy name to pressure level
        """
        pressures = {}

        for strategy, fitness in self.strategy_fitness.items():
            if fitness.growth_rate > 0.2:
                pressures[strategy] = "thriving"
            elif fitness.growth_rate > -0.2:
                pressures[strategy] = "stable"
            elif fitness.growth_rate > -0.5:
                pressures[strategy] = "declining"
            else:
                pressures[strategy] = "at_risk"

        return pressures

    def recommend_ecological_balance(self) -> List[str]:
        """Recommend adjustments for ecological balance.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not self.strategy_fitness:
            return recommendations

        # Check diversity
        if self.diversity_index < 1.0:
            recommendations.append(
                f"Low diversity ({self.diversity_index:.2f}): Consider adding strategies "
                f"with different opportunity profiles"
            )

        # Check for competitive relationships
        competitive = [
            r for r in self.relationships.values()
            if r.relationship_type == RelationshipType.COMPETITIVE
            and r.strength > 0.7
        ]
        if competitive:
            competitive_pairs = [(r.strategy_a, r.strategy_b) for r in competitive]
            recommendations.append(
                f"High competition detected between {competitive_pairs}: "
                f"Consider reducing allocation to one of each pair"
            )

        # Check for declining strategies
        pressures = self.get_evolutionary_pressure()
        declining = [s for s, p in pressures.items() if p in ["declining", "at_risk"]]
        if declining:
            recommendations.append(
                f"Strategies under evolutionary pressure: {declining}. "
                f"Consider investigation or reduction"
            )

        # Check for symbiotic relationships to exploit
        symbiotic = [
            r for r in self.relationships.values()
            if r.relationship_type == RelationshipType.SYMBIOTIC
            and r.strength > 0.7
        ]
        if symbiotic:
            symbiotic_pairs = [(r.strategy_a, r.strategy_b) for r in symbiotic]
            recommendations.append(
                f"Symbiotic relationships detected: {symbiotic_pairs}. "
                f"Consider increasing allocation to both"
            )

        return recommendations

    def get_ecology_report(self) -> str:
        """Generate an ecology report."""
        lines = []
        lines.append("=" * 70)
        lines.append("STRATEGY ECOLOGY REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Ecological metrics
        self.calculate_ecological_metrics()

        lines.append("ECOLOGICAL METRICS")
        lines.append("-" * 70)
        lines.append(f"Diversity Index: {self.diversity_index:.3f}")
        lines.append(f"Stability Index: {self.stability_index:.3f}")
        lines.append(f"Resilience Index: {self.resilience_index:.3f}")
        lines.append("")

        # Strategy fitness
        lines.append("STRATEGY FITNESS")
        lines.append("-" * 70)
        for strategy, fitness in sorted(
            self.strategy_fitness.items(),
            key=lambda x: x[1].fitness_score,
            reverse=True
        ):
            lines.append(
                f"{strategy}: Fitness={fitness.fitness_score:.3f}, "
                f"Sharpe={fitness.sharpe_ratio:.2f}, "
                f"Growth={fitness.growth_rate:.2%}"
            )
        lines.append("")

        # Evolutionary pressure
        pressures = self.get_evolutionary_pressure()
        lines.append("EVOLUTIONARY PRESSURE")
        lines.append("-" * 70)
        for strategy, pressure in sorted(pressures.items()):
            lines.append(f"  {strategy}: {pressure}")
        lines.append("")

        # Relationships
        lines.append("ECOLOGICAL RELATIONSHIPS")
        lines.append("-" * 70)
        for rel in sorted(self.relationships.values(), key=lambda x: x.strength, reverse=True)[:10]:
            lines.append(
                f"{rel.strategy_a} <-> {rel.strategy_b}: "
                f"{rel.relationship_type.value} (strength={rel.strength:.2f}, "
                f"overlap={rel.opportunity_overlap:.2f})"
            )
        lines.append("")

        # Recommendations
        recommendations = self.recommend_ecological_balance()
        if recommendations:
            lines.append("ECOLOGICAL BALANCE RECOMMENDATIONS")
            lines.append("-" * 70)
            for rec in recommendations:
                lines.append(f"  • {rec}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

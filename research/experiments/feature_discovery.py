"""
Continuous Feature Discovery Engine
Automatically discovers new features using genetic programming and SHAP importance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import random
import hashlib
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from time_machine_simulator import TimeMachineSimulator, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStatus(Enum):
    """Feature status"""
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


@dataclass
class FeatureCandidate:
    """Feature candidate from discovery"""
    feature_id: str
    name: str
    definition: str
    base_features: List[str]
    created_at: datetime
    generation: int
    ic: float  # Information Coefficient
    rank_ic: float  # Rank IC
    correlation_with_existing: float
    importance_score: float
    status: FeatureStatus = FeatureStatus.CANDIDATE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of feature discovery run"""
    run_id: str
    timestamp: datetime
    total_candidates: int
    passed_validation: int
    approved_features: int
    top_features: List[FeatureCandidate]
    statistics: Dict[str, Any]


class FeatureGeneticProgramming:
    """Genetic programming for feature discovery"""
    
    def __init__(
        self,
        base_features: List[str],
        max_depth: int = 4,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        self.base_features = base_features
        self.max_depth = max_depth
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # GP operators
        self.operators = ['+', '-', '*', '/', 'log', 'sqrt', 'rank', 'delay', 'diff', 'mean', 'std', 'max', 'min', 'correlation']
        
        logger.info(f"Feature GP Engine initialized: pop_size={population_size}, max_depth={max_depth}")
    
    def generate_random_expression(self, depth: int = 0) -> str:
        """Generate random feature expression"""
        if depth >= self.max_depth:
            return random.choice(self.base_features)
        
        if random.random() < 0.4:
            op = random.choice(self.operators)
            
            if op in ['+', '-', '*', '/']:
                left = self.generate_random_expression(depth + 1)
                right = self.generate_random_expression(depth + 1)
                return f"({left} {op} {right})"
            elif op in ['log', 'sqrt', 'rank']:
                operand = self.generate_random_expression(depth + 1)
                return f"{op}({operand})"
            elif op in ['delay', 'diff']:
                operand = self.generate_random_expression(depth + 1)
                period = random.randint(1, 20)
                return f"{op}({operand}, {period})"
            elif op in ['mean', 'std', 'max', 'min']:
                operand = self.generate_random_expression(depth + 1)
                period = random.randint(5, 30)
                return f"{op}({operand}, {period})"
            elif op == 'correlation':
                operand1 = self.generate_random_expression(depth + 1)
                operand2 = self.generate_random_expression(depth + 1)
                return f"{op}({operand1}, {operand2})"
        else:
            return random.choice(self.base_features)
    
    def compute_feature(
        self,
        expression: str,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute feature from expression
        Simplified implementation for simulation
        """
        # In production, this would use a proper expression evaluator
        # For simulation, generate random feature values
        return pd.Series(np.random.randn(len(data)), index=data.index)
    
    def evaluate_fitness(
        self,
        expression: str,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> float:
        """
        Evaluate fitness of feature
        Fitness = Information Coefficient (IC)
        """
        try:
            # Compute feature
            feature = self.compute_feature(expression, features)
            
            # Align with labels
            common_index = feature.index.intersection(labels.index)
            feature_aligned = feature.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            # Calculate IC
            if len(feature_aligned) > 10:
                ic = feature_aligned.corr(labels_aligned)
                return abs(ic) if not np.isnan(ic) else 0.0
            return 0.0
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def mutate(self, expression: str) -> str:
        """Mutate expression"""
        parts = expression.split()
        if len(parts) > 2:
            idx = random.randint(0, len(parts) - 1)
            parts[idx] = random.choice(self.base_features)
        return ' '.join(parts)
    
    def crossover(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """Crossover two expressions"""
        parts1 = expr1.split()
        parts2 = expr2.split()
        
        if len(parts1) > 4 and len(parts2) > 4:
            idx1 = random.randint(1, len(parts1) - 2)
            idx2 = random.randint(1, len(parts2) - 2)
            
            new_expr1 = ' '.join(parts1[:idx1] + parts2[idx2:])
            new_expr2 = ' '.join(parts2[:idx2] + parts1[idx1:])
            
            return new_expr1, new_expr2
        
        return expr1, expr2
    
    def run_generation(
        self,
        population: List[str],
        features: pd.DataFrame,
        labels: pd.Series
    ) -> List[str]:
        """Run one generation of GP"""
        # Evaluate fitness
        fitness_scores = []
        for expr in population:
            fitness = self.evaluate_fitness(expr, features, labels)
            fitness_scores.append(fitness)
        
        # Sort by fitness
        sorted_pop = [expr for _, expr in sorted(zip(fitness_scores, population), reverse=True)]
        
        # Select top 50%
        survivors = sorted_pop[:self.population_size // 2]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.population_size - len(survivors):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.append(parent1)
                offspring.append(parent2)
        
        # Mutate
        mutated = []
        for expr in offspring:
            if random.random() < self.mutation_rate:
                mutated.append(self.mutate(expr))
            else:
                mutated.append(expr)
        
        new_population = survivors + mutated[:self.population_size - len(survivors)]
        
        return new_population


class FeatureDiscoveryEngine:
    """
    Continuous Feature Discovery Engine
    """
    
    def __init__(
        self,
        time_machine: TimeMachineSimulator,
        base_features: List[str],
        existing_features: Optional[List[FeatureCandidate]] = None
    ):
        self.time_machine = time_machine
        self.base_features = base_features
        self.existing_features = existing_features or []
        self.discovered_features: List[FeatureCandidate] = []
        
        self.gp_engine = FeatureGeneticProgramming(base_features)
        
        # Validation thresholds
        self.validation_thresholds = {
            'min_ic': 0.02,
            'min_rank_ic': 0.015,
            'max_correlation': 0.7,
        }
        
        logger.info("Feature Discovery Engine initialized")
    
    def discover_features(
        self,
        start_date: datetime,
        end_date: datetime,
        num_generations: int = 10,
        population_size: int = 100
    ) -> DiscoveryResult:
        """
        Run feature discovery process
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            num_generations: Number of GP generations
            population_size: Population size per generation
            
        Returns:
            DiscoveryResult
        """
        run_id = f"FEATURE_DISCOVERY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting feature discovery {run_id}")
        
        # Get training data
        snapshots = self.time_machine.get_snapshot_range(
            start_date=start_date,
            end_date=end_date,
            frequency='1D',
            symbols=['NIFTY'],
            data_types=[DataType.OHLCV],
            lookback_days=20
        )
        
        features = self.time_machine.get_feature_matrix(snapshots)
        labels = self.time_machine.get_labels(snapshots, forward_periods=1)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            expr = self.gp_engine.generate_random_expression()
            population.append(expr)
        
        # Run GP generations
        for generation in range(num_generations):
            population = self.gp_engine.run_generation(population, features, labels)
            logger.info(f"Generation {generation + 1}/{num_generations} complete")
        
        # Evaluate final population
        candidates = []
        for expr in population:
            ic = self.gp_engine.evaluate_fitness(expr, features, labels)
            rank_ic = ic * 0.9  # Simulated rank IC
            
            # Calculate correlation with existing features
            correlation = self._calculate_correlation_with_existing(expr, features)
            
            # Calculate importance score
            importance_score = self._calculate_importance_score(expr, features, labels)
            
            # Create feature candidate
            feature_id = f"FEAT_{hashlib.sha256(expr.encode()).hexdigest()[:8]}"
            
            candidate = FeatureCandidate(
                feature_id=feature_id,
                name=f"feature_{feature_id}",
                definition=expr,
                base_features=self._extract_features(expr),
                created_at=datetime.now(),
                generation=num_generations,
                ic=ic,
                rank_ic=rank_ic,
                correlation_with_existing=correlation,
                importance_score=importance_score,
                status=FeatureStatus.CANDIDATE
            )
            
            candidates.append(candidate)
        
        # Filter by validation thresholds
        passed_validation = self._validate_candidates(candidates)
        
        # Remove redundant features using Boruta-like approach
        approved_features = self._remove_redundant_features(passed_validation)
        
        # Store discovered features
        self.discovered_features.extend(approved_features)
        
        result = DiscoveryResult(
            run_id=run_id,
            timestamp=datetime.now(),
            total_candidates=len(candidates),
            passed_validation=len(passed_validation),
            approved_features=len(approved_features),
            top_features=sorted(approved_features, key=lambda x: x.importance_score, reverse=True)[:10],
            statistics={
                'avg_ic': np.mean([c.ic for c in candidates]),
                'max_ic': max([c.ic for c in candidates]),
                'avg_importance': np.mean([c.importance_score for c in candidates]),
            }
        )
        
        logger.info(
            f"Discovery complete: {len(candidates)} candidates, "
            f"{len(approved_features)} approved"
        )
        
        return result
    
    def _extract_features(self, expression: str) -> List[str]:
        """Extract base features from expression"""
        features = []
        for feat in self.base_features:
            if feat in expression:
                features.append(feat)
        return features
    
    def _calculate_correlation_with_existing(
        self,
        expression: str,
        features: pd.DataFrame
    ) -> float:
        """Calculate max correlation with existing features"""
        if not self.existing_features:
            return 0.0
        
        # Compute new feature
        new_feature = self.gp_engine.compute_feature(expression, features)
        
        # Calculate max correlation with existing
        max_corr = 0.0
        for existing in self.existing_features:
            existing_feature = self.gp_engine.compute_feature(existing.definition, features)
            corr = new_feature.corr(existing_feature)
            max_corr = max(max_corr, abs(corr) if not np.isnan(corr) else 0.0)
        
        return max_corr
    
    def _calculate_importance_score(
        self,
        expression: str,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> float:
        """Calculate feature importance using random forest"""
        try:
            # Compute feature
            new_feature = self.gp_engine.compute_feature(expression, features)
            
            # Combine with base features
            X = features.copy()
            X['new_feature'] = new_feature
            
            # Align with labels
            common_index = X.index.intersection(labels.index)
            X_aligned = X.loc[common_index]
            y_aligned = labels.loc[common_index]
            
            # Train random forest
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X_aligned, y_aligned)
            
            # Get feature importance
            importance = rf.feature_importances_
            new_feature_importance = importance[-1]  # Last feature is the new one
            
            return new_feature_importance
            
        except Exception as e:
            logger.error(f"Importance calculation failed: {e}")
            return 0.0
    
    def _validate_candidates(self, candidates: List[FeatureCandidate]) -> List[FeatureCandidate]:
        """Validate candidates by thresholds"""
        passed = []
        
        for candidate in candidates:
            if (candidate.ic >= self.validation_thresholds['min_ic'] and
                candidate.rank_ic >= self.validation_thresholds['min_rank_ic'] and
                candidate.correlation_with_existing <= self.validation_thresholds['max_correlation']):
                
                candidate.status = FeatureStatus.VALIDATED
                passed.append(candidate)
        
        logger.info(f"Validation: {len(passed)}/{len(candidates)} passed")
        
        return passed
    
    def _remove_redundant_features(
        self,
        candidates: List[FeatureCandidate]
    ) -> List[FeatureCandidate]:
        """Remove redundant features using Boruta-like approach"""
        # Sort by importance
        sorted_candidates = sorted(candidates, key=lambda x: x.importance_score, reverse=True)
        
        approved = []
        for candidate in sorted_candidates:
            # Check if redundant with already approved features
            is_redundant = False
            for approved_candidate in approved:
                if candidate.correlation_with_existing > 0.8:
                    is_redundant = True
                    break
            
            if not is_redundant:
                candidate.status = FeatureStatus.APPROVED
                approved.append(candidate)
        
        logger.info(f"Redundancy removal: {len(approved)}/{len(candidates)} approved")
        
        return approved
    
    def get_feature_registry(self) -> pd.DataFrame:
        """Get feature registry"""
        registry_data = []
        
        all_features = self.existing_features + self.discovered_features
        
        for feature in all_features:
            registry_data.append({
                'feature_id': feature.feature_id,
                'name': feature.name,
                'definition': feature.definition,
                'base_features': ', '.join(feature.base_features),
                'ic': feature.ic,
                'rank_ic': feature.rank_ic,
                'importance_score': feature.importance_score,
                'correlation_with_existing': feature.correlation_with_existing,
                'status': feature.status.value,
                'created_at': feature.created_at.isoformat(),
            })
        
        return pd.DataFrame(registry_data)
    
    def save_discovery_result(self, result: DiscoveryResult, save_path: str) -> None:
        """Save discovery result to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            'run_id': result.run_id,
            'timestamp': result.timestamp.isoformat(),
            'total_candidates': result.total_candidates,
            'passed_validation': result.passed_validation,
            'approved_features': result.approved_features,
            'statistics': result.statistics,
            'top_features': [
                {
                    'feature_id': f.feature_id,
                    'name': f.name,
                    'definition': f.definition,
                    'ic': f.ic,
                    'importance_score': f.importance_score,
                    'status': f.status.value,
                }
                for f in result.top_features
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved discovery result to {save_path}")


def simulate_feature_discovery():
    """Simulate feature discovery"""
    
    print("="*60)
    print("FEATURE DISCOVERY ENGINE SIMULATION")
    print("="*60)
    
    # Initialize time machine
    time_machine = TimeMachineSimulator()
    
    # Initialize discovery engine
    base_features = [
        'close', 'open', 'high', 'low', 'volume',
        'returns_1d', 'returns_5d', 'volatility_5d'
    ]
    
    discovery_engine = FeatureDiscoveryEngine(
        time_machine=time_machine,
        base_features=base_features
    )
    
    # Run discovery
    print("\n1. Running feature discovery...")
    result = discovery_engine.discover_features(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 1, 1),
        num_generations=5,  # Reduced for simulation
        population_size=50
    )
    
    print(f"  Run ID: {result.run_id}")
    print(f"  Total candidates: {result.total_candidates}")
    print(f"  Passed validation: {result.passed_validation}")
    print(f"  Approved features: {result.approved_features}")
    
    # Show statistics
    print("\n2. Discovery statistics:")
    for stat, value in result.statistics.items():
        print(f"  {stat}: {value:.4f}")
    
    # Show top features
    print("\n3. Top features:")
    for i, feature in enumerate(result.top_features[:5], 1):
        print(f"  {i}. {feature.name}")
        print(f"     Definition: {feature.definition}")
        print(f"     IC: {feature.ic:.4f}")
        print(f"     Importance: {feature.importance_score:.4f}")
        print(f"     Status: {feature.status.value}")
    
    # Get feature registry
    print("\n4. Feature registry:")
    registry = discovery_engine.get_feature_registry()
    print(f"  Total features: {len(registry)}")
    if not registry.empty:
        print(f"  Status distribution:")
        print(registry['status'].value_counts())
    
    # Save result
    print("\n5. Saving discovery result...")
    discovery_engine.save_discovery_result(result, "data/feature_discovery_result.json")
    print("  Saved to data/feature_discovery_result.json")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_feature_discovery()

"""
Continuous Alpha Discovery Engine
Automatically discovers new alphas using genetic programming
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import random
import hashlib
from scipy import stats

from time_machine_simulator import TimeMachineSimulator, DataType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaStatus(Enum):
    """Alpha status"""
    CANDIDATE = "candidate"
    SCREENING = "screening"
    VALIDATION = "validation"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class AlphaCandidate:
    """Alpha candidate from discovery"""
    alpha_id: str
    formula: str
    features: List[str]
    created_at: datetime
    generation: int
    fitness_score: float
    sharpe: float
    max_drawdown: float
    turnover: float
    capacity_cr: float
    correlation_with_existing: float
    status: AlphaStatus = AlphaStatus.CANDIDATE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Result of alpha discovery run"""
    run_id: str
    timestamp: datetime
    total_candidates: int
    passed_screening: int
    passed_validation: int
    approved_for_paper: int
    top_candidates: List[AlphaCandidate]
    statistics: Dict[str, Any]


class GeneticProgrammingEngine:
    """Genetic programming for alpha discovery"""
    
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
        self.operators = ['+', '-', '*', '/', 'log', 'sqrt', 'rank', 'delay', 'diff', 'mean', 'std']
        
        logger.info(f"GP Engine initialized: pop_size={population_size}, max_depth={max_depth}")
    
    def generate_random_expression(self, depth: int = 0) -> str:
        """Generate random expression using GP"""
        if depth >= self.max_depth:
            # Return a base feature
            return random.choice(self.base_features)
        
        if random.random() < 0.3:
            # Return operator
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
            elif op in ['mean', 'std']:
                operand = self.generate_random_expression(depth + 1)
                period = random.randint(5, 30)
                return f"{op}({operand}, {period})"
        else:
            return random.choice(self.base_features)
    
    def evaluate_fitness(
        self,
        expression: str,
        features: pd.DataFrame,
        labels: pd.Series
    ) -> float:
        """
        Evaluate fitness of expression
        Fitness = Information Coefficient (IC)
        """
        try:
            # Compute signal from expression
            signal = self._compute_signal(expression, features)
            
            # Align with labels
            common_index = signal.index.intersection(labels.index)
            signal_aligned = signal.loc[common_index]
            labels_aligned = labels.loc[common_index]
            
            # Calculate IC (correlation)
            if len(signal_aligned) > 10:
                ic = signal_aligned.corr(labels_aligned)
                return abs(ic) if not np.isnan(ic) else 0.0
            return 0.0
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def _compute_signal(self, expression: str, features: pd.DataFrame) -> pd.Series:
        """Compute signal from expression"""
        # This is a simplified implementation
        # In production, this would use a proper expression evaluator
        
        # For simulation, return a random signal
        return pd.Series(np.random.randn(len(features)), index=features.index)
    
    def mutate(self, expression: str) -> str:
        """Mutate expression"""
        # Simple mutation: replace a random sub-expression
        parts = expression.split()
        if len(parts) > 2:
            idx = random.randint(0, len(parts) - 1)
            parts[idx] = random.choice(self.base_features)
        return ' '.join(parts)
    
    def crossover(self, expr1: str, expr2: str) -> Tuple[str, str]:
        """Crossover two expressions"""
        # Simple crossover: swap sub-expressions
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


class AlphaDiscoveryEngine:
    """
    Continuous Alpha Discovery Engine
    """
    
    def __init__(
        self,
        time_machine: TimeMachineSimulator,
        base_features: List[str],
        existing_alphas: Optional[List[AlphaCandidate]] = None
    ):
        self.time_machine = time_machine
        self.base_features = base_features
        self.existing_alphas = existing_alphas or []
        self.discovered_alphas: List[AlphaCandidate] = []
        
        self.gp_engine = GeneticProgrammingEngine(base_features)
        
        # Screening thresholds
        self.screening_thresholds = {
            'min_sharpe': 1.0,
            'max_drawdown': 0.20,
            'max_turnover': 10.0,  # 1000% per year
        }
        
        # Validation thresholds
        self.validation_thresholds = {
            'min_sharpe': 1.2,
            'max_p_value': 0.05,
            'max_correlation': 0.6,
        }
        
        logger.info("Alpha Discovery Engine initialized")
    
    def discover_alphas(
        self,
        start_date: datetime,
        end_date: datetime,
        num_generations: int = 10,
        population_size: int = 100
    ) -> DiscoveryResult:
        """
        Run alpha discovery process
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            num_generations: Number of GP generations
            population_size: Population size per generation
            
        Returns:
            DiscoveryResult
        """
        run_id = f"DISCOVERY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting alpha discovery {run_id}")
        
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
            fitness = self.gp_engine.evaluate_fitness(expr, features, labels)
            
            # Create alpha candidate
            alpha_id = f"ALPHA_{hashlib.sha256(expr.encode()).hexdigest()[:8]}"
            
            candidate = AlphaCandidate(
                alpha_id=alpha_id,
                formula=expr,
                features=self._extract_features(expr),
                created_at=datetime.now(),
                generation=num_generations,
                fitness_score=fitness,
                sharpe=fitness * 2.0,  # Simulated
                max_drawdown=abs(np.random.normal(0.1, 0.05)),
                turnover=np.random.uniform(2.0, 8.0),
                capacity_cr=np.random.uniform(50, 200),
                correlation_with_existing=np.random.uniform(0.1, 0.5),
                status=AlphaStatus.CANDIDATE
            )
            
            candidates.append(candidate)
        
        # Filter by screening thresholds
        passed_screening = self._screen_candidates(candidates)
        
        # Filter by validation thresholds
        passed_validation = self._validate_candidates(passed_screening)
        
        # Approve for paper trading
        approved_for_paper = self._approve_for_paper_trading(passed_validation)
        
        # Store discovered alphas
        self.discovered_alphas.extend(approved_for_paper)
        
        result = DiscoveryResult(
            run_id=run_id,
            timestamp=datetime.now(),
            total_candidates=len(candidates),
            passed_screening=len(passed_screening),
            passed_validation=len(passed_validation),
            approved_for_paper=len(approved_for_paper),
            top_candidates=sorted(approved_for_paper, key=lambda x: x.fitness_score, reverse=True)[:10],
            statistics={
                'avg_fitness': np.mean([c.fitness_score for c in candidates]),
                'max_fitness': max([c.fitness_score for c in candidates]),
                'avg_sharpe': np.mean([c.sharpe for c in candidates]),
            }
        )
        
        logger.info(
            f"Discovery complete: {len(candidates)} candidates, "
            f"{len(approved_for_paper)} approved for paper trading"
        )
        
        return result
    
    def _extract_features(self, expression: str) -> List[str]:
        """Extract feature names from expression"""
        features = []
        for feat in self.base_features:
            if feat in expression:
                features.append(feat)
        return features
    
    def _screen_candidates(self, candidates: List[AlphaCandidate]) -> List[AlphaCandidate]:
        """Screen candidates by basic thresholds"""
        passed = []
        
        for candidate in candidates:
            if (candidate.sharpe >= self.screening_thresholds['min_sharpe'] and
                candidate.max_drawdown <= self.screening_thresholds['max_drawdown'] and
                candidate.turnover <= self.screening_thresholds['max_turnover']):
                
                candidate.status = AlphaStatus.SCREENING
                passed.append(candidate)
        
        logger.info(f"Screening: {len(passed)}/{len(candidates)} passed")
        
        return passed
    
    def _validate_candidates(self, candidates: List[AlphaCandidate]) -> List[AlphaCandidate]:
        """Validate candidates with stricter thresholds"""
        passed = []
        
        for candidate in candidates:
            # Simulate p-value calculation
            p_value = np.random.uniform(0.01, 0.1)
            
            if (candidate.sharpe >= self.validation_thresholds['min_sharpe'] and
                p_value <= self.validation_thresholds['max_p_value'] and
                candidate.correlation_with_existing <= self.validation_thresholds['max_correlation']):
                
                candidate.status = AlphaStatus.VALIDATION
                candidate.metadata['p_value'] = p_value
                passed.append(candidate)
        
        logger.info(f"Validation: {len(passed)}/{len(candidates)} passed")
        
        return passed
    
    def _approve_for_paper_trading(self, candidates: List[AlphaCandidate]) -> List[AlphaCandidate]:
        """Approve candidates for paper trading"""
        approved = []
        
        for candidate in candidates:
            # Additional check: paper Sharpe > 1.0
            paper_sharpe = candidate.sharpe * 0.9  # Simulated degradation
            
            if paper_sharpe >= 1.0:
                candidate.status = AlphaStatus.PAPER_TRADING
                candidate.metadata['paper_sharpe'] = paper_sharpe
                approved.append(candidate)
        
        logger.info(f"Paper trading approval: {len(approved)}/{len(candidates)} approved")
        
        return approved
    
    def estimate_capacity(
        self,
        candidate: AlphaCandidate,
        base_capacity: float = 100_000_000
    ) -> float:
        """
        Estimate capacity of alpha
        Simulate increasing trade size until Sharpe drops by 20%
        """
        # Simplified capacity estimation
        # In production, this would run actual simulations
        
        capacity = base_capacity * (candidate.sharpe / 2.0)
        
        return capacity
    
    def get_alpha_genome(self) -> pd.DataFrame:
        """Get alpha genome database"""
        genome_data = []
        
        all_alphas = self.existing_alphas + self.discovered_alphas
        
        for alpha in all_alphas:
            genome_data.append({
                'alpha_id': alpha.alpha_id,
                'formula': alpha.formula,
                'features': ', '.join(alpha.features),
                'sharpe': alpha.sharpe,
                'max_drawdown': alpha.max_drawdown,
                'capacity_cr': alpha.capacity_cr,
                'correlation_with_existing': alpha.correlation_with_existing,
                'status': alpha.status.value,
                'created_at': alpha.created_at.isoformat(),
            })
        
        return pd.DataFrame(genome_data)
    
    def save_discovery_result(self, result: DiscoveryResult, save_path: str) -> None:
        """Save discovery result to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            'run_id': result.run_id,
            'timestamp': result.timestamp.isoformat(),
            'total_candidates': result.total_candidates,
            'passed_screening': result.passed_screening,
            'passed_validation': result.passed_validation,
            'approved_for_paper': result.approved_for_paper,
            'statistics': result.statistics,
            'top_candidates': [
                {
                    'alpha_id': c.alpha_id,
                    'formula': c.formula,
                    'fitness_score': c.fitness_score,
                    'sharpe': c.sharpe,
                    'status': c.status.value,
                }
                for c in result.top_candidates
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved discovery result to {save_path}")


def simulate_alpha_discovery():
    """Simulate alpha discovery"""
    
    print("="*60)
    print("ALPHA DISCOVERY ENGINE SIMULATION")
    print("="*60)
    
    # Initialize time machine
    time_machine = TimeMachineSimulator()
    
    # Initialize discovery engine
    base_features = [
        'close', 'volume', 'returns_1d', 'returns_5d',
        'volatility_5d', 'volume_ratio', 'rsi'
    ]
    
    discovery_engine = AlphaDiscoveryEngine(
        time_machine=time_machine,
        base_features=base_features
    )
    
    # Run discovery
    print("\n1. Running alpha discovery...")
    result = discovery_engine.discover_alphas(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 1, 1),
        num_generations=5,  # Reduced for simulation
        population_size=50
    )
    
    print(f"  Run ID: {result.run_id}")
    print(f"  Total candidates: {result.total_candidates}")
    print(f"  Passed screening: {result.passed_screening}")
    print(f"  Passed validation: {result.passed_validation}")
    print(f"  Approved for paper: {result.approved_for_paper}")
    
    # Show statistics
    print("\n2. Discovery statistics:")
    for stat, value in result.statistics.items():
        print(f"  {stat}: {value:.4f}")
    
    # Show top candidates
    print("\n3. Top candidates:")
    for i, candidate in enumerate(result.top_candidates[:5], 1):
        print(f"  {i}. {candidate.alpha_id}")
        print(f"     Formula: {candidate.formula}")
        print(f"     Sharpe: {candidate.sharpe:.2f}")
        print(f"     Status: {candidate.status.value}")
    
    # Get alpha genome
    print("\n4. Alpha genome database:")
    genome = discovery_engine.get_alpha_genome()
    print(f"  Total alphas: {len(genome)}")
    if not genome.empty:
        print(f"  Status distribution:")
        print(genome['status'].value_counts())
    
    # Save result
    print("\n5. Saving discovery result...")
    discovery_engine.save_discovery_result(result, "data/alpha_discovery_result.json")
    print("  Saved to data/alpha_discovery_result.json")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    simulate_alpha_discovery()

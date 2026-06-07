"""
MadEvolve-Style Alpha Evolution

This module implements genetic programming-based alpha evolution for automated
alpha discovery and optimization as specified in the V4 Institutional Architecture.

Key Features:
- Genetic programming for feature expression evolution
- Crossover, mutation, and selection operations
- Fitness evaluation via backtesting
- Population-based optimization
- Expected Sharpe improvement: +0.1–0.2

Based on V4 Blueprint - Institutional Architecture
Priority: Medium (Phase 2)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in expression tree."""
    CONSTANT = "constant"
    FEATURE = "feature"
    OPERATOR = "operator"


@dataclass
class Node:
    """Node in expression tree."""
    node_type: NodeType
    value: str
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    
    def __str__(self):
        if self.node_type == NodeType.CONSTANT:
            return self.value
        elif self.node_type == NodeType.FEATURE:
            return self.value
        elif self.node_type == NodeType.OPERATOR:
            if self.left and self.right:
                return f"({self.left} {self.value} {self.right})"
            elif self.left:
                return f"{self.value}({self.left})"
            return self.value
        return ""


@dataclass
class FeatureExpression:
    """Feature expression for alpha."""
    expression: Node
    fitness: float = 0.0
    sharpe: float = 0.0
    ic: float = 0.0
    generation: int = 0
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate expression on data."""
        return self._evaluate_node(self.expression, data)
    
    def _evaluate_node(self, node: Node, data: pd.DataFrame) -> pd.Series:
        """Recursively evaluate node."""
        if node.node_type == NodeType.CONSTANT:
            return pd.Series([float(node.value)] * len(data), index=data.index)
        elif node.node_type == NodeType.FEATURE:
            if node.value in data.columns:
                return data[node.value]
            return pd.Series([0.0] * len(data), index=data.index)
        elif node.node_type == NodeType.OPERATOR:
            left = self._evaluate_node(node.left, data) if node.left else None
            right = self._evaluate_node(node.right, data) if node.right else None
            
            if node.value == '+':
                return left + right
            elif node.value == '-':
                return left - right
            elif node.value == '*':
                return left * right
            elif node.value == '/':
                return left / (right + 1e-10)
            elif node.value == 'abs':
                return left.abs()
            elif node.value == 'log':
                return np.log(left.abs() + 1e-10)
            elif node.value == 'sqrt':
                return np.sqrt(left.abs())
            elif node.value == 'rank':
                return left.rank(pct=True)
            elif node.value == 'delay':
                return left.shift(1)
            elif node.value == 'delta':
                return left - left.shift(1)
            elif node.value == 'ts_rank':
                return left.rolling(window=20).rank(pct=True)
            elif node.value == 'ts_mean':
                return left.rolling(window=20).mean()
            elif node.value == 'ts_std':
                return left.rolling(window=20).std()
            elif node.value == 'ts_max':
                return left.rolling(window=20).max()
            elif node.value == 'ts_min':
                return left.rolling(window=20).min()
            elif node.value == 'sign':
                return np.sign(left)
            elif node.value == 'sigmoid':
                return 1 / (1 + np.exp(-left))
        
        return pd.Series([0.0] * len(data), index=data.index)


class MadEvolveEvolution:
    """
    MadEvolve-style feature evolution engine.
    
    Uses genetic programming to evolve feature expressions for alpha generation.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        elite_size: int = 5
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.population: List[FeatureExpression] = []
        self.best_expression: Optional[FeatureExpression] = None
        self.generation = 0
        
        # Available features
        self.features = [
            'returns_5d', 'returns_20d', 'returns_60d',
            'rv_5d', 'rv_20d',
            'realized_vol_5d', 'realized_vol_20d',
            'spread', 'ofi', 'vwap_deviation',
            'ibs', 'close_position',
            'amihud', 'turnover'
        ]
        
        # Available operators
        self.operators = ['+', '-', '*', '/', 'abs', 'log', 'sqrt', 'rank', 
                        'delay', 'delta', 'ts_rank', 'ts_mean', 'ts_std', 
                        'ts_max', 'ts_min', 'sign', 'sigmoid']
        
        # Constants
        self.constants = ['-1.0', '-0.5', '0.0', '0.5', '1.0', '2.0', '5.0', '10.0']
        
        logger.info(f"MadEvolveEvolution initialized: population={population_size}, generations={max_generations}")
    
    def initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            expression = self._generate_random_expression(max_depth=4)
            feature_expr = FeatureExpression(expression=expression, generation=0)
            self.population.append(feature_expr)
        
        logger.info(f"Initialized population with {len(self.population)} expressions")
    
    def _generate_random_expression(self, max_depth: int = 4, current_depth: int = 0) -> Node:
        """Generate random expression tree."""
        if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.3):
            # Leaf node
            if random.random() < 0.5:
                return Node(NodeType.FEATURE, random.choice(self.features))
            else:
                return Node(NodeType.CONSTANT, random.choice(self.constants))
        
        # Operator node
        operator = random.choice(self.operators)
        
        # Determine arity
        if operator in ['abs', 'log', 'sqrt', 'rank', 'delay', 'delta', 'sign', 'sigmoid']:
            # Unary operator
            left = self._generate_random_expression(max_depth, current_depth + 1)
            return Node(NodeType.OPERATOR, operator, left)
        else:
            # Binary operator
            left = self._generate_random_expression(max_depth, current_depth + 1)
            right = self._generate_random_expression(max_depth, current_depth + 1)
            return Node(NodeType.OPERATOR, operator, left, right)
    
    def evaluate_fitness(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        fitness_func: Optional[Callable] = None
    ) -> None:
        """
        Evaluate fitness of population.
        
        Args:
            data: Feature data
            target: Target returns
            fitness_func: Custom fitness function (default: IC)
        """
        for expr in self.population:
            try:
                signal = expr.evaluate(data)
                
                # Calculate IC (Information Coefficient)
                ic = signal.corr(target)
                expr.ic = ic if not np.isnan(ic) else 0.0
                
                # Calculate Sharpe (simplified)
                returns = signal * target
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
                expr.sharpe = sharpe if not np.isnan(sharpe) else 0.0
                
                # Fitness = IC + 0.5 * Sharpe (normalized)
                expr.fitness = expr.ic + 0.5 * (expr.sharpe / 10.0)
                
            except Exception as e:
                logger.warning(f"Error evaluating expression: {e}")
                expr.fitness = -1.0
                expr.ic = 0.0
                expr.sharpe = 0.0
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best expression
        if self.population:
            self.best_expression = self.population[0]
    
    def selection(self, tournament_size: int = 5) -> FeatureExpression:
        """Tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: FeatureExpression, parent2: FeatureExpression) -> Tuple[FeatureExpression, FeatureExpression]:
        """Crossover two expressions."""
        child1_expr = self._crossover_node(parent1.expression, parent2.expression)
        child2_expr = self._crossover_node(parent2.expression, parent1.expression)
        
        child1 = FeatureExpression(expression=child1_expr, generation=self.generation + 1)
        child2 = FeatureExpression(expression=child2_expr, generation=self.generation + 1)
        
        return child1, child2
    
    def _crossover_node(self, node1: Node, node2: Node) -> Node:
        """Crossover two nodes recursively."""
        if random.random() < 0.5:
            return self._copy_node(node2)
        
        if node1.node_type == NodeType.OPERATOR:
            left = self._crossover_node(node1.left, node2) if node1.left else None
            right = self._crossover_node(node1.right, node2) if node1.right else None
            return Node(node1.node_type, node1.value, left, right)
        
        return self._copy_node(node1)
    
    def _copy_node(self, node: Node) -> Node:
        """Deep copy node."""
        if node.node_type == NodeType.OPERATOR:
            left = self._copy_node(node.left) if node.left else None
            right = self._copy_node(node.right) if node.right else None
            return Node(node.node_type, node.value, left, right)
        return Node(node.node_type, node.value)
    
    def mutate(self, expression: FeatureExpression) -> FeatureExpression:
        """Mutate expression."""
        mutated_expr = self._mutate_node(expression.expression)
        return FeatureExpression(expression=mutated_expr, generation=self.generation + 1)
    
    def _mutate_node(self, node: Node, depth: int = 0) -> Node:
        """Mutate node recursively."""
        if random.random() < self.mutation_rate or depth > 10:
            # Replace with random subtree
            return self._generate_random_expression(max_depth=3)
        
        if node.node_type == NodeType.OPERATOR:
            left = self._mutate_node(node.left, depth + 1) if node.left else None
            right = self._mutate_node(node.right, depth + 1) if node.right else None
            return Node(node.node_type, node.value, left, right)
        
        return node
    
    def evolve(self, data: pd.DataFrame, target: pd.Series) -> FeatureExpression:
        """
        Run evolution process.
        
        Args:
            data: Feature data
            target: Target returns
            
        Returns:
            Best evolved expression
        """
        logger.info("Starting evolution process...")
        
        # Initialize population
        self.initialize_population()
        
        # Initial fitness evaluation
        self.evaluate_fitness(data, target)
        
        logger.info(f"Generation 0: Best fitness={self.population[0].fitness:.4f}, IC={self.population[0].ic:.4f}")
        
        # Evolution loop
        for self.generation in range(1, self.max_generations + 1):
            new_population = []
            
            # Elitism: keep best expressions
            new_population.extend(self.population[:self.elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.selection()
                parent2 = self.selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                else:
                    new_population.append(parent1)
                    if len(new_population) < self.population_size:
                        new_population.append(parent2)
            
            # Mutation
            for i in range(self.elite_size, len(new_population)):
                if random.random() < self.mutation_rate:
                    new_population[i] = self.mutate(new_population[i])
            
            self.population = new_population
            
            # Evaluate fitness
            self.evaluate_fitness(data, target)
            
            logger.info(f"Generation {self.generation}: Best fitness={self.population[0].fitness:.4f}, IC={self.population[0].ic:.4f}")
        
        logger.info(f"Evolution complete. Best IC: {self.best_expression.ic:.4f}, Best Sharpe: {self.best_expression.sharpe:.4f}")
        
        return self.best_expression
    
    def get_expression_string(self, expression: FeatureExpression) -> str:
        """Get expression as string."""
        return str(expression.expression)
    
    def print_population_report(self) -> None:
        """Print population report."""
        print("\n" + "="*60)
        print("MADEVOLVE POPULATION REPORT")
        print("="*60)
        print(f"Generation: {self.generation}")
        print(f"Population Size: {len(self.population)}")
        print(f"Best Fitness: {self.population[0].fitness:.4f}")
        print(f"Best IC: {self.population[0].ic:.4f}")
        print(f"Best Sharpe: {self.population[0].sharpe:.4f}")
        
        print("\nTop 5 Expressions:")
        for i, expr in enumerate(self.population[:5], 1):
            print(f"  {i}. {self.get_expression_string(expr)}")
            print(f"     Fitness: {expr.fitness:.4f}, IC: {expr.ic:.4f}, Sharpe: {expr.sharpe:.4f}")
        
        print("="*60)


def run_sample_madevolve():
    """Demonstrate MadEvolve evolution."""
    print("=== MadEvolve-Style Alpha Evolution Demo ===\n")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'returns_5d': np.random.randn(n_samples) * 0.02,
        'returns_20d': np.random.randn(n_samples) * 0.03,
        'returns_60d': np.random.randn(n_samples) * 0.04,
        'rv_5d': np.random.rand(n_samples) * 2.0 + 0.5,
        'rv_20d': np.random.rand(n_samples) * 2.0 + 0.5,
        'realized_vol_5d': np.random.rand(n_samples) * 0.02,
        'realized_vol_20d': np.random.rand(n_samples) * 0.03,
        'spread': np.random.rand(n_samples) * 0.001,
        'ofi': np.random.randn(n_samples) * 100000,
        'vwap_deviation': np.random.randn(n_samples) * 0.01,
        'ibs': np.random.rand(n_samples),
        'close_position': np.random.rand(n_samples),
        'amihud': np.random.rand(n_samples) * 1e-10,
        'turnover': np.random.rand(n_samples) * 1e8
    })
    
    # Generate target (future returns)
    target = data['returns_5d'] * 0.5 + data['returns_20d'] * 0.3 + np.random.randn(n_samples) * 0.01
    
    # Initialize evolution
    evolution = MadEvolveEvolution(
        population_size=20,
        max_generations=10,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elite_size=3
    )
    
    # Run evolution
    print("Running evolution...")
    best_expression = evolution.evolve(data, target)
    
    # Print report
    evolution.print_population_report()
    
    print("\nBest Expression:")
    print(f"  {evolution.get_expression_string(best_expression)}")
    print(f"  IC: {best_expression.ic:.4f}")
    print(f"  Sharpe: {best_expression.sharpe:.4f}")
    
    print("\n=== MadEvolve Evolution Demo Complete ===")
    print("Expected Sharpe Improvement: +0.1–0.2")


if __name__ == "__main__":
    run_sample_madevolve()

"""
Alpha Factory (Genetic Programming)
Based on V3 Blueprint - Automated Alpha Generation

Key findings from research:
- Alphas are manually researched, not systematically generated
- Genetic programming for systematic alpha generation
- Input: 50 core features, operators: +, -, *, /, log, rank, lag, rolling_mean, rolling_std
- Population: 1000, Generations: 30, Fitness: OOS Sharpe (3-year train, 1-year test)
- Constraints: max features 5, max depth 3, min OOS Sharpe > 0.8
- Expected yield: 1-2 new production alphas per quarter

V3 Upgrade - Expected Sharpe increase: +0.1 (future, research only)
Priority: Medium
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
import random
import copy


@dataclass
class AlphaExpression:
    """Alpha expression tree"""
    expression: str  # String representation
    features: List[str]  # Features used
    depth: int  # Tree depth
    sharpe_train: float  # Training Sharpe
    sharpe_test: float  # Test Sharpe
    fitness: float  # Overall fitness score


class AlphaFactory:
    """
    Alpha Factory using Genetic Programming.
    
    Input: 50 core features (1-min)
    Operators: +, -, *, /, log, rank, lag, rolling_mean, rolling_std, sign
    Population: 1000
    Generations: 30
    Fitness: OOS Sharpe (3-year train, 1-year test)
    Constraints: max features 5, max depth 3, min OOS Sharpe > 0.8
    """
    
    def __init__(self):
        self.operators = ["+", "-", "*", "/"]
        self.functions = ["log", "rank", "lag", "rolling_mean", "rolling_std", "sign"]
        self.features = [
            "returns_1d", "returns_5d", "returns_21d", "volatility_20d",
            "rv_ratio", "volume_impulse", "vwap_distance", "vwap_slope",
            "pcr_oi", "iv_skew", "adv_dec_ratio", "new_highs_pct",
            "rsi_14", "macd", "momentum_20d", "reversal_5d"
        ]
        self.population: List[AlphaExpression] = []
        self.generation = 0
    
    def generate_random_expression(self, max_depth: int = 3) -> str:
        """
        Generate a random alpha expression.
        
        Args:
            max_depth: Maximum tree depth
            
        Returns:
            Expression string
        """
        depth = random.randint(1, max_depth)
        
        if depth == 1:
            # Leaf node: feature or constant
            if random.random() < 0.7:
                return random.choice(self.features)
            else:
                return str(round(random.uniform(-1, 1), 4))
        
        # Internal node: operator or function
        if random.random() < 0.5:
            # Binary operator
            op = random.choice(self.operators)
            left = self.generate_random_expression(max_depth - 1)
            right = self.generate_random_expression(max_depth - 1)
            return f"({left} {op} {right})"
        else:
            # Unary function
            func = random.choice(self.functions)
            operand = self.generate_random_expression(max_depth - 1)
            if func in ["lag", "rolling_mean", "rolling_std"]:
                # These need a parameter
                param = random.randint(1, 20)
                return f"{func}({operand}, {param})"
            return f"{func}({operand})"
    
    def evaluate_expression(
        self,
        expression: str,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Evaluate an expression on data.
        
        Args:
            expression: Expression string
            data: DataFrame with features
            
        Returns:
            Result series
        """
        try:
            # Safe evaluation (simplified)
            # In production, use ast.literal_eval or a proper expression parser
            # For now, return random values as placeholder
            n = len(data)
            return pd.Series(np.random.normal(0, 0.01, n), index=data.index)
        except:
            n = len(data)
            return pd.Series(np.zeros(n), index=data.index)
    
    def calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe
    
    def mutate(self, expression: str, mutation_rate: float = 0.1) -> str:
        """
        Mutate an expression.
        
        Args:
            expression: Expression string
            mutation_rate: Mutation probability
            
        Returns:
            Mutated expression
        """
        if random.random() < mutation_rate:
            # Replace a random part with new expression
            return self.generate_random_expression()
        return expression
    
    def crossover(self, expr1: str, expr2: str) -> str:
        """
        Crossover two expressions.
        
        Args:
            expr1: First expression
            expr2: Second expression
            
        Returns:
            Crossover expression
        """
        # Simplified crossover: swap subtrees
        # In production, parse as AST and swap subtrees
        if random.random() < 0.5:
            return expr1
        else:
            return expr2
    
    def calculate_fitness(
        self,
        expression: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Calculate fitness score.
        
        Args:
            expression: Expression string
            train_data: Training data
            test_data: Test data
            
        Returns:
            (sharpe_train, sharpe_test, fitness)
        """
        # Evaluate on train data
        train_returns = self.evaluate_expression(expression, train_data)
        sharpe_train = self.calculate_sharpe(train_returns)
        
        # Evaluate on test data
        test_returns = self.evaluate_expression(expression, test_data)
        sharpe_test = self.calculate_sharpe(test_returns)
        
        # Fitness = test Sharpe (penalize if test Sharpe < 0.8)
        if sharpe_test < 0.8:
            fitness = sharpe_test * 0.5
        else:
            fitness = sharpe_test
        
        return sharpe_train, sharpe_test, fitness
    
    def initialize_population(self, size: int = 1000) -> None:
        """Initialize random population."""
        self.population = []
        
        for _ in range(size):
            expression = self.generate_random_expression(max_depth=3)
            
            # Count features
            features_used = [f for f in self.features if f in expression]
            
            alpha = AlphaExpression(
                expression=expression,
                features=features_used,
                depth=expression.count("("),
                sharpe_train=0.0,
                sharpe_test=0.0,
                fitness=0.0
            )
            
            self.population.append(alpha)
    
    def evolve(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        generations: int = 30,
        elite_pct: float = 0.1,
        crossover_rate: float = 0.6,
        mutation_rate: float = 0.1
    ) -> List[AlphaExpression]:
        """
        Run genetic programming evolution.
        
        Args:
            train_data: Training data
            test_data: Test data
            generations: Number of generations
            elite_pct: Elite percentage
            crossover_rate: Crossover rate
            mutation_rate: Mutation rate
            
        Returns:
            Best alphas from final generation
        """
        self.initialize_population(size=1000)
        
        for gen in range(generations):
            self.generation = gen + 1
            
            # Evaluate fitness
            for alpha in self.population:
                sharpe_train, sharpe_test, fitness = self.calculate_fitness(
                    alpha.expression, train_data, test_data
                )
                alpha.sharpe_train = sharpe_train
                alpha.sharpe_test = sharpe_test
                alpha.fitness = fitness
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Select elite
            elite_size = int(len(self.population) * elite_pct)
            elite = self.population[:elite_size]
            
            # Create new population
            new_population = elite.copy()
            
            while len(new_population) < len(self.population):
                # Select parents from elite
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover
                if random.random() < crossover_rate:
                    child_expr = self.crossover(parent1.expression, parent2.expression)
                else:
                    child_expr = parent1.expression
                
                # Mutate
                child_expr = self.mutate(child_expr, mutation_rate)
                
                # Count features
                features_used = [f for f in self.features if f in child_expr]
                
                # Check constraints
                if len(features_used) <= 5 and child_expr.count("(") <= 3:
                    new_alpha = AlphaExpression(
                        expression=child_expr,
                        features=features_used,
                        depth=child_expr.count("("),
                        sharpe_train=0.0,
                        sharpe_test=0.0,
                        fitness=0.0
                    )
                    new_population.append(new_alpha)
            
            self.population = new_population
            
            if gen % 5 == 0:
                print(f"Generation {gen}: Best fitness = {self.population[0].fitness:.4f}")
        
        # Return top alphas
        return self.population[:10]
    
    def print_results(self, alphas: List[AlphaExpression]) -> None:
        """Print top alphas."""
        print("\n" + "="*60)
        print("ALPHA FACTORY - TOP CANDIDATES")
        print("="*60)
        
        for i, alpha in enumerate(alphas):
            print(f"\n#{i+1}:")
            print(f"  Expression: {alpha.expression}")
            print(f"  Features: {alpha.features}")
            print(f"  Depth: {alpha.depth}")
            print(f"  Train Sharpe: {alpha.sharpe_train:.4f}")
            print(f"  Test Sharpe: {alpha.sharpe_test:.4f}")
            print(f"  Fitness: {alpha.fitness:.4f}")
        
        print("="*60)


def run_sample_alpha_factory():
    """Run sample alpha factory."""
    factory = AlphaFactory()
    
    # Generate sample data
    np.random.seed(42)
    n_train = 756  # 3 years
    n_test = 252  # 1 year
    
    train_data = pd.DataFrame({
        f: np.random.normal(0, 0.01, n_train) for f in factory.features
    })
    train_data.index = pd.date_range("2020-01-01", periods=n_train, freq="D")
    
    test_data = pd.DataFrame({
        f: np.random.normal(0, 0.01, n_test) for f in factory.features
    })
    test_data.index = pd.date_range("2023-01-01", periods=n_test, freq="D")
    
    # Run evolution (reduced generations for demo)
    print("Running Alpha Factory...")
    top_alphas = factory.evolve(
        train_data,
        test_data,
        generations=10,  # Reduced for demo
        elite_pct=0.1,
        crossover_rate=0.6,
        mutation_rate=0.1
    )
    
    factory.print_results(top_alphas)
    
    return factory


if __name__ == "__main__":
    run_sample_alpha_factory()

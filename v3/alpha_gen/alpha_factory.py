"""
Alpha Factory
Automated alpha generation using genetic programming.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
import random
from copy import deepcopy


class Operator(Enum):
    """Operators for genetic programming"""
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    LOG = "log"
    RANK = "rank"
    LAG = "lag"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_STD = "rolling_std"
    CROSSOVER = "crossover"


@dataclass
class AlphaExpression:
    """Alpha expression tree node"""
    operator: Operator
    operands: List[Any]  # Can be other AlphaExpression or feature names
    depth: int = 0
    
    def to_string(self) -> str:
        """Convert expression to string representation"""
        if self.operator in [Operator.LOG, Operator.RANK]:
            return f"{self.operator.value}({self.operands[0]})"
        elif self.operator in [Operator.LAG]:
            return f"{self.operator.value}({self.operands[0]}, {self.operands[1]})"
        elif self.operator in [Operator.ROLLING_MEAN, Operator.ROLLING_STD]:
            return f"{self.operator.value}({self.operands[0]}, {self.operands[1]})"
        elif len(self.operands) == 2:
            return f"({self.operands[0]} {self.operator.value} {self.operands[1]})"
        else:
            return str(self.operands[0])
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate expression on data.
        
        Args:
            data: DataFrame with feature columns
        
        Returns:
            Series with computed values
        """
        if self.operator == Operator.ADD:
            return self.operands[0].evaluate(data) + self.operands[1].evaluate(data)
        elif self.operator == Operator.SUBTRACT:
            return self.operands[0].evaluate(data) - self.operands[1].evaluate(data)
        elif self.operator == Operator.MULTIPLY:
            return self.operands[0].evaluate(data) * self.operands[1].evaluate(data)
        elif self.operator == Operator.DIVIDE:
            # Safe division
            numerator = self.operands[0].evaluate(data)
            denominator = self.operands[1].evaluate(data)
            return numerator / (denominator + 1e-10)
        elif self.operator == Operator.LOG:
            operand = self.operands[0].evaluate(data)
            return np.log(np.abs(operand) + 1e-10)
        elif self.operator == Operator.RANK:
            operand = self.operands[0].evaluate(data)
            return operand.rank(pct=True)
        elif self.operator == Operator.LAG:
            operand = self.operands[0].evaluate(data)
            lag = self.operands[1]
            return operand.shift(lag)
        elif self.operator == Operator.ROLLING_MEAN:
            operand = self.operands[0].evaluate(data)
            window = self.operands[1]
            return operand.rolling(window=window).mean()
        elif self.operator == Operator.ROLLING_STD:
            operand = self.operands[0].evaluate(data)
            window = self.operands[1]
            return operand.rolling(window=window).std()
        else:
            # Leaf node (feature name)
            return data[self.operands[0]]
    
    def get_depth(self) -> int:
        """Get depth of expression tree"""
        if not self.operands or isinstance(self.operands[0], str):
            return 0
        
        max_child_depth = 0
        for operand in self.operands:
            if isinstance(operand, AlphaExpression):
                max_child_depth = max(max_child_depth, operand.get_depth())
        
        return max_child_depth + 1


@dataclass
class AlphaCandidate:
    """Candidate alpha from genetic programming"""
    expression: AlphaExpression
    sharpe: float = 0.0
    turnover: float = 0.0
    efficiency: float = 0.0  # Sharpe / turnover
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "expression": self.expression.to_string(),
            "sharpe": self.sharpe,
            "turnover": self.turnover,
            "efficiency": self.efficiency,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }


@dataclass
class GenerationResult:
    """Result of alpha generation run"""
    generation: int
    candidates: List[AlphaCandidate] = field(default_factory=list)
    best_candidate: Optional[AlphaCandidate] = None
    avg_sharpe: float = 0.0
    avg_efficiency: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "generation": self.generation,
            "num_candidates": len(self.candidates),
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "avg_sharpe": self.avg_sharpe,
            "avg_efficiency": self.avg_efficiency,
            "generated_at": self.generated_at.isoformat(),
        }


class AlphaFactory:
    """
    Automated alpha generation using genetic programming.
    Generates candidate alphas, evaluates them, and selects top performers.
    """
    
    def __init__(
        self,
        population_size: int = 500,
        generations: int = 20,
        max_depth: int = 3,
        max_turnover: float = 2.0,  # 200% per day
        feature_names: Optional[List[str]] = None
    ):
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.max_turnover = max_turnover
        self.feature_names = feature_names or [
            "close", "volume", "vwap", "high", "low",
            "returns_1d", "returns_5d", "volatility_20d",
            "rsi_14", "macd", "bollinger_upper", "bollinger_lower"
        ]
        
        self.operators = [
            Operator.ADD, Operator.SUBTRACT, Operator.MULTIPLY, Operator.DIVIDE,
            Operator.LOG, Operator.RANK, Operator.LAG,
            Operator.ROLLING_MEAN, Operator.ROLLING_STD
        ]
        
        self.population: List[AlphaCandidate] = []
        self.generation_history: List[GenerationResult] = []
        self.next_id = 0
    
    def generate_random_expression(self, max_depth: int = None) -> AlphaExpression:
        """
        Generate a random expression tree.
        
        Args:
            max_depth: Maximum depth of tree
        
        Returns:
            AlphaExpression
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        # Randomly choose depth
        depth = random.randint(0, max_depth)
        
        if depth == 0:
            # Leaf node: feature name
            feature = random.choice(self.feature_names)
            return AlphaExpression(operator=Operator.ADD, operands=[feature], depth=0)
        
        # Internal node: operator
        operator = random.choice(self.operators)
        
        if operator in [Operator.LOG, Operator.RANK]:
            # Unary operator
            operand = self.generate_random_expression(max_depth - 1)
            return AlphaExpression(operator=operator, operands=[operand], depth=depth)
        elif operator in [Operator.LAG, Operator.ROLLING_MEAN, Operator.ROLLING_STD]:
            # Binary operator with parameter
            operand = self.generate_random_expression(max_depth - 1)
            param = random.randint(1, 20)  # Random lag or window
            return AlphaExpression(operator=operator, operands=[operand, param], depth=depth)
        else:
            # Binary operator
            left = self.generate_random_expression(max_depth - 1)
            right = self.generate_random_expression(max_depth - 1)
            return AlphaExpression(operator=operator, operands=[left, right], depth=depth)
    
    def mutate(self, expression: AlphaExpression, mutation_rate: float = 0.1) -> AlphaExpression:
        """
        Mutate an expression.
        
        Args:
            expression: Expression to mutate
            mutation_rate: Probability of mutation
        
        Returns:
            Mutated expression
        """
        if random.random() < mutation_rate:
            # Replace with random expression
            return self.generate_random_expression()
        
        # Recursively mutate operands
        new_operands = []
        for operand in expression.operands:
            if isinstance(operand, AlphaExpression):
                new_operands.append(self.mutate(operand, mutation_rate))
            else:
                new_operands.append(operand)
        
        return AlphaExpression(
            operator=expression.operator,
            operands=new_operands,
            depth=expression.depth
        )
    
    def crossover(self, parent1: AlphaExpression, parent2: AlphaExpression) -> AlphaExpression:
        """
        Crossover two expressions.
        
        Args:
            parent1: First parent expression
            parent2: Second parent expression
        
        Returns:
            Child expression
        """
        # Simple crossover: swap subtrees at random depth
        child = deepcopy(parent1)
        
        # Find a random subtree in child
        if isinstance(child.operands[0], AlphaExpression):
            subtree = self._get_random_subtree(parent2)
            child.operands[0] = subtree
        
        return child
    
    def _get_random_subtree(self, expression: AlphaExpression) -> AlphaExpression:
        """Get a random subtree from expression"""
        if not isinstance(expression.operands[0], AlphaExpression):
            return expression
        
        if random.random() < 0.5:
            return expression
        else:
            return self._get_random_subtree(random.choice(expression.operands))
    
    def evaluate_candidate(
        self,
        expression: AlphaExpression,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> AlphaCandidate:
        """
        Evaluate an alpha candidate.
        
        Args:
            expression: Alpha expression
            train_data: Training data
            test_data: Test data
        
        Returns:
            AlphaCandidate with performance metrics
        """
        try:
            # Compute alpha values
            train_alpha = expression.evaluate(train_data)
            test_alpha = expression.evaluate(test_data)
            
            # Calculate returns (simplified: use next day return)
            train_returns = train_data['close'].pct_change().shift(-1)
            test_returns = test_data['close'].pct_change().shift(-1)
            
            # Calculate Sharpe
            train_ic = train_alpha.corr(train_returns)
            test_ic = test_alpha.corr(test_returns)
            
            # Use IC as proxy for Sharpe (simplified)
            sharpe = abs(test_ic) * 10  # Scale to reasonable Sharpe range
            
            # Calculate turnover (simplified)
            turnover = np.mean(np.abs(train_alpha.diff().dropna()))
            
            # Calculate efficiency
            efficiency = sharpe / (turnover + 1e-10)
            
            candidate = AlphaCandidate(
                expression=expression,
                sharpe=sharpe,
                turnover=turnover,
                efficiency=efficiency
            )
            
            return candidate
            
        except Exception as e:
            # Return invalid candidate if evaluation fails
            return AlphaCandidate(
                expression=expression,
                sharpe=0.0,
                turnover=float('inf'),
                efficiency=0.0
            )
    
    def select_parents(self, population: List[AlphaCandidate], n_parents: int) -> List[AlphaCandidate]:
        """
        Select parents for next generation using tournament selection.
        
        Args:
            population: Current population
            n_parents: Number of parents to select
        
        Returns:
            Selected parents
        """
        parents = []
        tournament_size = 5
        
        for _ in range(n_parents):
            # Tournament selection
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.efficiency)
            parents.append(winner)
        
        return parents
    
    def generate_generation(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        generation: int
    ) -> GenerationResult:
        """
        Generate a new generation of alphas.
        
        Args:
            train_data: Training data (3 years)
            test_data: Test data (1 year)
            generation: Generation number
        
        Returns:
            GenerationResult with candidates
        """
        candidates = []
        
        if generation == 0:
            # Initial population: random expressions
            for _ in range(self.population_size):
                expression = self.generate_random_expression()
                candidate = self.evaluate_candidate(expression, train_data, test_data)
                candidate.generation = generation
                candidates.append(candidate)
        else:
            # Evolution: select parents, crossover, mutate
            parents = self.select_parents(self.population, self.population_size // 2)
            
            for i in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                if random.random() < 0.7:
                    child_expr = self.crossover(parent1.expression, parent2.expression)
                else:
                    child_expr = deepcopy(parent1.expression)
                
                # Mutate
                child_expr = self.mutate(child_expr, mutation_rate=0.1)
                
                # Evaluate
                candidate = self.evaluate_candidate(child_expr, train_data, test_data)
                candidate.generation = generation
                candidate.parent_ids = [id(parent1), id(parent2)]
                candidates.append(candidate)
        
        # Filter valid candidates
        valid_candidates = [
            c for c in candidates
            if c.sharpe > 0 and c.turnover < self.max_turnover and c.expression.get_depth() <= self.max_depth
        ]
        
        # Sort by efficiency
        valid_candidates.sort(key=lambda x: x.efficiency, reverse=True)
        
        # Keep top candidates
        self.population = valid_candidates[:self.population_size]
        
        # Calculate statistics
        avg_sharpe = np.mean([c.sharpe for c in valid_candidates]) if valid_candidates else 0.0
        avg_efficiency = np.mean([c.efficiency for c in valid_candidates]) if valid_candidates else 0.0
        best_candidate = valid_candidates[0] if valid_candidates else None
        
        result = GenerationResult(
            generation=generation,
            candidates=valid_candidates,
            best_candidate=best_candidate,
            avg_sharpe=avg_sharpe,
            avg_efficiency=avg_efficiency
        )
        
        self.generation_history.append(result)
        
        return result
    
    def run_full_evolution(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> List[GenerationResult]:
        """
        Run full genetic programming evolution.
        
        Args:
            train_data: Training data (3 years)
            test_data: Test data (1 year)
        
        Returns:
            List of generation results
        """
        results = []
        
        for generation in range(self.generations):
            print(f"Generating generation {generation + 1}/{self.generations}")
            result = self.generate_generation(train_data, test_data, generation)
            results.append(result)
            print(f"  Best Sharpe: {result.best_candidate.sharpe if result.best_candidate else 0:.3f}")
            print(f"  Avg Efficiency: {result.avg_efficiency:.3f}")
        
        return results
    
    def get_top_alphas(self, n: int = 10) -> List[Dict]:
        """
        Get top N alphas from final population.
        
        Args:
            n: Number of top alphas to return
        
        Returns:
            List of alpha candidates as dicts
        """
        top_candidates = sorted(self.population, key=lambda x: x.efficiency, reverse=True)[:n]
        return [c.to_dict() for c in top_candidates]
    
    def get_generation_history(self) -> List[Dict]:
        """Get history of all generations"""
        return [r.to_dict() for r in self.generation_history]
    
    def reset(self) -> None:
        """Reset factory state"""
        self.population.clear()
        self.generation_history.clear()
        self.next_id = 0


def mock_train_data() -> pd.DataFrame:
    """Generate mock training data for testing"""
    np.random.seed(42)
    n_samples = 750  # 3 years of daily data
    
    data = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_samples)),
        'volume': np.random.uniform(1e6, 5e6, n_samples),
        'vwap': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n_samples)),
        'high': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_samples)) * 1.01,
        'low': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_samples)) * 0.99,
        'returns_1d': np.random.normal(0.0005, 0.02, n_samples),
        'returns_5d': np.random.normal(0.0025, 0.04, n_samples),
        'volatility_20d': np.random.uniform(0.01, 0.03, n_samples),
        'rsi_14': np.random.uniform(30, 70, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'bollinger_upper': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_samples)) * 1.02,
        'bollinger_lower': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_samples)) * 0.98,
    })
    
    return data


def mock_test_data() -> pd.DataFrame:
    """Generate mock test data for testing"""
    np.random.seed(43)
    n_samples = 250  # 1 year of daily data
    
    data = pd.DataFrame({
        'close': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.022, n_samples)),
        'volume': np.random.uniform(1e6, 5e6, n_samples),
        'vwap': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.017, n_samples)),
        'high': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.022, n_samples)) * 1.01,
        'low': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.022, n_samples)) * 0.99,
        'returns_1d': np.random.normal(0.0003, 0.022, n_samples),
        'returns_5d': np.random.normal(0.0015, 0.045, n_samples),
        'volatility_20d': np.random.uniform(0.012, 0.032, n_samples),
        'rsi_14': np.random.uniform(30, 70, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'bollinger_upper': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.022, n_samples)) * 1.02,
        'bollinger_lower': 100 * np.cumprod(1 + np.random.normal(0.0003, 0.022, n_samples)) * 0.98,
    })
    
    return data

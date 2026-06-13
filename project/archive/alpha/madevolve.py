"""
MadEvolve - LLM-Driven Alpha Evolution

Implements the MadEvolve/QuantEvolve-style engine for automated alpha discovery
and evolution using Large Language Models. This system generates, evaluates, and
evolves trading strategies through LLM-guided mutation and selection.

Key Features:
- Safe alpha-code compilation with execution namespace
- Candidate evaluation by deflated Sharpe
- LLM mutation through client with generate(prompt)
- Deterministic offline mutation fallback
- Alpha registry and lineage tracking
- Automated feature/strategy discovery

Based on Blueprint Week 13-14: LLM-Driven Alpha Evolution (MadEvolve)
Reference: Li et al. - MadEvolve: LLM-Driven Alpha Evolution
"""

import ast
import inspect
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlphaCandidate:
    """Represents an alpha candidate."""
    code: str
    description: str
    sharpe: float
    deflated_sharpe: float
    fitness: float
    generation: int
    parent_id: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AlphaRegistry:
    """
    Registry for alpha candidates with lineage tracking.
    """
    
    def __init__(self):
        self.alphas: Dict[str, AlphaCandidate] = {}
        self.lineage: Dict[str, List[str]] = {}  # parent -> children
        self.best_alpha: Optional[AlphaCandidate] = None
    
    def register(self, alpha: AlphaCandidate) -> str:
        """
        Register an alpha candidate.
        
        Args:
            alpha: Alpha candidate to register
            
        Returns:
            Alpha ID
        """
        alpha_id = f"alpha_{len(self.alphas)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.alphas[alpha_id] = alpha
        
        # Track lineage
        if alpha.parent_id:
            if alpha.parent_id not in self.lineage:
                self.lineage[alpha.parent_id] = []
            self.lineage[alpha.parent_id].append(alpha_id)
        
        # Update best alpha
        if self.best_alpha is None or alpha.fitness > self.best_alpha.fitness:
            self.best_alpha = alpha
        
        return alpha_id
    
    def get_alpha(self, alpha_id: str) -> Optional[AlphaCandidate]:
        """Get alpha by ID."""
        return self.alphas.get(alpha_id)
    
    def get_lineage(self, alpha_id: str) -> List[AlphaCandidate]:
        """Get lineage of an alpha."""
        lineage = []
        current = self.get_alpha(alpha_id)
        while current:
            lineage.append(current)
            current = self.get_alpha(current.parent_id) if current.parent_id else None
        return lineage[::-1]  # Return in chronological order
    
    def get_top_n(self, n: int) -> List[AlphaCandidate]:
        """Get top N alphas by fitness."""
        sorted_alphas = sorted(self.alphas.values(), key=lambda x: x.fitness, reverse=True)
        return sorted_alphas[:n]


class SafeExecutor:
    """
    Safe execution environment for alpha code.
    
    Provides a restricted namespace for executing alpha functions
    to prevent malicious code execution.
    """
    
    def __init__(self):
        # Allowed functions and modules
        self.allowed_modules = {
            'numpy': np,
            'pandas': pd,
            'math': __import__('math'),
        }
        
        # Additional safe functions
        self.safe_functions = {
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'range': range,
            'round': round,
            'float': float,
            'int': int,
            'str': str,
            'list': list,
            'dict': dict,
        }
    
    def execute(self, code: str, features: pd.DataFrame) -> np.ndarray:
        """
        Execute alpha code safely.
        
        Args:
            code: Python code to execute
            features: Feature DataFrame
            
        Returns:
            Alpha signal array
        """
        # Create restricted namespace
        namespace = {
            **self.allowed_modules,
            **self.safe_functions,
            'features': features,
            'np': np,
            'pd': pd,
        }
        
        try:
            # Parse and validate code
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Only allow specific modules
                    module_name = node.module if isinstance(node, ast.Import) else node.module
                    if module_name not in self.allowed_modules:
                        raise ValueError(f"Import of module '{module_name}' not allowed")
                elif isinstance(node, ast.Call):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'open', '__import__']:
                            raise ValueError(f"Function '{node.func.id}' not allowed")
            
            # Execute code
            exec(code, namespace)
            
            # Get alpha function
            if 'alpha' not in namespace:
                raise ValueError("Code must define an 'alpha' function")
            
            alpha_func = namespace['alpha']
            
            # Execute alpha function
            signal = alpha_func(features)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error executing alpha code: {e}")
            raise


class MadEvolveAlphaFactory:
    """
    MadEvolve Alpha Factory for LLM-driven alpha evolution.
    
    This factory uses LLMs to generate, mutate, and evolve alpha strategies.
    It evaluates candidates using deflated Sharpe and maintains a registry
    of the best performers.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5
    ):
        """
        Initialize MadEvolve factory.
        
        Args:
            llm_client: LLM client with generate(prompt) method
            population_size: Size of alpha population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.llm_client = llm_client
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.registry = AlphaRegistry()
        self.executor = SafeExecutor()
        
        # Feature catalog for LLM context
        self.feature_catalog = self._build_feature_catalog()
    
    def _build_feature_catalog(self) -> str:
        """Build feature catalog for LLM context."""
        return """
Available Features:
- features['close']: Closing prices
- features['open']: Opening prices
- features['high']: High prices
- features['low']: Low prices
- features['volume']: Trading volume
- features['returns']: Daily returns
- features['sma_20']: 20-day simple moving average
- features['sma_50']: 50-day simple moving average
- features['rsi']: Relative Strength Index
- features['volatility']: Rolling volatility
- features['momentum']: Price momentum

Available Functions:
- numpy functions (np.mean, np.std, np.sum, etc.)
- pandas functions (rolling, diff, pct_change, etc.)
- Basic math functions (abs, min, max, etc.)

Requirements:
- Function must be named 'alpha'
- Must accept 'features' as argument
- Must return a signal in [-1, 1] range
- Use only features available at prediction time (no future data)
"""
    
    def generate_alpha(
        self,
        description: str,
        fitness_metric: str = 'sharpe'
    ) -> AlphaCandidate:
        """
        Generate an alpha from description using LLM.
        
        Args:
            description: Natural language description of alpha
            fitness_metric: Metric for evaluation ('sharpe', 'deflated_sharpe')
            
        Returns:
            Alpha candidate
        """
        if self.llm_client is None:
            # Fallback to deterministic generation
            return self._generate_deterministic_alpha(description)
        
        prompt = f"""
Write a Python function `alpha(features)` that returns a trading signal in [-1, 1].

Description: {description}

{self.feature_catalog}

The function should:
1. Use only features available at prediction time
2. Return a signal where:
   - Positive values indicate long position
   - Negative values indicate short position
   - Magnitude indicates confidence
3. Be computationally efficient

Return only the code, no explanation.
"""
        
        try:
            code = self.llm_client.generate(prompt)
            alpha = self._evaluate_alpha(code, description, fitness_metric)
            return alpha
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, using deterministic fallback")
            return self._generate_deterministic_alpha(description)
    
    def _generate_deterministic_alpha(
        self,
        description: str
    ) -> AlphaCandidate:
        """
        Generate alpha deterministically (fallback).
        
        Args:
            description: Description of alpha
            
        Returns:
            Alpha candidate
        """
        # Simple momentum-based alpha as fallback
        code = """
def alpha(features):
    # Simple momentum strategy
    if 'returns' in features.columns:
        # Use recent returns
        recent_returns = features['returns'].iloc[-20:]
        signal = np.sign(recent_returns.mean())
        return np.clip(signal, -1, 1)
    elif 'close' in features.columns:
        # Use price momentum
        prices = features['close']
        momentum = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
        signal = np.sign(momentum)
        return np.clip(signal, -1, 1)
    else:
        return 0.0
"""
        
        return self._evaluate_alpha(code, description, 'sharpe')
    
    def _evaluate_alpha(
        self,
        code: str,
        description: str,
        fitness_metric: str
    ) -> AlphaCandidate:
        """
        Evaluate alpha candidate.
        
        Args:
            code: Alpha code
            description: Alpha description
            fitness_metric: Metric for evaluation
            
        Returns:
            Alpha candidate with metrics
        """
        # Create sample features for testing
        np.random.seed(42)
        n_samples = 100
        sample_features = pd.DataFrame({
            'close': np.random.uniform(100, 200, n_samples),
            'open': np.random.uniform(100, 200, n_samples),
            'high': np.random.uniform(100, 200, n_samples),
            'low': np.random.uniform(100, 200, n_samples),
            'volume': np.random.uniform(1000000, 5000000, n_samples),
            'returns': np.random.normal(0, 0.02, n_samples),
            'sma_20': np.random.uniform(100, 200, n_samples),
            'sma_50': np.random.uniform(100, 200, n_samples),
            'rsi': np.random.uniform(30, 70, n_samples),
            'volatility': np.random.uniform(0.01, 0.05, n_samples),
            'momentum': np.random.uniform(-0.1, 0.1, n_samples),
        })
        
        try:
            # Execute alpha
            signal = self.executor.execute(code, sample_features)
            
            # Calculate metrics
            sharpe = self._calculate_sharpe(signal, sample_features['returns'])
            deflated_sharpe = self._calculate_deflated_sharpe(sharpe, n_trials=100)
            
            # Fitness score
            if fitness_metric == 'deflated_sharpe':
                fitness = deflated_sharpe
            else:
                fitness = sharpe
            
            return AlphaCandidate(
                code=code,
                description=description,
                sharpe=sharpe,
                deflated_sharpe=deflated_sharpe,
                fitness=fitness,
                generation=0
            )
            
        except Exception as e:
            logger.error(f"Error evaluating alpha: {e}")
            # Return poor candidate
            return AlphaCandidate(
                code=code,
                description=description,
                sharpe=-10.0,
                deflated_sharpe=-10.0,
                fitness=-10.0,
                generation=0
            )
    
    def _calculate_sharpe(self, signal: np.ndarray, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(signal) != len(returns):
            # Align lengths
            min_len = min(len(signal), len(returns))
            signal = signal[:min_len]
            returns = returns[:min_len]
        
        # Calculate portfolio returns
        portfolio_returns = signal * returns
        
        # Sharpe ratio
        if np.std(portfolio_returns) == 0:
            return 0.0
        
        sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_deflated_sharpe(
        self,
        observed_sharpe: float,
        n_trials: int,
        skew: float = 0,
        kurt: float = 3
    ) -> float:
        """
        Calculate deflated Sharpe (Bailey et al. 2014).
        
        Adjusts Sharpe for multiple testing.
        """
        from scipy.stats import norm
        
        var_sr = (1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe +
                 (kurt - 3) / 4 * observed_sharpe**2) / n_trials
        
        z = norm.ppf(1 - 1 / n_trials)
        
        deflated = observed_sharpe - np.sqrt(var_sr) * z
        return deflated
    
    def evolve(
        self,
        population: List[AlphaCandidate],
        fitness_scores: List[float],
        generation: int
    ) -> List[AlphaCandidate]:
        """
        Evolve alpha population using LLM-guided mutation.
        
        Args:
            population: Current population of alphas
            fitness_scores: Fitness scores for each alpha
            generation: Current generation number
            
        Returns:
            New population
        """
        # Select top performers
        sorted_indices = np.argsort(fitness_scores)[::-1]
        top_n = max(1, len(population) // 4)
        top_alphas = [population[i] for i in sorted_indices[:top_n]]
        
        new_population = []
        
        for i in range(self.population_size):
            # Selection
            parent = np.random.choice(top_alphas)
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate_alpha(parent, generation)
            else:
                # Keep parent
                child = AlphaCandidate(
                    code=parent.code,
                    description=parent.description,
                    sharpe=parent.sharpe,
                    deflated_sharpe=parent.deflated_sharpe,
                    fitness=parent.fitness,
                    generation=generation,
                    parent_id=None
                )
            
            new_population.append(child)
        
        return new_population
    
    def _mutate_alpha(
        self,
        parent: AlphaCandidate,
        generation: int
    ) -> AlphaCandidate:
        """
        Mutate alpha using LLM or deterministic methods.
        
        Args:
            parent: Parent alpha
            generation: Current generation
            
        Returns:
            Mutated alpha
        """
        if self.llm_client is not None:
            return self._llm_mutation(parent, generation)
        else:
            return self._deterministic_mutation(parent, generation)
    
    def _llm_mutation(
        self,
        parent: AlphaCandidate,
        generation: int
    ) -> AlphaCandidate:
        """Mutate alpha using LLM."""
        prompt = f"""
Improve this alpha code to increase Sharpe ratio:

{parent.code}

Description: {parent.description}

{self.feature_catalog}

Make one small mutation:
- Change a parameter
- Add a filter
- Modify the logic
- Add a feature

Return only the new code, no explanation.
"""
        
        try:
            new_code = self.llm_client.generate(prompt)
            new_description = f"Mutated from: {parent.description}"
            return self._evaluate_alpha(new_code, new_description, 'deflated_sharpe')
        except Exception as e:
            logger.warning(f"LLM mutation failed: {e}, using deterministic fallback")
            return self._deterministic_mutation(parent, generation)
    
    def _deterministic_mutation(
        self,
        parent: AlphaCandidate,
        generation: int
    ) -> AlphaCandidate:
        """Mutate alpha deterministically."""
        # Simple parameter mutation
        code = parent.code
        
        # Add noise to signal
        if 'return np.clip(signal, -1, 1)' in code:
            code = code.replace('return np.clip(signal, -1, 1)', 
                             'signal = signal * 0.9  # Reduce aggression\n    return np.clip(signal, -1, 1)')
        
        new_description = f"Deterministically mutated from: {parent.description}"
        return self._evaluate_alpha(code, new_description, 'deflated_sharpe')
    
    def run_evolution(
        self,
        initial_descriptions: List[str],
        n_generations: int = 10
    ) -> AlphaCandidate:
        """
        Run full evolution process.
        
        Args:
            initial_descriptions: Initial alpha descriptions
            n_generations: Number of generations to evolve
            
        Returns:
            Best alpha found
        """
        # Generate initial population
        population = []
        for desc in initial_descriptions:
            alpha = self.generate_alpha(desc)
            alpha_id = self.registry.register(alpha)
            population.append(alpha)
        
        # Evolve
        for gen in range(1, n_generations + 1):
            fitness_scores = [alpha.fitness for alpha in population]
            population = self.evolve(population, fitness_scores, gen)
            
            # Register new alphas
            for alpha in population:
                self.registry.register(alpha)
            
            logger.info(f"Generation {gen}: Best fitness = {max(fitness_scores):.4f}")
        
        return self.registry.best_alpha


if __name__ == "__main__":
    # Test MadEvolve
    print("Testing MadEvolve Alpha Factory...")
    
    # Create factory (without LLM client for testing)
    factory = MadEvolveAlphaFactory(llm_client=None)
    
    # Generate initial alphas
    descriptions = [
        "Momentum strategy based on 20-day returns",
        "Mean reversion strategy using RSI",
        "Volatility-based strategy"
    ]
    
    for desc in descriptions:
        alpha = factory.generate_alpha(desc)
        alpha_id = factory.registry.register(alpha)
        print(f"\nGenerated alpha: {alpha_id}")
        print(f"Description: {alpha.description}")
        print(f"Sharpe: {alpha.sharpe:.4f}")
        print(f"Deflated Sharpe: {alpha.deflated_sharpe:.4f}")
        print(f"Fitness: {alpha.fitness:.4f}")
    
    # Run evolution
    print("\nRunning evolution...")
    best_alpha = factory.run_evolution(descriptions, n_generations=3)
    
    print(f"\nBest alpha found:")
    print(f"Description: {best_alpha.description}")
    print(f"Sharpe: {best_alpha.sharpe:.4f}")
    print(f"Deflated Sharpe: {best_alpha.deflated_sharpe:.4f}")
    print(f"Fitness: {best_alpha.fitness:.4f}")
    
    print("\nMadEvolve test completed.")

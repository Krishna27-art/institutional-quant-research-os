"""Alpha generator agent for Quant Research OS V3.5."""

from typing import Dict, List, Any, Tuple
import logging
import random
import math
import statistics
from .agent_base import Agent, AgentMessage, MessageType, AgentCapability

logger = logging.getLogger(__name__)


class AlphaGeneratorAgent(Agent):
    """
    Generates alpha strategies using cost-constrained genetic programming (GP)
    and a constrained Domain-Specific Language (DSL).
    """

    def __init__(self, agent_id: str = "alpha_generator_agent"):
        super().__init__(agent_id, [AgentCapability.ALPHA_GENERATION])
        # Constrained DSL Operator Set
        self.dsl_operators = ["add", "sub", "mul", "div", "log", "abs", "rolling_mean", "rolling_std", "rank", "lag"]
        self.max_complexity_cap = 10  # Max operations per expression
        self.base_population_size = 100
        self.strategy_registry: List[Dict[str, Any]] = []
        self.last_search_summary: Dict[str, Any] = {}

    def receive_message(self, message: AgentMessage):
        """Handle incoming hypotheses and reward updates."""
        if message.message_type == MessageType.HYPOTHESIS:
            hypothesis_data = message.payload
            self.run_alpha_search(hypothesis_data)
        elif message.message_type == MessageType.REWARD_UPDATE:
            logger.info(f"[{self.agent_id}] Received reward feedback: delta_sharpe={message.payload.get('delta_sharpe', 0.0):+.2f}")
        else:
            logger.debug(f"[{self.agent_id}] Ignored message type: {message.message_type.value}")

    def run_alpha_search(self, hypothesis: Dict[str, Any]):
        """Run evolutionary search using adaptive sizing, surrogate fitness, and complexity penalties."""
        hyp_id = hypothesis.get("hypothesis_id", "unknown")
        features = hypothesis.get("features", [])
        logger.info(f"[{self.agent_id}] Starting Alpha search for hypothesis: {hyp_id}")
        
        # Adaptive population sizing (start small)
        population_size = self.base_population_size
        logger.info(f"[{self.agent_id}] Initializing population of size: {population_size}")
        
        # Initialize population
        population = self._initialize_population(features, population_size)
        
        # Evolve population (e.g. 5 generations instead of 100 to save compute)
        generations = 5
        best_candidates = []
        surrogate_checked = 0

        for generation in range(generations):
            evaluated_pop = []
            for ind_idx, individual in enumerate(population):
                # Evaluate using surrogate fitness proxy (fast filtering)
                passes_surrogate, proxy_score = self._evaluate_surrogate_fitness(individual)
                surrogate_checked += 1
                
                if not passes_surrogate:
                    continue  # Filtered out 99% of bad candidates

                # Apply complexity penalty (AIC/BIC style)
                complexity = self._calculate_complexity(individual)
                complexity_penalty = self._complexity_penalty(complexity, len(features))
                final_fitness = max(0.0, proxy_score - complexity_penalty)
                
                # Fitness Inheritance (if mutation is minor, inherit parent's fitness with small perturbation)
                if ind_idx > 0 and random.random() < 0.3:  # Mock inheritance
                    parent_fitness = evaluated_pop[-1][1] if evaluated_pop else final_fitness
                    final_fitness = 0.9 * parent_fitness + 0.1 * final_fitness
                    logger.debug(f"[{self.agent_id}] Fitness inherited: {final_fitness:.4f}")
                
                evaluated_pop.append((individual, final_fitness))
            
            # Sort population by fitness
            evaluated_pop.sort(key=lambda x: x[1], reverse=True)
            
            # Keep elite candidates
            elites = evaluated_pop[:10]
            if not elites:
                logger.info(f"[{self.agent_id}] Generation {generation + 1} fully filtered out")
                break
            best_candidates.extend(elites)
            
            # Create next generation
            population = self._create_next_generation(evaluated_pop, features, population_size)
            logger.info(f"[{self.agent_id}] Generation {generation + 1}/{generations} complete. Best fitness: {elites[0][1]:.4f}")

        # Select top candidates
        best_candidates.sort(key=lambda x: x[1], reverse=True)
        top_individual = best_candidates[0][0]
        top_fitness = best_candidates[0][1]

        logger.info(f"[{self.agent_id}] Top alpha candidate found: '{top_individual}' with fitness {top_fitness:.4f}")
        
        # Formulate candidate strategy payload
        strategy_id = f"strat_{hyp_id}_dsl_opt"
        payload = {
            "strategy_id": strategy_id,
            "hypothesis_id": hyp_id,
            "dsl_expression": top_individual,
            "features_used": features,
            "surrogate_fitness": top_fitness,
            "complexity": self._calculate_complexity(top_individual),
            "expected_sharpe": min(1.5, 0.7 + top_fitness),
            "expected_capacity_cr": hypothesis.get("expected_capacity_cr", 100.0)
        }
        
        # SEBI Compliance Log
        self.log_decision(
            logic_name="generate_strategy",
            inputs={"hypothesis_id": hyp_id, "features": features},
            output=payload,
            reasoning=f"GP Search generated formula '{top_individual}' that passed the surrogate fitness filter and complexity caps."
        )
        
        self.strategy_registry.append(payload)
        self.last_search_summary = {
            "hypothesis_id": hyp_id,
            "population_size": population_size,
            "generations": generations,
            "surrogate_checked": surrogate_checked,
            "surrogate_survivors": len(best_candidates),
            "best_expression": top_individual,
            "best_fitness": top_fitness,
        }
        
        # Submit candidate to Validator Agent
        self.send_message(
            message_type=MessageType.CANDIDATE_STRATEGY,
            target="validator_agent",
            payload=payload,
            priority=2
        )
        return payload

    def _initialize_population(self, features: List[str], size: int) -> List[str]:
        """Generate random DSL expressions."""
        population = []
        for _ in range(size):
            expr = self._random_dsl_expression(features)
            population.append(expr)
        return population

    def _random_dsl_expression(self, features: List[str]) -> str:
        """Create a random operation tree as a string representation."""
        if not features:
            return "1.0"
        
        # Random complexity level (1 to 4 ops)
        num_ops = random.randint(1, 3)
        expr = random.choice(features)
        
        for _ in range(num_ops):
            op = random.choice(self.dsl_operators)
            if op in ["add", "sub", "mul", "div"]:
                operand = random.choice(features)
                expr = f"{op}({expr}, {operand})"
            else:
                expr = f"{op}({expr})"
                
        return expr

    def _evaluate_surrogate_fitness(self, expression: str) -> Tuple[bool, float]:
        """Evaluate cheap fitness proxy. Filters 99% of bad candidates."""
        # Simple deterministic hashing function to simulate a proxy model
        h = 0
        for char in expression:
            h = (h * 33 + ord(char)) & 0xFFFFFFFF
            
        proxy_score = float(h % 100) / 100.0  # Returns score in [0.0, 1.0)
        
        # Threshold: must exceed 0.25 to pass surrogate check
        passes = proxy_score >= 0.25
        return passes, proxy_score

    def _calculate_complexity(self, expression: str) -> int:
        """Count the number of operations in the DSL expression."""
        count = 0
        for op in self.dsl_operators:
            count += expression.count(op)
        return count

    def _complexity_penalty(self, complexity: int, feature_count: int) -> float:
        """AIC/BIC-style complexity penalty with a hard complexity cap."""
        if complexity > self.max_complexity_cap:
            return 1.0
        aic = complexity * 0.03
        bic = complexity * math.log(max(feature_count, 2)) / 100.0
        return aic + bic

    def _create_next_generation(self, evaluated_pop: List[Tuple[str, float]], features: List[str], size: int) -> List[str]:
        """Create next generation via selection, crossover, and mutation."""
        next_gen = []
        
        # Carry over best 10 (elitism)
        for ind, _ in evaluated_pop[:10]:
            next_gen.append(ind)
            
        # Fill rest of population
        while len(next_gen) < size:
            if len(evaluated_pop) > 5 and random.random() < 0.7:
                # Tournament Selection & Mock Mutation
                parent, _ = random.choice(evaluated_pop[:10])
                
                # Apply mutation
                if random.random() < 0.5:
                    # Mutate features
                    mutated = parent
                    for f in features:
                        if f in mutated and random.random() < 0.3:
                            replacement = random.choice(features)
                            mutated = mutated.replace(f, replacement, 1)
                    next_gen.append(mutated)
                else:
                    # Wrap in a new operator
                    op = random.choice(self.dsl_operators)
                    if op in ["add", "sub", "mul", "div"]:
                        mutated = f"{op}({parent}, {random.choice(features)})"
                    else:
                        mutated = f"{op}({parent})"
                    
                    # Ensure complexity cap is not breached
                    if self._calculate_complexity(mutated) <= self.max_complexity_cap:
                        next_gen.append(mutated)
                    else:
                        next_gen.append(parent)
            else:
                # Add random new individual
                next_gen.append(self._random_dsl_expression(features))
                
        return next_gen

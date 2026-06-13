"""
LLM Client for MadEvolve Alpha Evolution

Integrates with GPT-4 and Claude API for automated alpha discovery
and evolution using Large Language Models.

Key Features:
- GPT-4 API integration
- Claude API integration
- Alpha code generation
- Alpha mutation and optimization
- Prompt engineering for financial tasks
- Error handling and fallbacks
- Rate limiting and cost management

Based on Blueprint Week 13-14: LLM-Driven Alpha Evolution
"""

import os
import time
from typing import Optional, Dict, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLM provider."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMClient:
    """
    LLM client for GPT-4 and Claude API integration.
    
    Provides a unified interface for interacting with different LLM providers
    for alpha code generation and evolution.
    """
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 2000,
        temperature: float = 0.7
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider (OPENAI or ANTHROPIC)
            api_key: API key (reads from env if not provided)
            model: Model name
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif provider == LLMProvider.OPENAI:
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif provider == LLMProvider.ANTHROPIC:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.api_key = None
        
        if not self.api_key:
            logger.warning(f"No API key provided for {provider}")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))
        
        self.last_request_time = time.time()
        
        if not self.api_key:
            # Fallback to deterministic generation
            return self._deterministic_generation(prompt)
        
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._call_openai(prompt)
            elif self.provider == LLMProvider.ANTHROPIC:
                return self._call_claude(prompt)
            else:
                return self._deterministic_generation(prompt)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._deterministic_generation(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI GPT-4 API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quantitative finance expert specializing in alpha generation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except ImportError:
            logger.warning("OpenAI library not installed")
            return self._deterministic_generation(prompt)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._deterministic_generation(prompt)
    
    def _call_claude(self, prompt: str) -> str:
        """
        Call Anthropic Claude API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except ImportError:
            logger.warning("Anthropic library not installed")
            return self._deterministic_generation(prompt)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return self._deterministic_generation(prompt)
    
    def _deterministic_generation(self, prompt: str) -> str:
        """
        Deterministic fallback generation when API unavailable.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated code
        """
        # Simple template-based generation
        if "momentum" in prompt.lower():
            return """
def alpha(features):
    # Momentum strategy
    if 'returns' in features.columns:
        recent_returns = features['returns'].iloc[-20:]
        signal = np.sign(recent_returns.mean())
        return np.clip(signal, -1, 1)
    return 0.0
"""
        elif "mean reversion" in prompt.lower():
            return """
def alpha(features):
    # Mean reversion strategy
    if 'close' in features.columns:
        prices = features['close']
        sma = prices.rolling(window=20).mean()
        signal = -np.sign(prices.iloc[-1] - sma.iloc[-1])
        return np.clip(signal, -1, 1)
    return 0.0
"""
        elif "volatility" in prompt.lower():
            return """
def alpha(features):
    # Volatility strategy
    if 'volatility' in features.columns:
        vol = features['volatility'].iloc[-1]
        signal = -np.sign(vol - features['volatility'].mean())
        return np.clip(signal, -1, 1)
    return 0.0
"""
        else:
            return """
def alpha(features):
    # Default strategy
    if 'returns' in features.columns:
        return np.sign(features['returns'].iloc[-1])
    return 0.0
"""
    
    def generate_alpha(
        self,
        description: str,
        strategy_type: str = "momentum"
    ) -> str:
        """
        Generate alpha code from description.
        
        Args:
            description: Natural language description
            strategy_type: Type of strategy
            
        Returns:
            Generated alpha code
        """
        prompt = f"""
Write a Python function `alpha(features)` that returns a trading signal in [-1, 1].

Description: {description}
Strategy Type: {strategy_type}

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

Requirements:
1. Use only features available at prediction time (no future data)
2. Return a signal where:
   - Positive values indicate long position
   - Negative values indicate short position
   - Magnitude indicates confidence
3. Be computationally efficient
4. Handle edge cases (missing data, division by zero)

Return only the code, no explanation.
"""
        
        return self.generate(prompt)
    
    def mutate_alpha(
        self,
        current_code: str,
        mutation_type: str = "parameter"
    ) -> str:
        """
        Mutate existing alpha code.
        
        Args:
            current_code: Current alpha code
            mutation_type: Type of mutation (parameter, logic, feature)
            
        Returns:
            Mutated alpha code
        """
        prompt = f"""
Improve this alpha code to increase Sharpe ratio:

{current_code}

Make one small mutation:
- If {mutation_type} == 'parameter': Change a numerical parameter
- If {mutation_type} == 'logic': Modify the logic slightly
- If {mutation_type} == 'feature': Add or replace a feature

Return only the new code, no explanation.
"""
        
        return self.generate(prompt)
    
    def explain_alpha(
        self,
        code: str
    ) -> str:
        """
        Explain alpha code logic.
        
        Args:
            code: Alpha code to explain
            
        Returns:
            Explanation
        """
        prompt = f"""
Explain this alpha trading strategy code:

{code}

Explain:
1. What the strategy does
2. What signals it generates
3. What market conditions it works best in
4. Potential risks and limitations
"""
        
        return self.generate(prompt)


class MadEvolveLLMIntegration:
    """
    Integration of LLM with MadEvolve alpha evolution.
    
    This class connects the LLM client with the MadEvolve system
    for automated alpha discovery and evolution.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_generations: int = 10,
        mutation_rate: float = 0.3
    ):
        """
        Initialize MadEvolve LLM integration.
        
        Args:
            llm_client: LLM client instance
            max_generations: Maximum generations per evolution
            mutation_rate: Probability of mutation
        """
        self.llm_client = llm_client
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
    
    def discover_alpha(
        self,
        description: str,
        strategy_type: str = "momentum"
    ) -> Dict:
        """
        Discover new alpha using LLM.
        
        Args:
            description: Strategy description
            strategy_type: Type of strategy
            
        Returns:
            Dictionary with alpha code and metadata
        """
        code = self.llm_client.generate_alpha(description, strategy_type)
        
        return {
            'code': code,
            'description': description,
            'strategy_type': strategy_type,
            'generation': 0,
            'source': 'LLM'
        }
    
    def evolve_alpha(
        self,
        current_alpha: Dict,
        n_mutations: int = 3
    ) -> List[Dict]:
        """
        Evolve alpha using LLM-guided mutations.
        
        Args:
            current_alpha: Current alpha dictionary
            n_mutations: Number of mutations to generate
            
        Returns:
            List of mutated alphas
        """
        mutated_alphas = []
        
        mutation_types = ['parameter', 'logic', 'feature']
        
        for i in range(n_mutations):
            mutation_type = mutation_types[i % len(mutation_types)]
            
            mutated_code = self.llm_client.mutate_alpha(
                current_alpha['code'],
                mutation_type
            )
            
            mutated_alphas.append({
                'code': mutated_code,
                'description': f"Mutated from: {current_alpha['description']}",
                'strategy_type': current_alpha['strategy_type'],
                'generation': current_alpha['generation'] + 1,
                'mutation_type': mutation_type,
                'parent_id': current_alpha.get('id'),
                'source': 'LLM'
            })
        
        return mutated_alphas
    
    def weekly_alpha_discovery(
        self,
        strategy_descriptions: List[str]
    ) -> List[Dict]:
        """
        Perform weekly automated alpha discovery.
        
        Args:
            strategy_descriptions: List of strategy descriptions
            
        Returns:
            List of discovered alphas
        """
        discovered_alphas = []
        
        for description in strategy_descriptions:
            alpha = self.discover_alpha(description)
            discovered_alphas.append(alpha)
        
        return discovered_alphas


if __name__ == "__main__":
    # Test LLM client
    print("Testing LLM Client for MadEvolve...")
    
    # Create client (without API key for testing)
    client = LLMClient(
        provider=LLMProvider.OPENAI,
        api_key=None,  # Will use deterministic fallback
        model="gpt-4"
    )
    
    # Test generation
    print("\nTesting alpha generation...")
    code = client.generate_alpha(
        "Momentum strategy based on 20-day returns",
        strategy_type="momentum"
    )
    print(f"Generated code:\n{code}")
    
    # Test mutation
    print("\nTesting alpha mutation...")
    mutated = client.mutate_alpha(code, mutation_type="parameter")
    print(f"Mutated code:\n{mutated}")
    
    # Test MadEvolve integration
    print("\nTesting MadEvolve LLM Integration...")
    mad_evolve = MadEvolveLLMIntegration(client)
    
    alpha = mad_evolve.discover_alpha("Mean reversion using RSI", "mean_reversion")
    print(f"Discovered alpha: {alpha['description']}")
    
    mutations = mad_evolve.evolve_alpha(alpha, n_mutations=2)
    print(f"Generated {len(mutations)} mutations")
    
    # Test weekly discovery
    print("\nTesting weekly alpha discovery...")
    descriptions = [
        "Volatility breakout strategy",
        "Volume-weighted price action",
        "Sector rotation strategy"
    ]
    
    discovered = mad_evolve.weekly_alpha_discovery(descriptions)
    print(f"Discovered {len(discovered)} alphas")
    
    print("\nLLM Client for MadEvolve test completed.")

"""
Alpha Marketplace Registry.
Treats alphas as trackable assets. Each alpha must define capacity, 
decay, expected returns, and continuous self-evaluation metrics.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlphaMetadata:
    expected_sharpe: float
    capacity_limit_usd: float
    turnover_daily_pct: float
    decay_half_life_days: int
    author: str
    causal_hypothesis_id: str

@dataclass
class AlphaPerformance:
    observed_sharpe: float = 0.0
    current_capacity_usage: float = 0.0
    is_active: bool = True
    failure_reason: Optional[str] = None

class AlphaMarketplace:
    """Central registry and evaluator for all deployed alphas."""
    def __init__(self):
        self.registered_alphas: Dict[str, AlphaMetadata] = {}
        self.performance_stats: Dict[str, AlphaPerformance] = {}
        
    def register_alpha(self, alpha_id: str, metadata: AlphaMetadata) -> None:
        """Register a new alpha into the marketplace."""
        if metadata.expected_sharpe < 1.0:
            logger.warning(f"Alpha {alpha_id} has sub-1.0 Sharpe. Subject to strict capital limits.")
            
        self.registered_alphas[alpha_id] = metadata
        self.performance_stats[alpha_id] = AlphaPerformance()
        logger.info(f"Registered alpha {alpha_id} in marketplace.")
        
    def evaluate_alphas(self) -> None:
        """Continuous Self-Evaluation System."""
        for alpha_id, meta in self.registered_alphas.items():
            perf = self.performance_stats[alpha_id]
            if not perf.is_active:
                continue
                
            # Simulate evaluation logic
            deviation = meta.expected_sharpe - perf.observed_sharpe
            
            if deviation > 1.0:
                # Institutional "What assumption failed?" check
                perf.is_active = False
                perf.failure_reason = "Regime shift or Spread expansion exceeded tolerance."
                logger.error(f"Alpha {alpha_id} deactivated. Reason: {perf.failure_reason}")
                
    def get_available_alphas(self) -> List[str]:
        """Returns alphas ready for the Capital Allocation Engine."""
        return [aid for aid, perf in self.performance_stats.items() if perf.is_active]

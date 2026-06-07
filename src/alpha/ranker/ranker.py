"""
Alpha Ranker - Multi-metric scoring and ranking of alpha strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AlphaPerformance:
    """Performance metrics for an alpha"""
    alpha_id: str
    regime_id: int
    is_online: bool
    date: str
    daily_return: float
    sharpe_rolling_21d: float
    hit_rate_21d: float
    turnover_21d: float
    drawdown_21d: float


class AlphaRanker:
    """Rank alphas based on multi-metric scoring"""
    
    def __init__(self, decay_half_life: int = 21):
        self.decay_half_life = decay_half_life
        self.weights = {
            'sharpe': 0.4,
            'hit_rate': 0.2,
            'turnover_penalty': -0.2,
            'drawdown_penalty': -0.2
        }
    
    def score(self, alpha_perf: List[AlphaPerformance], regime: Optional[str] = None) -> float:
        """
        Compute composite score for an alpha
        
        Args:
            alpha_perf: List of performance metrics
            regime: Current regime (optional for regime-specific scoring)
            
        Returns:
            Composite score (higher is better)
        """
        if not alpha_perf:
            return 0.0
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame([p.__dict__ for p in alpha_perf])
        
        # Compute individual component scores
        sharpe_score = df['sharpe_rolling_21d'] * self.weights['sharpe']
        hit_rate_score = df['hit_rate_21d'] * self.weights['hit_rate']
        turnover_penalty = df['turnover_21d'] * self.weights['turnover_penalty']
        drawdown_penalty = df['drawdown_21d'] * self.weights['drawdown_penalty']
        
        # Combine scores
        combined_score = sharpe_score + hit_rate_score + turnover_penalty + drawdown_penalty
        
        # Apply exponential decay to older performance
        if len(combined_score) > 1:
            decay_weights = np.exp(-np.arange(len(combined_score)) / self.decay_half_life)
            decay_weights = decay_weights / decay_weights.sum()
            weighted_score = np.average(combined_score, weights=decay_weights)
        else:
            weighted_score = combined_score.iloc[0]
        
        return weighted_score
    
    def rank_alphas(self, alpha_perfs: Dict[str, List[AlphaPerformance]], 
                   regime: Optional[str] = None) -> List[tuple]:
        """
        Rank multiple alphas
        
        Args:
            alpha_perfs: Dict mapping alpha_id to performance metrics
            regime: Current regime
            
        Returns:
            List of (alpha_id, score) tuples sorted by score (descending)
        """
        scores = []
        for alpha_id, perf in alpha_perfs.items():
            score = self.score(perf, regime)
            scores.append((alpha_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def compute_regime_weights(self, regime: str, alpha_regime_deps: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Compute regime-based weights for alphas
        
        Args:
            regime: Current regime
            alpha_regime_deps: Dict mapping alpha_id to regime dependencies
            
        Returns:
            Dict mapping alpha_id to weight
        """
        weights = {}
        for alpha_id, deps in alpha_regime_deps.items():
            if not deps or regime in deps:
                weights[alpha_id] = 1.0
            else:
                weights[alpha_id] = 0.0
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def adjust_for_correlation(self, scores: List[tuple], 
                              correlation_matrix: pd.DataFrame) -> List[tuple]:
        """
        Adjust scores based on alpha correlations
        
        Args:
            scores: List of (alpha_id, score) tuples
            correlation_matrix: Correlation matrix of alpha returns
            
        Returns:
            Adjusted list of (alpha_id, score) tuples
        """
        if len(scores) <= 1:
            return scores
        
        alpha_ids = [s[0] for s in scores]
        original_scores = np.array([s[1] for s in scores])
        
        # Compute diversification penalty
        corr_subset = correlation_matrix.loc[alpha_ids, alpha_ids]
        avg_corr = corr_subset.values[np.triu_indices_from(corr_subset.values, k=1)].mean()
        
        # Penalize highly correlated alphas
        correlation_penalty = 1.0 - avg_corr
        adjusted_scores = original_scores * correlation_penalty
        
        # Re-sort
        adjusted = list(zip(alpha_ids, adjusted_scores))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        
        return adjusted

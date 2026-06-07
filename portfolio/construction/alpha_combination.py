"""
Alpha Combination Engine
Risk-parity + Kelly (15%) portfolio construction

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm

from alpha_engines.base import AlphaSignal, AlphaMetrics


@dataclass
class AlphaCombinationConfig:
    """Configuration for Alpha Combination Engine"""
    method: str = "risk_parity_kelly"
    kelly_fraction: float = 0.15  # 15% of optimal Kelly
    
    # Risk parity parameters
    target_volatility: float = 0.15  # 15% annual
    
    # Correlation penalty
    use_correlation_penalty: bool = True
    correlation_threshold: float = 0.5
    correlation_penalty_factor: float = 0.5
    
    # Constraints
    max_single_alpha_weight: float = 0.50
    min_alpha_weight: float = 0.05
    
    # Regime-based weights
    use_regime_weights: bool = True
    regime_weights: Dict[str, Dict[str, float]] = None


@dataclass
class CombinedSignal:
    """Combined alpha signal"""
    timestamp: datetime
    alpha_weights: Dict[str, float]
    regime: Optional[str] = None
    total_confidence: float = 0.0
    expected_return: float = 0.0
    risk_budget: float = 0.0


class AlphaCombinationEngine:
    """
    Alpha Combination Engine
    
    Combines multiple alpha signals using:
    1. Risk-parity allocation (equal volatility contribution)
    2. Kelly criterion (15% of optimal)
    3. Regime-based weighting
    4. Correlation penalty
    
    Methods:
    - Risk-parity: Scale each alpha to contribute equal volatility
    - Kelly: f* = (μ - r) / σ²
    - Combined: Risk-parity weights × Kelly fraction
    """
    
    def __init__(self, config: dict):
        self.config = AlphaCombinationConfig(**config)
        
        # Set default regime weights if not provided
        if self.config.regime_weights is None:
            self.config.regime_weights = {
                "bull_trend": {
                    "ORB": 0.40,
                    "VWAP": 0.30,
                    "PCP": 0.15,
                    "VolCarry": 0.10
                },
                "bear_trend": {
                    "ORB": 0.20,
                    "VWAP": 0.40,
                    "PCP": 0.20,
                    "VolCarry": 0.15
                },
                "sideways": {
                    "ORB": 0.10,
                    "VWAP": 0.10,
                    "PCP": 0.30,
                    "VolCarry": 0.40
                },
                "high_vol": {
                    "ORB": 0.15,
                    "VWAP": 0.15,
                    "PCP": 0.20,
                    "VolCarry": 0.40
                }
            }
        
        # Alpha performance tracking
        self.alpha_metrics: Dict[str, AlphaMetrics] = {}
        self.alpha_returns: Dict[str, List[float]] = {}
        
        # Correlation matrix
        self.correlation_matrix: Optional[np.ndarray] = None
        self.alpha_names: List[str] = []
    
    def update_alpha_metrics(self, alpha_name: str, metrics: AlphaMetrics) -> None:
        """Update performance metrics for an alpha"""
        self.alpha_metrics[alpha_name] = metrics
        self.alpha_names = list(self.alpha_metrics.keys())
    
    def update_alpha_return(self, alpha_name: str, return_pct: float) -> None:
        """Update return history for an alpha"""
        if alpha_name not in self.alpha_returns:
            self.alpha_returns[alpha_name] = []
        
        self.alpha_returns[alpha_name].append(return_pct)
        
        # Keep last 252 returns (1 year)
        if len(self.alpha_returns[alpha_name]) > 252:
            self.alpha_returns[alpha_name] = self.alpha_returns[alpha_name][-252:]
    
    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix between alphas"""
        n = len(self.alpha_names)
        if n < 2:
            return np.eye(n)
        
        # Build return matrix
        min_length = min(len(self.alpha_returns.get(name, [])) for name in self.alpha_names)
        if min_length < 30:
            return np.eye(n)
        
        returns_matrix = []
        for name in self.alpha_names:
            returns = self.alpha_returns[name][-min_length:]
            returns_matrix.append(returns)
        
        returns_matrix = np.array(returns_matrix)
        
        # Calculate correlation
        corr_matrix = np.corrcoef(returns_matrix)
        
        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def calculate_risk_parity_weights(
        self,
        volatilities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate risk-parity weights.
        
        Risk-parity: w_i = (1/σ_i) / Σ(1/σ_j)
        Each alpha contributes equal volatility to portfolio.
        """
        inv_vols = {name: 1.0 / vol if vol > 0 else 0 for name, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol == 0:
            # Equal weights if all vols are zero
            n = len(volatilities)
            return {name: 1.0 / n for name in volatilities.keys()}
        
        weights = {name: inv_vol / total_inv_vol for name, inv_vol in inv_vols.items()}
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights)
        
        return weights
    
    def calculate_kelly_fraction(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate Kelly criterion fraction.
        
        Kelly: f* = (μ - r) / σ²
        """
        if volatility == 0:
            return 0.0
        
        excess_return = expected_return - risk_free_rate
        kelly = excess_return / (volatility ** 2)
        
        # Apply 15% fraction (conservative)
        kelly = kelly * self.config.kelly_fraction
        
        # Cap at 1.0 (no leverage > 1x for single alpha)
        return min(1.0, max(0.0, kelly))
    
    def apply_correlation_penalty(
        self,
        weights: Dict[str, float],
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Apply correlation penalty to weights.
        
        If correlation between alphas > threshold, reduce weights.
        """
        if not self.config.use_correlation_penalty:
            return weights
        
        n = len(self.alpha_names)
        if n < 2 or correlation_matrix is None:
            return weights
        
        # Calculate average correlation for each alpha
        avg_correlations = {}
        for i, name in enumerate(self.alpha_names):
            correlations = correlation_matrix[i, :]
            correlations[i] = 0  # Exclude self-correlation
            avg_corr = np.mean(np.abs(correlations))
            avg_correlations[name] = avg_corr
        
        # Apply penalty
        penalized_weights = {}
        for name, weight in weights.items():
            avg_corr = avg_correlations.get(name, 0)
            if avg_corr > self.config.correlation_threshold:
                penalty = self.config.correlation_penalty_factor
                penalized_weights[name] = weight * penalty
            else:
                penalized_weights[name] = weight
        
        # Renormalize
        total = sum(penalized_weights.values())
        if total > 0:
            penalized_weights = {name: w / total for name, w in penalized_weights.items()}
        
        return penalized_weights
    
    def combine_signals(
        self,
        signals: Dict[str, List[AlphaSignal]],
        regime: Optional[str] = None
    ) -> CombinedSignal:
        """
        Combine alpha signals into portfolio weights.
        
        Args:
            signals: Dictionary mapping alpha name to list of signals
            regime: Current market regime
            
        Returns:
            CombinedSignal with weights and metadata
        """
        # Get volatilities from alpha metrics
        volatilities = {}
        expected_returns = {}
        
        for alpha_name in self.alpha_names:
            metrics = self.alpha_metrics.get(alpha_name)
            if metrics:
                # Convert Sharpe to expected return: μ = Sharpe × σ
                vol = 1.0 / metrics.sharpe_ratio if metrics.sharpe_ratio > 0 else 0.2
                volatilities[alpha_name] = vol
                expected_returns[alpha_name] = metrics.sharpe_ratio * vol
        
        # If no metrics available, use equal volatilities
        if not volatilities:
            for alpha_name in signals.keys():
                volatilities[alpha_name] = 0.2  # Default 20% vol
                expected_returns[alpha_name] = 0.15  # Default 15% return
        
        # Calculate risk-parity weights
        rp_weights = self.calculate_risk_parity_weights(volatilities)
        
        # Apply regime-based adjustment
        if self.config.use_regime_weights and regime:
            regime_weights = self.config.regime_weights.get(regime, {})
            rp_weights = self._apply_regime_adjustment(rp_weights, regime_weights)
        
        # Calculate correlation matrix and apply penalty
        corr_matrix = self.calculate_correlation_matrix()
        weights = self.apply_correlation_penalty(rp_weights, corr_matrix)
        
        # Calculate Kelly fractions
        kelly_weights = {}
        for name in weights.keys():
            exp_ret = expected_returns.get(name, 0.15)
            vol = volatilities.get(name, 0.2)
            kelly = self.calculate_kelly_fraction(exp_ret, vol)
            kelly_weights[name] = kelly
        
        # Final weights: Risk-parity × Kelly
        final_weights = {}
        for name in weights.keys():
            final_weights[name] = weights[name] * kelly_weights[name]
        
        # Normalize
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {name: w / total for name, w in final_weights.items()}
        
        # Calculate portfolio metrics
        portfolio_expected_return = sum(
            final_weights[name] * expected_returns.get(name, 0)
            for name in final_weights
        )
        
        portfolio_volatility = np.sqrt(sum(
            final_weights[i] * final_weights[j] * volatilities.get(self.alpha_names[i], 0.2) * volatilities.get(self.alpha_names[j], 0.2) * corr_matrix[i, j]
            for i in range(len(self.alpha_names))
            for j in range(len(self.alpha_names))
        )) if corr_matrix is not None else 0.2
        
        total_confidence = np.mean([
            s.confidence for signals_list in signals.values() for s in signals_list
        ]) if signals else 0.5
        
        return CombinedSignal(
            timestamp=datetime.now(),
            alpha_weights=final_weights,
            regime=regime,
            total_confidence=total_confidence,
            expected_return=portfolio_expected_return,
            risk_budget=portfolio_volatility
        )
    
    def _apply_regime_adjustment(
        self,
        base_weights: Dict[str, float],
        regime_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply regime-based weight adjustment"""
        adjusted = {}
        
        for name in base_weights:
            regime_weight = regime_weights.get(name, 0.25)
            adjusted[name] = base_weights[name] * regime_weight
        
        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {name: w / total for name, w in adjusted.items()}
        
        return adjusted
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints"""
        constrained = {}
        
        for name, weight in weights.items():
            # Cap max weight
            if weight > self.config.max_single_alpha_weight:
                weight = self.config.max_single_alpha_weight
            # Ensure min weight
            if weight < self.config.min_alpha_weight and weight > 0:
                weight = self.config.min_alpha_weight
            
            constrained[name] = weight
        
        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            constrained = {name: w / total for name, w in constrained.items()}
        
        return constrained
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        target_vol: float = 0.15
    ) -> np.ndarray:
        """
        Optimize portfolio using mean-variance optimization.
        
        Objective: Minimize variance subject to target volatility
        """
        n = len(expected_returns)
        
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Bounds: 0 <= w <= max_weight
        bounds = [(0, self.config.max_single_alpha_weight) for _ in range(n)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n) / n
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            return initial_weights

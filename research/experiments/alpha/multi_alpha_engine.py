"""
Multi-Alpha Combination Engine with Dynamic Regime-Based Weighting
Based on Blueprint V1.0

Architecture:
- 20 alphas across 7 categories
- Dynamic weighting based on regime
- Performance-based adjustment
- Turnover constraints
- Risk-adjusted combination

CRITICAL FIX: Added weekly rebalancing to reduce turnover and costs.
Daily rebalancing causes excessive transaction costs.

Regime-Specific Weights:
- Bull Trend: Momentum 50%, MeanRev 5%, Vol 15%, Options 10%, Micro 10%, Factor 10%
- Bear Trend: Momentum 40%, MeanRev 10%, Vol 20%, Options 15%, Micro 5%, Factor 10%
- Sideways: Momentum 10%, MeanRev 50%, Vol 15%, Options 15%, Micro 5%, Factor 5%
- High Vol: Momentum 20%, MeanRev 10%, Vol 40%, Options 20%, Micro 5%, Factor 5%
- Low Vol: Momentum 30%, MeanRev 20%, Vol 10%, Options 10%, Micro 20%, Factor 10%
- Panic: Momentum 0%, MeanRev 20%, Vol 30%, Options 30%, Micro 10%, Factor 10%
- Euphoria: Momentum 20%, MeanRev 20%, Vol 10%, Options 10%, Micro 20%, Factor 20%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class RegimeType(Enum):
    """Market regime types."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    PANIC = "panic"
    EUPHORIA = "euphoria"


class AlphaCategory(Enum):
    """Alpha categories."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    OPTIONS = "options"
    MICROSTRUCTURE = "microstructure"
    FACTOR = "factor"


@dataclass
class AlphaSignal:
    """Individual alpha signal."""
    alpha_name: str
    alpha_category: AlphaCategory
    signal: float  # Normalized signal (-1 to 1)
    confidence: float  # Confidence score (0 to 1)
    timestamp: datetime
    metadata: Dict = None


@dataclass
class CombinedSignal:
    """Combined alpha signal."""
    signal: float
    confidence: float
    weights: Dict[str, float]
    regime: RegimeType
    timestamp: datetime
    component_signals: Dict[str, AlphaSignal]


@dataclass
class AlphaConfig:
    """Configuration for multi-alpha engine."""
    
    # Weight constraints
    max_single_alpha_weight: float = 0.2  # Max 20% per alpha
    max_category_weight: float = 0.5  # Max 50% per category
    min_weight: float = 0.01  # Min 1% weight
    
    # Turnover constraints
    max_daily_weight_change: float = 0.2  # Max 20% daily change
    rebalance_frequency: str = "weekly"  # CRITICAL FIX: "daily" or "weekly"
    
    # Performance adjustment
    performance_lookback: int = 20  # Days
    performance_decay: float = 0.95  # Decay factor
    min_sharpe_for_inclusion: float = 0.5
    
    # Regime override
    use_regime_weights: bool = True
    regime_adjustment_strength: float = 0.7  # How much to follow regime weights
    
    # Risk adjustment
    use_risk_parity: bool = False
    risk_target: float = 0.15  # 15% vol target


class MultiAlphaEngine:
    """
    Multi-Alpha Combination Engine with Dynamic Regime-Based Weighting.
    
    Combines 20 alphas across 7 categories using:
    1. Regime-based base weights
    2. Performance-based adjustments
    3. Risk parity (optional)
    4. Turnover constraints
    """
    
    def __init__(self, config: AlphaConfig = None):
        self.config = config or AlphaConfig()
        
        # Alpha registry
        self.alpha_registry = self._build_alpha_registry()
        
        # Regime weights from blueprint
        self.regime_weights = self._build_regime_weights()
        
        # Current weights
        self.current_weights = {alpha: 1.0/len(self.alpha_registry) for alpha in self.alpha_registry}
        
        # Performance tracking
        self.alpha_performance = {alpha: {'sharpe': 0.5, 'returns': [], 'last_update': datetime.now()} 
                                 for alpha in self.alpha_registry}
        
        # Weight history for turnover constraint
        self.weight_history = []
        
        # Track last rebalance date for weekly rebalancing
        self.last_rebalance_date = None
        
    def _build_alpha_registry(self) -> Dict[str, AlphaCategory]:
        """Build registry of actually available alphas."""
        return {
            'orb_zarattini': AlphaCategory.MEAN_REVERSION,
            'vwap_trend_zarattini': AlphaCategory.MOMENTUM
        }
    
    def _build_regime_weights(self) -> Dict[RegimeType, Dict[str, float]]:
        """Build regime-specific weights from blueprint."""
        return {
            RegimeType.BULL_TREND: { 'momentum': 0.80, 'mean_reversion': 0.20 },
            RegimeType.BEAR_TREND: { 'momentum': 0.80, 'mean_reversion': 0.20 },
            RegimeType.SIDEWAYS: { 'momentum': 0.20, 'mean_reversion': 0.80 },
            RegimeType.HIGH_VOL: { 'momentum': 0.30, 'mean_reversion': 0.70 },
            RegimeType.LOW_VOL: { 'momentum': 0.60, 'mean_reversion': 0.40 },
            RegimeType.PANIC: { 'momentum': 0.10, 'mean_reversion': 0.90 },
            RegimeType.EUPHORIA: { 'momentum': 0.50, 'mean_reversion': 0.50 }
        }
    
    def _should_rebalance(self, current_date: datetime) -> bool:
        """Check if we should rebalance based on frequency setting."""
        if self.config.rebalance_frequency == "daily":
            return True
        
        if self.config.rebalance_frequency == "weekly":
            if self.last_rebalance_date is None:
                return True
            # Rebalance if it's been at least 7 days since last rebalance
            days_since_rebalance = (current_date - self.last_rebalance_date).days
            return days_since_rebalance >= 7
        
        return True
    
    def combine_signals(
        self,
        alpha_signals: List[AlphaSignal],
        regime: RegimeType = RegimeType.SIDEWAYS,
        regime_multiplier: float = 1.0
    ) -> CombinedSignal:
        """
        Combine multiple alpha signals into a single signal.
        
        Args:
            alpha_signals: List of individual alpha signals
            regime: Current market regime
            regime_multiplier: Position multiplier from regime engine
            
        Returns:
            Combined signal with weights and metadata
        """
        if not alpha_signals:
            return CombinedSignal(
                signal=0.0, confidence=0.0, weights={}, regime=regime,
                timestamp=datetime.now(), component_signals={}
            )
        
        current_date = datetime.now()
        
        # Check if we should rebalance (CRITICAL FIX: weekly rebalancing to reduce turnover)
        if self._should_rebalance(current_date):
            # Update weights based on regime and performance
            self._update_weights(regime)
            self.last_rebalance_date = current_date
        
        # Calculate weighted signal
        weighted_signal = 0.0
        total_confidence = 0.0
        component_dict = {}
        
        for signal in alpha_signals:
            if signal.alpha_name not in self.current_weights:
                continue
            
            weight = self.current_weights[signal.alpha_name]
            weighted_signal += weight * signal.signal * signal.confidence
            total_confidence += weight * signal.confidence
            component_dict[signal.alpha_name] = signal
        
        # Normalize
        if total_confidence > 0:
            weighted_signal /= total_confidence
            total_confidence /= len(alpha_signals)
        
        # Apply regime multiplier
        weighted_signal *= regime_multiplier
        
        return CombinedSignal(
            signal=weighted_signal,
            confidence=total_confidence,
            weights=self.current_weights.copy(),
            regime=regime,
            timestamp=datetime.now(),
            component_signals=component_dict
        )
    
    def _update_weights(self, regime: RegimeType) -> None:
        """Update weights based on regime and performance."""
        # Get regime-based category weights
        regime_cat_weights = self.regime_weights.get(regime, self.regime_weights[RegimeType.SIDEWAYS])
        
        # Calculate new weights
        new_weights = {}
        
        for alpha_name, category in self.alpha_registry.items():
            # Base weight from regime
            cat_weight = regime_cat_weights.get(category.value, 1.0/6)
            
            # Performance adjustment
            perf = self.alpha_performance[alpha_name]
            perf_adj = self._calculate_performance_adjustment(perf)
            
            # Combine
            base_weight = cat_weight / self._count_alphas_in_category(category)
            new_weights[alpha_name] = base_weight * (1 + perf_adj)
        
        # Apply constraints
        new_weights = self._apply_weight_constraints(new_weights)
        
        # Apply turnover constraint
        new_weights = self._apply_turnover_constraint(new_weights)
        
        # Normalize to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        # Update current weights
        self.current_weights = new_weights
        self.weight_history.append(new_weights.copy())
        
        # Keep only last 30 days of history
        if len(self.weight_history) > 30:
            self.weight_history = self.weight_history[-30:]
    
    def _count_alphas_in_category(self, category: AlphaCategory) -> int:
        """Count number of alphas in a category."""
        return sum(1 for cat in self.alpha_registry.values() if cat == category)
    
    def _calculate_performance_adjustment(self, performance: Dict) -> float:
        """Calculate performance-based weight adjustment."""
        sharpe = performance['sharpe']
        
        # Sharpe-based adjustment
        if sharpe > 1.0:
            return 0.2  # Boost by 20%
        elif sharpe > 0.8:
            return 0.1  # Boost by 10%
        elif sharpe > 0.5:
            return 0.0  # No adjustment
        elif sharpe > 0.3:
            return -0.1  # Reduce by 10%
        else:
            return -0.2  # Reduce by 20%
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints."""
        constrained = weights.copy()
        
        # Single alpha max
        for alpha, weight in constrained.items():
            constrained[alpha] = min(weight, self.config.max_single_alpha_weight)
        
        # Category max
        for category in AlphaCategory:
            cat_alphas = [a for a, c in self.alpha_registry.items() if c == category]
            cat_weight = sum(constrained[a] for a in cat_alphas)
            
            if cat_weight > self.config.max_category_weight:
                scale = self.config.max_category_weight / cat_weight
                for alpha in cat_alphas:
                    constrained[alpha] *= scale
        
        # Min weight
        for alpha, weight in constrained.items():
            constrained[alpha] = max(weight, self.config.min_weight)
        
        return constrained
    
    def _apply_turnover_constraint(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply turnover constraint to limit daily weight changes."""
        if not self.weight_history:
            return new_weights
        
        prev_weights = self.weight_history[-1]
        constrained = new_weights.copy()
        
        for alpha in constrained:
            if alpha in prev_weights:
                change = abs(constrained[alpha] - prev_weights[alpha])
                max_change = self.config.max_daily_weight_change
                
                if change > max_change:
                    # Cap the change
                    direction = 1 if constrained[alpha] > prev_weights[alpha] else -1
                    constrained[alpha] = prev_weights[alpha] + direction * max_change
        
        return constrained
    
    def update_performance(self, alpha_name: str, returns: float) -> None:
        """
        Update performance tracking for an alpha.
        
        Args:
            alpha_name: Name of the alpha
            returns: Daily return
        """
        if alpha_name not in self.alpha_performance:
            return
        
        perf = self.alpha_performance[alpha_name]
        perf['returns'].append(returns)
        
        # Keep only lookback period
        if len(perf['returns']) > self.config.performance_lookback:
            perf['returns'] = perf['returns'][-self.config.performance_lookback:]
        
        # Update Sharpe
        if len(perf['returns']) > 1:
            returns_array = np.array(perf['returns'])
            sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252) if returns_array.std() > 0 else 0
            perf['sharpe'] = sharpe
        
        perf['last_update'] = datetime.now()
    
    def get_category_weights(self, regime: RegimeType = RegimeType.SIDEWAYS) -> Dict[str, float]:
        """Get current category-level weights."""
        regime_cat_weights = self.regime_weights.get(regime, self.regime_weights[RegimeType.SIDEWAYS])
        return regime_cat_weights
    
    def get_alpha_weights(self) -> Dict[str, float]:
        """Get current alpha-level weights."""
        return self.current_weights.copy()
    
    def get_top_alphas(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N alphas by weight."""
        sorted_weights = sorted(self.current_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_weights[:n]
    
    def get_regime_recommendation(self, regime: RegimeType) -> Dict[str, str]:
        """
        Get regime-based strategy recommendations.
        
        Args:
            regime: Current regime
            
        Returns:
            Dictionary of recommendations
        """
        recommendations = {
            RegimeType.BULL_TREND: {
                'primary': 'Focus on momentum strategies',
                'secondary': 'Reduce mean reversion',
                'risk': 'Moderate risk, trend-following',
                'position_sizing': 'Full position sizes'
            },
            RegimeType.BEAR_TREND: {
                'primary': 'Focus on volatility and options',
                'secondary': 'Reduce momentum exposure',
                'risk': 'Higher risk, defensive positioning',
                'position_sizing': 'Reduce position sizes'
            },
            RegimeType.SIDEWAYS: {
                'primary': 'Focus on mean reversion',
                'secondary': 'Use range-bound strategies',
                'risk': 'Low risk, range trading',
                'position_sizing': 'Normal position sizes'
            },
            RegimeType.HIGH_VOL: {
                'primary': 'Focus on volatility strategies',
                'secondary': 'Use options for hedging',
                'risk': 'High risk, volatility trading',
                'position_sizing': 'Reduce position sizes'
            },
            RegimeType.LOW_VOL: {
                'primary': 'Focus on momentum and microstructure',
                'secondary': 'Use market making strategies',
                'risk': 'Low risk, mean reversion',
                'position_sizing': 'Increase position sizes'
            },
            RegimeType.PANIC: {
                'primary': 'Defensive positioning, reduce exposure',
                'secondary': 'Focus on volatility and options',
                'risk': 'Very high risk, defensive',
                'position_sizing': 'Minimum position sizes'
            },
            RegimeType.EUPHORIA: {
                'primary': 'Reduce risk, prepare for reversal',
                'secondary': 'Diversify across categories',
                'risk': 'High risk, contrarian positioning',
                'position_sizing': 'Reduce position sizes'
            }
        }
        
        return recommendations.get(regime, recommendations[RegimeType.SIDEWAYS])


class AlphaPerformanceTracker:
    """Track and evaluate alpha performance over time."""
    
    def __init__(self, alpha_names: List[str]):
        self.alpha_names = alpha_names
        self.returns_history = {alpha: [] for alpha in alpha_names}
        self.sharpe_history = {alpha: [] for alpha in alpha_names}
        self.drawdown_history = {alpha: [] for alpha in alpha_names}
        
    def update(self, alpha_name: str, return_val: float) -> None:
        """Update performance for an alpha."""
        if alpha_name not in self.alpha_names:
            return
        
        self.returns_history[alpha_name].append(return_val)
        
        # Calculate rolling Sharpe
        if len(self.returns_history[alpha_name]) >= 20:
            returns = np.array(self.returns_history[alpha_name][-20:])
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            self.sharpe_history[alpha_name].append(sharpe)
            
            # Calculate drawdown
            cum_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
            self.drawdown_history[alpha_name].append(max_dd)
    
    def get_summary(self) -> pd.DataFrame:
        """Get performance summary for all alphas."""
        summary = []
        
        for alpha in self.alpha_names:
            if not self.returns_history[alpha]:
                continue
            
            returns = np.array(self.returns_history[alpha])
            total_return = (1 + returns).prod() - 1
            avg_return = returns.mean()
            std_return = returns.std()
            
            sharpe = self.sharpe_history[alpha][-1] if self.sharpe_history[alpha] else 0
            max_dd = self.drawdown_history[alpha][-1] if self.drawdown_history[alpha] else 0
            
            summary.append({
                'alpha': alpha,
                'total_return': total_return,
                'avg_return': avg_return,
                'std_return': std_return,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'num_trades': len(returns)
            })
        
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test the multi-alpha engine
    print("Testing Multi-Alpha Engine...")
    
    engine = MultiAlphaEngine()
    
    # Create sample signals
    signals = []
    for alpha_name in list(engine.alpha_registry.keys())[:5]:
        signal = AlphaSignal(
            alpha_name=alpha_name,
            alpha_category=engine.alpha_registry[alpha_name],
            signal=np.random.uniform(-1, 1),
            confidence=np.random.uniform(0.5, 0.9),
            timestamp=datetime.now()
        )
        signals.append(signal)
    
    # Combine signals
    combined = engine.combine_signals(signals, regime=RegimeType.BULL_TREND)
    
    print(f"Combined signal: {combined.signal:.4f}")
    print(f"Confidence: {combined.confidence:.4f}")
    print(f"Top alphas: {engine.get_top_alphas(3)}")
    print(f"Category weights (Bull): {engine.get_category_weights(RegimeType.BULL_TREND)}")

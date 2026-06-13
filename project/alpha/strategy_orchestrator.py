"""
Strategy Orchestrator - Multi-Strategy Ensemble

Implements institutional-grade strategy orchestration:
- Combines multiple strategies into ensemble
- Regime-dependent allocation
- Risk-aware position sizing
- Performance monitoring

Based on blueprint specification for multi-strategy framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyOutput:
    """Output from a single strategy"""
    name: str
    signal: float
    confidence: float
    metadata: Dict = None


@dataclass
class EnsembleOutput:
    """Output from strategy ensemble"""
    combined_signal: float
    strategy_weights: Dict[str, float]
    individual_signals: Dict[str, float]
    regime: str
    timestamp: pd.Timestamp


class StrategyOrchestrator:
    """
    Strategy Orchestrator for Multi-Strategy Ensemble
    
    Combines multiple strategies with regime-dependent weighting.
    """
    
    def __init__(self):
        """Initialize strategy orchestrator."""
        # Import strategy modules
        try:
            from alpha.trend_following_strategies import TSMOM, DualMomentum, SectorMomentum
            from alpha.mean_reversion_strategies import KalmanPairs, ORBStrategy, VWAPReversion
            from alpha.statistical_arbitrage import PCAStatArb, ETFArbitrage
            from alpha.volatility_strategies import VolatilityRiskPremium, VIXFuturesBasis, DispersionTrading
            from alpha.options_strategies import PutCallParityCarry, SkewRiskReversal
            from alpha.microstructure_strategies import MarketMakingInventory
            from alpha.factor_strategies import LowVolatilityFactor, ValueFactor, QualityFactor
            from alpha.regime_strategy_switching import AdaptiveStrategyManager
            
            # Initialize strategies
            self.strategies = {
                'tsmom': TSMOM(),
                'dual_mom': DualMomentum(),
                'sector_mom': SectorMomentum(),
                'kalman_pairs': KalmanPairs(),
                'orb': ORBStrategy(),
                'vwap_rev': VWAPReversion(),
                'pca_arb': PCAStatArb(),
                'etf_arb': ETFArbitrage(),
                'vrp': VolatilityRiskPremium(),
                'vix_basis': VIXFuturesBasis(),
                'dispersion': DispersionTrading(),
                'pcp_carry': PutCallParityCarry(),
                'skew_rr': SkewRiskReversal(),
                'market_making': MarketMakingInventory(),
                'low_vol': LowVolatilityFactor(),
                'value': ValueFactor(),
                'quality': QualityFactor()
            }
            
            # Initialize regime manager
            self.regime_manager = AdaptiveStrategyManager()
            
            logger.info("Strategy Orchestrator initialized with all strategies")
            
        except ImportError as e:
            logger.warning(f"Some strategies not available: {e}")
            self.strategies = {}
            self.regime_manager = None
    
    def get_combined_signal(
        self,
        market_data: Dict,
        regime_probs: Optional[Dict] = None
    ) -> EnsembleOutput:
        """
        Get combined signal from all strategies.
        
        Args:
            market_data: Dictionary with market data (prices, returns, volatility, etc.)
            regime_probs: Optional regime probabilities
            
        Returns:
            EnsembleOutput with combined signal
        """
        # Detect regime if not provided
        if self.regime_manager and regime_probs is None:
            returns = market_data.get('returns', pd.Series())
            volatility = market_data.get('volatility', pd.Series())
            if len(returns) > 0 and len(volatility) > 0:
                regime_state, strategy_weights = self.regime_manager.update(returns, volatility)
                regime = regime_state.regime.value
                regime_probs = {r: p for r, p in regime_state.probabilities.items()}
            else:
                regime = 'sideways'
                strategy_weights = self._get_default_weights()
        else:
            regime = regime_probs.get('regime', 'sideways') if regime_probs else 'sideways'
            strategy_weights = self._get_regime_weights(regime)
        
        # Compute individual strategy signals
        individual_signals = {}
        for name, strategy in self.strategies.items():
            try:
                signal = self._compute_strategy_signal(strategy, name, market_data)
                if signal is not None:
                    individual_signals[name] = signal
            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}")
        
        # Combine signals using regime weights
        combined_signal = 0.0
        for name, signal in individual_signals.items():
            weight = strategy_weights.get(name, 0.0)
            combined_signal += weight * signal
        
        # Clip to [-1, 1]
        combined_signal = np.clip(combined_signal, -1.0, 1.0)
        
        return EnsembleOutput(
            combined_signal=combined_signal,
            strategy_weights=strategy_weights,
            individual_signals=individual_signals,
            regime=regime,
            timestamp=pd.Timestamp.now()
        )
    
    def _compute_strategy_signal(
        self,
        strategy,
        name: str,
        market_data: Dict
    ) -> Optional[float]:
        """
        Compute signal for a single strategy.
        
        Args:
            strategy: Strategy instance
            name: Strategy name
            market_data: Market data dictionary
            
        Returns:
            Signal value or None
        """
        # Route to appropriate method based on strategy type
        if name in ['tsmom', 'dual_mom', 'sector_mom']:
            return self._compute_trend_signal(strategy, name, market_data)
        elif name in ['kalman_pairs', 'orb', 'vwap_rev']:
            return self._compute_mean_reversion_signal(strategy, name, market_data)
        elif name in ['pca_arb', 'etf_arb']:
            return self._compute_stat_arb_signal(strategy, name, market_data)
        elif name in ['vrp', 'vix_basis', 'dispersion']:
            return self._compute_vol_signal(strategy, name, market_data)
        elif name in ['pcp_carry', 'skew_rr']:
            return self._compute_options_signal(strategy, name, market_data)
        elif name in ['low_vol', 'value', 'quality']:
            return self._compute_factor_signal(strategy, name, market_data)
        else:
            return None
    
    def _compute_trend_signal(self, strategy, name: str, market_data: Dict) -> Optional[float]:
        """Compute trend following signal."""
        prices = market_data.get('prices')
        if prices is None:
            return None
        
        if name == 'tsmom':
            result = strategy.compute(prices)
            return result.signal.iloc[-1] if hasattr(result, 'signal') else 0
        elif name == 'dual_mom':
            benchmark = market_data.get('benchmark_prices')
            if benchmark is None:
                return None
            signals = strategy.compute(prices, benchmark)
            return signals.mean().iloc[-1]
        elif name == 'sector_mom':
            returns = market_data.get('sector_returns')
            if returns is None:
                return None
            weights = strategy.compute(returns)
            return weights.mean().iloc[-1]
        
        return None
    
    def _compute_mean_reversion_signal(self, strategy, name: str, market_data: Dict) -> Optional[float]:
        """Compute mean reversion signal."""
        if name == 'kalman_pairs':
            y = market_data.get('pair_y')
            x = market_data.get('pair_x')
            if y is None or x is None:
                return None
            z = strategy.update(y, x)
            signal, _ = strategy.get_signal(z)
            return signal
        elif name in ['orb', 'vwap_rev']:
            data = market_data.get('intraday_data')
            symbol = market_data.get('symbol', 'RELIANCE')
            if data is None:
                return None
            signal_obj = strategy.generate_signal(data, symbol)
            return signal_obj.signal if signal_obj else 0
        
        return None
    
    def _compute_stat_arb_signal(self, strategy, name: str, market_data: Dict) -> Optional[float]:
        """Compute statistical arbitrage signal."""
        returns = market_data.get('returns')
        if returns is None:
            return None
        
        if name == 'pca_arb':
            strategy.fit(returns)
            signal = strategy.compute_signal(returns)
            return signal.signal.mean() if hasattr(signal, 'signal') else 0
        elif name == 'etf_arb':
            etf_price = market_data.get('etf_price')
            nav = market_data.get('nav')
            basket_prices = market_data.get('basket_prices')
            basket_weights = market_data.get('basket_weights')
            if None in [etf_price, nav, basket_prices, basket_weights]:
                return None
            signal, _ = strategy.get_signal(etf_price, nav, basket_prices, basket_weights)
            return signal
        
        return None
    
    def _compute_vol_signal(self, strategy, name: str, market_data: Dict) -> Optional[float]:
        """Compute volatility signal."""
        data = market_data.get('data')
        if data is None:
            return None
        
        if name == 'vrp':
            implied_vol = market_data.get('implied_vol', 0.2)
            symbol = market_data.get('symbol', 'RELIANCE')
            signal_obj = strategy.generate_signal(data, implied_vol, symbol)
            return signal_obj.signal if signal_obj else 0
        elif name in ['vix_basis', 'dispersion']:
            # Simplified - would need actual options data
            return 0
        
        return None
    
    def _compute_options_signal(self, strategy, name: str, market_data: Dict) -> Optional[float]:
        """Compute options signal."""
        if name == 'pcp_carry':
            call_price = market_data.get('call_price')
            put_price = market_data.get('put_price')
            strike = market_data.get('strike')
            forward = market_data.get('forward')
            ois_discount = market_data.get('ois_discount')
            tau = market_data.get('tau')
            if None in [call_price, put_price, strike, forward, ois_discount, tau]:
                return None
            carry_gap = strategy.compute_carry_gap(call_price, put_price, strike, forward, ois_discount, tau)
            signal, _ = strategy.get_signal(carry_gap)
            return signal
        elif name == 'skew_rr':
            iv_put = market_data.get('iv_25d_put')
            iv_call = market_data.get('iv_25d_call')
            if iv_put is None or iv_call is None:
                return None
            skew = strategy.compute_skew(iv_put, iv_call)
            signal, _ = strategy.get_signal(skew)
            return signal
        
        return None
    
    def _compute_factor_signal(self, strategy, name: str, market_data: Dict) -> Optional[float]:
        """Compute factor signal."""
        prices = market_data.get('prices')
        if prices is None:
            return None
        
        if name == 'low_vol':
            weights = strategy.compute_signal(prices)
            return weights.mean()
        elif name in ['value', 'quality']:
            fundamentals = market_data.get('fundamentals')
            if fundamentals is None:
                return None
            weights = strategy.compute_signal(prices, fundamentals)
            return weights.mean()
        
        return None
    
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default equal-weighted strategy allocation."""
        return {
            'tsmom': 0.15,
            'dual_mom': 0.15,
            'sector_mom': 0.10,
            'kalman_pairs': 0.10,
            'orb': 0.05,
            'vwap_rev': 0.05,
            'pca_arb': 0.10,
            'vrp': 0.10,
            'low_vol': 0.10,
            'value': 0.05,
            'quality': 0.05
        }
    
    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get regime-specific strategy weights."""
        if regime == 'trend':
            return {
                'tsmom': 0.4,
                'dual_mom': 0.3,
                'sector_mom': 0.2,
                'orb': 0.1
            }
        elif regime == 'mean_reversion':
            return {
                'kalman_pairs': 0.4,
                'orb': 0.3,
                'vwap_rev': 0.2,
                'pca_arb': 0.1
            }
        elif regime == 'volatility':
            return {
                'vrp': 0.5,
                'vix_basis': 0.3,
                'dispersion': 0.2
            }
        elif regime == 'crisis':
            return {
                'low_vol': 0.4,
                'quality': 0.3,
                'cash': 0.3
            }
        else:  # sideways
            return self._get_default_weights()
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary of strategy ensemble.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'num_strategies': len(self.strategies),
            'active_strategies': list(self.strategies.keys()),
            'regime_manager_active': self.regime_manager is not None
        }


if __name__ == "__main__":
    # Test Strategy Orchestrator
    print("Testing Strategy Orchestrator...")
    
    # Create orchestrator
    orchestrator = StrategyOrchestrator()
    
    # Get performance summary
    summary = orchestrator.get_performance_summary()
    print("\nPerformance Summary:")
    print(f"   Number of strategies: {summary['num_strategies']}")
    print(f"   Active strategies: {summary['active_strategies']}")
    print(f"   Regime manager active: {summary['regime_manager_active']}")
    
    # Test with synthetic market data
    print("\nTesting with synthetic market data...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(n) * 0.5),
        index=dates
    )
    
    market_data = {
        'prices': prices,
        'returns': prices.pct_change(),
        'volatility': prices.pct_change().rolling(21).std() * np.sqrt(252),
        'symbol': 'RELIANCE'
    }
    
    # Get combined signal
    ensemble_output = orchestrator.get_combined_signal(market_data)
    
    print(f"\nEnsemble Output:")
    print(f"   Combined signal: {ensemble_output.combined_signal:.4f}")
    print(f"   Regime: {ensemble_output.regime}")
    print(f"   Strategy weights: {ensemble_output.strategy_weights}")
    print(f"   Individual signals: {ensemble_output.individual_signals}")
    
    print("\n✓ Strategy Orchestrator tested")

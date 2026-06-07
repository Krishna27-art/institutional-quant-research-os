"""
TSMOM and Dual Momentum Strategies

Implements Time Series Momentum (TSMOM) and Dual Momentum strategies
following the academic literature and institutional best practices.

Key Features:
- TSMOM (Moskowitz et al. 2012) - Sign-based momentum
- Dual Momentum (Antonacci) - Absolute + relative momentum
- Volatility-managed momentum
- Cross-sectional momentum (Jegadeesh & Titman 1993)
- Risk-adjusted position sizing
- Lookback period optimization

Based on Blueprint Week 5-6: Alpha Models (Classical)
References:
- Moskowitz et al. (2012) - Time Series Momentum
- Antonacci (2014) - Dual Momentum
- Jegadeesh & Titman (1993) - Returns to Buying Winners and Selling Losers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MomentumSignal(Enum):
    """Momentum signal classification."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class TSMOMStrategy:
    """
    Time Series Momentum (TSMOM) Strategy.
    
    TSMOM goes long on assets with positive past returns and short on
    assets with negative past returns, scaled by volatility.
    
    Signal = sign(Excess Return over lookback period)
    Position = Signal / Volatility
    """
    
    def __init__(
        self,
        lookback: int = 252,
        volatility_window: int = 60,
        target_volatility: float = 0.15
    ):
        """
        Initialize TSMOM strategy.
        
        Args:
            lookback: Lookback period for momentum signal (days)
            volatility_window: Window for volatility estimation
            target_volatility: Target annualized volatility
        """
        self.lookback = lookback
        self.volatility_window = volatility_window
        self.target_volatility = target_volatility
    
    def generate_signal(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Generate TSMOM signal.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with signal and position
        """
        if len(returns) < self.lookback:
            return {
                'signal': MomentumSignal.NEUTRAL,
                'position': 0.0,
                'excess_return': 0.0,
                'volatility': 0.0,
                'leverage': 0.0
            }
        
        # Calculate excess return over lookback
        excess_return = returns.iloc[-self.lookback:].sum() - risk_free_rate * self.lookback / 252
        
        # Calculate volatility
        volatility = returns.iloc[-self.volatility_window:].std() * np.sqrt(252)
        
        # Generate signal
        if excess_return > 0:
            signal = MomentumSignal.LONG
        elif excess_return < 0:
            signal = MomentumSignal.SHORT
        else:
            signal = MomentumSignal.NEUTRAL
        
        # Calculate position size (volatility-scaled)
        if volatility > 0:
            leverage = self.target_volatility / volatility
            leverage = np.clip(leverage, 0.0, 2.0)  # Cap at 2x leverage
            position = signal.value * leverage
        else:
            position = 0.0
            leverage = 0.0
        
        return {
            'signal': signal,
            'position': position,
            'excess_return': excess_return,
            'volatility': volatility,
            'leverage': leverage
        }
    
    def backtest(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Backtest TSMOM strategy.
        
        Args:
            returns: DataFrame of returns for multiple assets
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame with strategy returns
        """
        positions = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback, len(returns)):
            for asset in returns.columns:
                signal_dict = self.generate_signal(returns[asset].iloc[:i], risk_free_rate)
                positions.iloc[i, asset] = signal_dict['position']
        
        # Calculate strategy returns
        strategy_returns = (positions.shift(1) * returns).sum(axis=1)
        
        return strategy_returns


class DualMomentumStrategy:
    """
    Dual Momentum Strategy (Antonacci 2014).
    
    Combines absolute momentum (trend following) with relative momentum
    (cross-sectional) to select the best performing assets.
    
    1. Apply absolute momentum filter (only invest in assets with positive momentum)
    2. Apply relative momentum (select top performing assets)
    """
    
    def __init__(
        self,
        lookback: int = 252,
        n_assets: int = 3,
        volatility_window: int = 60,
        target_volatility: float = 0.15
    ):
        """
        Initialize dual momentum strategy.
        
        Args:
            lookback: Lookback period for momentum
            n_assets: Number of top assets to select
            volatility_window: Window for volatility estimation
            target_volatility: Target annualized volatility
        """
        self.lookback = lookback
        self.n_assets = n_assets
        self.volatility_window = volatility_window
        self.target_volatility = target_volatility
    
    def generate_signals(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Generate dual momentum signals.
        
        Args:
            returns: DataFrame of returns for multiple assets
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with selected assets and positions
        """
        if len(returns) < self.lookback:
            return {
                'selected_assets': [],
                'positions': pd.Series(0.0, index=returns.columns),
                'momentum_scores': pd.Series(0.0, index=returns.columns)
            }
        
        # Calculate momentum scores (excess returns over lookback)
        momentum_scores = {}
        for asset in returns.columns:
            excess_return = returns[asset].iloc[-self.lookback:].sum() - risk_free_rate * self.lookback / 252
            momentum_scores[asset] = excess_return
        
        momentum_series = pd.Series(momentum_scores)
        
        # Absolute momentum filter (only positive momentum)
        positive_momentum = momentum_series[momentum_series > 0]
        
        if len(positive_momentum) == 0:
            return {
                'selected_assets': [],
                'positions': pd.Series(0.0, index=returns.columns),
                'momentum_scores': momentum_series
            }
        
        # Relative momentum (select top n)
        if len(positive_momentum) > self.n_assets:
            selected = positive_momentum.nlargest(self.n_assets)
        else:
            selected = positive_momentum
        
        # Equal-weight positions
        positions = pd.Series(0.0, index=returns.columns)
        positions[selected.index] = 1.0 / len(selected)
        
        # Volatility scaling
        volatilities = {}
        for asset in selected.index:
            vol = returns[asset].iloc[-self.volatility_window:].std() * np.sqrt(252)
            volatilities[asset] = vol
        
        if volatilities:
            avg_vol = np.mean(list(volatilities.values()))
            if avg_vol > 0:
                scaling = self.target_volatility / avg_vol
                scaling = np.clip(scaling, 0.5, 2.0)
                positions[selected.index] *= scaling
        
        return {
            'selected_assets': list(selected.index),
            'positions': positions,
            'momentum_scores': momentum_series
        }
    
    def backtest(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0
    ) -> pd.DataFrame:
        """
        Backtest dual momentum strategy.
        
        Args:
            returns: DataFrame of returns for multiple assets
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame with strategy returns
        """
        positions = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(self.lookback, len(returns)):
            signal_dict = self.generate_signals(returns.iloc[:i], risk_free_rate)
            positions.iloc[i] = signal_dict['positions']
        
        # Calculate strategy returns
        strategy_returns = (positions.shift(1) * returns).sum(axis=1)
        
        return strategy_returns


class VolatilityManagedMomentum:
    """
    Volatility-Managed Momentum Strategy.
    
    Adjusts momentum positions based on realized volatility to
    improve risk-adjusted returns (Barroso & Santa-Clara 2015).
    
    Position = Momentum Signal / (Volatility^γ)
    where γ is the volatility scaling parameter.
    """
    
    def __init__(
        self,
        lookback: int = 252,
        volatility_window: int = 60,
        gamma: float = 0.5,
        target_volatility: float = 0.15
    ):
        """
        Initialize volatility-managed momentum.
        
        Args:
            lookback: Lookback period for momentum
            volatility_window: Window for volatility estimation
            gamma: Volatility scaling parameter
            target_volatility: Target annualized volatility
        """
        self.lookback = lookback
        self.volatility_window = volatility_window
        self.gamma = gamma
        self.target_volatility = target_volatility
    
    def generate_signal(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> Dict:
        """
        Generate volatility-managed momentum signal.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with signal and position
        """
        if len(returns) < self.lookback:
            return {
                'signal': MomentumSignal.NEUTRAL,
                'position': 0.0,
                'excess_return': 0.0,
                'volatility': 0.0,
                'volatility_scaled_position': 0.0
            }
        
        # Calculate excess return
        excess_return = returns.iloc[-self.lookback:].sum() - risk_free_rate * self.lookback / 252
        
        # Calculate volatility
        volatility = returns.iloc[-self.volatility_window:].std() * np.sqrt(252)
        
        # Generate signal
        if excess_return > 0:
            signal = MomentumSignal.LONG
        elif excess_return < 0:
            signal = MomentumSignal.SHORT
        else:
            signal = MomentumSignal.NEUTRAL
        
        # Volatility scaling
        if volatility > 0:
            vol_scaled = 1.0 / (volatility ** self.gamma)
            # Normalize to target volatility
            scaling = self.target_volatility / (volatility ** (1 - self.gamma))
            scaling = np.clip(scaling, 0.0, 2.0)
            position = signal.value * scaling
        else:
            vol_scaled = 0.0
            position = 0.0
        
        return {
            'signal': signal,
            'position': position,
            'excess_return': excess_return,
            'volatility': volatility,
            'volatility_scaled_position': vol_scaled
        }


class CrossSectionalMomentum:
    """
    Cross-Sectional Momentum Strategy (Jegadeesh & Titman 1993).
    
    Ranks assets based on past returns and goes long on winners,
    short on losers. This is relative momentum across assets.
    """
    
    def __init__(
        self,
        lookback: int = 126,
        holding_period: int = 21,
        n_winners: int = 10,
        n_losers: int = 10
    ):
        """
        Initialize cross-sectional momentum.
        
        Args:
            lookback: Lookback period for ranking
            holding_period: Holding period after ranking
            n_winners: Number of winners to go long
            n_losers: Number of losers to short
        """
        self.lookback = lookback
        self.holding_period = holding_period
        self.n_winners = n_winners
        self.n_losers = n_losers
    
    def generate_signals(
        self,
        returns: pd.DataFrame
    ) -> Dict:
        """
        Generate cross-sectional momentum signals.
        
        Args:
            returns: DataFrame of returns for multiple assets
            
        Returns:
            Dictionary with long and short positions
        """
        if len(returns) < self.lookback:
            return {
                'long_positions': pd.Series(0.0, index=returns.columns),
                'short_positions': pd.Series(0.0, index=returns.columns),
                'momentum_rankings': pd.Series(0.0, index=returns.columns)
            }
        
        # Calculate momentum (cumulative returns over lookback)
        momentum = {}
        for asset in returns.columns:
            momentum[asset] = returns[asset].iloc[-self.lookback:].sum()
        
        momentum_series = pd.Series(momentum)
        
        # Rank assets
        rankings = momentum_series.rank(ascending=False)
        
        # Select winners and losers
        winners = rankings.nsmallest(self.n_winners).index
        losers = rankings.nlargest(self.n_losers).index
        
        # Equal-weight positions
        long_positions = pd.Series(0.0, index=returns.columns)
        short_positions = pd.Series(0.0, index=returns.columns)
        
        long_positions[winners] = 1.0 / len(winners)
        short_positions[losers] = -1.0 / len(losers)
        
        return {
            'long_positions': long_positions,
            'short_positions': short_positions,
            'momentum_rankings': rankings
        }


class MomentumFactory:
    """
    Factory for creating momentum strategies.
    
    Provides a unified interface for different momentum strategies.
    """
    
    @staticmethod
    def create_tsmom(
        lookback: int = 252,
        volatility_window: int = 60,
        target_volatility: float = 0.15
    ) -> TSMOMStrategy:
        """Create TSMOM strategy."""
        return TSMOMStrategy(lookback, volatility_window, target_volatility)
    
    @staticmethod
    def create_dual_momentum(
        lookback: int = 252,
        n_assets: int = 3,
        volatility_window: int = 60,
        target_volatility: float = 0.15
    ) -> DualMomentumStrategy:
        """Create dual momentum strategy."""
        return DualMomentumStrategy(lookback, n_assets, volatility_window, target_volatility)
    
    @staticmethod
    def create_volatility_managed(
        lookback: int = 252,
        volatility_window: int = 60,
        gamma: float = 0.5,
        target_volatility: float = 0.15
    ) -> VolatilityManagedMomentum:
        """Create volatility-managed momentum."""
        return VolatilityManagedMomentum(lookback, volatility_window, gamma, target_volatility)
    
    @staticmethod
    def create_cross_sectional(
        lookback: int = 126,
        holding_period: int = 21,
        n_winners: int = 10,
        n_losers: int = 10
    ) -> CrossSectionalMomentum:
        """Create cross-sectional momentum."""
        return CrossSectionalMomentum(lookback, holding_period, n_winners, n_losers)


if __name__ == "__main__":
    # Test momentum strategies
    print("Testing TSMOM and Dual Momentum Strategies...")
    
    # Create sample returns
    np.random.seed(42)
    n_samples = 500
    n_assets = 5
    
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            np.zeros(n_assets),
            np.eye(n_assets) * 0.02,
            n_samples
        ),
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Test TSMOM
    print("\nTesting TSMOM Strategy...")
    tsmom = TSMOMStrategy(lookback=252, volatility_window=60, target_volatility=0.15)
    signal = tsmom.generate_signal(returns['Asset_0'])
    print(f"TSMOM Signal: {signal['signal']}")
    print(f"TSMOM Position: {signal['position']:.4f}")
    print(f"Excess Return: {signal['excess_return']:.4f}")
    print(f"Volatility: {signal['volatility']:.4f}")
    
    # Test Dual Momentum
    print("\nTesting Dual Momentum Strategy...")
    dual_mom = DualMomentumStrategy(lookback=252, n_assets=3, target_volatility=0.15)
    signals = dual_mom.generate_signals(returns)
    print(f"Selected Assets: {signals['selected_assets']}")
    print(f"Positions:\n{signals['positions']}")
    
    # Test Volatility-Managed Momentum
    print("\nTesting Volatility-Managed Momentum...")
    vol_mom = VolatilityManagedMomentum(lookback=252, gamma=0.5, target_volatility=0.15)
    signal = vol_mom.generate_signal(returns['Asset_0'])
    print(f"Signal: {signal['signal']}")
    print(f"Position: {signal['position']:.4f}")
    print(f"Volatility-Scaled Position: {signal['volatility_scaled_position']:.4f}")
    
    # Test Cross-Sectional Momentum
    print("\nTesting Cross-Sectional Momentum...")
    cross_mom = CrossSectionalMomentum(lookback=126, n_winners=2, n_losers=2)
    signals = cross_mom.generate_signals(returns)
    print(f"Long Positions:\n{signals['long_positions']}")
    print(f"Short Positions:\n{signals['short_positions']}")
    
    print("\nTSMOM and Dual Momentum Strategies test completed.")

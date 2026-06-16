"""
Volatility Alpha Strategies

This module implements volatility-based alpha strategies including:
- Volatility Risk Premium (VRP) - Carr & Wu (2009)
- Volatility Targeting
- VIX Futures Basis Trading - Simon & Campasano (2014)
- Gamma Scalping
- Dispersion Trading (Index vs Components)

Based on Audit Report Priority 2: Alpha Generation
Research Papers: Bollerslev et al (2009), Carr & Wu (2009), Simon & Campasano (2014)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolatilitySignal:
    """Volatility trading signal."""
    symbol: str
    strategy: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    implied_vol: float
    realized_vol: float
    vol_premium: float
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VolatilityRiskPremium:
    """
    Volatility Risk Premium (VRP) Strategy.
    
    Exploits the difference between implied volatility (from options)
    and realized volatility (from historical returns).
    """
    
    def __init__(self, lookback_days: int = 20, premium_threshold: float = 0.05):
        """
        Initialize VRP strategy.
        
        Args:
            lookback_days: Lookback period for realized volatility
            premium_threshold: Minimum premium for entry
        """
        self.lookback_days = lookback_days
        self.premium_threshold = premium_threshold
        
        logger.info(f"VRPStrategy initialized with {lookback_days} day lookback")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        implied_vol: float,
        symbol: str
    ) -> Optional[VolatilitySignal]:
        """
        Generate VRP signal.
        
        Args:
            data: DataFrame with OHLCV data
            implied_vol: Implied volatility from options
            symbol: Stock symbol
            
        Returns:
            VolatilitySignal
        """
        if len(data) < self.lookback_days:
            return None
        
        # Calculate realized volatility
        recent_data = data.tail(self.lookback_days)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return None
        
        realized_vol = returns.std() * np.sqrt(252)
        
        # Calculate volatility risk premium
        vol_premium = implied_vol - realized_vol
        
        # Generate signal
        if vol_premium > self.premium_threshold:
            # Implied vol > realized vol - sell volatility (short options)
            signal = -1.0
            confidence = min(1.0, (vol_premium - self.premium_threshold) / 0.1)
        elif vol_premium < -self.premium_threshold:
            # Implied vol < realized vol - buy volatility (long options)
            signal = 1.0
            confidence = min(1.0, (abs(vol_premium) - self.premium_threshold) / 0.1)
        else:
            signal = 0.0
            confidence = 0.0
        
        return VolatilitySignal(
            symbol=symbol,
            strategy="VRP",
            signal=signal,
            confidence=confidence,
            implied_vol=implied_vol,
            realized_vol=realized_vol,
            vol_premium=vol_premium,
            timestamp=datetime.now(),
            metadata={
                'lookback_days': self.lookback_days,
                'premium_threshold': self.premium_threshold
            }
        )


class VolatilityTargeting:
    """
    Volatility Targeting Strategy.
    
    Adjusts position sizes to maintain target portfolio volatility.
    """
    
    def __init__(self, target_vol: float = 0.15, lookback_days: int = 20):
        """
        Initialize volatility targeting strategy.
        
        Args:
            target_vol: Target annualized volatility
            lookback_days: Lookback period for volatility calculation
        """
        self.target_vol = target_vol
        self.lookback_days = lookback_days
        
        logger.info(f"VolatilityTargeting initialized with target vol {target_vol}")
    
    def calculate_position_scale(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Tuple[float, float]:
        """
        Calculate position scaling factor based on volatility.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Tuple of (scale_factor, current_volatility)
        """
        if len(data) < self.lookback_days:
            return 1.0, 0.0
        
        # Calculate realized volatility
        recent_data = data.tail(self.lookback_days)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return 1.0, 0.0
        
        current_vol = returns.std() * np.sqrt(252)
        
        # Calculate scaling factor
        if current_vol > 0:
            scale_factor = self.target_vol / current_vol
            # Clamp to reasonable range
            scale_factor = min(2.0, max(0.5, scale_factor))
        else:
            scale_factor = 1.0
        
        return scale_factor, current_vol
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        base_signal: float
    ) -> Optional[VolatilitySignal]:
        """
        Generate volatility-targeted signal.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            base_signal: Base trading signal (-1 to 1)
            
        Returns:
            VolatilitySignal
        """
        scale_factor, current_vol = self.calculate_position_scale(data, symbol)
        
        # Scale the base signal
        scaled_signal = base_signal * scale_factor
        
        # Confidence based on volatility deviation from target
        vol_deviation = abs(current_vol - self.target_vol) / self.target_vol
        confidence = min(1.0, vol_deviation)
        
        return VolatilitySignal(
            symbol=symbol,
            strategy="VolatilityTargeting",
            signal=scaled_signal,
            confidence=confidence,
            implied_vol=current_vol,
            realized_vol=current_vol,
            vol_premium=0.0,
            timestamp=datetime.now(),
            metadata={
                'scale_factor': scale_factor,
                'target_vol': self.target_vol,
                'base_signal': base_signal
            }
        )


class VIXFuturesBasis:
    """
    VIX Futures Basis Trading Strategy.
    
    Trades the spread between VIX futures and spot VIX.
    """
    
    def __init__(self, lookback_days: int = 20, basis_threshold: float = 2.0):
        """
        Initialize VIX futures basis strategy.
        
        Args:
            lookback_days: Lookback period for basis calculation
            basis_threshold: Basis threshold for entry
        """
        self.lookback_days = lookback_days
        self.basis_threshold = basis_threshold
        
        logger.info(f"VIXFuturesBasis initialized with {lookback_days} day lookback")
    
    def generate_signal(
        self,
        vix_spot: float,
        vix_futures: float,
        vix_data: pd.DataFrame
    ) -> Optional[VolatilitySignal]:
        """
        Generate VIX futures basis signal.
        
        Args:
            vix_spot: Current VIX spot value
            vix_futures: Current VIX futures value
            vix_data: DataFrame with historical VIX data
            
        Returns:
            VolatilitySignal
        """
        if len(vix_data) < self.lookback_days:
            return None
        
        # Calculate basis
        basis = vix_futures - vix_spot
        basis_pct = (basis / vix_spot) * 100 if vix_spot > 0 else 0
        
        # Calculate historical basis
        recent_vix = vix_data.tail(self.lookback_days)
        historical_basis_mean = recent_vix['basis'].mean() if 'basis' in recent_vix.columns else 0
        historical_basis_std = recent_vix['basis'].std() if 'basis' in recent_vix.columns else 1.0
        
        # Calculate z-score of current basis
        z_score = (basis - historical_basis_mean) / historical_basis_std if historical_basis_std > 0 else 0
        
        # Generate signal
        if z_score > self.basis_threshold:
            # Futures at large premium - short futures
            signal = -1.0
            confidence = min(1.0, (z_score - self.basis_threshold) / 2.0)
        elif z_score < -self.basis_threshold:
            # Futures at large discount - long futures
            signal = 1.0
            confidence = min(1.0, (abs(z_score) - self.basis_threshold) / 2.0)
        else:
            signal = 0.0
            confidence = 0.0
        
        return VolatilitySignal(
            symbol="VIX",
            strategy="VIXFuturesBasis",
            signal=signal,
            confidence=confidence,
            implied_vol=vix_futures,
            realized_vol=vix_spot,
            vol_premium=basis_pct,
            timestamp=datetime.now(),
            metadata={
                'basis': basis,
                'basis_pct': basis_pct,
                'z_score': z_score,
                'historical_mean': historical_basis_mean
            }
        )


class GammaScalping:
    """
    Gamma Scalping Strategy.
    
    Delta-hedged options position to profit from gamma.
    """
    
    def __init__(self, option_type: str = "straddle", delta_threshold: float = 0.1):
        """
        Initialize gamma scalping strategy.
        
        Args:
            option_type: Type of option (straddle, strangle)
            delta_threshold: Delta threshold for rebalancing
        """
        self.option_type = option_type
        self.delta_threshold = delta_threshold
        
        logger.info(f"GammaScalping initialized with {option_type} options")
    
    def generate_signal(
        self,
        spot_price: float,
        option_delta: float,
        option_gamma: float,
        symbol: str
    ) -> Optional[VolatilitySignal]:
        """
        Generate gamma scalping signal.
        
        Args:
            spot_price: Current spot price
            option_delta: Current option delta
            option_gamma: Current option gamma
            symbol: Underlying symbol
            
        Returns:
            VolatilitySignal
        """
        # Check if delta exceeds threshold
        if abs(option_delta) > self.delta_threshold:
            # Rebalance delta - trade underlying
            signal = -np.sign(option_delta)  # Trade opposite to delta
            confidence = min(1.0, (abs(option_delta) - self.delta_threshold) / 0.2)
        else:
            signal = 0.0
            confidence = 0.0
        
        # Estimate implied vol from delta/gamma
        implied_vol = abs(option_delta) / (option_gamma * spot_price) if option_gamma > 0 else 0.2
        
        return VolatilitySignal(
            symbol=symbol,
            strategy="GammaScalping",
            signal=signal,
            confidence=confidence,
            implied_vol=implied_vol,
            realized_vol=0.0,
            vol_premium=0.0,
            timestamp=datetime.now(),
            metadata={
                'option_delta': option_delta,
                'option_gamma': option_gamma,
                'spot_price': spot_price,
                'option_type': self.option_type
            }
        )


class DispersionTrading:
    """
    Dispersion Trading (Index vs Components)
    
    Formula:
    Index IV > weighted avg component IV → short index vol, long component vols
    
    Expected Sharpe: 0.6
    Capacity: 1,000 Cr
    Turnover: 40%/month
    Best Regime: Low correlation
    Failure: Crisis (correlations → 1)
    """
    
    def __init__(
        self,
        entry_threshold: float = 0.02,
        lookback: int = 21
    ):
        """
        Initialize dispersion trading.
        
        Args:
            entry_threshold: IV difference threshold for entry (2%)
            lookback: Lookback period for IV calculation
        """
        self.entry_threshold = entry_threshold
        self.lookback = lookback
        
    def compute_signal(
        self,
        index_iv: float,
        component_ivs: np.ndarray,
        component_weights: np.ndarray
    ) -> Tuple[float, str]:
        """
        Compute dispersion trading signal.
        
        Args:
            index_iv: Index implied volatility
            component_ivs: Array of component implied volatilities
            component_weights: Array of component weights
            
        Returns:
            Tuple of (signal, direction)
        """
        # Calculate weighted average component IV
        avg_component_iv = np.sum(component_ivs * component_weights)
        
        # Calculate dispersion
        dispersion = index_iv - avg_component_iv
        
        if dispersion > self.entry_threshold:
            # Index IV too high - short index vol, long component vols
            signal = -1.0
            direction = "SHORT_INDEX_LONG_COMPONENTS"
        elif dispersion < -self.entry_threshold:
            # Index IV too low - long index vol, short component vols
            signal = 1.0
            direction = "LONG_INDEX_SHORT_COMPONENTS"
        else:
            signal = 0.0
            direction = "HOLD"
        
        return signal, direction


def get_volatility_signals(
    data_dict: Dict[str, pd.DataFrame],
    vix_data: pd.DataFrame = None,
    strategies: List[str] = None
) -> Dict[str, List[VolatilitySignal]]:
    """
    Generate volatility signals using multiple strategies.
    
    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        vix_data: DataFrame with VIX data
        strategies: List of strategy names to use
        
    Returns:
        Dictionary mapping strategy names to signal lists
    """
    if strategies is None:
        strategies = ["VRP", "VolatilityTargeting", "VIXFuturesBasis"]
    
    results = {}
    
    # Volatility Risk Premium
    if "VRP" in strategies:
        vrp = VolatilityRiskPremium()
        vrp_signals = []
        for symbol, data in data_dict.items():
            # Use a proxy for implied vol (e.g., historical vol * 1.2)
            if len(data) >= 20:
                recent_data = data.tail(20)
                realized_vol = recent_data['close'].pct_change().std() * np.sqrt(252)
                implied_vol = realized_vol * 1.2  # Proxy
                
                signal = vrp.generate_signal(data, implied_vol, symbol)
                if signal and signal.signal != 0:
                    vrp_signals.append(signal)
        results["VRP"] = vrp_signals
    
    # Volatility Targeting
    if "VolatilityTargeting" in strategies:
        vt = VolatilityTargeting()
        vt_signals = []
        for symbol, data in data_dict.items():
            scale_factor, current_vol = vt.calculate_position_scale(data, symbol)
            if abs(scale_factor - 1.0) > 0.1:  # Only if significant adjustment needed
                signal = vt.generate_signal(data, symbol, 1.0)
                if signal:
                    vt_signals.append(signal)
        results["VolatilityTargeting"] = vt_signals
    
    # VIX Futures Basis
    if "VIXFuturesBasis" in strategies and vix_data is not None:
        vfb = VIXFuturesBasis()
        if len(vix_data) >= 20:
            vix_spot = vix_data['close'].iloc[-1]
            vix_futures = vix_spot * 1.05  # Proxy
            signal = vfb.generate_signal(vix_spot, vix_futures, vix_data)
            if signal:
                results["VIXFuturesBasis"] = [signal]
    
    return results


if __name__ == "__main__":
    # Test volatility strategies
    print("Testing Volatility Strategies...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=300, freq='1D')
    np.random.seed(42)
    
    data_dict = {
        'RELIANCE': pd.DataFrame({
            'close': np.cumprod(1 + np.random.normal(0.001, 0.02, 300)) * 1000
        }, index=dates)
    }
    
    # Generate signals
    signals = get_volatility_signals(data_dict)
    
    print(f"\nGenerated signals:")
    for strategy, signal_list in signals.items():
        print(f"  {strategy}: {len(signal_list)} signals")
        for signal in signal_list[:3]:
            print(f"    {signal.symbol}: {signal.signal:.3f} (confidence: {signal.confidence:.2f})")

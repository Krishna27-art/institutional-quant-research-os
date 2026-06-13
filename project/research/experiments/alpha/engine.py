"""
Alpha Calculation Framework
Implements 20+ alpha strategies across 7 categories for Indian markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

from scipy import stats
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class AlphaType(Enum):
    """Alpha strategy categories"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    OPTIONS = "options"
    MICROSTRUCTURE = "microstructure"
    FACTOR = "factor"
    REGIME = "regime"


@dataclass
class AlphaSignal:
    """Alpha signal structure"""
    symbol: str
    alpha_name: str
    alpha_type: AlphaType
    signal: float  # -1 to 1, negative = short, positive = long
    confidence: float  # 0 to 1
    timestamp: pd.Timestamp
    metadata: Dict


class BaseAlpha(ABC):
    """Base class for alpha strategies"""
    
    def __init__(self, name: str, alpha_type: AlphaType):
        self.name = name
        self.alpha_type = alpha_type
    
    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        """Compute alpha signal"""
        pass
    
    def normalize_signal(self, raw_signal: float) -> float:
        """Normalize signal to [-1, 1] range"""
        return np.tanh(raw_signal)


# ==================== TREND/MOMENTUM ALPHAS ====================

class TimeSeriesMomentum(BaseAlpha):
    """12-1 Momentum (Jegadeesh & Titman)"""
    
    def __init__(self):
        super().__init__("time_series_momentum_12_1", AlphaType.MOMENTUM)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 13:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        
        # 12-1 momentum: return from t-12 to t-1
        momentum_12_1 = np.log(close[-2] / close[-13]) if len(close) >= 13 else 0
        
        # Normalize signal
        signal = self.normalize_signal(momentum_12_1 * 10)  # Scale for tanh
        
        # Confidence based on consistency
        returns = pd.Series(close).pct_change().dropna()
        confidence = min(abs(momentum_12_1) / returns.std(), 1.0) if len(returns) > 0 and returns.std() > 0 else 0.5
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'momentum_12_1': momentum_12_1}
        )


class DualMomentum(BaseAlpha):
    """Dual Momentum (Absolute + Relative)"""
    
    def __init__(self):
        super().__init__("dual_momentum", AlphaType.MOMENTUM)
    
    def compute(self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None, **kwargs) -> AlphaSignal:
        if len(df) < 200:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        
        # Absolute momentum: SMA(200) crossover
        sma_200 = close[-200:].mean()
        abs_momentum = 1 if close[-1] > sma_200 else -1
        
        # Relative momentum (if benchmark provided)
        rel_momentum = 0
        if benchmark_df is not None and len(benchmark_df) >= 200:
            bench_close = benchmark_df['close'].values
            asset_return = close[-1] / close[-200] - 1
            bench_return = bench_close[-1] / bench_close[-200] - 1
            rel_momentum = 1 if asset_return > bench_return else -1
        
        # Combine
        combined_signal = (abs_momentum + rel_momentum) / 2 if benchmark_df is not None else abs_momentum
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=combined_signal,
            confidence=0.7,
            timestamp=pd.Timestamp.now(),
            metadata={'abs_momentum': abs_momentum, 'rel_momentum': rel_momentum}
        )


class VolatilityManagedMomentum(BaseAlpha):
    """Volatility-Managed Momentum (Moreira & Muir)"""
    
    def __init__(self):
        super().__init__("volatility_managed_momentum", AlphaType.MOMENTUM)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 60:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        
        # Momentum signal
        momentum = close[-1] / close[-20] - 1
        
        # Volatility scaling
        returns = pd.Series(close).pct_change().dropna()
        vol = returns.rolling(20).std()[-1] if len(returns) >= 20 else 0.01
        
        # Inverse volatility scaling
        vol_scaled = momentum / (vol + 0.01) if vol > 0 else momentum
        
        signal = self.normalize_signal(vol_scaled * 5)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=0.6,
            timestamp=pd.Timestamp.now(),
            metadata={'momentum': momentum, 'volatility': vol}
        )


class SectorMomentum(BaseAlpha):
    """Sector Momentum Rotation"""
    
    def __init__(self):
        super().__init__("sector_momentum", AlphaType.MOMENTUM)
    
    def compute(self, df: pd.DataFrame, sector_returns: Optional[Dict[str, float]] = None, **kwargs) -> AlphaSignal:
        if len(df) < 126:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        
        # 6-month return
        six_month_return = close[-1] / close[-126] - 1
        
        # Rank within sector (if sector data provided)
        rank_percentile = 0.5  # Default
        if sector_returns:
            all_returns = list(sector_returns.values())
            rank = sum(1 for r in all_returns if r < six_month_return)
            rank_percentile = rank / len(all_returns) if all_returns else 0.5
        
        # Signal based on percentile
        signal = (rank_percentile - 0.5) * 2  # Map to [-1, 1]
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=0.5,
            timestamp=pd.Timestamp.now(),
            metadata={'six_month_return': six_month_return, 'rank_percentile': rank_percentile}
        )


class ResidualMomentum(BaseAlpha):
    """Residual Momentum (Blitz)"""
    
    def __init__(self):
        super().__init__("residual_momentum", AlphaType.MOMENTUM)
    
    def compute(self, df: pd.DataFrame, factor_returns: Optional[pd.DataFrame] = None, **kwargs) -> AlphaSignal:
        if len(df) < 60 or factor_returns is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data_or_no_factors'}
            )
        
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Regress returns on factors
        if len(returns) == len(factor_returns):
            model = LinearRegression()
            model.fit(factor_returns, returns)
            predicted = model.predict(factor_returns)
            residual = returns - predicted
            
            # Momentum of residuals
            residual_momentum = residual.rolling(20).sum()[-1] if len(residual) >= 20 else 0
            
            signal = self.normalize_signal(residual_momentum * 10)
            
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=signal,
                confidence=0.5,
                timestamp=pd.Timestamp.now(),
                metadata={'residual_momentum': residual_momentum}
            )
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=0.0,
            confidence=0.0,
            timestamp=pd.Timestamp.now(),
            metadata={'error': 'mismatched_data'}
        )


# ==================== MEAN REVERSION ALPHAS ====================

class PairsTrading(BaseAlpha):
    """Cointegration Pairs Trading"""
    
    def __init__(self):
        super().__init__("pairs_trading", AlphaType.MEAN_REVERSION)
    
    def compute(self, df: pd.DataFrame, pair_df: Optional[pd.DataFrame] = None, **kwargs) -> AlphaSignal:
        if len(df) < 60 or pair_df is None or len(pair_df) < 60:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data_or_no_pair'}
            )
        
        close = df['close'].values
        pair_close = pair_df['close'].values
        
        # Check cointegration
        score, pvalue, _ = coint(close[-60:], pair_close[-60:])
        
        if pvalue > 0.05:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'not_cointegrated', 'pvalue': pvalue}
            )
        
        # Calculate spread
        model = LinearRegression()
        model.fit(pair_close[-60:].reshape(-1, 1), close[-60:])
        hedge_ratio = model.coef_[0]
        spread = close[-60:] - hedge_ratio * pair_close[-60:]
        
        # Z-score of spread
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Signal: short if z-score > 2, long if z-score < -2
        signal = -np.sign(z_score) * min(abs(z_score) / 2, 1)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=min(abs(z_score) / 3, 1.0),
            timestamp=pd.Timestamp.now(),
            metadata={'z_score': z_score, 'hedge_ratio': hedge_ratio, 'pvalue': pvalue}
        )


class ORB(BaseAlpha):
    """Opening Range Breakout (Zarattini)"""
    
    def __init__(self):
        super().__init__("opening_range_breakout", AlphaType.MEAN_REVERSION)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 10:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
        
        # First 5-minute range (assuming 1-min bars)
        orb_high = high[-5:].max()
        orb_low = low[-5:].min()
        orb_range = orb_high - orb_low
        
        if orb_range == 0:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'zero_range'}
            )
        
        # Current price position
        price_position = (close[-1] - orb_low) / orb_range
        
        # Volume check
        avg_volume = volume[-20:].mean()
        volume_ratio = volume[-1] / avg_volume if avg_volume > 0 else 1
        
        # Signal: breakout if volume > 100% avg
        if volume_ratio > 1.0:
            signal = (price_position - 0.5) * 2  # Map to [-1, 1]
        else:
            signal = 0
        
        confidence = min(volume_ratio - 0.5, 1.0) if volume_ratio > 0.5 else 0
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'orb_high': orb_high, 'orb_low': orb_low, 'volume_ratio': volume_ratio}
        )


class VWAPReversion(BaseAlpha):
    """VWAP Reversion"""
    
    def __init__(self):
        super().__init__("vwap_reversion", AlphaType.MEAN_REVERSION)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 20 or 'volume' not in df.columns:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data_or_no_volume'}
            )
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Distance from VWAP
        dist_from_vwap = (close[-1] - vwap[-1]) / vwap[-1] if vwap[-1] != 0 else 0
        
        # Reversion signal
        signal = -self.normalize_signal(dist_from_vwap * 100)
        
        # Confidence based on distance
        confidence = min(abs(dist_from_vwap) * 50, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'dist_from_vwap': dist_from_vwap, 'vwap': vwap[-1]}
        )


class IBS(BaseAlpha):
    """Internal Bar Strength (Pagonidis)"""
    
    def __init__(self):
        super().__init__("internal_bar_strength", AlphaType.MEAN_REVERSION)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 1:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # IBS calculation
        if high[-1] != low[-1]:
            ibs = (close[-1] - low[-1]) / (high[-1] - low[-1])
        else:
            ibs = 0.5
        
        # Reversion signal: short if IBS > 0.8, long if IBS < 0.2
        if ibs > 0.8:
            signal = -1
            confidence = (ibs - 0.8) / 0.2
        elif ibs < 0.2:
            signal = 1
            confidence = (0.2 - ibs) / 0.2
        else:
            signal = 0
            confidence = 0
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'ibs': ibs}
        )


class ETFArbitrage(BaseAlpha):
    """ETF Arbitrage (NAV vs Market Price)"""
    
    def __init__(self):
        super().__init__("etf_arbitrage", AlphaType.MEAN_REVERSION)
    
    def compute(self, df: pd.DataFrame, nav: Optional[float] = None, **kwargs) -> AlphaSignal:
        if nav is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_nav'}
            )
        
        close = df['close'].values[-1]
        
        # Calculate discount/premium
        discount = (nav - close) / nav if nav != 0 else 0
        
        # Signal: long if discount > 1%, short if premium > 1%
        if discount > 0.01:
            signal = 1
            confidence = min(discount / 0.02, 1.0)
        elif discount < -0.01:
            signal = -1
            confidence = min(abs(discount) / 0.02, 1.0)
        else:
            signal = 0
            confidence = 0
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'nav': nav, 'discount': discount}
        )


# ==================== VOLATILITY ALPHAS ====================

class VolatilityRiskPremium(BaseAlpha):
    """Volatility Risk Premium (Short Vol)"""
    
    def __init__(self):
        super().__init__("volatility_risk_premium", AlphaType.VOLATILITY)
    
    def compute(self, df: pd.DataFrame, iv: Optional[float] = None, **kwargs) -> AlphaSignal:
        if iv is None or len(df) < 20:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_iv_or_insufficient_data'}
            )
        
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Realized volatility
        realized_vol = returns.rolling(20).std()[-1] * np.sqrt(252) if len(returns) >= 20 else 0.15
        
        # VRP = IV - Realized Vol
        vrp = iv - realized_vol
        
        # Signal: short vol if VRP > 0
        signal = -1 if vrp > 0 else 1
        confidence = min(abs(vrp) / 0.1, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'iv': iv, 'realized_vol': realized_vol, 'vrp': vrp}
        )


class VIXFuturesBasis(BaseAlpha):
    """VIX Futures Basis"""
    
    def __init__(self):
        super().__init__("vix_futures_basis", AlphaType.VOLATILITY)
    
    def compute(self, df: pd.DataFrame, vix_futures: Optional[float] = None, vix_spot: Optional[float] = None, **kwargs) -> AlphaSignal:
        if vix_futures is None or vix_spot is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_vix_data'}
            )
        
        # Basis = Futures - Spot
        basis = vix_futures - vix_spot
        
        # Signal: short VIX if contango (basis > 0)
        signal = -1 if basis > 0 else 1
        confidence = min(abs(basis) / 2.0, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'basis': basis, 'vix_futures': vix_futures, 'vix_spot': vix_spot}
        )


class VolatilityTargeting(BaseAlpha):
    """Volatility Targeting"""
    
    def __init__(self):
        super().__init__("volatility_targeting", AlphaType.VOLATILITY)
    
    def compute(self, df: pd.DataFrame, target_vol: float = 0.15, **kwargs) -> AlphaSignal:
        if len(df) < 20:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Current volatility
        current_vol = returns.rolling(20).std()[-1] * np.sqrt(252) if len(returns) >= 20 else 0.15
        
        # Target leverage
        leverage = target_vol / current_vol if current_vol > 0 else 1.0
        leverage = min(leverage, 4.0)  # Cap at 4x
        
        # Signal based on leverage
        signal = min(leverage / 2, 1.0)  # Normalize to [-1, 1]
        confidence = 0.8
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'current_vol': current_vol, 'target_vol': target_vol, 'leverage': leverage}
        )


class DispersionTrading(BaseAlpha):
    """Dispersion Trading"""
    
    def __init__(self):
        super().__init__("dispersion_trading", AlphaType.VOLATILITY)
    
    def compute(self, df: pd.DataFrame, index_iv: Optional[float] = None, constituent_iv: Optional[float] = None, **kwargs) -> AlphaSignal:
        if index_iv is None or constituent_iv is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_iv_data'}
            )
        
        # Dispersion = Index IV - Constituent IV
        dispersion = index_iv - constituent_iv
        
        # Signal: long dispersion if positive (buy index vol, sell constituent vol)
        signal = 1 if dispersion > 0 else -1
        confidence = min(abs(dispersion) / 0.05, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'dispersion': dispersion, 'index_iv': index_iv, 'constituent_iv': constituent_iv}
        )


class VolatilityCarry(BaseAlpha):
    """Volatility Carry (VXX/VXZ)"""
    
    def __init__(self):
        super().__init__("volatility_carry", AlphaType.VOLATILITY)
    
    def compute(self, df: pd.DataFrame, vxx: Optional[float] = None, vxz: Optional[float] = None, **kwargs) -> AlphaSignal:
        if vxx is None or vxz is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_etf_data'}
            )
        
        # Carry trade: short VXX, long VXZ in contango
        signal = -1  # Default short VXX
        confidence = 0.6
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'vxx': vxx, 'vxz': vxz}
        )


# ==================== OPTIONS/SKEW ALPHAS ====================

class PutCallParityCarry(BaseAlpha):
    """Put-Call Parity Carry Gap (Shin 2026)"""
    
    def __init__(self):
        super().__init__("put_call_parity_carry", AlphaType.OPTIONS)
    
    def compute(self, df: pd.DataFrame, carry_gap: Optional[float] = None, **kwargs) -> AlphaSignal:
        if carry_gap is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_carry_gap'}
            )
        
        # Signal: long if positive carry
        signal = 1 if carry_gap > 0 else -1
        confidence = min(abs(carry_gap) / 0.01, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'carry_gap': carry_gap}
        )


class SkewRiskReversal(BaseAlpha):
    """Skew - Long Risk Reversal"""
    
    def __init__(self):
        super().__init__("skew_risk_reversal", AlphaType.OPTIONS)
    
    def compute(self, df: pd.DataFrame, skew: Optional[float] = None, **kwargs) -> AlphaSignal:
        if skew is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_skew'}
            )
        
        # Signal: long risk reversal if skew is steep (positive)
        signal = 1 if skew > 0.05 else -1 if skew < -0.05 else 0
        confidence = min(abs(skew) / 0.1, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'skew': skew}
        )


class GammaScalping(BaseAlpha):
    """Gamma Scalping"""
    
    def __init__(self):
        super().__init__("gamma_scalping", AlphaType.OPTIONS)
    
    def compute(self, df: pd.DataFrame, gamma: Optional[float] = None, realized_vol: Optional[float] = None, **kwargs) -> AlphaSignal:
        if gamma is None or realized_vol is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_greek_data'}
            )
        
        # Signal: long gamma if realized vol > implied vol
        signal = 1 if gamma > 0 and realized_vol > 0.2 else 0
        confidence = min(abs(gamma) * 10, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'gamma': gamma, 'realized_vol': realized_vol}
        )


class VarianceSwapTermStructure(BaseAlpha):
    """Variance Swap Term Structure"""
    
    def __init__(self):
        super().__init__("variance_swap_term_structure", AlphaType.OPTIONS)
    
    def compute(self, df: pd.DataFrame, term_slope: Optional[float] = None, **kwargs) -> AlphaSignal:
        if term_slope is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_term_structure'}
            )
        
        # Signal: long term structure if upward sloping
        signal = 1 if term_slope > 0 else -1
        confidence = min(abs(term_slope) / 0.01, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'term_slope': term_slope}
        )


class ButterflyVolatility(BaseAlpha):
    """Butterfly (Vol Convexity)"""
    
    def __init__(self):
        super().__init__("butterfly_volatility", AlphaType.OPTIONS)
    
    def compute(self, df: pd.DataFrame, vol_convexity: Optional[float] = None, **kwargs) -> AlphaSignal:
        if vol_convexity is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_convexity'}
            )
        
        # Signal: long butterfly if convexity is high
        signal = 1 if vol_convexity > 0 else -1
        confidence = min(abs(vol_convexity) * 100, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'vol_convexity': vol_convexity}
        )


# ==================== MICROSTRUCTURE ALPHAS ====================

class OptimalQuoting(BaseAlpha):
    """Optimal Quoting (Signal-Adaptive)"""
    
    def __init__(self):
        super().__init__("optimal_quoting", AlphaType.MICROSTRUCTURE)
    
    def compute(self, df: pd.DataFrame, signal_strength: Optional[float] = None, inventory: Optional[float] = None, **kwargs) -> AlphaSignal:
        if signal_strength is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_signal'}
            )
        
        # Optimal delta based on signal and inventory
        inventory_penalty = -inventory if inventory else 0
        optimal_delta = signal_strength + inventory_penalty
        
        signal = self.normalize_signal(optimal_delta)
        confidence = 0.7
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'signal_strength': signal_strength, 'inventory': inventory}
        )


class MarketMaking(BaseAlpha):
    """Market Making (Bid-Ask Spread Capture)"""
    
    def __init__(self):
        super().__init__("market_making", AlphaType.MICROSTRUCTURE)
    
    def compute(self, df: pd.DataFrame, spread: Optional[float] = None, fill_prob: Optional[float] = None, **kwargs) -> AlphaSignal:
        if spread is None or fill_prob is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_micro_data'}
            )
        
        # Expected profit = spread * fill_prob
        expected_profit = spread * fill_prob
        
        # Signal: provide liquidity if expected profit > 0
        signal = 1 if expected_profit > 0 else 0
        confidence = min(expected_profit / 0.001, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'spread': spread, 'fill_prob': fill_prob, 'expected_profit': expected_profit}
        )


class OrderImbalance(BaseAlpha):
    """Order Imbalance"""
    
    def __init__(self):
        super().__init__("order_imbalance", AlphaType.MICROSTRUCTURE)
    
    def compute(self, df: pd.DataFrame, buy_vol: Optional[float] = None, sell_vol: Optional[float] = None, **kwargs) -> AlphaSignal:
        if buy_vol is None or sell_vol is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_order_data'}
            )
        
        total_vol = buy_vol + sell_vol
        if total_vol == 0:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'zero_volume'}
            )
        
        # Order imbalance
        imbalance = (buy_vol - sell_vol) / total_vol
        
        # Signal: follow imbalance
        signal = self.normalize_signal(imbalance * 2)
        confidence = min(abs(imbalance) * 2, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'imbalance': imbalance}
        )


class VWAPExecution(BaseAlpha):
    """VWAP Execution"""
    
    def __init__(self):
        super().__init__("vwap_execution", AlphaType.MICROSTRUCTURE)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 20 or 'volume' not in df.columns:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Price vs VWAP
        price_vs_vwap = (close[-1] - vwap[-1]) / vwap[-1] if vwap[-1] != 0 else 0
        
        # Signal: buy if below VWAP
        signal = -self.normalize_signal(price_vs_vwap * 100)
        confidence = min(abs(price_vs_vwap) * 50, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'price_vs_vwap': price_vs_vwap}
        )


class InventoryManagement(BaseAlpha):
    """Inventory Management"""
    
    def __init__(self):
        super().__init__("inventory_management", AlphaType.MICROSTRUCTURE)
    
    def compute(self, df: pd.DataFrame, inventory: Optional[float] = None, **kwargs) -> AlphaSignal:
        if inventory is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_inventory'}
            )
        
        # Skew quotes based on inventory
        # If long inventory, skew towards selling
        signal = -self.normalize_signal(inventory / 1000)
        confidence = min(abs(inventory) / 1000, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'inventory': inventory}
        )


# ==================== FACTOR/STYLE ALPHAS ====================

class LowVolatilityAnomaly(BaseAlpha):
    """Low Volatility Anomaly (Ang)"""
    
    def __init__(self):
        super().__init__("low_volatility_anomaly", AlphaType.FACTOR)
    
    def compute(self, df: pd.DataFrame, **kwargs) -> AlphaSignal:
        if len(df) < 60:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'insufficient_data'}
            )
        
        close = df['close'].values
        returns = pd.Series(close).pct_change().dropna()
        
        # Volatility
        vol = returns.rolling(60).std()[-1] if len(returns) >= 60 else 0.01
        
        # Signal: long low vol stocks
        signal = -self.normalize_signal(vol * 100)
        confidence = 0.5
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'volatility': vol}
        )


class ValueFactor(BaseAlpha):
    """Value (B/P, Earnings Yield)"""
    
    def __init__(self):
        super().__init__("value_factor", AlphaType.FACTOR)
    
    def compute(self, df: pd.DataFrame, book_to_price: Optional[float] = None, earnings_yield: Optional[float] = None, **kwargs) -> AlphaSignal:
        if book_to_price is None and earnings_yield is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_value_metrics'}
            )
        
        # Combine value metrics
        value_score = 0
        if book_to_price:
            value_score += book_to_price
        if earnings_yield:
            value_score += earnings_yield
        
        # Signal: long value stocks
        signal = self.normalize_signal(value_score * 10)
        confidence = 0.5
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'book_to_price': book_to_price, 'earnings_yield': earnings_yield}
        )


class QualityFactor(BaseAlpha):
    """Quality (Profitability, Low Debt)"""
    
    def __init__(self):
        super().__init__("quality_factor", AlphaType.FACTOR)
    
    def compute(self, df: pd.DataFrame, roe: Optional[float] = None, debt_equity: Optional[float] = None, **kwargs) -> AlphaSignal:
        if roe is None and debt_equity is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_quality_metrics'}
            )
        
        # Quality score
        quality_score = 0
        if roe:
            quality_score += roe / 100  # Normalize
        if debt_equity:
            quality_score -= debt_equity / 100  # Penalize high debt
        
        # Signal: long quality stocks
        signal = self.normalize_signal(quality_score)
        confidence = 0.5
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'roe': roe, 'debt_equity': debt_equity}
        )


class CarryFactor(BaseAlpha):
    """Carry (Roll Yield in Futures)"""
    
    def __init__(self):
        super().__init__("carry_factor", AlphaType.FACTOR)
    
    def compute(self, df: pd.DataFrame, roll_yield: Optional[float] = None, **kwargs) -> AlphaSignal:
        if roll_yield is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_roll_yield'}
            )
        
        # Signal: long positive carry
        signal = 1 if roll_yield > 0 else -1
        confidence = min(abs(roll_yield) / 0.01, 1.0)
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'roll_yield': roll_yield}
        )


class MultiFactorCombination(BaseAlpha):
    """Multi-Factor Combination"""
    
    def __init__(self):
        super().__init__("multi_factor_combination", AlphaType.FACTOR)
    
    def compute(self, df: pd.DataFrame, factor_scores: Optional[Dict[str, float]] = None, **kwargs) -> AlphaSignal:
        if factor_scores is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_factor_scores'}
            )
        
        # Equal weight combination
        combined_score = sum(factor_scores.values()) / len(factor_scores) if factor_scores else 0
        
        # Signal
        signal = self.normalize_signal(combined_score)
        confidence = 0.6
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'factor_scores': factor_scores}
        )


# ==================== REGIME/MACRO ALPHAS ====================

class VolatilityRegimeFilter(BaseAlpha):
    """Volatility Regime Filter"""
    
    def __init__(self):
        super().__init__("volatility_regime_filter", AlphaType.REGIME)
    
    def compute(self, df: pd.DataFrame, regime: Optional[str] = None, **kwargs) -> AlphaSignal:
        if regime is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_regime'}
            )
        
        # Signal based on regime
        if regime == 'high_volatility':
            signal = -0.5  # Reduce exposure
            confidence = 0.8
        elif regime == 'low_volatility':
            signal = 0.5  # Increase exposure
            confidence = 0.8
        else:
            signal = 0
            confidence = 0.5
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'regime': regime}
        )


class TrendRegimeFilter(BaseAlpha):
    """Trend/Bear Regime Filter"""
    
    def __init__(self):
        super().__init__("trend_regime_filter", AlphaType.REGIME)
    
    def compute(self, df: pd.DataFrame, regime: Optional[str] = None, **kwargs) -> AlphaSignal:
        if regime is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_regime'}
            )
        
        # Signal based on regime
        if regime == 'bull_trend':
            signal = 0.8  # Long bias
            confidence = 0.8
        elif regime == 'bear_trend':
            signal = -0.8  # Short bias
            confidence = 0.8
        elif regime == 'sideways':
            signal = 0  # Neutral
            confidence = 0.6
        else:
            signal = 0
            confidence = 0.5
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'regime': regime}
        )


class LiquidityRegime(BaseAlpha):
    """Liquidity Regime"""
    
    def __init__(self):
        super().__init__("liquidity_regime", AlphaType.REGIME)
    
    def compute(self, df: pd.DataFrame, liquidity_score: Optional[float] = None, **kwargs) -> AlphaSignal:
        if liquidity_score is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_liquidity_score'}
            )
        
        # Signal: reduce size in low liquidity
        signal = self.normalize_signal(liquidity_score)
        confidence = 0.7
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'liquidity_score': liquidity_score}
        )


class MacroFactor(BaseAlpha):
    """Macro/Cross-Asset Factor"""
    
    def __init__(self):
        super().__init__("macro_factor", AlphaType.REGIME)
    
    def compute(self, df: pd.DataFrame, macro_signals: Optional[Dict[str, float]] = None, **kwargs) -> AlphaSignal:
        if macro_signals is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_macro_signals'}
            )
        
        # Combine macro signals
        combined = sum(macro_signals.values()) / len(macro_signals) if macro_signals else 0
        
        signal = self.normalize_signal(combined)
        confidence = 0.6
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'macro_signals': macro_signals}
        )


class InterestRateSensitivity(BaseAlpha):
    """Interest Rate Sensitivity (CIR)"""
    
    def __init__(self):
        super().__init__("interest_rate_sensitivity", AlphaType.REGIME)
    
    def compute(self, df: pd.DataFrame, rate_change: Optional[float] = None, duration: Optional[float] = None, **kwargs) -> AlphaSignal:
        if rate_change is None or duration is None:
            return AlphaSignal(
                symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
                alpha_name=self.name,
                alpha_type=self.alpha_type,
                signal=0.0,
                confidence=0.0,
                timestamp=pd.Timestamp.now(),
                metadata={'error': 'no_rate_data'}
            )
        
        # Impact = -duration * rate_change
        impact = -duration * rate_change
        
        signal = self.normalize_signal(impact * 100)
        confidence = 0.6
        
        return AlphaSignal(
            symbol=df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'UNKNOWN',
            alpha_name=self.name,
            alpha_type=self.alpha_type,
            signal=signal,
            confidence=confidence,
            timestamp=pd.Timestamp.now(),
            metadata={'rate_change': rate_change, 'duration': duration, 'impact': impact}
        )


# ==================== ALPHA ENGINE ====================

class AlphaEngine:
    """Main alpha engine that orchestrates all alpha strategies"""
    
    def __init__(self):
        self.alphas = {
            # Momentum
            'time_series_momentum_12_1': TimeSeriesMomentum(),
            'dual_momentum': DualMomentum(),
            'volatility_managed_momentum': VolatilityManagedMomentum(),
            'sector_momentum': SectorMomentum(),
            'residual_momentum': ResidualMomentum(),
            
            # Mean Reversion
            'pairs_trading': PairsTrading(),
            'orb': ORB(),
            'vwap_reversion': VWAPReversion(),
            'ibs': IBS(),
            'etf_arbitrage': ETFArbitrage(),
            
            # Volatility
            'volatility_risk_premium': VolatilityRiskPremium(),
            'vix_futures_basis': VIXFuturesBasis(),
            'volatility_targeting': VolatilityTargeting(),
            'dispersion_trading': DispersionTrading(),
            'volatility_carry': VolatilityCarry(),
            
            # Options
            'put_call_parity_carry': PutCallParityCarry(),
            'skew_risk_reversal': SkewRiskReversal(),
            'gamma_scalping': GammaScalping(),
            'variance_swap_term_structure': VarianceSwapTermStructure(),
            'butterfly_volatility': ButterflyVolatility(),
            
            # Microstructure
            'optimal_quoting': OptimalQuoting(),
            'market_making': MarketMaking(),
            'order_imbalance': OrderImbalance(),
            'vwap_execution': VWAPExecution(),
            'inventory_management': InventoryManagement(),
            
            # Factor
            'low_volatility_anomaly': LowVolatilityAnomaly(),
            'value_factor': ValueFactor(),
            'quality_factor': QualityFactor(),
            'carry_factor': CarryFactor(),
            'multi_factor_combination': MultiFactorCombination(),
            
            # Regime
            'volatility_regime_filter': VolatilityRegimeFilter(),
            'trend_regime_filter': TrendRegimeFilter(),
            'liquidity_regime': LiquidityRegime(),
            'macro_factor': MacroFactor(),
            'interest_rate_sensitivity': InterestRateSensitivity()
        }
    
    def compute_alpha(
        self,
        alpha_name: str,
        df: pd.DataFrame,
        **kwargs
    ) -> Optional[AlphaSignal]:
        """Compute a specific alpha signal"""
        if alpha_name not in self.alphas:
            logger.warning(f"Alpha {alpha_name} not found")
            return None
        
        alpha = self.alphas[alpha_name]
        return alpha.compute(df, **kwargs)
    
    def compute_all_alphas(
        self,
        df: pd.DataFrame,
        alpha_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, AlphaSignal]:
        """Compute all alpha signals or a subset"""
        if alpha_names is None:
            alpha_names = list(self.alphas.keys())
        
        signals = {}
        for name in alpha_names:
            signal = self.compute_alpha(name, df, **kwargs)
            if signal:
                signals[name] = signal
        
        return signals
    
    def combine_signals(
        self,
        signals: Dict[str, AlphaSignal],
        weights: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None
    ) -> float:
        """
        Combine alpha signals with dynamic weighting
        
        Args:
            signals: Dictionary of alpha signals
            weights: Optional custom weights
            regime: Current market regime for regime-based weighting
        """
        if not signals:
            return 0.0
        
        # Default equal weights
        if weights is None:
            weights = {name: 1.0 / len(signals) for name in signals.keys()}
        
        # Apply regime-based weights if regime provided
        if regime:
            regime_weights = self._get_regime_weights(regime)
            # Adjust weights based on alpha type
            for name, signal in signals.items():
                alpha_type_weight = regime_weights.get(signal.alpha_type.value, 1.0)
                weights[name] = weights.get(name, 0) * alpha_type_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Weighted combination
        combined = sum(
            weights[name] * signal.signal * signal.confidence
            for name, signal in signals.items()
        )
        
        return combined
    
    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        """Get regime-specific alpha weights"""
        # Simplified regime weights
        regime_weights = {
            'bull_trend': {'momentum': 2.0, 'mean_reversion': 0.5, 'volatility': 1.0},
            'bear_trend': {'momentum': 1.5, 'mean_reversion': 1.0, 'volatility': 1.5},
            'sideways': {'momentum': 0.5, 'mean_reversion': 2.0, 'volatility': 1.0},
            'high_volatility': {'momentum': 0.5, 'mean_reversion': 0.5, 'volatility': 2.0},
            'low_volatility': {'momentum': 1.5, 'mean_reversion': 1.5, 'volatility': 0.5}
        }
        return regime_weights.get(regime, {})
    
    def get_alpha_list(self) -> List[str]:
        """Get list of all available alphas"""
        return list(self.alphas.keys())
    
    def get_alphas_by_type(self, alpha_type: AlphaType) -> List[str]:
        """Get alphas by type"""
        return [name for name, alpha in self.alphas.items() if alpha.alpha_type == alpha_type]

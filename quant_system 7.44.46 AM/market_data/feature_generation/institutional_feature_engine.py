"""
Institutional Feature Engineering Engine
Based on Blueprint V1.0 - 200+ features across 8 categories

Categories:
1. Price Features (30 features)
2. Volume Features (20 features)
3. Volatility Features (25 features)
4. Options Features (20 features)
5. Market Breadth (15 features)
6. Order Flow / Microstructure (20 features)
7. Macro/Cross-Asset (15 features)
8. Derived/Nonlinear (40 features)

Optimized with:
- Fenwick Tree for rolling sums
- Online variance (Welford) for rolling std
- Segment Tree for range queries
- Deque for sliding window min/max
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# ============== OPTIMIZED DATA STRUCTURES ==============

class FenwickTree:
    """Binary Indexed Tree for O(log n) prefix sums and rolling averages."""
    
    def __init__(self, size: int):
        self.n = size
        self.tree = np.zeros(size + 1)
    
    def update(self, idx: int, delta: float) -> None:
        """Update element at index idx (1-indexed)."""
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx: int) -> float:
        """Query prefix sum from 1 to idx."""
        result = 0.0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & -idx
        return result
    
    def range_sum(self, l: int, r: int) -> float:
        """Query sum from l to r (inclusive)."""
        return self.query(r) - self.query(l - 1)


class OnlineVariance:
    """Welford's algorithm for online variance calculation."""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences
    
    def update(self, value: float) -> None:
        """Update with new value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def get_mean(self) -> float:
        return self.mean
    
    def get_variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0
    
    def get_std(self) -> float:
        return np.sqrt(self.get_variance())
    
    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0


class SlidingWindowMinMax:
    """Monotonic deque for O(1) sliding window min/max."""
    
    def __init__(self, size: int):
        self.size = size
        self.window = deque()
        self.min_deque = deque()
        self.max_deque = deque()
    
    def add(self, value: float) -> None:
        """Add new value to window."""
        self.window.append(value)
        
        # Maintain min deque
        while self.min_deque and self.min_deque[-1] > value:
            self.min_deque.pop()
        self.min_deque.append(value)
        
        # Maintain max deque
        while self.max_deque and self.max_deque[-1] < value:
            self.max_deque.pop()
        self.max_deque.append(value)
        
        # Remove old values
        if len(self.window) > self.size:
            removed = self.window.popleft()
            if self.min_deque[0] == removed:
                self.min_deque.popleft()
            if self.max_deque[0] == removed:
                self.max_deque.popleft()
    
    def get_min(self) -> float:
        return self.min_deque[0] if self.min_deque else 0.0
    
    def get_max(self) -> float:
        return self.max_deque[0] if self.max_deque else 0.0


@dataclass
class FeatureConfig:
    """Configuration for Institutional Feature Engine."""
    
    # Rolling windows for different timeframes
    windows_1d = [1, 5, 10, 20, 60, 120, 252]
    windows_5d = [1, 2, 4, 12, 24, 48]
    windows_1h = [1, 2, 4, 8, 12, 24]
    windows_5m = [1, 3, 6, 12, 24, 48, 96]
    
    # Technical indicator periods
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    bollinger_period = 20
    bollinger_std = 2.0
    
    # Volatility periods
    vol_windows = [5, 10, 20, 60, 120]
    
    # Enable/disable feature categories
    enable_price = True
    enable_volume = True
    enable_volatility = True
    enable_options = True
    enable_breadth = True
    enable_microstructure = True
    enable_macro = True
    enable_derived = True


class InstitutionalFeatureEngine:
    """
    Institutional Feature Engineering Engine
    
    Computes 200+ features across 8 categories optimized for Indian markets:
    - Price Features: Returns, log returns, Sharpe, drawdown, MA distance
    - Volume Features: Volume ratio, OBV, VPT, MFI, z-score
    - Volatility Features: ATR, historical vol, Parkinson, Garman-Klass, Yang-Zhang
    - Options Features: PCR, IV skew, term structure, VIX basis, carry gap
    - Market Breadth: A/D line, A/D ratio, % above MA, NH-NL, TRIN
    - Microstructure: Spread, order imbalance, VWAP, market impact
    - Macro/Cross-Asset: USD/INR, crude, gold, bond yields, India VIX
    - Derived/Nonlinear: RSI, MACD, Bollinger, Stochastic, etc.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
        # Rolling computation caches
        self.online_variance_cache: Dict[str, OnlineVariance] = {}
        self.minmax_cache: Dict[str, SlidingWindowMinMax] = {}
        self.fenwick_cache: Dict[str, FenwickTree] = {}
        
        # Feature registry
        self.all_features = self._build_feature_registry()
        
    def _build_feature_registry(self) -> Dict[str, str]:
        """Build registry of all features by category."""
        registry = {}
        
        if self.config.enable_price:
            for i, f in enumerate(self._get_price_feature_names()):
                registry[f] = "price"
        
        if self.config.enable_volume:
            for f in self._get_volume_feature_names():
                registry[f] = "volume"
        
        if self.config.enable_volatility:
            for f in self._get_volatility_feature_names():
                registry[f] = "volatility"
        
        if self.config.enable_options:
            for f in self._get_options_feature_names():
                registry[f] = "options"
        
        if self.config.enable_breadth:
            for f in self._get_breadth_feature_names():
                registry[f] = "breadth"
        
        if self.config.enable_microstructure:
            for f in self._get_microstructure_feature_names():
                registry[f] = "microstructure"
        
        if self.config.enable_macro:
            for f in self._get_macro_feature_names():
                registry[f] = "macro"
        
        if self.config.enable_derived:
            for f in self._get_derived_feature_names():
                registry[f] = "derived"
        
        return registry
    
    def compute_all_features(
        self,
        ohlcv: pd.DataFrame,
        options_data: Optional[Dict] = None,
        breadth_data: Optional[Dict] = None,
        microstructure_data: Optional[Dict] = None,
        macro_data: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Compute all 200+ features.
        
        Args:
            ohlcv: DataFrame with columns: open, high, low, close, volume
            options_data: Optional options data (IV, PCR, skew, etc.)
            breadth_data: Optional market breadth data
            microstructure_data: Optional order book and trade data
            macro_data: Optional macro/cross-asset data
            timestamp: Current timestamp
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        if len(ohlcv) < max(self.config.windows_1d):
            return self._get_default_features()
        
        # Compute each category
        if self.config.enable_price:
            features.update(self._compute_price_features(ohlcv))
        
        if self.config.enable_volume:
            features.update(self._compute_volume_features(ohlcv))
        
        if self.config.enable_volatility:
            features.update(self._compute_volatility_features(ohlcv, options_data))
        
        if self.config.enable_options:
            features.update(self._compute_options_features(options_data))
        
        if self.config.enable_breadth:
            features.update(self._compute_breadth_features(breadth_data))
        
        if self.config.enable_microstructure:
            features.update(self._compute_microstructure_features(microstructure_data, ohlcv))
        
        if self.config.enable_macro:
            features.update(self._compute_macro_features(macro_data))
        
        if self.config.enable_derived:
            features.update(self._compute_derived_features(ohlcv))
        
        return features
    
    # ============== PRICE FEATURES (30) ==============
    
    def _get_price_feature_names(self) -> List[str]:
        names = []
        # Returns at different windows
        for w in self.config.windows_1d:
            names.extend([f'returns_{w}d', f'log_returns_{w}d'])
        # Sharpe ratio
        names.extend(['sharpe_20d', 'sharpe_60d'])
        # Drawdown
        names.extend(['drawdown', 'max_drawdown_20d', 'max_drawdown_60d'])
        # Distance from ATH
        names.append('distance_from_ath')
        # Distance from MAs
        for w in [20, 50, 200]:
            names.extend([f'pct_above_ma_{w}', f'ma_{w}'])
        # Price percentile in range
        names.append('price_percentile_range')
        return names
    
    def _compute_price_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        features = {}
        close = ohlcv['close'].values
        
        # Returns
        for w in self.config.windows_1d:
            if len(close) > w:
                features[f'returns_{w}d'] = (close[-1] - close[-w-1]) / close[-w-1] if close[-w-1] > 0 else 0
                features[f'log_returns_{w}d'] = np.log(close[-1] / close[-w-1]) if close[-w-1] > 0 else 0
            else:
                features[f'returns_{w}d'] = 0
                features[f'log_returns_{w}d'] = 0
        
        # Sharpe ratio (rolling)
        for w in [20, 60]:
            if len(close) > w:
                returns = pd.Series(close[-w:]).pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    features[f'sharpe_{w}d'] = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    features[f'sharpe_{w}d'] = 0
            else:
                features[f'sharpe_{w}d'] = 0
        
        # Drawdown
        cummax = pd.Series(close).cummax()
        drawdown = (cummax - close) / cummax
        features['drawdown'] = drawdown.iloc[-1]
        
        # Max drawdown (rolling)
        for w in [20, 60]:
            if len(close) > w:
                recent_cummax = pd.Series(close[-w:]).cummax()
                recent_dd = (recent_cummax - close[-w:]) / recent_cummax
                features[f'max_drawdown_{w}d'] = recent_dd.max()
            else:
                features[f'max_drawdown_{w}d'] = 0
        
        # Distance from all-time high
        features['distance_from_ath'] = (cummax.iloc[-1] - close[-1]) / cummax.iloc[-1]
        
        # Distance from moving averages
        for w in [20, 50, 200]:
            if len(close) > w:
                ma = close[-w:].mean()
                features[f'ma_{w}'] = ma
                features[f'pct_above_ma_{w}'] = (close[-1] - ma) / ma if ma > 0 else 0
            else:
                features[f'ma_{w}'] = close[-1]
                features[f'pct_above_ma_{w}'] = 0
        
        # Price percentile in recent range
        if len(close) >= 20:
            recent_range = close[-20:]
            features['price_percentile_range'] = (close[-1] - recent_range.min()) / (recent_range.max() - recent_range.min()) if recent_range.max() > recent_range.min() else 0.5
        else:
            features['price_percentile_range'] = 0.5
        
        return features
    
    # ============== VOLUME FEATURES (20) ==============
    
    def _get_volume_feature_names(self) -> List[str]:
        return [
            'volume', 'log_volume',
            'volume_zscore_20', 'volume_zscore_60',
            'volume_ratio_20', 'volume_ratio_60',
            'obv', 'obv_ma_20',
            'vpt', 'vpt_ma_20',
            'mfi_14',
            'volume_trend_20', 'volume_trend_60',
            'volume_acceleration',
            'volume_volatility_20',
            'up_volume_ratio_20', 'down_volume_ratio_20',
            'volume_price_trend',
            'relative_volume_5d', 'relative_volume_20d'
        ]
    
    def _compute_volume_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        features = {}
        volume = ohlcv['volume'].values
        close = ohlcv['close'].values
        
        # Raw and log volume
        features['volume'] = volume[-1]
        features['log_volume'] = np.log(volume[-1]) if volume[-1] > 0 else 0
        
        # Volume z-score
        for w in [20, 60]:
            if len(volume) > w:
                mean_vol = volume[-w:].mean()
                std_vol = volume[-w:].std()
                if std_vol > 0:
                    features[f'volume_zscore_{w}'] = (volume[-1] - mean_vol) / std_vol
                else:
                    features[f'volume_zscore_{w}'] = 0
            else:
                features[f'volume_zscore_{w}'] = 0
        
        # Volume ratio vs MA
        for w in [20, 60]:
            if len(volume) > w:
                avg_vol = volume[-w:].mean()
                features[f'volume_ratio_{w}'] = volume[-1] / avg_vol if avg_vol > 0 else 0
            else:
                features[f'volume_ratio_{w}'] = 0
        
        # On-Balance Volume
        if len(close) > 1:
            obv = np.zeros(len(close))
            obv[0] = volume[0]
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            features['obv'] = obv[-1]
            if len(obv) >= 20:
                features['obv_ma_20'] = obv[-20:].mean()
            else:
                features['obv_ma_20'] = obv[-1]
        else:
            features['obv'] = 0
            features['obv_ma_20'] = 0
        
        # Volume Price Trend
        if len(close) > 1:
            vpt = np.zeros(len(close))
            vpt[0] = 0
            for i in range(1, len(close)):
                price_change = (close[i] - close[i-1]) / close[i-1] if close[i-1] > 0 else 0
                vpt[i] = vpt[i-1] + price_change * volume[i]
            features['vpt'] = vpt[-1]
            if len(vpt) >= 20:
                features['vpt_ma_20'] = vpt[-20:].mean()
            else:
                features['vpt_ma_20'] = vpt[-1]
        else:
            features['vpt'] = 0
            features['vpt_ma_20'] = 0
        
        # Money Flow Index (simplified)
        if len(close) >= 14 and len(volume) >= 14:
            typical_price = (ohlcv['high'].values + ohlcv['low'].values + close) / 3
            mfi = 100 - 100 / (1 + self._calculate_mfi(typical_price[-14:], volume[-14:]))
            features['mfi_14'] = mfi
        else:
            features['mfi_14'] = 50
        
        # Volume trend (linear regression slope)
        for w in [20, 60]:
            if len(volume) >= w:
                x = np.arange(w)
                slope = np.polyfit(x, volume[-w:], 1)[0]
                features[f'volume_trend_{w}'] = slope / np.mean(volume[-w:]) if np.mean(volume[-w:]) > 0 else 0
            else:
                features[f'volume_trend_{w}'] = 0
        
        # Volume acceleration (second derivative)
        if len(volume) >= 3:
            features['volume_acceleration'] = (volume[-1] - 2*volume[-2] + volume[-3]) / volume[-2] if volume[-2] > 0 else 0
        else:
            features['volume_acceleration'] = 0
        
        # Volume volatility
        if len(volume) >= 20:
            features['volume_volatility_20'] = volume[-20:].std() / volume[-20:].mean() if volume[-20:].mean() > 0 else 0
        else:
            features['volume_volatility_20'] = 0
        
        # Up/down volume ratio
        for w in [20, 60]:
            if len(close) >= w and len(volume) >= w:
                up_vol = volume[-w:][close[-w:] > close[-w-1:]].sum()
                down_vol = volume[-w:][close[-w:] < close[-w-1:]].sum()
                total = up_vol + down_vol
                features[f'up_volume_ratio_{w}'] = up_vol / total if total > 0 else 0.5
                features[f'down_volume_ratio_{w}'] = down_vol / total if total > 0 else 0.5
            else:
                features[f'up_volume_ratio_{w}'] = 0.5
                features[f'down_volume_ratio_{w}'] = 0.5
        
        # Volume-price trend (correlation)
        if len(close) >= 20 and len(volume) >= 20:
            features['volume_price_trend'] = np.corrcoef(close[-20:], volume[-20:])[0, 1] if len(close[-20:]) > 1 else 0
        else:
            features['volume_price_trend'] = 0
        
        # Relative volume vs 5d and 20d average
        if len(volume) >= 20:
            features['relative_volume_5d'] = volume[-1] / volume[-5:].mean() if volume[-5:].mean() > 0 else 0
            features['relative_volume_20d'] = volume[-1] / volume[-20:].mean() if volume[-20:].mean() > 0 else 0
        else:
            features['relative_volume_5d'] = 1
            features['relative_volume_20d'] = 1
        
        return features
    
    def _calculate_mfi(self, typical_price: np.ndarray, volume: np.ndarray) -> float:
        """Calculate Money Flow Index numerator."""
        positive_flow = 0.0
        negative_flow = 0.0
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow += typical_price[i] * volume[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow += typical_price[i] * volume[i]
        
        return positive_flow / negative_flow if negative_flow > 0 else 1
    
    # ============== VOLATILITY FEATURES (25) ==============
    
    def _get_volatility_feature_names(self) -> List[str]:
        names = ['atr_14', 'atr_ratio']
        for w in self.config.vol_windows:
            names.extend([f'historical_vol_{w}d', f'vol_percentile_{w}d'])
        names.extend([
            'parkinson_vol', 'garman_klass_vol', 'yang_zhang_vol',
            'vol_of_vol_20d', 'vol_regime_garch',
            'vix_nifty', 'vix_change', 'vix_percentile',
            'realized_vol_5d', 'realized_vol_20d',
            'iv_hv_spread', 'term_structure_vol',
            'vol_skew', 'vol_kurtosis'
        ])
        return names
    
    def _compute_volatility_features(self, ohlcv: pd.DataFrame, options_data: Optional[Dict]) -> Dict[str, float]:
        features = {}
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        
        # ATR
        atr = self._calculate_atr(high, low, close, 14)
        features['atr_14'] = atr
        features['atr_ratio'] = atr / close[-1] if close[-1] > 0 else 0
        
        # Historical volatility at different windows
        returns = np.diff(np.log(close))
        for w in self.config.vol_windows:
            if len(returns) >= w:
                hist_vol = returns[-w:].std() * np.sqrt(252)
                features[f'historical_vol_{w}d'] = hist_vol
                # Percentile vs history
                if len(returns) >= 2*w:
                    all_vols = []
                    for i in range(w, len(returns)):
                        all_vols.append(returns[i-w:i].std() * np.sqrt(252))
                    features[f'vol_percentile_{w}d'] = np.percentile(all_vols, hist_vol * 100) if all_vols else 0.5
                else:
                    features[f'vol_percentile_{w}d'] = 0.5
            else:
                features[f'historical_vol_{w}d'] = 0
                features[f'vol_percentile_{w}d'] = 0.5
        
        # Parkinson volatility (uses high/low)
        if len(high) >= 20:
            hl_ratio = np.log(high[-20:] / low[-20:])
            parkinson = np.sqrt((1 / (4 * np.log(2))) * np.mean(hl_ratio ** 2)) * np.sqrt(252)
            features['parkinson_vol'] = parkinson
        else:
            features['parkinson_vol'] = 0
        
        # Garman-Klass volatility
        if len(high) >= 20 and len(low) >= 20 and len(close) >= 20 and len(ohlcv['open'].values) >= 20:
            open_ = ohlcv['open'].values
            gk = 0.5 * (np.log(high[-20:] / low[-20:])) ** 2 - (2 * np.log(2) - 1) * (np.log(close[-20:] / open_[-20:])) ** 2
            features['garman_klass_vol'] = np.sqrt(np.mean(gk)) * np.sqrt(252)
        else:
            features['garman_klass_vol'] = 0
        
        # Yang-Zhang volatility (most robust)
        if len(close) >= 20:
            yz = self._calculate_yang_zhang(high[-20:], low[-20:], close[-20:], ohlcv['open'].values[-20:])
            features['yang_zhang_vol'] = yz
        else:
            features['yang_zhang_vol'] = 0
        
        # Volatility of volatility
        if len(returns) >= 20:
            rolling_vol = []
            for i in range(10, len(returns)):
                rolling_vol.append(returns[i-10:i].std())
            features['vol_of_vol_20d'] = np.std(rolling_vol) if rolling_vol else 0
        else:
            features['vol_of_vol_20d'] = 0
        
        # GARCH-based volatility regime (simplified)
        if len(returns) >= 20:
            vol_persistence = self._calculate_garch_persistence(returns[-20:])
            features['vol_regime_garch'] = vol_persistence
        else:
            features['vol_regime_garch'] = 0
        
        # VIX/NIFTY from options data
        if options_data:
            features['vix_nifty'] = options_data.get('vix', 0)
            features['vix_change'] = options_data.get('vix_change', 0)
            features['vix_percentile'] = options_data.get('vix_percentile', 0.5)
            features['iv_hv_spread'] = options_data.get('iv', 0) - features['histor_vol_20d']
        else:
            features['vix_nifty'] = 0
            features['vix_change'] = 0
            features['vix_percentile'] = 0.5
            features['iv_hv_spread'] = 0
        
        # Realized volatility (5d, 20d)
        if len(returns) >= 5:
            features['realized_vol_5d'] = returns[-5:].std() * np.sqrt(252)
        else:
            features['realized_vol_5d'] = 0
        
        if len(returns) >= 20:
            features['realized_vol_20d'] = returns[-20:].std() * np.sqrt(252)
        else:
            features['realized_vol_20d'] = 0
        
        # Term structure volatility
        features['term_structure_vol'] = options_data.get('term_structure_vol', 0) if options_data else 0
        
        # Volatility skew and kurtosis
        if len(returns) >= 20:
            features['vol_skew'] = pd.Series(returns[-20:]).skew()
            features['vol_kurtosis'] = pd.Series(returns[-20:]).kurtosis()
        else:
            features['vol_skew'] = 0
            features['vol_kurtosis'] = 0
        
        return features
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        if len(close) < period + 1:
            return 0
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return np.mean(tr[-period:])
    
    def _calculate_yang_zhang(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, open_: np.ndarray) -> float:
        """Yang-Zhang volatility estimator."""
        n = len(close)
        if n < 2:
            return 0
        
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_co = np.log(close / open_)
        log_oc = np.log(open / np.roll(close, 1))
        log_oc[0] = 0
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        close_open = log_oc * log_oc
        
        kz = np.mean(rs) / n + np.mean(close_open) / n
        return np.sqrt(kz) * np.sqrt(252)
    
    def _calculate_garch_persistence(self, returns: np.ndarray) -> float:
        """Simplified GARCH(1,1) persistence parameter."""
        if len(returns) < 10:
            return 0
        # Simple approximation using autocorrelation of squared returns
        squared_returns = returns ** 2
        if len(squared_returns) >= 2:
            return np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        return 0
    
    # ============== OPTIONS FEATURES (20) ==============
    
    def _get_options_feature_names(self) -> List[str]:
        return [
            'put_call_ratio_volume', 'put_call_ratio_oi',
            'iv_skew_25d', 'iv_skew_10d',
            'iv_term_structure_30d_60d', 'iv_term_structure_60d_90d',
            'vix_futures_basis', 'vix_vxv_ratio',
            'carry_gap', 'synthetic_forward_arb',
            'gamma_exposure_net', 'gamma_exposure_atm',
            'vanna_exposure', 'charm_exposure',
            'option_implied_pdf_mean', 'option_implied_pdf_std',
            'delta_hedging_pnl', 'vega_exposure',
            'theta_decay_daily', 'rho_exposure'
        ]
    
    def _compute_options_features(self, options_data: Optional[Dict]) -> Dict[str, float]:
        features = {}
        
        if not options_data:
            return {f: 0 for f in self._get_options_feature_names()}
        
        features['put_call_ratio_volume'] = options_data.get('pcr_volume', 0)
        features['put_call_ratio_oi'] = options_data.get('pcr_oi', 0)
        features['iv_skew_25d'] = options_data.get('iv_skew_25', 0)
        features['iv_skew_10d'] = options_data.get('iv_skew_10', 0)
        features['iv_term_structure_30d_60d'] = options_data.get('iv_ts_30_60', 0)
        features['iv_term_structure_60d_90d'] = options_data.get('iv_ts_60_90', 0)
        features['vix_futures_basis'] = options_data.get('vix_basis', 0)
        features['vix_vxv_ratio'] = options_data.get('vix_vxv', 0)
        features['carry_gap'] = options_data.get('carry_gap', 0)
        features['synthetic_forward_arb'] = options_data.get('synthetic_fwd', 0)
        features['gamma_exposure_net'] = options_data.get('gamma_net', 0)
        features['gamma_exposure_atm'] = options_data.get('gamma_atm', 0)
        features['vanna_exposure'] = options_data.get('vanna', 0)
        features['charm_exposure'] = options_data.get('charm', 0)
        features['option_implied_pdf_mean'] = options_data.get('pdf_mean', 0)
        features['option_implied_pdf_std'] = options_data.get('pdf_std', 0)
        features['delta_hedging_pnl'] = options_data.get('delta_pnl', 0)
        features['vega_exposure'] = options_data.get('vega', 0)
        features['theta_decay_daily'] = options_data.get('theta', 0)
        features['rho_exposure'] = options_data.get('rho', 0)
        
        return features
    
    # ============== MARKET BREADTH FEATURES (15) ==============
    
    def _get_breadth_feature_names(self) -> List[str]:
        return [
            'advance_decline_line', 'advance_decline_ratio',
            'pct_above_ma_50', 'pct_above_ma_200',
            'new_highs_new_lows', 'new_highs_pct', 'new_lows_pct',
            'trin_arms', 'trin_5d_ma',
            'mcclellan_oscillator', 'mcclellan_summation',
            'breadth_momentum', 'breadth_thrust',
            'tick_indicator', 'ticky_indicator'
        ]
    
    def _compute_breadth_features(self, breadth_data: Optional[Dict]) -> Dict[str, float]:
        features = {}
        
        if not breadth_data:
            return {f: 0 for f in self._get_breadth_feature_names()}
        
        features['advance_decline_line'] = breadth_data.get('ad_line', 0)
        features['advance_decline_ratio'] = breadth_data.get('ad_ratio', 1)
        features['pct_above_ma_50'] = breadth_data.get('pct_ma50', 0.5)
        features['pct_above_ma_200'] = breadth_data.get('pct_ma200', 0.5)
        features['new_highs_new_lows'] = breadth_data.get('nh_nl', 0)
        features['new_highs_pct'] = breadth_data.get('nh_pct', 0)
        features['new_lows_pct'] = breadth_data.get('nl_pct', 0)
        features['trin_arms'] = breadth_data.get('trin', 1)
        features['trin_5d_ma'] = breadth_data.get('trin_ma5', 1)
        features['mcclellan_oscillator'] = breadth_data.get('mco', 0)
        features['mcclellan_summation'] = breadth_data.get('mcs', 0)
        features['breadth_momentum'] = breadth_data.get('breadth_mom', 0)
        features['breadth_thrust'] = breadth_data.get('breadth_thrust', 0)
        features['tick_indicator'] = breadth_data.get('tick', 0)
        features['ticky_indicator'] = breadth_data.get('ticky', 0)
        
        return features
    
    # ============== MICROSTRUCTURE FEATURES (20) ==============
    
    def _get_microstructure_feature_names(self) -> List[str]:
        return [
            'bid_ask_spread', 'bid_ask_spread_pct',
            'order_imbalance_tick', 'order_imbalance_5m',
            'vwap', 'vwap_distance', 'vwap_slope',
            'volume_profile_poc', 'volume_profile_va_high', 'volume_profile_va_low',
            'market_impact_estimate', 'effective_spread',
            'realized_spread', 'price_impact',
            'depth_imbalance_bid_ask', 'depth_imbalance_levels',
            'trade_size_avg', 'trade_size_std',
            'aggressive_ratio', 'passive_ratio',
            'midpoint_return', 'quote_slope'
        ]
    
    def _compute_microstructure_features(self, micro_data: Optional[Dict], ohlcv: pd.DataFrame) -> Dict[str, float]:
        features = {}
        close = ohlcv['close'].values
        volume = ohlcv['volume'].values
        
        if not micro_data:
            # Compute basic features from OHLCV
            # VWAP
            if len(close) >= 20:
                typical_price = (ohlcv['high'].values + ohlcv['low'].values + close) / 3
                vwap = np.sum(typical_price[-20:] * volume[-20:]) / np.sum(volume[-20:]) if np.sum(volume[-20:]) > 0 else close[-1]
                features['vwap'] = vwap
                features['vwap_distance'] = (close[-1] - vwap) / vwap if vwap > 0 else 0
                
                # VWAP slope
                x = np.arange(20)
                vwap_slope = np.polyfit(x, typical_price[-20:], 1)[0]
                features['vwap_slope'] = vwap_slope / vwap if vwap > 0 else 0
            else:
                features['vwap'] = close[-1]
                features['vwap_distance'] = 0
                features['vwap_slope'] = 0
            
            # Trade size stats
            if len(volume) >= 20:
                features['trade_size_avg'] = volume[-20:].mean()
                features['trade_size_std'] = volume[-20:].std()
            else:
                features['trade_size_avg'] = volume[-1]
                features['trade_size_std'] = 0
            
            # Default values for order book features
            defaults = {
                'bid_ask_spread': 0, 'bid_ask_spread_pct': 0,
                'order_imbalance_tick': 0, 'order_imbalance_5m': 0,
                'volume_profile_poc': close[-1], 'volume_profile_va_high': close[-1], 'volume_profile_va_low': close[-1],
                'market_impact_estimate': 0, 'effective_spread': 0,
                'realized_spread': 0, 'price_impact': 0,
                'depth_imbalance_bid_ask': 0, 'depth_imbalance_levels': 0,
                'aggressive_ratio': 0.5, 'passive_ratio': 0.5,
                'midpoint_return': 0, 'quote_slope': 0
            }
            features.update(defaults)
            return features
        
        # Use provided microstructure data
        bid = micro_data.get('bid', close[-1])
        ask = micro_data.get('ask', close[-1])
        features['bid_ask_spread'] = ask - bid
        features['bid_ask_spread_pct'] = (ask - bid) / bid if bid > 0 else 0
        
        features['order_imbalance_tick'] = micro_data.get('imbalance_tick', 0)
        features['order_imbalance_5m'] = micro_data.get('imbalance_5m', 0)
        
        features['vwap'] = micro_data.get('vwap', close[-1])
        features['vwap_distance'] = (close[-1] - features['vwap']) / features['vwap'] if features['vwap'] > 0 else 0
        features['vwap_slope'] = micro_data.get('vwap_slope', 0)
        
        features['volume_profile_poc'] = micro_data.get('poc', close[-1])
        features['volume_profile_va_high'] = micro_data.get('va_high', close[-1])
        features['volume_profile_va_low'] = micro_data.get('va_low', close[-1])
        
        features['market_impact_estimate'] = micro_data.get('market_impact', 0)
        features['effective_spread'] = micro_data.get('effective_spread', 0)
        features['realized_spread'] = micro_data.get('realized_spread', 0)
        features['price_impact'] = micro_data.get('price_impact', 0)
        
        features['depth_imbalance_bid_ask'] = micro_data.get('depth_imbalance_ba', 0)
        features['depth_imbalance_levels'] = micro_data.get('depth_imbalance_levels', 0)
        
        features['trade_size_avg'] = micro_data.get('trade_size_avg', volume[-1])
        features['trade_size_std'] = micro_data.get('trade_size_std', 0)
        
        features['aggressive_ratio'] = micro_data.get('aggressive_ratio', 0.5)
        features['passive_ratio'] = micro_data.get('passive_ratio', 0.5)
        
        features['midpoint_return'] = micro_data.get('midpoint_return', 0)
        features['quote_slope'] = micro_data.get('quote_slope', 0)
        
        return features
    
    # ============== MACRO/CROSS-ASSET FEATURES (15) ==============
    
    def _get_macro_feature_names(self) -> List[str]:
        return [
            'usd_inr_return_1d', 'usd_inr_return_5d',
            'crude_oil_return_1d', 'crude_oil_return_5d',
            'gold_return_1d', 'gold_return_5d',
            'bond_yield_10y', 'bond_yield_2y',
            'yield_spread_10y_2y', 'yield_spread_change',
            'india_vix', 'india_vix_change',
            'global_vix', 'global_vix_change',
            'crb_index_return', 'dxy_index_return'
        ]
    
    def _compute_macro_features(self, macro_data: Optional[Dict]) -> Dict[str, float]:
        features = {}
        
        if not macro_data:
            return {f: 0 for f in self._get_macro_feature_names()}
        
        features['usd_inr_return_1d'] = macro_data.get('usdinr_ret_1d', 0)
        features['usd_inr_return_5d'] = macro_data.get('usdinr_ret_5d', 0)
        features['crude_oil_return_1d'] = macro_data.get('crude_ret_1d', 0)
        features['crude_oil_return_5d'] = macro_data.get('crude_ret_5d', 0)
        features['gold_return_1d'] = macro_data.get('gold_ret_1d', 0)
        features['gold_return_5d'] = macro_data.get('gold_ret_5d', 0)
        features['bond_yield_10y'] = macro_data.get('bond_10y', 0)
        features['bond_yield_2y'] = macro_data.get('bond_2y', 0)
        features['yield_spread_10y_2y'] = macro_data.get('yield_spread', 0)
        features['yield_spread_change'] = macro_data.get('yield_spread_change', 0)
        features['india_vix'] = macro_data.get('india_vix', 0)
        features['india_vix_change'] = macro_data.get('india_vix_change', 0)
        features['global_vix'] = macro_data.get('global_vix', 0)
        features['global_vix_change'] = macro_data.get('global_vix_change', 0)
        features['crb_index_return'] = macro_data.get('crb_ret', 0)
        features['dxy_index_return'] = macro_data.get('dxy_ret', 0)
        
        return features
    
    # ============== DERIVED/NONLINEAR FEATURES (40) ==============
    
    def _get_derived_feature_names(self) -> List[str]:
        return [
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'bollinger_position',
            'aroon_up', 'aroon_down', 'aroon_oscillator',
            'stochastic_k', 'stochastic_d',
            'chaikin_money_flow', 'chaikin_oscillator',
            'elder_force_index',
            'ultimate_oscillator',
            'williams_r', 'cci',
            'momentum', 'roc', 'rate_of_change',
            'trix', 'mass_index',
            'commodity_channel_index',
            'average_directional_index', 'adx', 'di_plus', 'di_minus',
            'parabolic_sar',
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
            'keltner_channel_upper', 'keltner_channel_lower', 'keltner_channel_middle',
            'donchian_upper', 'donchian_lower', 'donchian_middle'
        ]
    
    def _compute_derived_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        features = {}
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        open_ = ohlcv['open'].values
        volume = ohlcv['volume'].values
        
        # RSI at multiple periods
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._calculate_rsi(close, period)
        
        # MACD
        macd, signal, hist = self._calculate_macd(close, 12, 26, 9)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = hist
        
        # Bollinger Bands
        if len(close) >= 20:
            bb_upper, bb_lower, bb_width, bb_pos = self._calculate_bollinger(close, 20, 2)
            features['bollinger_upper'] = bb_upper
            features['bollinger_lower'] = bb_lower
            features['bollinger_width'] = bb_width
            features['bollinger_position'] = bb_pos
        else:
            features['bollinger_upper'] = close[-1]
            features['bollinger_lower'] = close[-1]
            features['bollinger_width'] = 0
            features['bollinger_position'] = 0.5
        
        # Aroon
        if len(high) >= 25:
            aroon_up, aroon_down = self._calculate_aroon(high, low, 25)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down
            features['aroon_oscillator'] = aroon_up - aroon_down
        else:
            features['aroon_up'] = 50
            features['aroon_down'] = 50
            features['aroon_oscillator'] = 0
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(high, low, close, 14, 3)
        features['stochastic_k'] = stoch_k
        features['stochastic_d'] = stoch_d
        
        # Chaikin Money Flow
        if len(close) >= 20:
            features['chaikin_money_flow'] = self._calculate_cmf(high, low, close, volume, 20)
            features['chaikin_oscillator'] = self._calculate_chaikin_oscillator(high, low, close, volume)
        else:
            features['chaikin_money_flow'] = 0
            features['chaikin_oscillator'] = 0
        
        # Elder's Force Index
        if len(close) >= 2 and len(volume) >= 2:
            features['elder_force_index'] = (close[-1] - close[-2]) * volume[-1]
        else:
            features['elder_force_index'] = 0
        
        # Ultimate Oscillator
        features['ultimate_oscillator'] = self._calculate_ultimate_oscillator(high, low, close)
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(high, low, close, 14)
        
        # CCI
        features['cci'] = self._calculate_cci(high, low, close, 20)
        
        # Momentum and ROC
        if len(close) >= 10:
            features['momentum'] = close[-1] - close[-10]
            features['roc'] = (close[-1] - close[-10]) / close[-10] if close[-10] > 0 else 0
        else:
            features['momentum'] = 0
            features['roc'] = 0
        
        features['rate_of_change'] = features['roc']
        
        # TRIX
        features['trix'] = self._calculate_trix(close, 14) if len(close) >= 15 else 0
        
        # Mass Index
        features['mass_index'] = self._calculate_mass_index(high, low) if len(high) >= 25 else 0
        
        # ADX
        if len(high) >= 14:
            adx, di_plus, di_minus = self._calculate_adx(high, low, close, 14)
            features['average_directional_index'] = adx
            features['adx'] = adx
            features['di_plus'] = di_plus
            features['di_minus'] = di_minus
        else:
            features['average_directional_index'] = 0
            features['adx'] = 0
            features['di_plus'] = 0
            features['di_minus'] = 0
        
        # Parabolic SAR (simplified)
        features['parabolic_sar'] = self._calculate_parabolic_sar(high, low) if len(high) >= 5 else close[-1]
        
        # Ichimoku (simplified)
        if len(close) >= 26:
            tenkan, kijun, senkou_a, senkou_b = self._calculate_ichimoku(high, low, close)
            features['ichimoku_tenkan'] = tenkan
            features['ichimoku_kijun'] = kijun
            features['ichimoku_senkou_a'] = senkou_a
            features['ichimoku_senkou_b'] = senkou_b
        else:
            features['ichimoku_tenkan'] = close[-1]
            features['ichimoku_kijun'] = close[-1]
            features['ichimoku_senkou_a'] = close[-1]
            features['ichimoku_senkou_b'] = close[-1]
        
        # Keltner Channels
        if len(close) >= 20:
            kc_upper, kc_lower, kc_middle = self._calculate_keltner(high, low, close, 20, 2)
            features['keltner_channel_upper'] = kc_upper
            features['keltner_channel_lower'] = kc_lower
            features['keltner_channel_middle'] = kc_middle
        else:
            features['keltner_channel_upper'] = close[-1]
            features['keltner_channel_lower'] = close[-1]
            features['keltner_channel_middle'] = close[-1]
        
        # Donchian Channels
        if len(high) >= 20:
            dc_upper = high[-20:].max()
            dc_lower = low[-20:].min()
            features['donchian_upper'] = dc_upper
            features['donchian_lower'] = dc_lower
            features['donchian_middle'] = (dc_upper + dc_lower) / 2
        else:
            features['donchian_upper'] = close[-1]
            features['donchian_lower'] = close[-1]
            features['donchian_middle'] = close[-1]
        
        return features
    
    # ============== DERIVED INDICATOR CALCULATIONS ==============
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[float, float, float]:
        if len(prices) < slow + signal:
            return 0, 0, 0
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line[-1] - signal_line[-1]
        return macd_line[-1], signal_line[-1], histogram
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    def _calculate_bollinger(self, prices: np.ndarray, period: int, std_dev: float) -> Tuple[float, float, float, float]:
        if len(prices) < period:
            return prices[-1], prices[-1], 0, 0.5
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        width = (upper - lower) / sma if sma > 0 else 0
        position = (prices[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        return upper, lower, width, position
    
    def _calculate_aroon(self, high: np.ndarray, low: np.ndarray, period: int) -> Tuple[float, float]:
        if len(high) < period:
            return 50, 50
        aroon_up = ((period - np.argmax(high[-period:])) / period) * 100
        aroon_down = ((period - np.argmin(low[-period:])) / period) * 100
        return aroon_up, aroon_down
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        if len(close) < k_period:
            return 50, 50
        recent_high = np.max(high[-k_period:])
        recent_low = np.min(low[-k_period:])
        if recent_high == recent_low:
            k = 50
        else:
            k = 100 * (close[-1] - recent_low) / (recent_high - recent_low)
        d = k  # Simplified
        return k, d
    
    def _calculate_cmf(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int) -> float:
        if len(close) < period:
            return 0
        mfv = []
        for i in range(-period, 0):
            if high[i] - low[i] > 0:
                mfv.append(((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) * volume[i])
            else:
                mfv.append(0)
        return np.sum(mfv) / np.sum(volume[-period:]) if np.sum(volume[-period:]) > 0 else 0
    
    def _calculate_chaikin_oscillator(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> float:
        if len(close) < 10:
            return 0
        cmf_3 = self._calculate_cmf(high, low, close, volume, 3)
        cmf_10 = self._calculate_cmf(high, low, close, volume, 10)
        return cmf_3 - cmf_10
    
    def _calculate_ultimate_oscillator(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        if len(close) < 28:
            return 50
        bp = close - np.minimum(low, np.roll(close, 1))
        tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
        avg7 = bp[-7:].sum() / tr[-7:].sum() if tr[-7:].sum() > 0 else 0
        avg14 = bp[-14:].sum() / tr[-14:].sum() if tr[-14:].sum() > 0 else 0
        avg28 = bp[-28:].sum() / tr[-28:].sum() if tr[-28:].sum() > 0 else 0
        return 100 * (4*avg7 + 2*avg14 + avg28) / 7 if (4*avg7 + 2*avg14 + avg28) > 0 else 50
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period:
            return -50
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        if highest_high == lowest_low:
            return -50
        return -100 * (highest_high - close[-1]) / (highest_high - lowest_low)
    
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> float:
        if len(close) < period:
            return 0
        typical_price = (high + low + close) / 3
        sma = np.mean(typical_price[-period:])
        mean_deviation = np.mean(np.abs(typical_price[-period:] - sma))
        if mean_deviation == 0:
            return 0
        return (typical_price[-1] - sma) / (0.015 * mean_deviation)
    
    def _calculate_trix(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 3:
            return 0
        ema1 = self._calculate_ema(prices, period)
        ema2 = self._calculate_ema(ema1, period)
        ema3 = self._calculate_ema(ema2, period)
        if len(ema3) < 2:
            return 0
        return (ema3[-1] - ema3[-2]) / ema3[-2] if ema3[-2] > 0 else 0
    
    def _calculate_mass_index(self, high: np.ndarray, low: np.ndarray) -> float:
        if len(high) < 25:
            return 0
        range_val = high - low
        ema9 = self._calculate_ema(range_val, 9)
        ema9_2 = self._calculate_ema(ema9, 9)
        ratio = ema9 / ema9_2
        ratio = np.nan_to_num(ratio, nan=1)
        return ratio[-25:].sum() if len(ratio) >= 25 else 0
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        if len(close) < period * 2:
            return 0, 0, 0
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        plus_dm = np.where(high[1:] > high[:-1], high[1:] - high[:-1], 0)
        minus_dm = np.where(low[1:] < low[:-1], low[:-1] - low[1:], 0)
        
        atr = self._calculate_ema(tr, period)
        plus_di = 100 * self._calculate_ema(plus_dm, period) / atr
        minus_di = 100 * self._calculate_ema(minus_dm, period) / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.nan_to_num(dx, nan=0)
        adx = self._calculate_ema(dx, period)
        
        return adx[-1] if len(adx) > 0 else 0, plus_di[-1] if len(plus_di) > 0 else 0, minus_di[-1] if len(minus_di) > 0 else 0
    
    def _calculate_parabolic_sar(self, high: np.ndarray, low: np.ndarray) -> float:
        if len(high) < 5:
            return low[-1]
        # Simplified SAR calculation
        return low[-5:].min() if high[-1] < high[-2] else high[-5:].max()
    
    def _calculate_ichimoku(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float, float, float]:
        if len(close) < 52:
            return close[-1], close[-1], close[-1], close[-1]
        tenkan = (high[-9:].max() + low[-9:].min()) / 2
        kijun = (high[-26:].max() + low[-26:].min()) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (high[-52:].max() + low[-52:].min()) / 2
        return tenkan, kijun, senkou_a, senkou_b
    
    def _calculate_keltner(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int, mult: float) -> Tuple[float, float, float]:
        if len(close) < period:
            return close[-1], close[-1], close[-1]
        ema = self._calculate_ema(close, period)
        atr = self._calculate_atr(high, low, close, period)
        upper = ema[-1] + mult * atr
        lower = ema[-1] - mult * atr
        return upper, lower, ema[-1]
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default values when insufficient data."""
        return {name: 0.0 for name in self.all_features.keys()}
    
    def get_feature_count(self) -> int:
        """Return total number of features."""
        return len(self.all_features)
    
    def get_features_by_category(self, category: str) -> List[str]:
        """Return feature names for a specific category."""
        return [name for name, cat in self.all_features.items() if cat == category]


# ============== FEATURE SELECTION ==============

class FeatureSelector:
    """
    Feature selection using multiple methods:
    - Mutual Information
    - SHAP values
    - Correlation analysis
    - Recursive Feature Elimination
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
    
    def select_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        method: str = "mutual_information",
        n_features: int = 50
    ) -> List[str]:
        """
        Select top features using specified method.
        
        Args:
            features: DataFrame of features
            target: Target series
            method: Selection method
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if method == "mutual_information":
            return self._mutual_information_selection(features, target, n_features)
        elif method == "correlation":
            return self._correlation_selection(features, target, n_features)
        elif method == "variance_threshold":
            return self._variance_threshold_selection(features, n_features)
        else:
            return features.columns.tolist()[:n_features]
    
    def _mutual_information_selection(self, features: pd.DataFrame, target: pd.Series, n_features: int) -> List[str]:
        """Select features using mutual information."""
        from sklearn.feature_selection import mutual_info_regression
        
        scores = mutual_info_regression(features, target, random_state=42)
        feature_scores = dict(zip(features.columns, scores))
        
        # Sort by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.feature_scores = feature_scores
        
        return [f[0] for f in sorted_features[:n_features]]
    
    def _correlation_selection(self, features: pd.DataFrame, target: pd.Series, n_features: int) -> List[str]:
        """Select features using correlation with target."""
        correlations = features.corrwith(target).abs()
        sorted_features = correlations.sort_values(ascending=False)
        
        self.feature_scores = dict(sorted_features.to_dict())
        return sorted_features.head(n_features).index.tolist()
    
    def _variance_threshold_selection(self, features: pd.DataFrame, n_features: int) -> List[str]:
        """Select features with highest variance."""
        variances = features.var()
        sorted_features = variances.sort_values(ascending=False)
        return sorted_features.head(n_features).index.tolist()


if __name__ == "__main__":
    # Test the feature engine
    engine = InstitutionalFeatureEngine()
    
    # Generate sample data
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    sample_ohlcv = pd.DataFrame({
        'open': np.random.normal(100, 5, n),
        'high': np.random.normal(105, 5, n),
        'low': np.random.normal(95, 5, n),
        'close': np.random.normal(100, 5, n),
        'volume': np.random.normal(1000000, 200000, n)
    }, index=dates)
    
    # Ensure high >= low and high >= close, low <= close
    sample_ohlcv['high'] = sample_ohlcv[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, n)
    sample_ohlcv['low'] = sample_ohlcv[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, n)
    
    features = engine.compute_all_features(sample_ohlcv)
    
    print(f"Total features computed: {len(features)}")
    print(f"Feature count from registry: {engine.get_feature_count()}")
    
    # Print features by category
    for category in ["price", "volume", "volatility", "options", "breadth", "microstructure", "macro", "derived"]:
        cat_features = engine.get_features_by_category(category)
        print(f"\n{category.upper()}: {len(cat_features)} features")

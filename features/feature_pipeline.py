"""
Feature Pipeline for Architecture V2
50 core features for alpha generation

Features include:
- Relative Volume, VWAP distance, realized vol, IV, PCR
- FII/DII flow, day-of-week, time-of-day
- ATR, RSI, MACD, Bollinger Band width
- Volume profile, tick volume ratio, bid-ask spread
- Order flow imbalance, momentum, volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass
import polars as pl


@dataclass
class FeatureConfig:
    """Configuration for feature pipeline"""
    n_features: int = 50
    feature_selection_method: str = "Boruta"
    
    # Rolling windows
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 60
    
    # Volume parameters
    volume_avg_window: int = 20
    rv_threshold: float = 2.0
    
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0


class FeaturePipeline:
    """
    Feature Pipeline for Architecture V2
    
    Computes 50 core features for alpha generation:
    1. Volume features: RV, volume ratio, tick volume
    2. Price features: VWAP distance, ATR, momentum
    3. Volatility features: Realized vol, IV, IV percentile
    4. Options features: PCR, skew, term structure
    5. Flow features: FII/DII flow, order flow imbalance
    6. Time features: Day-of-week, time-of-day
    7. Technical features: RSI, MACD, Bollinger Bands
    8. Microstructure features: Bid-ask spread, depth imbalance
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        # Feature definitions
        self.feature_names = self._get_feature_names()
        
        # Feature cache for rolling computations
        self.feature_cache: Dict[str, List[float]] = {}
    
    def _get_feature_names(self) -> List[str]:
        """Return list of all feature names."""
        return [
            # Volume features (5)
            "relative_volume",
            "volume_ratio",
            "volume_avg_20",
            "tick_volume_ratio",
            "volume_profile_slope",
            
            # Price features (8)
            "vwap_distance_pct",
            "vwap",
            "atr_14",
            "atr_ratio",
            "price_momentum_5d",
            "price_momentum_20d",
            "high_low_ratio",
            "close_open_ratio",
            
            # Volatility features (6)
            "realized_volatility_5d",
            "realized_volatility_20d",
            "implied_volatility",
            "iv_percentile",
            "iv_rv_spread",
            "volatility_regime",
            
            # Options features (5)
            "put_call_ratio",
            "iv_skew",
            "term_structure_slope",
            "gamma_exposure",
            "vix",
            
            # Flow features (4)
            "fii_dii_flow",
            "fii_flow",
            "dii_flow",
            "order_flow_imbalance",
            
            # Time features (3)
            "day_of_week",
            "time_of_day",
            "is_expiry_week",
            
            # Technical features (10)
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bollinger_band_width",
            "bollinger_position",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci",
            
            # Microstructure features (4)
            "bid_ask_spread",
            "bid_ask_spread_pct",
            "depth_imbalance",
            "trade_size_avg",
            
            # Market structure features (5)
            "gap_pct",
            "gap_fill",
            "inside_bar",
            "outside_bar",
            "engulfing_pattern"
        ]
    
    def compute_features(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        options_data: Optional[Dict] = None,
        flow_data: Optional[Dict] = None,
        order_book: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Compute all features for a symbol.
        
        Args:
            symbol: Symbol name
            ohlcv: OHLCV DataFrame with columns: open, high, low, close, volume
            options_data: Optional options data (IV, PCR, etc.)
            flow_data: Optional FII/DII flow data
            order_book: Optional order book data
            timestamp: Current timestamp
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Ensure we have enough data
        if len(ohlcv) < self.config.long_window:
            return self._get_default_features()
        
        # Volume features
        features.update(self._compute_volume_features(ohlcv))
        
        # Price features
        features.update(self._compute_price_features(ohlcv))
        
        # Volatility features
        features.update(self._compute_volatility_features(ohlcv, options_data))
        
        # Options features
        features.update(self._compute_options_features(options_data))
        
        # Flow features
        features.update(self._compute_flow_features(flow_data))
        
        # Time features
        features.update(self._compute_time_features(timestamp))
        
        # Technical features
        features.update(self._compute_technical_features(ohlcv))
        
        # Microstructure features
        features.update(self._compute_microstructure_features(order_book, ohlcv))
        
        # Market structure features
        features.update(self._compute_market_structure_features(ohlcv))
        
        return features
    
    def _compute_volume_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute volume-related features."""
        features = {}
        
        volume = ohlcv['volume'].values
        close = ohlcv['close'].values
        
        # Relative Volume
        avg_volume_20 = np.mean(volume[-20:])
        expected_5min_volume = avg_volume_20 / 78  # 390 min / 5
        relative_volume = volume[-1] / expected_5min_volume if expected_5min_volume > 0 else 0
        features['relative_volume'] = relative_volume
        
        # Volume ratio (current vs average)
        features['volume_ratio'] = volume[-1] / avg_volume_20 if avg_volume_20 > 0 else 0
        features['volume_avg_20'] = avg_volume_20
        
        # Tick volume ratio (approximate using volume changes)
        if len(volume) > 1:
            tick_volume_ratio = volume[-1] / volume[-2] if volume[-2] > 0 else 0
            features['tick_volume_ratio'] = tick_volume_ratio
        else:
            features['tick_volume_ratio'] = 0
        
        # Volume profile slope (trend in volume)
        if len(volume) >= 10:
            recent_volume = volume[-10:]
            x = np.arange(len(recent_volume))
            slope = np.polyfit(x, recent_volume, 1)[0]
            features['volume_profile_slope'] = slope / np.mean(recent_volume) if np.mean(recent_volume) > 0 else 0
        else:
            features['volume_profile_slope'] = 0
        
        return features
    
    def _compute_price_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute price-related features."""
        features = {}
        
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        open_ = ohlcv['open'].values
        volume = ohlcv['volume'].values
        
        # VWAP
        typical_price = (high + low + close) / 3
        vwap = np.sum(typical_price * volume) / np.sum(volume) if np.sum(volume) > 0 else close[-1]
        features['vwap'] = vwap
        features['vwap_distance_pct'] = (close[-1] - vwap) / vwap if vwap > 0 else 0
        
        # ATR
        atr = self._calculate_atr(high, low, close, self.config.atr_period)
        features['atr_14'] = atr
        features['atr_ratio'] = atr / close[-1] if close[-1] > 0 else 0
        
        # Momentum
        if len(close) >= 5:
            features['price_momentum_5d'] = (close[-1] - close[-5]) / close[-5] if close[-5] > 0 else 0
        else:
            features['price_momentum_5d'] = 0
        
        if len(close) >= 20:
            features['price_momentum_20d'] = (close[-1] - close[-20]) / close[-20] if close[-20] > 0 else 0
        else:
            features['price_momentum_20d'] = 0
        
        # High/Low ratio
        features['high_low_ratio'] = (high[-1] - low[-1]) / close[-1] if close[-1] > 0 else 0
        
        # Close/Open ratio
        features['close_open_ratio'] = close[-1] / open_[-1] if open_[-1] > 0 else 0
        
        return features
    
    def _compute_volatility_features(
        self,
        ohlcv: pd.DataFrame,
        options_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Compute volatility-related features."""
        features = {}
        
        close = ohlcv['close'].values
        
        # Realized volatility
        returns = np.diff(np.log(close))
        
        if len(returns) >= 5:
            rv_5d = np.std(returns[-5:]) * np.sqrt(252)
            features['realized_volatility_5d'] = rv_5d
        else:
            features['realized_volatility_5d'] = 0
        
        if len(returns) >= 20:
            rv_20d = np.std(returns[-20:]) * np.sqrt(252)
            features['realized_volatility_20d'] = rv_20d
        else:
            features['realized_volatility_20d'] = 0
        
        # Implied volatility from options data
        if options_data:
            features['implied_volatility'] = options_data.get('iv', 0)
            features['iv_percentile'] = options_data.get('iv_percentile', 0)
            features['vix'] = options_data.get('vix', 0)
        else:
            features['implied_volatility'] = 0
            features['iv_percentile'] = 0
            features['vix'] = 0
        
        # IV-RV spread
        iv = features['implied_volatility']
        rv = features['realized_volatility_5d']
        features['iv_rv_spread'] = iv - rv
        
        # Volatility regime
        if rv > 0.25:
            features['volatility_regime'] = 2  # High
        elif rv > 0.15:
            features['volatility_regime'] = 1  # Medium
        else:
            features['volatility_regime'] = 0  # Low
        
        return features
    
    def _compute_options_features(self, options_data: Optional[Dict]) -> Dict[str, float]:
        """Compute options-related features."""
        features = {}
        
        if options_data:
            features['put_call_ratio'] = options_data.get('pcr', 0)
            features['iv_skew'] = options_data.get('skew', 0)
            features['term_structure_slope'] = options_data.get('term_structure', 0)
            features['gamma_exposure'] = options_data.get('gamma', 0)
        else:
            features['put_call_ratio'] = 0
            features['iv_skew'] = 0
            features['term_structure_slope'] = 0
            features['gamma_exposure'] = 0
        
        return features
    
    def _compute_flow_features(self, flow_data: Optional[Dict]) -> Dict[str, float]:
        """Compute flow-related features."""
        features = {}
        
        if flow_data:
            features['fii_dii_flow'] = flow_data.get('fii_dii', 0)
            features['fii_flow'] = flow_data.get('fii', 0)
            features['dii_flow'] = flow_data.get('dii', 0)
            features['order_flow_imbalance'] = flow_data.get('order_flow', 0)
        else:
            features['fii_dii_flow'] = 0
            features['fii_flow'] = 0
            features['dii_flow'] = 0
            features['order_flow_imbalance'] = 0
        
        return features
    
    def _compute_time_features(self, timestamp: Optional[datetime]) -> Dict[str, float]:
        """Compute time-related features."""
        features = {}
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = timestamp.weekday()
        
        # Time of day (minutes from midnight)
        features['time_of_day'] = timestamp.hour * 60 + timestamp.minute
        
        # Is expiry week (Thursday expiry in India)
        day_of_week = timestamp.weekday()
        days_to_thursday = (3 - day_of_week) % 7
        features['is_expiry_week'] = 1.0 if days_to_thursday <= 3 else 0.0
        
        return features
    
    def _compute_technical_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute technical indicator features."""
        features = {}
        
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        
        # RSI
        features['rsi_14'] = self._calculate_rsi(close, self.config.rsi_period)
        
        # MACD
        macd, signal, histogram = self._calculate_macd(
            close,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_width, bb_position = self._calculate_bollinger_bands(
            close,
            self.config.bollinger_period,
            self.config.bollinger_std
        )
        features['bollinger_band_width'] = bb_width
        features['bollinger_position'] = bb_position
        
        # Stochastic
        stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(high, low, close)
        
        # CCI
        features['cci'] = self._calculate_cci(high, low, close)
        
        return features
    
    def _compute_microstructure_features(
        self,
        order_book: Optional[Dict],
        ohlcv: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute microstructure features."""
        features = {}
        
        if order_book:
            bid = order_book.get('bid_price', 0)
            ask = order_book.get('ask_price', 0)
            bid_size = order_book.get('bid_size', 0)
            ask_size = order_book.get('ask_size', 0)
            
            # Bid-ask spread
            if bid > 0 and ask > 0:
                features['bid_ask_spread'] = ask - bid
                features['bid_ask_spread_pct'] = (ask - bid) / bid
            else:
                features['bid_ask_spread'] = 0
                features['bid_ask_spread_pct'] = 0
            
            # Depth imbalance
            total_depth = bid_size + ask_size
            if total_depth > 0:
                features['depth_imbalance'] = (bid_size - ask_size) / total_depth
            else:
                features['depth_imbalance'] = 0
        else:
            features['bid_ask_spread'] = 0
            features['bid_ask_spread_pct'] = 0
            features['depth_imbalance'] = 0
        
        # Trade size average
        volume = ohlcv['volume'].values
        if len(volume) >= 20:
            features['trade_size_avg'] = np.mean(volume[-20:])
        else:
            features['trade_size_avg'] = np.mean(volume) if len(volume) > 0 else 0
        
        return features
    
    def _compute_market_structure_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute market structure features."""
        features = {}
        
        open_ = ohlcv['open'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values
        
        if len(ohlcv) < 2:
            return features
        
        # Gap
        prev_close = close[-2]
        curr_open = open_[-1]
        gap_pct = (curr_open - prev_close) / prev_close if prev_close > 0 else 0
        features['gap_pct'] = gap_pct
        
        # Gap fill (if price has crossed previous close)
        curr_high = high[-1]
        curr_low = low[-1]
        features['gap_fill'] = 1.0 if (curr_low <= prev_close <= curr_high) else 0.0
        
        # Inside bar
        prev_high = high[-2]
        prev_low = low[-2]
        features['inside_bar'] = 1.0 if (curr_high <= prev_high and curr_low >= prev_low) else 0.0
        
        # Outside bar
        features['outside_bar'] = 1.0 if (curr_high >= prev_high and curr_low <= prev_low) else 0.0
        
        # Engulfing pattern
        prev_open = open_[-2]
        prev_close_body = abs(prev_close - prev_open)
        curr_close_body = abs(close[-1] - curr_open)
        
        if prev_close_body > 0 and curr_close_body > 0:
            bullish_engulfing = (prev_close < prev_open and close[-1] > curr_open and 
                                curr_low <= prev_close and curr_high >= prev_open)
            bearish_engulfing = (prev_close > prev_open and close[-1] < curr_open and 
                                curr_high >= prev_close and curr_low <= prev_open)
            features['engulfing_pattern'] = 1.0 if (bullish_engulfing or bearish_engulfing) else 0.0
        else:
            features['engulfing_pattern'] = 0.0
        
        return features
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate Average True Range."""
        if len(close) < period + 1:
            return 0
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        
        return atr
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index."""
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
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[float, float, float]:
        """Calculate MACD."""
        if len(prices) < slow + signal:
            return 0, 0, 0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line[-1] - signal_line[-1]
        
        return macd_line[-1], signal_line[-1], histogram
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float) -> Tuple[float, float]:
        """Calculate Bollinger Band width and position."""
        if len(prices) < period:
            return 0, 0
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + std_dev * std
        lower_band = sma - std_dev * std
        
        width = (upper_band - lower_band) / sma if sma > 0 else 0
        position = (prices[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
        
        return width, position
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(close) < k_period + d_period:
            return 50, 50
        
        recent_high = np.max(high[-k_period:])
        recent_low = np.min(low[-k_period:])
        
        if recent_high == recent_low:
            k = 50
        else:
            k = 100 * (close[-1] - recent_low) / (recent_high - recent_low)
        
        # Simple moving average for %D
        d = k  # Simplified
        
        return k, d
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Williams %R."""
        if len(close) < period:
            return -50
        
        highest_high = np.max(high[-period:])
        lowest_low = np.min(low[-period:])
        
        if highest_high == lowest_low:
            return -50
        
        williams_r = -100 * (highest_high - close[-1]) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> float:
        """Calculate Commodity Channel Index."""
        if len(close) < period:
            return 0
        
        typical_price = (high + low + close) / 3
        sma = np.mean(typical_price[-period:])
        mean_deviation = np.mean(np.abs(typical_price[-period:] - sma))
        
        if mean_deviation == 0:
            return 0
        
        cci = (typical_price[-1] - sma) / (0.015 * mean_deviation)
        return cci
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when insufficient data."""
        return {name: 0.0 for name in self.feature_names}
    
    def select_features(self, features: Dict[str, float], method: str = "Boruta") -> List[str]:
        """
        Select most important features.
        
        Args:
            features: Dictionary of all features
            method: Feature selection method
            
        Returns:
            List of selected feature names
        """
        # For now, return all features
        # In production, implement Boruta or other feature selection
        return list(features.keys())

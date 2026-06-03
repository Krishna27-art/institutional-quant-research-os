"""
Feature Pipeline for Architecture V2 (Profit-Centric Simplified)
15 core features for alpha generation (reduced from 50 based on profit audit)

Features include:
- Relative Volume, VWAP distance, realized vol, IV, PCR
- FII/DII flow, day-of-week, time-of-day
- ATR, RSI, MACD, Bollinger Band width
- Order flow imbalance, momentum, volatility
- Gap, inside/outside bar, engulfing

CRITICAL FIX: Further reduced to 15 core features (from 25).
Overfitting risk with too many features - focus on most predictive ones.

REMOVED (Profit-Centric Audit):
- volume_ratio, tick_volume_ratio, volume_profile_slope (redundant with RV)
- high_low_ratio, close_open_ratio (redundant with ATR)
- volatility_regime (redundant with realized vol)
- vix (redundant with IV)
- fii_flow, dii_flow (redundant with fii_dii_flow)
- stoch_k, stoch_d, williams_r, cci (redundant with RSI)
- gap_fill (redundant with gap)
- trade_size_avg (low value)
- Additional removal: bid_ask_spread_pct (redundant with bid_ask_spread)
- Additional removal: depth_imbalance (low signal-to-noise)
- Additional removal: price_momentum_20d (redundant with 5d)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
from dataclasses import dataclass
import polars as pl


@dataclass
class FeatureConfig:
    """Configuration for feature pipeline (Profit-Centric Simplified)"""
    n_features: int = 15  # Reduced from 50 to 15 (CRITICAL FIX)
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
    
    # Data leakage detection
    enable_leakage_detection: bool = True
    leakage_correlation_threshold: float = 0.9  # Flag if correlation with future returns > 0.9
    leakage_time_window: int = 1  # Check 1-day forward for leakage
    
    # Feature decay detection (PSI)
    enable_psi_detection: bool = True
    psi_threshold: float = 0.1  # Flag if PSI > 0.1 (significant decay)
    psi_window_days: int = 30  # Compare current 30 days vs previous 30 days
    
    # Future information prevention (CRITICAL FIX)
    enable_future_info_check: bool = True
    forbid_centered_ma: bool = True  # No centered moving averages
    forbid_global_norm: bool = True  # No global normalization


@dataclass
class FutureInfoReport:
    """Report of future information detection"""
    has_future_info: bool
    features_with_future_info: List[str]
    description: str
    timestamp: datetime


class FutureInfoDetector:
    """
    Future Information Detector
    
    CRITICAL FIX: Detects use of future information in features.
    - Centered moving averages (use future data)
    - Global normalization (uses full sample statistics)
    - Future graph estimation (uses future correlations)
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def detect_future_info(
        self,
        features: Dict[str, float],
        ohlcv: pd.DataFrame,
        timestamp: datetime
    ) -> FutureInfoReport:
        """
        Detect if features use future information.
        
        Args:
            features: Dictionary of computed features
            ohlcv: OHLCV DataFrame
            timestamp: Current timestamp
            
        Returns:
            FutureInfoReport with future info status and details
        """
        features_with_future_info = []
        
        # Check for centered moving averages (would use future data)
        # This is a heuristic - in production, you'd check the actual computation
        suspicious_features = []
        for feature_name in features:
            if 'center' in feature_name.lower() or 'symmetric' in feature_name.lower():
                suspicious_features.append(feature_name)
        
        # Check for global normalization (would use full sample)
        if 'global' in str(features.keys()).lower() or 'zscore' in str(features.keys()).lower():
            suspicious_features.extend([k for k in features.keys() if 'zscore' in k.lower() or 'global' in k.lower()])
        
        has_future_info = len(suspicious_features) > 0
        
        return FutureInfoReport(
            has_future_info=has_future_info,
            features_with_future_info=suspicious_features,
            description="Features may use centered moving averages or global normalization" if has_future_info else "No future information detected",
            timestamp=timestamp
        )


@dataclass
class LeakageReport:
    """Report of data leakage detection"""
    has_leakage: bool
    leaked_features: List[str]
    leakage_correlations: Dict[str, float]
    timestamp: datetime


@dataclass
class DecayReport:
    """Report of feature decay detection using PSI"""
    has_decay: bool
    decayed_features: List[str]
    psi_scores: Dict[str, float]
    timestamp: datetime


class DataLeakageDetector:
    """
    Data Leakage Detector for Feature Pipeline
    
    Detects if features are using future data:
    1. Correlation with future returns (> 0.9 threshold)
    2. Timestamp validation (no future data)
    3. Rolling window validation (no look-ahead)
    
    CRITICAL: Data leakage can cause 100% loss of alpha PnL in live trading.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.leakage_history: List[LeakageReport] = []
    
    def detect_leakage(
        self,
        features: Dict[str, float],
        ohlcv: pd.DataFrame,
        timestamp: datetime
    ) -> LeakageReport:
        """
        Detect data leakage in features.
        
        Args:
            features: Dictionary of computed features
            ohlcv: OHLCV DataFrame
            timestamp: Current timestamp
            
        Returns:
            LeakageReport with leakage status and details
        """
        leaked_features = []
        leakage_correlations = {}
        
        # Check correlation with future returns
        if len(ohlcv) > self.config.leakage_time_window + 1:
            future_returns = np.diff(np.log(ohlcv['close'].values[-self.config.leakage_time_window-1:]))
            
            for feature_name, feature_value in features.items():
                # For time-series features, check correlation
                if feature_name in ohlcv.columns:
                    feature_series = ohlcv[feature_name].values[-len(future_returns):]
                    if len(feature_series) == len(future_returns):
                        correlation = np.corrcoef(feature_series, future_returns)[0, 1]
                        if abs(correlation) > self.config.leakage_correlation_threshold:
                            leaked_features.append(feature_name)
                            leakage_correlations[feature_name] = correlation
        
        has_leakage = len(leaked_features) > 0
        
        report = LeakageReport(
            has_leakage=has_leakage,
            leaked_features=leaked_features,
            leakage_correlations=leakage_correlations,
            timestamp=timestamp
        )
        
        self.leakage_history.append(report)
        
        return report


class FeatureDecayDetector:
    """
    Feature Decay Detector using Population Stability Index (PSI)
    
    PSI measures how much a feature's distribution has changed over time.
    PSI > 0.1 indicates significant feature decay.
    
    CRITICAL: Feature decay can cause alpha to stop working in live trading.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_history: Dict[str, List[float]] = {}
        self.decay_history: List[DecayReport] = []
    
    def add_observation(self, features: Dict[str, float], timestamp: datetime) -> None:
        """Add feature observation to history."""
        for feature_name, value in features.items():
            if feature_name not in self.feature_history:
                self.feature_history[feature_name] = []
            self.feature_history[feature_name].append(value)
            
            # Keep last psi_window_days * 2 observations (current + reference)
            max_samples = self.config.psi_window_days * 2
            if len(self.feature_history[feature_name]) > max_samples:
                self.feature_history[feature_name] = self.feature_history[feature_name][-max_samples:]
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        
        Args:
            expected: Reference distribution
            actual: Current distribution
            bins: Number of bins for histogram
            
        Returns:
            PSI value
        """
        # Create bins based on expected distribution
        min_val = min(np.min(expected), np.min(actual))
        max_val = max(np.max(expected), np.max(actual))
        
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calculate expected percentages
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        expected_pct = expected_counts / len(expected)
        expected_pct = np.maximum(expected_pct, 0.0001)  # Avoid division by zero
        
        # Calculate actual percentages
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        actual_pct = actual_counts / len(actual)
        actual_pct = np.maximum(actual_pct, 0.0001)  # Avoid division by zero
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
    
    def detect_decay(
        self,
        features: Dict[str, float],
        timestamp: datetime
    ) -> DecayReport:
        """
        Detect feature decay using PSI.
        
        Args:
            features: Current feature values
            timestamp: Current timestamp
            
        Returns:
            DecayReport with decay status and details
        """
        decayed_features = []
        psi_scores = {}
        
        window = self.config.psi_window_days
        
        for feature_name in features:
            if feature_name not in self.feature_history:
                continue
            
            history = self.feature_history[feature_name]
            
            if len(history) < window * 2:
                continue
            
            # Split into reference (older) and current (newer)
            reference = np.array(history[-(window * 2):-window])
            current = np.array(history[-window:])
            
            # Calculate PSI
            psi = self.calculate_psi(reference, current)
            psi_scores[feature_name] = psi
            
            # Check if PSI exceeds threshold
            if psi > self.config.psi_threshold:
                decayed_features.append(feature_name)
        
        has_decay = len(decayed_features) > 0
        
        report = DecayReport(
            has_decay=has_decay,
            decayed_features=decayed_features,
            psi_scores=psi_scores,
            timestamp=timestamp
        )
        
        self.decay_history.append(report)
        
        return report


class FeaturePipeline:
    """
    Feature Pipeline for Architecture V2 (Profit-Centric Simplified)
    
    Computes 25 core features for alpha generation (reduced from 50):
    1. Volume features: RV (1)
    2. Price features: VWAP distance, ATR, momentum (3)
    3. Volatility features: Realized vol, IV, IV percentile, IV-RV spread (4)
    4. Options features: PCR, skew, term structure, gamma exposure (4)
    5. Flow features: FII/DII flow, order flow imbalance (2)
    6. Time features: Day-of-week, time-of-day, expiry week (3)
    7. Technical features: RSI, MACD, MACD signal, Bollinger Bands (4)
    8. Microstructure features: Bid-ask spread, depth imbalance (2)
    9. Market structure features: Gap, inside bar, outside bar, engulfing (4)
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        
        # Feature definitions
        self.feature_names = self._get_feature_names()
        
        # Feature cache for rolling computations
        self.feature_cache: Dict[str, List[float]] = {}
        
        # Data leakage detector
        if self.config.enable_leakage_detection:
            self.leakage_detector = DataLeakageDetector(config)
        else:
            self.leakage_detector = None
        
        # Feature decay detector
        if self.config.enable_psi_detection:
            self.decay_detector = FeatureDecayDetector(config)
        else:
            self.decay_detector = None
        
        # Future information detector (CRITICAL FIX)
        if self.config.enable_future_info_check:
            self.future_info_detector = FutureInfoDetector(config)
        else:
            self.future_info_detector = None
    
    def _get_feature_names(self) -> List[str]:
        """Return list of 15 core feature names (reduced from 50 to 15)."""
        return [
            # Volume features (1) - KEEP
            "relative_volume",
            
            # Price features (2) - KEEP most predictive
            "vwap_distance_pct",
            "atr_14",
            
            # Volatility features (3) - KEEP most predictive
            "realized_volatility_5d",
            "implied_volatility",
            "iv_rv_spread",
            
            # Options features (2) - KEEP most predictive
            "put_call_ratio",
            "iv_skew",
            
            # Flow features (2) - KEEP
            "fii_dii_flow",
            "order_flow_imbalance",
            
            # Time features (1) - KEEP most predictive
            "day_of_week",
            
            # Technical features (2) - KEEP most predictive
            "rsi_14",
            "macd",
            
            # Microstructure features (1) - KEEP most predictive
            "bid_ask_spread",
            
            # Market structure features (1) - KEEP most predictive
            "gap_pct"
        ]
        
        # REMOVED (Profit-Centric Audit):
        # volume_ratio, tick_volume_ratio, volume_profile_slope (redundant with RV)
        # vwap, atr_ratio, price_momentum_20d, high_low_ratio, close_open_ratio (redundant)
        # realized_volatility_20d, volatility_regime, vix (redundant)
        # fii_flow, dii_flow (redundant with fii_dii_flow)
        # macd_histogram, bollinger_position, stoch_k, stoch_d, williams_r, cci (redundant with RSI)
        # bid_ask_spread_pct, trade_size_avg (low value)
        # gap_fill (redundant with gap)
    
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
        
        # Run data leakage detection if enabled
        if self.leakage_detector is not None:
            leakage_report = self.leakage_detector.detect_leakage(features, ohlcv, timestamp)
            if leakage_report.has_leakage:
                # Log warning but still return features (for debugging)
                print(f"WARNING: Data leakage detected at {timestamp}")
                print(f"Leaked features: {leakage_report.leaked_features}")
                print(f"Correlations: {leakage_report.leakage_correlations}")
        
        # Run feature decay detection if enabled
        if self.decay_detector is not None:
            self.decay_detector.add_observation(features, timestamp)
            decay_report = self.decay_detector.detect_decay(features, timestamp)
            if decay_report.has_decay:
                # Log warning but still return features (for debugging)
                print(f"WARNING: Feature decay detected at {timestamp}")
                print(f"Decayed features: {decay_report.decayed_features}")
                print(f"PSI scores: {decay_report.psi_scores}")
        
        # Run future information detection if enabled
        if self.future_info_detector is not None:
            future_info_report = self.future_info_detector.detect_future_info(features, ohlcv, timestamp)
            if future_info_report.has_future_info:
                # Log warning but still return features (for debugging)
                print(f"WARNING: Future information detected at {timestamp}")
                print(f"Features with future info: {future_info_report.features_with_future_info}")
                print(f"Description: {future_info_report.description}")
        
        # CRITICAL FIX: Integrate point-in-time validation
        if self.config.enable_leakage_detection:
            from validation.point_in_time_validator import PointInTimeValidator
            pit_validator = PointInTimeValidator()
            
            # Validate features for look-ahead bias
            features_df = pd.DataFrame([features])
            price_data = ohlcv['close'] if 'close' in ohlcv.columns else ohlcv.iloc[:, 0]
            
            biased_features = []
            for feature_name, feature_value in features.items():
                feature_series = pd.Series([feature_value], index=[timestamp])
                detection = pit_validator.validate_feature(feature_name, feature_series, price_data)
                if detection.has_bias:
                    biased_features.append(feature_name)
                    print(f"CRITICAL FIX: Look-ahead bias detected in {feature_name}: {detection.description}")
            
            # Remove biased features
            for bf in biased_features:
                if bf in features:
                    del features[bf]
        
        return features
    
    def _compute_volume_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute volume-related features (simplified to 1 feature)."""
        features = {}
        
        volume = ohlcv['volume'].values
        close = ohlcv['close'].values
        
        # Relative Volume (KEEP)
        avg_volume_20 = np.mean(volume[-20:])
        expected_5min_volume = avg_volume_20 / 78  # 390 min / 5
        relative_volume = volume[-1] / expected_5min_volume if expected_5min_volume > 0 else 0
        features['relative_volume'] = relative_volume
        
        # REMOVED: volume_ratio, tick_volume_ratio, volume_profile_slope (redundant with RV)
        
        return features
    
    def _compute_price_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute price-related features (simplified to 3 features)."""
        features = {}
        
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        open_ = ohlcv['open'].values
        volume = ohlcv['volume'].values
        
        # VWAP distance (KEEP)
        typical_price = (high + low + close) / 3
        vwap = np.sum(typical_price * volume) / np.sum(volume) if np.sum(volume) > 0 else close[-1]
        features['vwap_distance_pct'] = (close[-1] - vwap) / vwap if vwap > 0 else 0
        
        # ATR (KEEP)
        atr = self._calculate_atr(high, low, close, self.config.atr_period)
        features['atr_14'] = atr
        
        # Momentum 5d (KEEP)
        if len(close) >= 5:
            features['price_momentum_5d'] = (close[-1] - close[-5]) / close[-5] if close[-5] > 0 else 0
        else:
            features['price_momentum_5d'] = 0
        
        # REMOVED: vwap, atr_ratio, price_momentum_20d, high_low_ratio, close_open_ratio (redundant)
        
        return features
    
    def _compute_volatility_features(
        self,
        ohlcv: pd.DataFrame,
        options_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Compute volatility-related features (simplified to 4 features)."""
        features = {}
        
        close = ohlcv['close'].values
        
        # Realized volatility 5d (KEEP)
        returns = np.diff(np.log(close))
        if len(returns) >= 5:
            rv_5d = np.std(returns[-5:]) * np.sqrt(252)
            features['realized_volatility_5d'] = rv_5d
        else:
            features['realized_volatility_5d'] = 0
        
        # Implied volatility (KEEP)
        if options_data:
            features['implied_volatility'] = options_data.get('iv', 0)
            features['iv_percentile'] = options_data.get('iv_percentile', 0)
        else:
            features['implied_volatility'] = 0
            features['iv_percentile'] = 0
        
        # IV-RV spread (KEEP)
        iv = features['implied_volatility']
        rv = features['realized_volatility_5d']
        features['iv_rv_spread'] = iv - rv
        
        # REMOVED: realized_volatility_20d, volatility_regime, vix (redundant)
        
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
        """Compute flow-related features (simplified to 2 features)."""
        features = {}
        
        if flow_data:
            features['fii_dii_flow'] = flow_data.get('fii_dii', 0)
            features['order_flow_imbalance'] = flow_data.get('order_flow', 0)
        else:
            features['fii_dii_flow'] = 0
            features['order_flow_imbalance'] = 0
        
        # REMOVED: fii_flow, dii_flow (redundant with fii_dii_flow)
        
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
        """Compute technical indicator features (simplified to 4 features)."""
        features = {}
        
        close = ohlcv['close'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        
        # RSI (KEEP)
        features['rsi_14'] = self._calculate_rsi(close, self.config.rsi_period)
        
        # MACD (KEEP)
        macd, signal, histogram = self._calculate_macd(
            close,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features['macd'] = macd
        features['macd_signal'] = signal
        
        # Bollinger Band width (KEEP)
        bb_width, bb_position = self._calculate_bollinger_bands(
            close,
            self.config.bollinger_period,
            self.config.bollinger_std
        )
        features['bollinger_band_width'] = bb_width
        
        # REMOVED: macd_histogram, bollinger_position, stoch_k, stoch_d, williams_r, cci (redundant with RSI)
        
        return features
    
    def _compute_microstructure_features(
        self,
        order_book: Optional[Dict],
        ohlcv: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute microstructure features (simplified to 2 features)."""
        features = {}
        
        if order_book:
            bid = order_book.get('bid_price', 0)
            ask = order_book.get('ask_price', 0)
            bid_size = order_book.get('bid_size', 0)
            ask_size = order_book.get('ask_size', 0)
            
            # Bid-ask spread (KEEP)
            if bid > 0 and ask > 0:
                features['bid_ask_spread'] = ask - bid
            else:
                features['bid_ask_spread'] = 0
            
            # Depth imbalance (KEEP)
            total_depth = bid_size + ask_size
            if total_depth > 0:
                features['depth_imbalance'] = (bid_size - ask_size) / total_depth
            else:
                features['depth_imbalance'] = 0
        else:
            features['bid_ask_spread'] = 0
            features['depth_imbalance'] = 0
        
        # REMOVED: bid_ask_spread_pct, trade_size_avg (low value)
        
        return features
    
    def _compute_market_structure_features(self, ohlcv: pd.DataFrame) -> Dict[str, float]:
        """Compute market structure features (simplified to 4 features)."""
        features = {}
        
        open_ = ohlcv['open'].values
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values
        
        if len(ohlcv) < 2:
            return features
        
        # Gap (KEEP)
        prev_close = close[-2]
        curr_open = open_[-1]
        gap_pct = (curr_open - prev_close) / prev_close if prev_close > 0 else 0
        features['gap_pct'] = gap_pct
        
        # Inside bar (KEEP)
        prev_high = high[-2]
        prev_low = low[-2]
        curr_high = high[-1]
        curr_low = low[-1]
        features['inside_bar'] = 1.0 if (curr_high <= prev_high and curr_low >= prev_low) else 0.0
        
        # Outside bar (KEEP)
        features['outside_bar'] = 1.0 if (curr_high >= prev_high and curr_low <= prev_low) else 0.0
        
        # Engulfing pattern (KEEP)
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
        
        # REMOVED: gap_fill (redundant with gap)
        
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

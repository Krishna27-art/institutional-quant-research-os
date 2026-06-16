"""
Mean Reversion Alpha Strategies

This module implements mean reversion-based alpha strategies including:
- Pairs Trading (Cointegration)
- Pairs Trading with Kalman Filter (Dynamic Hedge Ratio)
- Opening Range Breakout (ORB) with Relative Volume Filter
- VWAP Reversion
- Bollinger Band Reversion

Based on Audit Report Priority 2: Alpha Generation
Research Papers: Gatev et al (2006), Zarattini & Aziz (2023-2025), Avellaneda & Lee (2010)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

def estimate_round_trip_costs(price: float) -> float:
    """
    Indian Transaction Costs (ported from orb_zarattini.py).
    Calculated on a per-share basis to estimate high-turnover drag.
    """
    stamp_duty_rate = 0.00015  # 0.015%
    stt_rate = 0.00025  # 0.025%
    exchange_rate = 0.0000345  # 0.00345%
    sebi_fees_rate = 0.000001  # 0.0001%
    slippage_bps = 2.0  # 0.0002
    
    # Variable cost multiplier per leg
    per_leg_cost = stamp_duty_rate + stt_rate + exchange_rate + sebi_fees_rate + (slippage_bps / 10000)
    # Round trip is 2 legs
    return price * (per_leg_cost * 2)


@dataclass
class MeanReversionSignal:
    """Mean reversion trading signal."""
    symbol: str
    strategy: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    z_score: float
    mean: float
    std: float
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PairsTradingStrategy:
    """
    Pairs Trading Strategy using Cointegration.
    
    Identifies cointegrated pairs and trades when the spread deviates from mean.
    """
    
    def __init__(self, lookback_days: int = 252, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        """
        Initialize pairs trading strategy.
        
        Args:
            lookback_days: Lookback period for cointegration test
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.lookback_days = lookback_days
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        self.cointegrated_pairs: Dict[Tuple[str, str], Dict] = {}
        
        logger.info(f"PairsTradingStrategy initialized with {lookback_days} day lookback")
    
    def find_cointegrated_pairs(
        self,
        data_dict: Dict[str, pd.DataFrame],
        significance_level: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs among stocks.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            significance_level: Significance level for cointegration test
            
        Returns:
            List of (symbol1, symbol2, p_value) tuples
        """
        symbols = list(data_dict.keys())
        cointegrated = []
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                
                # Get aligned price series
                prices1 = data_dict[s1]['close'].tail(self.lookback_days)
                prices2 = data_dict[s2]['close'].tail(self.lookback_days)
                
                # Align indices
                common_index = prices1.index.intersection(prices2.index)
                if len(common_index) < self.lookback_days * 0.8:
                    continue
                
                prices1 = prices1.loc[common_index]
                prices2 = prices2.loc[common_index]
                
                # Test for cointegration
                try:
                    score, p_value, _ = coint(prices1, prices2)
                    
                    if p_value < significance_level:
                        cointegrated.append((s1, s2, p_value))
                        
                        # Calculate hedge ratio
                        hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)
                        
                        # Store pair information
                        self.cointegrated_pairs[(s1, s2)] = {
                            'hedge_ratio': hedge_ratio,
                            'p_value': p_value,
                            'last_updated': datetime.now()
                        }
                        
                        logger.info(f"Found cointegrated pair: {s1}-{s2} (p={p_value:.4f})")
                        
                except Exception as e:
                    logger.warning(f"Cointegration test failed for {s1}-{s2}: {e}")
        
        return cointegrated
    
    def _calculate_hedge_ratio(self, prices1: pd.Series, prices2: pd.Series) -> float:
        """Calculate hedge ratio using OLS regression."""
        # prices1 = hedge_ratio * prices2 + residual
        X = sm.add_constant(prices2)
        model = sm.OLS(prices1, X).fit()
        return model.params[1]
    
    def generate_signal(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol1: str,
        symbol2: str
    ) -> Optional[MeanReversionSignal]:
        """
        Generate signal for a cointegrated pair.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            symbol1: First symbol
            symbol2: Second symbol
            
        Returns:
            MeanReversionSignal
        """
        pair_key = (symbol1, symbol2)
        if pair_key not in self.cointegrated_pairs:
            return None
        
        pair_info = self.cointegrated_pairs[pair_key]
        hedge_ratio = pair_info['hedge_ratio']
        
        # Get recent prices
        prices1 = data_dict[symbol1]['close'].tail(60)
        prices2 = data_dict[symbol2]['close'].tail(60)
        
        # Align indices
        common_index = prices1.index.intersection(prices2.index)
        prices1 = prices1.loc[common_index]
        prices2 = prices2.loc[common_index]
        
        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        
        # Calculate z-score
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (spread.iloc[-1] - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Generate signal
        if z_score > self.entry_threshold:
            # Spread is too high - short symbol1, long symbol2
            signal = -1.0
            confidence = min(1.0, (z_score - self.entry_threshold) / 2.0)
        elif z_score < -self.entry_threshold:
            # Spread is too low - long symbol1, short symbol2
            signal = 1.0
            confidence = min(1.0, (abs(z_score) - self.entry_threshold) / 2.0)
        else:
            signal = 0.0
            confidence = 0.0
            
        # Apply Indian Transaction Costs filter
        expected_profit = max(0, (abs(z_score) - self.exit_threshold)) * spread_std
        est_cost = estimate_round_trip_costs(prices1.iloc[-1]) + abs(hedge_ratio) * estimate_round_trip_costs(prices2.iloc[-1])
        if expected_profit < est_cost:
            signal = 0.0
            confidence = 0.0
        
        return MeanReversionSignal(
            symbol=f"{symbol1}-{symbol2}",
            strategy="PairsTrading",
            signal=signal,
            confidence=confidence,
            z_score=z_score,
            mean=spread_mean,
            std=spread_std,
            timestamp=datetime.now(),
            metadata={
                'hedge_ratio': hedge_ratio,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'current_spread': spread.iloc[-1]
            }
        )


class KalmanPairs:
    """
    Pairs Trading with Kalman Filter (Dynamic Hedge Ratio)
    
    Formula: y_t = α + β_t x_t + ε_t, β_t = β_{t-1} + η_t (random walk)
    
    Kalman filter estimates time-varying beta for dynamic hedging.
    
    Entry: |z| > 2 → short expensive leg, long cheap leg
    Exit: |z| < 0.5
    
    Expected Sharpe: 0.6
    Capacity: 500 Cr
    Turnover: 200%/month
    Best Regime: Sideways
    Failure: Structural break
    """
    
    def __init__(self, delta: float = 1e-5, vt: float = 1e-3, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        """
        Initialize Kalman Filter pairs trading.
        
        Args:
            delta: State transition variance (random walk noise)
            vt: Observation variance (measurement noise)
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        self.delta = delta
        self.vt = vt
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # Kalman filter state
        self.beta = 1.0
        self.P = 1.0  # Covariance matrix
        self.R = None  # Residuals for z-score calculation
        
        logger.info(f"KalmanPairs initialized with delta={delta}, vt={vt}")
    
    def update(self, y: float, x: float) -> float:
        """
        Update Kalman filter with new observation.
        
        Args:
            y: Price of asset y
            x: Price of asset x
            
        Returns:
            Residual (z-score for entry/exit)
        """
        # Prediction step
        beta_pred = self.beta
        P_pred = self.P + self.delta
        
        # Update step
        residual = y - beta_pred * x
        K = P_pred * x / (x * P_pred * x + self.vt)  # Kalman gain
        self.beta = beta_pred + K * residual
        self.P = (1 - K * x) * P_pred
        
        # Store residual for z-score calculation
        if self.R is None:
            self.R = []
        self.R.append(residual)
        if len(self.R) > 60:  # Keep last 60 observations
            self.R.pop(0)
        
        # Calculate z-score
        if len(self.R) >= 20:
            z = residual / (np.std(self.R) + 1e-8)
        else:
            z = 0
        
        return z
    
    def get_signal(
        self,
        z: float,
        y_price: Optional[float] = None,
        x_price: Optional[float] = None,
        spread_std: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Get trading signal from z-score.
        
        Args:
            z: Z-score
            y_price: Price of asset y (optional, for cost filtering)
            x_price: Price of asset x (optional, for cost filtering)
            spread_std: Standard deviation of residuals (optional, for cost filtering)
            
        Returns:
            Tuple of (signal, state)
        """
        if abs(z) > self.entry_threshold:
            # Apply cost filtering if parameters are provided
            if y_price is not None and x_price is not None and spread_std is not None:
                expected_profit = max(0, (abs(z) - self.exit_threshold)) * spread_std
                est_cost = estimate_round_trip_costs(y_price) + abs(self.beta) * estimate_round_trip_costs(x_price)
                if expected_profit < est_cost:
                    return 0.0, "COST_BARRIER"
            
            if z > 0:
                return -1.0, "SHORT_Y_LONG_X"  # y expensive, x cheap
            else:
                return 1.0, "LONG_Y_SHORT_X"  # y cheap, x expensive
        elif abs(z) < self.exit_threshold:
            return 0.0, "EXIT"
        else:
            return 0.0, "HOLD"
    
    def reset(self):
        """Reset Kalman filter state."""
        self.beta = 1.0
        self.P = 1.0
        self.R = None


class ORBStrategy:
    """
    Opening Range Breakout (ORB) Strategy with Relative Volume Filter
    
    Zarattini & Aziz 2023-2025
    
    First 5-min high/low. If first candle bullish, place buy stop at high; 
    if bearish, sell stop at low.
    
    Filter: Only trade if Relative Volume > 100% (volume in first 5m > 14-day average).
    
    Expected Sharpe: 0.9
    Capacity: 100 Cr
    Turnover: 5000%/month
    Best Regime: Trending
    Failure: Doji open
    """
    
    def __init__(
        self, 
        opening_range_minutes: int = 5, 
        breakout_threshold: float = 0.001,
        volume_lookback: int = 14,
        min_rel_volume: float = 1.0
    ):
        """
        Initialize ORB strategy with relative volume filter.
        
        Args:
            opening_range_minutes: Duration of opening range in minutes
            breakout_threshold: Minimum breakout percentage
            volume_lookback: Days for average volume calculation
            min_rel_volume: Minimum relative volume threshold
        """
        self.opening_range_minutes = opening_range_minutes
        self.breakout_threshold = breakout_threshold
        self.volume_lookback = volume_lookback
        self.min_rel_volume = min_rel_volume
        
        logger.info(f"ORBStrategy initialized with {opening_range_minutes} min opening range and RV filter")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str,
        current_time: datetime = None
    ) -> Optional[MeanReversionSignal]:
        """
        Generate ORB signal with relative volume filter.
        
        Args:
            data: DataFrame with intraday OHLCV data
            symbol: Stock symbol
            current_time: Current time
            
        Returns:
            MeanReversionSignal
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check if we're in trading hours
        if not self._is_trading_hours(current_time):
            return None
        
        # Get today's data
        today_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        today_data = data[data.index >= today_start]
        
        if len(today_data) < 30:  # Need at least 30 minutes of data
            return None
        
        # Calculate opening range (first N minutes)
        opening_range_end = today_start + timedelta(minutes=self.opening_range_minutes)
        opening_data = today_data[today_data.index <= opening_range_end]
        
        if len(opening_data) == 0:
            return None
        
        or_high = opening_data['high'].max()
        or_low = opening_data['low'].min()
        or_range = or_high - or_low
        or_volume = opening_data['volume'].sum()
        
        # Calculate average volume over lookback period
        # Get historical data for average volume calculation
        historical_data = data[data.index < today_start].tail(self.volume_lookback)
        if len(historical_data) > 0:
            avg_daily_volume = historical_data['volume'].mean()
            # Estimate 5-min average volume (assuming 375 trading minutes per day)
            avg_5min_volume = avg_daily_volume / (375 / self.opening_range_minutes)
            rel_volume = or_volume / (avg_5min_volume + 1e-8)
        else:
            rel_volume = 1.0
        
        # Relative volume filter
        if rel_volume < self.min_rel_volume:
            return None
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Check for breakout
        if current_price > or_high * (1 + self.breakout_threshold):
            # Breakout above opening range - long
            signal = 1.0
            z_score = (current_price - or_high) / or_range if or_range > 0 else 0
            confidence = min(1.0, z_score / 2.0)
        elif current_price < or_low * (1 - self.breakout_threshold):
            # Breakdown below opening range - short
            signal = -1.0
            z_score = (or_low - current_price) / or_range if or_range > 0 else 0
            confidence = min(1.0, z_score / 2.0)
        else:
            signal = 0.0
            z_score = 0.0
            confidence = 0.0
        
        return MeanReversionSignal(
            symbol=symbol,
            strategy="ORB",
            signal=signal,
            confidence=confidence,
            z_score=z_score,
            mean=(or_high + or_low) / 2,
            std=or_range / 4,
            timestamp=current_time,
            metadata={
                'or_high': or_high,
                'or_low': or_low,
                'or_range': or_range,
                'current_price': current_price,
                'rel_volume': rel_volume,
                'or_volume': or_volume
            }
        )
    
    def _is_trading_hours(self, dt: datetime) -> bool:
        """Check if given time is during trading hours."""
        time = dt.time()
        return time(9, 15) <= time <= time(15, 30)


class VWAPReversion:
    """
    VWAP Reversion Strategy.
    
    Trades when price deviates significantly from VWAP.
    """
    
    def __init__(self, lookback_minutes: int = 60, deviation_threshold: float = 0.002):
        """
        Initialize VWAP reversion strategy.
        
        Args:
            lookback_minutes: Lookback period for VWAP calculation
            deviation_threshold: Deviation threshold for entry
        """
        self.lookback_minutes = lookback_minutes
        self.deviation_threshold = deviation_threshold
        
        logger.info(f"VWAPReversion initialized with {lookback_minutes} min lookback")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Optional[MeanReversionSignal]:
        """
        Generate VWAP reversion signal.
        
        Args:
            data: DataFrame with intraday OHLCV data
            symbol: Stock symbol
            
        Returns:
            MeanReversionSignal
        """
        if len(data) < self.lookback_minutes:
            return None
        
        # Get recent data
        recent_data = data.tail(self.lookback_minutes)
        
        # Calculate VWAP
        typical_price = (recent_data['high'] + recent_data['low'] + recent_data['close']) / 3
        vwap = (typical_price * recent_data['volume']).sum() / recent_data['volume'].sum()
        
        # Calculate standard deviation of price around VWAP
        price_std = recent_data['close'].std()
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Calculate deviation from VWAP
        deviation = (current_price - vwap) / vwap
        
        # Generate signal
        if deviation > self.deviation_threshold:
            # Price above VWAP - expect reversion, short
            signal = -1.0
            z_score = deviation / self.deviation_threshold
            confidence = min(1.0, z_score / 2.0)
        elif deviation < -self.deviation_threshold:
            # Price below VWAP - expect reversion, long
            signal = 1.0
            z_score = abs(deviation) / self.deviation_threshold
            confidence = min(1.0, z_score / 2.0)
        else:
            signal = 0.0
            z_score = 0.0
            confidence = 0.0
            
        # Apply Indian Transaction Costs filter
        expected_profit = abs(deviation) * vwap
        est_cost = estimate_round_trip_costs(current_price)
        if expected_profit < est_cost:
            signal = 0.0
            confidence = 0.0
        
        return MeanReversionSignal(
            symbol=symbol,
            strategy="VWAPReversion",
            signal=signal,
            confidence=confidence,
            z_score=z_score,
            mean=vwap,
            std=price_std,
            timestamp=datetime.now(),
            metadata={
                'vwap': vwap,
                'current_price': current_price,
                'deviation': deviation
            }
        )


class BollingerReversion:
    """
    Bollinger Band Reversion Strategy.
    
    Trades when price hits Bollinger Bands.
    """
    
    def __init__(self, lookback_days: int = 20, std_multiplier: float = 2.0):
        """
        Initialize Bollinger band reversion strategy.
        
        Args:
            lookback_days: Lookback period for moving average
            std_multiplier: Standard deviation multiplier for bands
        """
        self.lookback_days = lookback_days
        self.std_multiplier = std_multiplier
        
        logger.info(f"BollingerReversion initialized with {lookback_days} day lookback")
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> Optional[MeanReversionSignal]:
        """
        Generate Bollinger band reversion signal.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            MeanReversionSignal
        """
        if len(data) < self.lookback_days:
            return None
        
        # Calculate Bollinger Bands
        recent_data = data.tail(self.lookback_days)
        sma = recent_data['close'].mean()
        std = recent_data['close'].std()
        
        upper_band = sma + self.std_multiplier * std
        lower_band = sma - self.std_multiplier * std
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Calculate z-score
        z_score = (current_price - sma) / std if std > 0 else 0
        
        # Generate signal
        if current_price > upper_band:
            # Price above upper band - expect reversion, short
            signal = -1.0
            confidence = min(1.0, (z_score - self.std_multiplier) / 2.0)
        elif current_price < lower_band:
            # Price below lower band - expect reversion, long
            signal = 1.0
            confidence = min(1.0, (self.std_multiplier - z_score) / 2.0)
        else:
            signal = 0.0
            confidence = 0.0
            
        # Apply Indian Transaction Costs filter
        expected_profit = max(0, (abs(z_score) - self.std_multiplier)) * std
        est_cost = estimate_round_trip_costs(current_price)
        if expected_profit < est_cost:
            signal = 0.0
            confidence = 0.0
        
        return MeanReversionSignal(
            symbol=symbol,
            strategy="BollingerReversion",
            signal=signal,
            confidence=confidence,
            z_score=z_score,
            mean=sma,
            std=std,
            timestamp=datetime.now(),
            metadata={
                'upper_band': upper_band,
                'lower_band': lower_band,
                'current_price': current_price
            }
        )


def get_mean_reversion_signals(
    data_dict: Dict[str, pd.DataFrame],
    strategies: List[str] = None
) -> Dict[str, List[MeanReversionSignal]]:
    """
    Generate mean reversion signals using multiple strategies.
    
    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        strategies: List of strategy names to use
        
    Returns:
        Dictionary mapping strategy names to signal lists
    """
    if strategies is None:
        strategies = ["ORB", "VWAPReversion", "BollingerReversion"]
    
    results = {}
    
    # ORB
    if "ORB" in strategies:
        orb = ORBStrategy()
        orb_signals = []
        for symbol, data in data_dict.items():
            signal = orb.generate_signal(data, symbol)
            if signal and signal.signal != 0:
                orb_signals.append(signal)
        results["ORB"] = orb_signals
    
    # VWAP Reversion
    if "VWAPReversion" in strategies:
        vwap = VWAPReversion()
        vwap_signals = []
        for symbol, data in data_dict.items():
            signal = vwap.generate_signal(data, symbol)
            if signal and signal.signal != 0:
                vwap_signals.append(signal)
        results["VWAPReversion"] = vwap_signals
    
    # Bollinger Reversion
    if "BollingerReversion" in strategies:
        bb = BollingerReversion()
        bb_signals = []
        for symbol, data in data_dict.items():
            signal = bb.generate_signal(data, symbol)
            if signal and signal.signal != 0:
                bb_signals.append(signal)
        results["BollingerReversion"] = bb_signals
    
    return results


if __name__ == "__main__":
    # Test mean reversion strategies
    print("Testing Mean Reversion Strategies...")
    
    # Create sample intraday data
    dates = pd.date_range(start='2024-01-01 09:15', periods=400, freq='1min')
    np.random.seed(42)
    
    data_dict = {
        'RELIANCE': pd.DataFrame({
            'open': np.random.uniform(2500, 2510, 400),
            'high': np.random.uniform(2510, 2520, 400),
            'low': np.random.uniform(2490, 2500, 400),
            'close': np.random.uniform(2495, 2515, 400),
            'volume': np.random.randint(100, 1000, 400)
        }, index=dates)
    }
    
    # Generate signals
    signals = get_mean_reversion_signals(data_dict)
    
    print(f"\nGenerated signals:")
    for strategy, signal_list in signals.items():
        print(f"  {strategy}: {len(signal_list)} signals")
        for signal in signal_list[:3]:
            print(f"    {signal.symbol}: {signal.signal:.3f} (confidence: {signal.confidence:.2f})")

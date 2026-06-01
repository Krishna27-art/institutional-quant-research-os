"""
VWAP Trend Alpha Engine
Volume Weighted Average Price trend following for NIFTY futures

Source: Volume Weighted Average Price (VWAP) strategy
Adapted for Indian Markets (NIFTY/BANKNIFTY futures)
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass

from .base import MicrostructureAlpha, AlphaSignal, SignalDirection, AlphaMetrics


@dataclass
class VWAPConfig:
    """Configuration for VWAP strategy"""
    # VWAP calculation
    vwap_period_minutes: int = 60  # 1-hour VWAP
    
    # Signal parameters
    vwap_distance_threshold_bps: float = 10.0  # 10 bps from VWAP
    min_volume_ratio: float = 0.8  # Volume must be 80% of average
    
    # Trend parameters
    trend_lookback_minutes: int = 30  # 30-min trend
    trend_strength_threshold: float = 0.002  # 0.2% trend strength
    
    # Risk parameters
    stop_loss_atr_pct: float = 0.10  # 10% ATR
    trailing_stop_pct: float = 0.005  # 0.5% trailing stop
    
    # Position sizing
    max_position_pct: float = 0.05  # 5% per position
    
    # Slippage (conservative)
    slippage_bps: float = 2.0  # 2 bps for large caps


class VWAPEngine(MicrostructureAlpha):
    """
    VWAP Trend Engine for NIFTY Futures
    
    Strategy:
    1. Calculate VWAP over specified period (default 1-hour)
    2. Enter long when price crosses above VWAP with volume confirmation
    3. Enter short when price crosses below VWAP with volume confirmation
    4. Use trailing stop loss (0.5%)
    5. Exit when price reverts to VWAP mean
    
    Market Condition: Liquidity expansion phases
    Failure Condition: Sideways markets with whipsaws
    """
    
    def __init__(self, config: dict):
        super().__init__("VWAP Trend (NIFTY futures)", config)
        
        # Store VWAP history
        self.vwap_history = {}  # symbol -> list of (timestamp, vwap)
        self.price_history = {}  # symbol -> list of (timestamp, price)
        
        # Track current positions
        self.current_positions = {}  # symbol -> direction
        
    def get_required_features(self) -> List[str]:
        """Return required features"""
        return [
            "vwap",
            "vwap_distance_pct",
            "volume_ratio",
            "atr_14",
            "close",
            "volume",
            "volume_avg_20"
        ]
    
    def calculate_vwap(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        period_minutes: int
    ) -> float:
        """
        Calculate VWAP over specified period.
        
        VWAP = Σ(Price × Volume) / Σ(Volume)
        """
        if len(prices) < period_minutes or len(volumes) < period_minutes:
            return 0.0
        
        # Take last N minutes
        recent_prices = prices[-period_minutes:]
        recent_volumes = volumes[-period_minutes:]
        
        typical_price = (recent_prices + np.zeros_like(recent_prices))  # Use close as typical price
        
        numerator = np.sum(typical_price * recent_volumes)
        denominator = np.sum(recent_volumes)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_vwap_distance(self, price: float, vwap: float) -> float:
        """Calculate percentage distance from VWAP"""
        if vwap == 0:
            return 0.0
        return (price - vwap) / vwap
    
    def detect_trend(
        self,
        symbol: str,
        current_price: float,
        lookback_minutes: int
    ) -> tuple:
        """
        Detect trend direction and strength.
        
        Returns:
            (direction, strength) where direction is 1 (up), -1 (down), 0 (flat)
        """
        if symbol not in self.price_history:
            return 0, 0.0
        
        history = self.price_history[symbol]
        if len(history) < lookback_minutes:
            return 0, 0.0
        
        # Get prices over lookback period
        recent_prices = [p for _, p in history[-lookback_minutes:]]
        
        # Calculate linear regression slope
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        
        if len(y) < 2:
            return 0, 0.0
        
        # Simple slope calculation
        slope = (y[-1] - y[0]) / len(y) if len(y) > 0 else 0
        avg_price = np.mean(y)
        
        # Normalize slope by average price
        strength = slope / avg_price if avg_price != 0 else 0
        
        # Determine direction
        if strength > self.config.get("trend_strength_threshold", 0.002):
            return 1, strength
        elif strength < -self.config.get("trend_strength_threshold", 0.002):
            return -1, abs(strength)
        else:
            return 0, 0.0
    
    def generate_signals(
        self,
        market_data: Dict[str, np.ndarray],
        features: Dict[str, np.ndarray],
        timestamp: datetime
    ) -> List[AlphaSignal]:
        """Generate VWAP trend signals"""
        signals = []
        
        for symbol in market_data:
            data = market_data[symbol]
            feat = features[symbol]
            
            current_price = data.get("close", 0)
            volume = data.get("volume", 0)
            vwap = feat.get("vwap", 0)
            vwap_distance = feat.get("vwap_distance_pct", 0)
            volume_ratio = feat.get("volume_ratio", 0)
            atr = feat.get("atr_14", 0)
            
            if current_price == 0 or vwap == 0 or atr == 0:
                continue
            
            # Update history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append((timestamp, current_price))
            
            # Keep only last 2 hours of history
            max_history = 120
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
            
            # Check volume confirmation
            if volume_ratio < self.config.get("min_volume_ratio", 0.8):
                continue
            
            # Detect trend
            trend_direction, trend_strength = self.detect_trend(
                symbol, current_price, 
                self.config.get("trend_lookback_minutes", 30)
            )
            
            # Generate signals based on VWAP distance and trend
            threshold_bps = self.config.get("vwap_distance_threshold_bps", 10.0)
            threshold_pct = threshold_bps / 10000.0
            
            # Long signal: Price above VWAP + uptrend
            if vwap_distance > threshold_pct and trend_direction == 1:
                # Check if we're not already long
                if self.current_positions.get(symbol) != SignalDirection.LONG:
                    signal = self._create_long_signal(
                        symbol, timestamp, current_price, vwap, 
                        vwap_distance, trend_strength, atr, feat
                    )
                    signals.append(signal)
                    self.current_positions[symbol] = SignalDirection.LONG
            
            # Short signal: Price below VWAP + downtrend
            elif vwap_distance < -threshold_pct and trend_direction == -1:
                # Check if we're not already short
                if self.current_positions.get(symbol) != SignalDirection.SHORT:
                    signal = self._create_short_signal(
                        symbol, timestamp, current_price, vwap,
                        abs(vwap_distance), trend_strength, atr, feat
                    )
                    signals.append(signal)
                    self.current_positions[symbol] = SignalDirection.SHORT
            
            # Exit signal: Price reverts to VWAP
            elif abs(vwap_distance) < threshold_pct * 0.5:
                # Close existing position
                if symbol in self.current_positions:
                    self.current_positions.pop(symbol)
        
        return self.filter_signals(signals)
    
    def _create_long_signal(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float,
        vwap: float,
        vwap_distance: float,
        trend_strength: float,
        atr: float,
        features: Dict[str, np.ndarray]
    ) -> AlphaSignal:
        """Create a long VWAP signal"""
        # Confidence based on distance and trend strength
        confidence = min(0.9, 0.5 + vwap_distance * 100 + trend_strength * 10)
        
        # Expected return based on distance
        expected_return = vwap_distance * 10000  # Convert to bps
        
        return AlphaSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=SignalDirection.LONG,
            confidence=confidence,
            expected_return=expected_return,
            holding_period_minutes=60,  # Average 1 hour
            features={
                "vwap_distance_pct": vwap_distance,
                "trend_strength": trend_strength,
                "atr": atr,
                "volume_ratio": features.get("volume_ratio", 0)
            },
            metadata={
                "entry_price": current_price,
                "vwap": vwap,
                "stop_loss": current_price * (1 - 0.10 * atr / current_price),
                "trailing_stop": current_price * 0.995
            }
        )
    
    def _create_short_signal(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float,
        vwap: float,
        vwap_distance: float,
        trend_strength: float,
        atr: float,
        features: Dict[str, np.ndarray]
    ) -> AlphaSignal:
        """Create a short VWAP signal"""
        confidence = min(0.9, 0.5 + vwap_distance * 100 + trend_strength * 10)
        expected_return = vwap_distance * 10000
        
        return AlphaSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=SignalDirection.SHORT,
            confidence=confidence,
            expected_return=expected_return,
            holding_period_minutes=60,
            features={
                "vwap_distance_pct": -vwap_distance,
                "trend_strength": trend_strength,
                "atr": atr,
                "volume_ratio": features.get("volume_ratio", 0)
            },
            metadata={
                "entry_price": current_price,
                "vwap": vwap,
                "stop_loss": current_price * (1 + 0.10 * atr / current_price),
                "trailing_stop": current_price * 1.005
            }
        )
    
    def get_metrics(self) -> AlphaMetrics:
        """Return performance metrics"""
        return AlphaMetrics(
            total_trades=0,
            win_rate=0.35,
            profit_factor=1.3,
            sharpe_ratio=0.9,
            max_drawdown=0.12,
            avg_holding_period_minutes=60,
            capacity_cr=500,
            decay_months=12
        )
    
    def reset_daily(self) -> None:
        """Reset daily state"""
        self.vwap_history.clear()
        self.price_history.clear()
        self.current_positions.clear()

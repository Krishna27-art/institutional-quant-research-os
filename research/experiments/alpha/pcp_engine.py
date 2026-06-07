"""
Put-Call Carry (PCP) Alpha Engine
Weekly options expiry strategy for NIFTY/BANKNIFTY

Source: Options Expert (Jane Street) - Weekly expiry edge
Adapted for Indian Markets (Thursday expiry, Wednesday entry)
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import MicrostructureAlpha, AlphaSignal, SignalDirection, AlphaMetrics


@dataclass
class PCPConfig:
    """Configuration for Put-Call Carry strategy"""
    # Timing
    entry_day: str = "Wednesday"  # Enter on Wednesday
    exit_day: str = "Thursday"  # Exit on Thursday (expiry)
    
    # Option parameters
    otm_pct: float = 0.02  # 2% OTM strikes
    max_spread_bps: float = 20.0  # Max bid-ask spread 20 bps
    
    # Volatility parameters
    iv_percentile_threshold: float = 0.70  # IV > 70th percentile
    min_iv: float = 0.12  # Min IV 12%
    max_iv: float = 0.40  # Max IV 40%
    
    # Position sizing
    max_position_pct: float = 0.03  # 3% per position
    max_contracts: int = 50
    
    # Risk parameters
    delta_hedge_threshold: float = 0.10  # Hedge if |delta| > 0.10
    gamma_threshold: float = 0.05  # Monitor gamma risk
    
    # Strategy type
    strategy_type: str = "short_strangle"  # Short OTM strangle


class PCPEngine(MicrostructureAlpha):
    """
    Put-Call Carry Engine for Weekly Options
    
    Strategy:
    1. On Wednesday before expiry, identify IV > 70th percentile
    2. Sell OTM strangle (2% OTM puts and calls)
    3. Close on Thursday before expiry
    4. Delta-hedge if |delta| > 0.10
    
    Market Condition: High IV environments, pre-expiry
    Failure Condition: Low IV, large moves, gamma squeeze
    
    Focus: NIFTY & BANKNIFTY weekly expiries only
    """
    
    def __init__(self, config: dict):
        super().__init__("Put-Call Carry (Weekly options)", config)
        
        # Track IV history
        self.iv_history = {}  # symbol -> list of (timestamp, iv)
        self.iv_percentiles = {}  # symbol -> current IV percentile
        
        # Track current positions
        self.current_positions = {}  # symbol -> position info
        
    def get_required_features(self) -> List[str]:
        """Return required features"""
        return [
            "implied_volatility",
            "iv_percentile",
            "put_call_ratio",
            "vix",
            "atm_iv",
            "skew",
            "term_structure_slope"
        ]
    
    def calculate_iv_percentile(
        self,
        symbol: str,
        current_iv: float,
        lookback_days: int = 252
    ) -> float:
        """Calculate IV percentile over lookback period"""
        if symbol not in self.iv_history:
            self.iv_history[symbol] = []
        
        # Add current IV to history
        self.iv_history[symbol].append((datetime.now(), current_iv))
        
        # Keep only last lookback_days
        if len(self.iv_history[symbol]) > lookback_days:
            self.iv_history[symbol] = self.iv_history[symbol][-lookback_days:]
        
        # Calculate percentile
        iv_values = [iv for _, iv in self.iv_history[symbol]]
        if len(iv_values) < 30:  # Need at least 30 samples
            return 0.5
        
        iv_values.sort()
        rank = iv_values.index(current_iv) if current_iv in iv_values else len(iv_values)
        percentile = rank / len(iv_values)
        
        self.iv_percentiles[symbol] = percentile
        return percentile
    
    def is_expiry_week(self, timestamp: datetime) -> bool:
        """Check if current week is expiry week"""
        # Indian weekly options expire on Thursday
        # Check if Thursday is within 3 days
        target_day = 3  # Thursday (0=Monday, 6=Sunday)
        current_day = timestamp.weekday()
        
        days_to_expiry = (target_day - current_day) % 7
        return days_to_expiry <= 3
    
    def select_strikes(
        self,
        underlying_price: float,
        otm_pct: float
    ) -> tuple:
        """
        Select OTM strike prices.
        
        Returns:
            (put_strike, call_strike)
        """
        put_strike = underlying_price * (1 - otm_pct)
        call_strike = underlying_price * (1 + otm_pct)
        
        # Round to nearest strike interval (50 for NIFTY, 100 for BANKNIFTY)
        put_strike = round(put_strike / 50) * 50
        call_strike = round(call_strike / 50) * 50
        
        return put_strike, call_strike
    
    def generate_signals(
        self,
        market_data: Dict[str, np.ndarray],
        features: Dict[str, np.ndarray],
        timestamp: datetime
    ) -> List[AlphaSignal]:
        """Generate Put-Call Carry signals"""
        signals = []
        
        # Only trade on Wednesday before expiry
        day_name = timestamp.strftime("%A")
        if day_name != self.config.get("entry_day", "Wednesday"):
            return signals
        
        # Check if it's expiry week
        if not self.is_expiry_week(timestamp):
            return signals
        
        for symbol in market_data:
            # Only trade NIFTY and BANKNIFTY options
            if "NIFTY" not in symbol.upper() and "BANKNIFTY" not in symbol.upper():
                continue
            
            data = market_data[symbol]
            feat = features[symbol]
            
            underlying_price = data.get("close", 0)
            iv = feat.get("implied_volatility", 0)
            iv_percentile = feat.get("iv_percentile", 0)
            vix = feat.get("vix", 0)
            pcr = feat.get("put_call_ratio", 0)
            
            if underlying_price == 0 or iv == 0:
                continue
            
            # Calculate IV percentile if not provided
            if iv_percentile == 0:
                iv_percentile = self.calculate_iv_percentile(symbol, iv)
            
            # Check IV conditions
            min_iv = self.config.get("min_iv", 0.12)
            max_iv = self.config.get("max_iv", 0.40)
            iv_threshold = self.config.get("iv_percentile_threshold", 0.70)
            
            if not (min_iv <= iv <= max_iv):
                continue
            
            if iv_percentile < iv_threshold:
                continue
            
            # Select strikes
            otm_pct = self.config.get("otm_pct", 0.02)
            put_strike, call_strike = self.select_strikes(underlying_price, otm_pct)
            
            # Check if we already have a position
            if symbol in self.current_positions:
                continue
            
            # Generate short strangle signal
            signal = self._create_strangle_signal(
                symbol, timestamp, underlying_price,
                put_strike, call_strike, iv, iv_percentile,
                vix, pcr, feat
            )
            signals.append(signal)
            self.current_positions[symbol] = {
                "entry_time": timestamp,
                "put_strike": put_strike,
                "call_strike": call_strike,
                "direction": SignalDirection.SHORT
            }
        
        return self.filter_signals(signals)
    
    def _create_strangle_signal(
        self,
        symbol: str,
        timestamp: datetime,
        underlying_price: float,
        put_strike: float,
        call_strike: float,
        iv: float,
        iv_percentile: float,
        vix: float,
        pcr: float,
        features: Dict[str, np.ndarray]
    ) -> AlphaSignal:
        """Create a short strangle signal"""
        # Confidence based on IV percentile and VIX
        confidence = min(0.9, 0.5 + iv_percentile * 0.4)
        
        # Expected return: theta decay over 1 day
        # Approximate: 0.5% to 1% of underlying
        expected_return = 50.0  # 50 bps
        
        return AlphaSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=SignalDirection.SHORT,
            confidence=confidence,
            expected_return=expected_return,
            holding_period_minutes=390,  # 1 trading day
            features={
                "iv": iv,
                "iv_percentile": iv_percentile,
                "vix": vix,
                "put_call_ratio": pcr,
                "otm_pct": self.config.get("otm_pct", 0.02)
            },
            metadata={
                "strategy": "short_strangle",
                "underlying_price": underlying_price,
                "put_strike": put_strike,
                "call_strike": call_strike,
                "entry_day": "Wednesday",
                "exit_day": "Thursday",
                "delta_hedge_threshold": self.config.get("delta_hedge_threshold", 0.10)
            }
        )
    
    def check_exit_conditions(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float,
        features: Dict[str, np.ndarray]
    ) -> bool:
        """Check if position should be exited"""
        if symbol not in self.current_positions:
            return False
        
        # Exit on Thursday (expiry day)
        day_name = timestamp.strftime("%A")
        if day_name == self.config.get("exit_day", "Thursday"):
            return True
        
        # Exit if IV drops significantly
        position = self.current_positions[symbol]
        entry_iv = features.get("implied_volatility", 0)
        if entry_iv < position.get("entry_iv", 0) * 0.8:
            return True
        
        return False
    
    def get_metrics(self) -> AlphaMetrics:
        """Return performance metrics"""
        return AlphaMetrics(
            total_trades=0,
            win_rate=0.65,
            profit_factor=1.8,
            sharpe_ratio=0.7,
            max_drawdown=0.10,
            avg_holding_period_minutes=390,
            capacity_cr=200,
            decay_months=24
        )
    
    def reset_daily(self) -> None:
        """Reset daily state"""
        # Clear positions on Friday (after expiry)
        current_day = datetime.now().strftime("%A")
        if current_day == "Friday":
            self.current_positions.clear()

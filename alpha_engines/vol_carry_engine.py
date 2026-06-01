"""
Volatility Carry Alpha Engine
Short straddle with delta hedging for NIFTY options

Source: Volatility risk premium capture
Adapted for Indian Markets (NIFTY/BANKNIFTY)
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base import MicrostructureAlpha, AlphaSignal, SignalDirection, AlphaMetrics


@dataclass
class VolCarryConfig:
    """Configuration for Volatility Carry strategy"""
    # Option parameters
    days_to_expiry: int = 5  # Target 5 DTE
    strike_selection: str = "atm"  # ATM or slightly OTM
    
    # Volatility parameters
    iv_rv_threshold: float = 0.10  # IV - RV > 10%
    min_iv: float = 0.15  # Min IV 15%
    max_iv: float = 0.35  # Max IV 35%
    
    # Delta hedging
    delta_hedge_threshold: float = 0.15  # Hedge if |delta| > 0.15
    hedge_frequency_minutes: int = 30  # Re-hedge every 30 min
    
    # Position sizing
    max_position_pct: float = 0.02  # 2% per position
    max_contracts: int = 30
    vega_limit_pct: float = 0.01  # 1% of AUM vega exposure
    
    # Risk parameters
    gamma_threshold: float = 0.10  # Monitor gamma risk
    theta_target_daily: float = 0.001  # Target 0.1% daily theta
    
    # Strategy type
    strategy_type: str = "short_straddle"  # Short ATM straddle


class VolCarryEngine(MicrostructureAlpha):
    """
    Volatility Carry Engine for NIFTY Options
    
    Strategy:
    1. Identify periods where IV > realized vol by >10%
    2. Sell ATM straddle on NIFTY with 5 DTE
    3. Delta-hedge when |delta| > 0.15
    4. Hold until expiry or 1 DTE
    
    Market Condition: Elevated IV vs realized vol
    Failure Condition: IV spikes, gamma squeeze, large moves
    
    Focus: NIFTY & BANKNIFTY liquid options
    """
    
    def __init__(self, config: dict):
        super().__init__("Volatility Carry (Short straddle)", config)
        
        # Track vol history
        self.iv_history = {}  # symbol -> list of (timestamp, iv)
        self.rv_history = {}  # symbol -> list of (timestamp, rv)
        
        # Track current positions and hedges
        self.current_positions = {}  # symbol -> position info
        self.last_hedge_time = {}  # symbol -> last hedge timestamp
        
    def get_required_features(self) -> List[str]:
        """Return required features"""
        return [
            "implied_volatility",
            "realized_volatility_5d",
            "realized_volatility_20d",
            "vix",
            "atm_iv",
            "delta",
            "gamma",
            "vega",
            "theta"
        ]
    
    def calculate_realized_vol(
        self,
        returns: np.ndarray,
        annualization_factor: int = 252
    ) -> float:
        """Calculate realized volatility from returns"""
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(annualization_factor)
    
    def calculate_vol_risk_premium(
        self,
        iv: float,
        rv: float
    ) -> float:
        """Calculate volatility risk premium (IV - RV)"""
        return iv - rv
    
    def select_strike(
        self,
        underlying_price: float,
        selection: str = "atm"
    ) -> float:
        """Select strike price"""
        if selection == "atm":
            # Round to nearest strike
            return round(underlying_price / 50) * 50
        elif selection == "slightly_otm":
            # 1% OTM
            otm_price = underlying_price * 1.01
            return round(otm_price / 50) * 50
        else:
            return underlying_price
    
    def should_hedge(
        self,
        symbol: str,
        current_delta: float,
        timestamp: datetime
    ) -> bool:
        """Check if position should be delta-hedged"""
        threshold = self.config.get("delta_hedge_threshold", 0.15)
        
        # Check delta threshold
        if abs(current_delta) > threshold:
            return True
        
        # Check hedge frequency
        if symbol in self.last_hedge_time:
            time_since_hedge = (timestamp - self.last_hedge_time[symbol]).total_seconds()
            freq_seconds = self.config.get("hedge_frequency_minutes", 30) * 60
            if time_since_hedge >= freq_seconds:
                return True
        
        return False
    
    def generate_signals(
        self,
        market_data: Dict[str, np.ndarray],
        features: Dict[str, np.ndarray],
        timestamp: datetime
    ) -> List[AlphaSignal]:
        """Generate Volatility Carry signals"""
        signals = []
        
        for symbol in market_data:
            # Only trade NIFTY and BANKNIFTY options
            if "NIFTY" not in symbol.upper() and "BANKNIFTY" not in symbol.upper():
                continue
            
            data = market_data[symbol]
            feat = features[symbol]
            
            underlying_price = data.get("close", 0)
            iv = feat.get("implied_volatility", 0)
            rv_5d = feat.get("realized_volatility_5d", 0)
            rv_20d = feat.get("realized_volatility_20d", 0)
            vix = feat.get("vix", 0)
            delta = feat.get("delta", 0)
            
            if underlying_price == 0 or iv == 0:
                continue
            
            # Use 5-day RV for comparison
            rv = rv_5d if rv_5d > 0 else rv_20d
            
            # Calculate vol risk premium
            vol_premium = self.calculate_vol_risk_premium(iv, rv)
            
            # Check conditions
            min_iv = self.config.get("min_iv", 0.15)
            max_iv = self.config.get("max_iv", 0.35)
            premium_threshold = self.config.get("iv_rv_threshold", 0.10)
            
            if not (min_iv <= iv <= max_iv):
                continue
            
            if vol_premium < premium_threshold:
                continue
            
            # Check if we already have a position
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                
                # Check if we need to hedge
                if self.should_hedge(symbol, delta, timestamp):
                    # Generate hedge signal
                    hedge_signal = self._create_hedge_signal(
                        symbol, timestamp, delta, feat
                    )
                    signals.append(hedge_signal)
                    self.last_hedge_time[symbol] = timestamp
                
                continue
            
            # Select strike
            selection = self.config.get("strike_selection", "atm")
            strike = self.select_strike(underlying_price, selection)
            
            # Generate short straddle signal
            signal = self._create_straddle_signal(
                symbol, timestamp, underlying_price, strike,
                iv, rv, vol_premium, vix, feat
            )
            signals.append(signal)
            
            self.current_positions[symbol] = {
                "entry_time": timestamp,
                "strike": strike,
                "entry_iv": iv,
                "entry_rv": rv,
                "direction": SignalDirection.SHORT
            }
            self.last_hedge_time[symbol] = timestamp
        
        return self.filter_signals(signals)
    
    def _create_straddle_signal(
        self,
        symbol: str,
        timestamp: datetime,
        underlying_price: float,
        strike: float,
        iv: float,
        rv: float,
        vol_premium: float,
        vix: float,
        features: Dict[str, np.ndarray]
    ) -> AlphaSignal:
        """Create a short straddle signal"""
        # Confidence based on vol premium
        confidence = min(0.9, 0.5 + vol_premium * 2.0)
        
        # Expected return: theta decay over 5 days
        # Approximate: 0.2% to 0.5% per day
        expected_return = 100.0  # 100 bps over holding period
        
        return AlphaSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=SignalDirection.SHORT,
            confidence=confidence,
            expected_return=expected_return,
            holding_period_minutes=5 * 390,  # 5 trading days
            features={
                "iv": iv,
                "rv": rv,
                "vol_premium": vol_premium,
                "vix": vix,
                "strike": strike
            },
            metadata={
                "strategy": "short_straddle",
                "underlying_price": underlying_price,
                "strike": strike,
                "days_to_expiry": self.config.get("days_to_expiry", 5),
                "delta_hedge_threshold": self.config.get("delta_hedge_threshold", 0.15),
                "hedge_frequency": self.config.get("hedge_frequency_minutes", 30)
            }
        )
    
    def _create_hedge_signal(
        self,
        symbol: str,
        timestamp: datetime,
        delta: float,
        features: Dict[str, np.ndarray]
    ) -> AlphaSignal:
        """Create a delta hedge signal"""
        # Hedge direction is opposite to delta
        hedge_direction = SignalDirection.SHORT if delta > 0 else SignalDirection.LONG
        
        return AlphaSignal(
            symbol=symbol,
            timestamp=timestamp,
            direction=hedge_direction,
            confidence=0.8,
            expected_return=0.0,  # Hedge is for risk management
            holding_period_minutes=30,  # Short-term hedge
            features={
                "current_delta": delta,
                "hedge_type": "delta_hedge"
            },
            metadata={
                "strategy": "delta_hedge",
                "hedge_size": abs(delta),
                "original_position": self.current_positions[symbol]
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
        
        position = self.current_positions[symbol]
        entry_time = position["entry_time"]
        
        # Exit at 1 DTE
        days_held = (timestamp - entry_time).days
        if days_held >= self.config.get("days_to_expiry", 5) - 1:
            return True
        
        # Exit if IV spikes (gamma risk)
        current_iv = features.get("implied_volatility", 0)
        if current_iv > position["entry_iv"] * 1.5:
            return True
        
        return False
    
    def get_metrics(self) -> AlphaMetrics:
        """Return performance metrics"""
        return AlphaMetrics(
            total_trades=0,
            win_rate=0.60,
            profit_factor=1.5,
            sharpe_ratio=0.6,
            max_drawdown=0.15,
            avg_holding_period_minutes=5 * 390,
            capacity_cr=150,
            decay_months=18
        )
    
    def reset_daily(self) -> None:
        """Reset daily state"""
        # Clear positions that have reached expiry
        current_time = datetime.now()
        to_remove = []
        
        for symbol, position in self.current_positions.items():
            days_held = (current_time - position["entry_time"]).days
            if days_held >= self.config.get("days_to_expiry", 5):
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del self.current_positions[symbol]
            if symbol in self.last_hedge_time:
                del self.last_hedge_time[symbol]

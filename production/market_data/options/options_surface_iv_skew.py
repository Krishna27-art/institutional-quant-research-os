"""
Options Surface and IV Skew History
Based on the critique: Build data moat with options data

Critical for institutional edge:
- Options surface (volatility smile/skew)
- Implied volatility (IV) history
- IV skew analysis
- Put-call ratio
- Options positioning
- Volatility regime detection

Data Sources:
- NSE options chain
- Third-party providers (NSE India, Bloomberg, Refinitiv)
- Historical IV data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OptionType(Enum):
    """Type of option."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionData:
    """Data for a single option contract."""
    timestamp: datetime
    symbol: str
    option_type: OptionType
    strike: float
    expiry: datetime
    spot_price: float
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid price of option."""
        return (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else self.last_price
    
    @property
    def moneyness(self) -> float:
        """Moneyness (strike / spot)."""
        return self.strike / self.spot_price if self.spot_price > 0 else 1.0


@dataclass
class IVSkew:
    """IV skew at a point in time."""
    timestamp: datetime
    symbol: str
    spot_price: float
    atm_iv: float  # At-the-money IV
    otm_call_iv: float  # Out-of-the-money call IV
    otm_put_iv: float  # Out-of-the-money put IV
    skew_value: float  # otm_put_iv - otm_call_iv
    put_call_ratio: float  # put_volume / call_volume
    total_call_volume: int
    total_put_volume: int
    total_call_oi: int
    total_put_oi: int


@dataclass
class VolatilitySurface:
    """Volatility surface at a point in time."""
    timestamp: datetime
    symbol: str
    strikes: List[float]
    call_ivs: List[float]
    put_ivs: List[float]
    expiries: List[datetime]
    
    def get_iv_at_moneyness(self, moneyness: float, option_type: OptionType) -> Optional[float]:
        """Get IV at a specific moneyness."""
        ivs = self.call_ivs if option_type == OptionType.CALL else self.put_ivs
        moneynesses = [s / self.strikes[0] for s in self.strikes]  # Approximate
        
        # Interpolate
        if len(ivs) < 2:
            return None
        
        return np.interp(moneyness, moneynesses, ivs)


class OptionsDataManager:
    """
    Manager for options surface and IV skew data.
    
    Features:
    - Options chain tracking
    - IV surface reconstruction
    - IV skew analysis
    - Put-call ratio calculation
    - Options positioning analysis
    - Volatility regime detection
    """
    
    def __init__(self):
        self.option_data: Dict[str, List[OptionData]] = {}  # symbol -> options
        self.iv_skew_history: Dict[str, List[IVSkew]] = {}  # symbol -> skew history
        self.volatility_surfaces: Dict[str, List[VolatilitySurface]] = {}  # symbol -> surfaces
        
        # Configuration
        self.skew_window_days = 20  # For calculating skew averages
        self.iv_percentile_window = 252  # For IV percentiles (1 year)
    
    def add_option_data(self, option: OptionData) -> None:
        """Add option data."""
        symbol = option.symbol
        
        if symbol not in self.option_data:
            self.option_data[symbol] = []
        
        self.option_data[symbol].append(option)
    
    def calculate_iv_skew(
        self,
        symbol: str,
        timestamp: datetime,
        spot_price: float,
        options: List[OptionData]
    ) -> Optional[IVSkew]:
        """
        Calculate IV skew from options data.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp of calculation
            spot_price: Current spot price
            options: List of option data
            
        Returns:
            IVSkew or None if insufficient data
        """
        # Separate calls and puts
        calls = [o for o in options if o.option_type == OptionType.CALL]
        puts = [o for o in options if o.option_type == OptionType.PUT]
        
        if not calls or not puts:
            return None
        
        # Find ATM options (closest to spot)
        atm_call = min(calls, key=lambda o: abs(o.strike - spot_price))
        atm_put = min(puts, key=lambda o: abs(o.strike - spot_price))
        
        # Find OTM options (5% OTM)
        otm_call_strike = spot_price * 1.05
        otm_put_strike = spot_price * 0.95
        
        otm_call = min(
            [o for o in calls if o.strike >= otm_call_strike],
            key=lambda o: abs(o.strike - otm_call_strike),
            default=None
        )
        otm_put = min(
            [o for o in puts if o.strike <= otm_put_strike],
            key=lambda o: abs(o.strike - otm_put_strike),
            default=None
        )
        
        # Calculate volumes
        total_call_volume = sum(o.volume for o in calls)
        total_put_volume = sum(o.volume for o in puts)
        total_call_oi = sum(o.open_interest for o in calls)
        total_put_oi = sum(o.open_interest for o in puts)
        
        # Put-call ratio
        put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        
        # Skew value
        otm_call_iv = otm_call.implied_volatility if otm_call else atm_call.implied_volatility
        otm_put_iv = otm_put.implied_volatility if otm_put else atm_put.implied_volatility
        skew_value = otm_put_iv - otm_call_iv
        
        skew = IVSkew(
            timestamp=timestamp,
            symbol=symbol,
            spot_price=spot_price,
            atm_iv=atm_call.implied_volatility,
            otm_call_iv=otm_call_iv,
            otm_put_iv=otm_put_iv,
            skew_value=skew_value,
            put_call_ratio=put_call_ratio,
            total_call_volume=total_call_volume,
            total_put_volume=total_put_volume,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi
        )
        
        # Store skew history
        if symbol not in self.iv_skew_history:
            self.iv_skew_history[symbol] = []
        self.iv_skew_history[symbol].append(skew)
        
        return skew
    
    def build_volatility_surface(
        self,
        symbol: str,
        timestamp: datetime,
        options: List[OptionData]
    ) -> Optional[VolatilitySurface]:
        """
        Build volatility surface from options data.
        
        Args:
            symbol: Trading symbol
            timestamp: Timestamp of surface
            options: List of option data
            
        Returns:
            VolatilitySurface or None
        """
        # Group by expiry
        expiries = list(set(o.expiry for o in options))
        expiries.sort()
        
        if not expiries:
            return None
        
        # Use nearest expiry
        nearest_expiry = min(expiries, key=lambda e: abs((e - timestamp).days))
        expiry_options = [o for o in options if o.expiry == nearest_expiry]
        
        # Group by strike
        strikes = sorted(list(set(o.strike for o in expiry_options)))
        
        if not strikes:
            return None
        
        # Get IVs for each strike
        call_ivs = []
        put_ivs = []
        
        for strike in strikes:
            call_opts = [o for o in expiry_options if o.strike == strike and o.option_type == OptionType.CALL]
            put_opts = [o for o in expiry_options if o.strike == strike and o.option_type == OptionType.PUT]
            
            call_iv = np.mean([o.implied_volatility for o in call_opts]) if call_opts else np.nan
            put_iv = np.mean([o.implied_volatility for o in put_opts]) if put_opts else np.nan
            
            call_ivs.append(call_iv)
            put_ivs.append(put_iv)
        
        surface = VolatilitySurface(
            timestamp=timestamp,
            symbol=symbol,
            strikes=strikes,
            call_ivs=call_ivs,
            put_ivs=put_ivs,
            expiries=[nearest_expiry]
        )
        
        # Store surface
        if symbol not in self.volatility_surfaces:
            self.volatility_surfaces[symbol] = []
        self.volatility_surfaces[symbol].append(surface)
        
        return surface
    
    def get_iv_percentile(
        self,
        symbol: str,
        current_iv: float,
        days: int = 252
    ) -> float:
        """
        Get IV percentile relative to historical IV.
        
        Args:
            symbol: Trading symbol
            current_iv: Current IV value
            days: Number of days of history
            
        Returns:
            IV percentile (0 to 1)
        """
        if symbol not in self.iv_skew_history:
            return 0.5
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        historical_ivs = [
            skew.atm_iv for skew in self.iv_skew_history[symbol]
            if start_date <= skew.timestamp <= end_date
        ]
        
        if not historical_ivs:
            return 0.5
        
        # Calculate percentile
        percentile = len([iv for iv in historical_ivs if iv < current_iv]) / len(historical_ivs)
        
        return percentile
    
    def analyze_skew_regime(
        self,
        symbol: str,
        days: int = 20
    ) -> Dict[str, any]:
        """
        Analyze skew regime.
        
        Args:
            symbol: Trading symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with regime analysis
        """
        if symbol not in self.iv_skew_history:
            return {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        skews = [
            skew for skew in self.iv_skew_history[symbol]
            if start_date <= skew.timestamp <= end_date
        ]
        
        if not skews:
            return {}
        
        skew_values = [s.skew_value for s in skews]
        put_call_ratios = [s.put_call_ratio for s in skews]
        
        avg_skew = np.mean(skew_values)
        avg_pcr = np.mean(put_call_ratios)
        
        # Regime classification
        if avg_skew > 0.05 and avg_pcr > 1.2:
            regime = "Fear / Hedging"
        elif avg_skew < -0.02 and avg_pcr < 0.8:
            regime = "Complacency / Speculation"
        elif abs(avg_skew) < 0.02:
            regime = "Balanced"
        else:
            regime = "Neutral"
        
        return {
            'avg_skew': avg_skew,
            'avg_put_call_ratio': avg_pcr,
            'skew_trend': np.polyfit(range(len(skew_values)), skew_values, 1)[0],
            'regime': regime,
            'latest_skew': skew_values[-1] if skew_values else 0,
            'latest_pcr': put_call_ratios[-1] if put_call_ratios else 0
        }
    
    def get_options_positioning(
        self,
        symbol: str
    ) -> Dict[str, float]:
        """
        Get options positioning metrics.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with positioning metrics
        """
        if symbol not in self.option_data:
            return {}
        
        # Get latest options
        latest_timestamp = max(o.timestamp for o in self.option_data[symbol])
        latest_options = [o for o in self.option_data[symbol] if o.timestamp == latest_timestamp]
        
        if not latest_options:
            return {}
        
        # Calculate positioning
        calls = [o for o in latest_options if o.option_type == OptionType.CALL]
        puts = [o for o in latest_options if o.option_type == OptionType.PUT]
        
        total_call_oi = sum(o.open_interest for o in calls)
        total_put_oi = sum(o.open_interest for o in puts)
        total_oi = total_call_oi + total_put_oi
        
        # Positioning bias
        call_bias = total_call_oi / total_oi if total_oi > 0 else 0.5
        put_bias = total_put_oi / total_oi if total_oi > 0 else 0.5
        
        return {
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'call_bias': call_bias,
            'put_bias': put_bias,
            'positioning': 'bullish' if call_bias > 0.6 else 'bearish' if put_bias > 0.6 else 'neutral'
        }


if __name__ == "__main__":
    # Test the Options Data Manager
    print("Testing Options Surface and IV Skew Manager...")
    
    manager = OptionsDataManager()
    
    # Generate sample options data
    print("\nGenerating sample options data...")
    base_time = datetime.now()
    spot_price = 2500
    
    for i in range(30):
        for strike in np.linspace(2300, 2700, 9):
            # Call option
            call = OptionData(
                timestamp=base_time - timedelta(days=i),
                symbol="RELIANCE",
                option_type=OptionType.CALL,
                strike=strike,
                expiry=base_time + timedelta(days=30),
                spot_price=spot_price,
                bid=max(0, strike - spot_price + 50),
                ask=max(0, strike - spot_price + 60),
                last_price=max(0, strike - spot_price + 55),
                volume=np.random.randint(100, 1000),
                open_interest=np.random.randint(1000, 10000),
                implied_volatility=np.random.uniform(0.15, 0.30)
            )
            manager.add_option_data(call)
            
            # Put option
            put = OptionData(
                timestamp=base_time - timedelta(days=i),
                symbol="RELIANCE",
                option_type=OptionType.PUT,
                strike=strike,
                expiry=base_time + timedelta(days=30),
                spot_price=spot_price,
                bid=max(0, spot_price - strike + 50),
                ask=max(0, spot_price - strike + 60),
                last_price=max(0, spot_price - strike + 55),
                volume=np.random.randint(100, 1000),
                open_interest=np.random.randint(1000, 10000),
                implied_volatility=np.random.uniform(0.15, 0.30)
            )
            manager.add_option_data(put)
    
    print(f"Added options data for {len(manager.option_data)} symbols")
    
    # Calculate IV skew
    print("\nCalculating IV skew...")
    latest_options = [o for o in manager.option_data['RELIANCE'] if o.timestamp == base_time]
    skew = manager.calculate_iv_skew('RELIANCE', base_time, spot_price, latest_options)
    
    if skew:
        print(f"ATM IV: {skew.atm_iv:.2%}")
        print(f"OTM Call IV: {skew.otm_call_iv:.2%}")
        print(f"OTM Put IV: {skew.otm_put_iv:.2%}")
        print(f"Skew Value: {skew.skew_value:.4f}")
        print(f"Put-Call Ratio: {skew.put_call_ratio:.2f}")
    
    # Build volatility surface
    print("\nBuilding volatility surface...")
    surface = manager.build_volatility_surface('RELIANCE', base_time, latest_options)
    
    if surface:
        print(f"Strikes: {len(surface.strikes)}")
        print(f"Expiries: {len(surface.expiries)}")
        print(f"Call IVs: {len(surface.call_ivs)}")
        print(f"Put IVs: {len(surface.put_ivs)}")
    
    # Get IV percentile
    print("\nGetting IV percentile...")
    current_iv = 0.25
    percentile = manager.get_iv_percentile('RELIANCE', current_iv)
    print(f"IV Percentile: {percentile:.2%}")
    
    # Analyze skew regime
    print("\nAnalyzing skew regime...")
    regime = manager.analyze_skew_regime('RELIANCE', days=20)
    for key, value in regime.items():
        print(f"  {key}: {value}")
    
    # Get options positioning
    print("\nGetting options positioning...")
    positioning = manager.get_options_positioning('RELIANCE')
    for key, value in positioning.items():
        print(f"  {key}: {value}")

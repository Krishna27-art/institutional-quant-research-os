"""
IV Surface and Black-Scholes Derivatives
Based on the critique: Use Black-Scholes for IV Surface, IV Rank, IV Percentile, Volatility Risk Premium, Gamma Exposure, Dealer Positioning

This is not just for option pricing - it's for understanding market structure and dealer positioning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import norm
from scipy.optimize import minimize_scalar


@dataclass
class IVSurfacePoint:
    """Point on IV surface."""
    timestamp: datetime
    symbol: str
    strike: float
    expiry: datetime
    days_to_expiry: float
    moneyness: float
    iv_call: float
    iv_put: float
    iv_skew: float
    iv_smile: float


@dataclass
class GammaExposure:
    """Gamma exposure analysis."""
    timestamp: datetime
    symbol: str
    total_gamma: float
    call_gamma: float
    put_gamma: float
    gamma_pnl: float
    dealer_gamma: float
    is_gamma_squeeze: bool


@dataclass
class DealerFlow:
    """Dealer flow analysis."""
    timestamp: datetime
    symbol: str
    dealer_position: str  # "long_gamma" or "short_gamma"
    hedging_flow: float
    delta_hedge_ratio: float
    is_dealer_selling: bool


@dataclass
class VolatilityRiskPremium:
    """Volatility risk premium."""
    timestamp: datetime
    symbol: str
    realized_vol: float
    implied_vol: float
    vr_premium: float
    is_vol_overpriced: bool


class BlackScholesModel:
    """Black-Scholes model for option pricing and Greeks."""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter."""
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate call option price."""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate put option price."""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='call'):
        """Calculate delta."""
        if T <= 0:
            return 1.0 if option_type == 'call' and S > K else 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate gamma."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate vega."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='call'):
        """Calculate theta."""
        if T <= 0:
            return 0.0
        d1 = BlackScholesModel.d1(S, K, T, r, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return theta / 365
    
    @staticmethod
    def implied_volatility(S, K, T, r, market_price, option_type='call'):
        """Calculate implied volatility using Newton-Raphson."""
        def objective(sigma):
            if option_type == 'call':
                price = BlackScholesModel.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholesModel.put_price(S, K, T, r, sigma)
            return price - market_price
        
        result = minimize_scalar(lambda x: objective(x) ** 2, bounds=(0.01, 5.0), method='bounded')
        return result.x


class IVSurfaceManager:
    """
    IV Surface Manager for understanding market structure.
    
    Features:
    - IV Surface reconstruction
    - IV Rank and IV Percentile
    - Volatility Risk Premium
    - Gamma Exposure
    - Dealer Positioning
    """
    
    def __init__(self):
        self.iv_surface_points: List[IVSurfacePoint] = []
        self.gamma_exposures: List[GammaExposure] = []
        self.dealer_flows: List[DealerFlow] = []
        self.vrp_history: List[VolatilityRiskPremium] = []
        
        self.bs_model = BlackScholesModel()
        self.risk_free_rate = 0.06  # 6% risk-free rate
    
    def build_iv_surface(
        self,
        symbol: str,
        spot_price: float,
        strikes: List[float],
        expiries: List[datetime],
        option_prices: Dict[Tuple[str, float, datetime], float]
    ) -> List[IVSurfacePoint]:
        """
        Build IV surface from option prices.
        
        Args:
            symbol: Trading symbol
            spot_price: Current spot price
            strikes: List of strike prices
            expiries: List of expiry dates
            option_prices: Dictionary mapping (option_type, strike, expiry) to price
            
        Returns:
            List of IV surface points
        """
        surface_points = []
        
        for expiry in expiries:
            T = (expiry - datetime.now()).days / 365.0
            
            for strike in strikes:
                # Get call and put prices
                call_price = option_prices.get(('call', strike, expiry), 0)
                put_price = option_prices.get(('put', strike, expiry), 0)
                
                # Calculate IVs
                iv_call = self.bs_model.implied_volatility(spot_price, strike, T, self.risk_free_rate, call_price, 'call') if call_price > 0 else 0
                iv_put = self.bs_model.implied_volatility(spot_price, strike, T, self.risk_free_rate, put_price, 'put') if put_price > 0 else 0
                
                # Calculate moneyness
                moneyness = strike / spot_price
                
                # Calculate skew and smile
                iv_skew = iv_put - iv_call
                iv_smile = (iv_call + iv_put) / 2
                
                point = IVSurfacePoint(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    days_to_expiry=T * 365,
                    moneyness=moneyness,
                    iv_call=iv_call,
                    iv_put=iv_put,
                    iv_skew=iv_skew,
                    iv_smile=iv_smile
                )
                
                surface_points.append(point)
        
        self.iv_surface_points.extend(surface_points)
        return surface_points
    
    def calculate_iv_rank(self, symbol: str, current_iv: float, lookback_days: int = 252) -> float:
        """
        Calculate IV Rank.
        
        IV Rank = (Current IV - Min IV) / (Max IV - Min IV) over lookback period.
        
        Args:
            symbol: Trading symbol
            current_iv: Current IV
            lookback_days: Number of days to look back
            
        Returns:
            IV Rank (0 to 1)
        """
        # Get historical IVs
        historical_ivs = [p.iv_smile for p in self.iv_surface_points if p.symbol == symbol]
        
        if len(historical_ivs) < 10:
            return 0.5
        
        min_iv = min(historical_ivs)
        max_iv = max(historical_ivs)
        
        if max_iv == min_iv:
            return 0.5
        
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv)
        return max(0, min(1, iv_rank))
    
    def calculate_iv_percentile(self, symbol: str, current_iv: float, lookback_days: int = 252) -> float:
        """
        Calculate IV Percentile.
        
        IV Percentile = Percentage of historical IVs below current IV.
        
        Args:
            symbol: Trading symbol
            current_iv: Current IV
            lookback_days: Number of days to look back
            
        Returns:
            IV Percentile (0 to 1)
        """
        historical_ivs = [p.iv_smile for p in self.iv_surface_points if p.symbol == symbol]
        
        if len(historical_ivs) < 10:
            return 0.5
        
        percentile = len([iv for iv in historical_ivs if iv < current_iv]) / len(historical_ivs)
        return max(0, min(1, percentile))
    
    def calculate_gamma_exposure(
        self,
        symbol: str,
        spot_price: float,
        strikes: List[float],
        expiries: List[datetime],
        open_interest: Dict[Tuple[str, float, datetime], int]
    ) -> GammaExposure:
        """
        Calculate total gamma exposure.
        
        Gamma exposure = Sum(Gamma * Open Interest * Spot Price)
        
        Args:
            symbol: Trading symbol
            spot_price: Current spot price
            strikes: List of strike prices
            expiries: List of expiry dates
            open_interest: Dictionary mapping (option_type, strike, expiry) to open interest
            
        Returns:
            GammaExposure with analysis
        """
        total_gamma = 0
        call_gamma = 0
        put_gamma = 0
        
        for expiry in expiries:
            T = (expiry - datetime.now()).days / 365.0
            
            for strike in strikes:
                # Calculate gamma
                gamma = self.bs_model.gamma(spot_price, strike, T, self.risk_free_rate, 0.2)
                
                # Get open interest
                call_oi = open_interest.get(('call', strike, expiry), 0)
                put_oi = open_interest.get(('put', strike, expiry), 0)
                
                # Calculate gamma exposure
                call_gamma += gamma * call_oi * spot_price
                put_gamma += gamma * put_oi * spot_price
        
        total_gamma = call_gamma + put_gamma
        
        # Estimate dealer gamma (dealers are typically short gamma)
        dealer_gamma = -total_gamma * 0.7  # Assume dealers hold 70% of short gamma
        
        # Check for gamma squeeze
        is_gamma_squeeze = abs(dealer_gamma) > 1e8  # Threshold for gamma squeeze
        
        exposure = GammaExposure(
            timestamp=datetime.now(),
            symbol=symbol,
            total_gamma=total_gamma,
            call_gamma=call_gamma,
            put_gamma=put_gamma,
            gamma_pnl=0,  # Would calculate from price changes
            dealer_gamma=dealer_gamma,
            is_gamma_squeeze=is_gamma_squeeze
        )
        
        self.gamma_exposures.append(exposure)
        return exposure
    
    def calculate_volatility_risk_premium(
        self,
        symbol: str,
        realized_vol: float,
        implied_vol: float
    ) -> VolatilityRiskPremium:
        """
        Calculate volatility risk premium.
        
        VRP = Realized Vol - Implied Vol
        
        Positive VRP means implied vol is overpriced (sell vol)
        Negative VRP means implied vol is underpriced (buy vol)
        
        Args:
            symbol: Trading symbol
            realized_vol: Realized volatility (annualized)
            implied_vol: Implied volatility (annualized)
            
        Returns:
            VolatilityRiskPremium
        """
        vrp = realized_vol - implied_vol
        is_vol_overpriced = vrp < 0
        
        premium = VolatilityRiskPremium(
            timestamp=datetime.now(),
            symbol=symbol,
            realized_vol=realized_vol,
            implied_vol=implied_vol,
            vr_premium=vrp,
            is_vol_overpriced=is_vol_overpriced
        )
        
        self.vrp_history.append(premium)
        return premium
    
    def analyze_dealer_positioning(
        self,
        symbol: str,
        spot_price: float,
        strikes: List[float],
        expiries: List[datetime],
        option_prices: Dict[Tuple[str, float, datetime], float],
        open_interest: Dict[Tuple[str, float, datetime], int]
    ) -> DealerFlow:
        """
        Analyze dealer positioning and hedging flows.
        
        Dealers are typically short options, so they need to hedge.
        When dealers are short gamma, they must buy when price falls and sell when price rises.
        This creates negative feedback (stabilizing) in normal conditions.
        When dealers are long gamma, they amplify moves (destabilizing).
        
        Args:
            symbol: Trading symbol
            spot_price: Current spot price
            strikes: List of strike prices
            expiries: List of expiry dates
            option_prices: Option prices
            open_interest: Open interest
            
        Returns:
            DealerFlow analysis
        """
        total_gamma = 0
        
        for expiry in expiries:
            T = (expiry - datetime.now()).days / 365.0
            
            for strike in strikes:
                gamma = self.bs_model.gamma(spot_price, strike, T, self.risk_free_rate, 0.2)
                
                call_oi = open_interest.get(('call', strike, expiry), 0)
                put_oi = open_interest.get(('put', strike, expiry), 0)
                
                # Assume dealers are short 70% of open interest
                dealer_gamma += -0.7 * gamma * (call_oi + put_oi) * spot_price
        
        # Determine dealer position
        dealer_position = "short_gamma" if dealer_gamma < 0 else "long_gamma"
        
        # Calculate hedging flow
        # Dealers hedge to maintain delta neutrality
        hedging_flow = abs(dealer_gamma) * 0.01  # Approximate daily hedge
        
        # Delta hedge ratio
        delta_hedge_ratio = min(1.0, hedging_flow / spot_price)
        
        # Check if dealers are selling (short gamma)
        is_dealer_selling = dealer_gamma < 0
        
        flow = DealerFlow(
            timestamp=datetime.now(),
            symbol=symbol,
            dealer_position=dealer_position,
            hedging_flow=hedging_flow,
            delta_hedge_ratio=delta_hedge_ratio,
            is_dealer_selling=is_dealer_selling
        )
        
        self.dealer_flows.append(flow)
        return flow


if __name__ == "__main__":
    # Test the IV Surface Manager
    print("Testing IV Surface and Black-Scholes Derivatives...")
    
    manager = IVSurfaceManager()
    
    # Generate sample data
    symbol = "RELIANCE"
    spot_price = 2500
    strikes = [2300, 2400, 2500, 2600, 2700]
    expiries = [datetime.now() + timedelta(days=30), datetime.now() + timedelta(days=60)]
    
    # Generate option prices
    option_prices = {}
    for expiry in expiries:
        for strike in strikes:
            T = (expiry - datetime.now()).days / 365.0
            option_prices[('call', strike, expiry)] = manager.bs_model.call_price(spot_price, strike, T, 0.06, 0.2)
            option_prices[('put', strike, expiry)] = manager.bs_model.put_price(spot_price, strike, T, 0.06, 0.2)
    
    # Build IV surface
    print("\nBuilding IV Surface...")
    surface = manager.build_iv_surface(symbol, spot_price, strikes, expiries, option_prices)
    print(f"Built {len(surface)} IV surface points")
    
    # Calculate IV Rank and Percentile
    print("\nIV Metrics:")
    current_iv = 0.25
    iv_rank = manager.calculate_iv_rank(symbol, current_iv)
    iv_percentile = manager.calculate_iv_percentile(symbol, current_iv)
    print(f"IV Rank: {iv_rank:.2%}")
    print(f"IV Percentile: {iv_percentile:.2%}")
    
    # Calculate Gamma Exposure
    print("\nGamma Exposure:")
    open_interest = {}
    for expiry in expiries:
        for strike in strikes:
            open_interest[('call', strike, expiry)] = np.random.randint(1000, 10000)
            open_interest[('put', strike, expiry)] = np.random.randint(1000, 10000)
    
    gamma_exposure = manager.calculate_gamma_exposure(symbol, spot_price, strikes, expiries, open_interest)
    print(f"Total Gamma: {gamma_exposure.total_gamma:.0f}")
    print(f"Dealer Gamma: {gamma_exposure.dealer_gamma:.0f}")
    print(f"Gamma Squeeze: {gamma_exposure.is_gamma_squeeze}")
    
    # Calculate Volatility Risk Premium
    print("\nVolatility Risk Premium:")
    vrp = manager.calculate_volatility_risk_premium(symbol, 0.18, 0.25)
    print(f"Realized Vol: {vrp.realized_vol:.2%}")
    print(f"Implied Vol: {vrp.implied_vol:.2%}")
    print(f"VRP: {vrp.vr_premium:.2%}")
    print(f"Vol Overpriced: {vrp.is_vol_overpriced}")
    
    # Analyze Dealer Positioning
    print("\nDealer Positioning:")
    dealer_flow = manager.analyze_dealer_positioning(symbol, spot_price, strikes, expiries, option_prices, open_interest)
    print(f"Dealer Position: {dealer_flow.dealer_position}")
    print(f"Hedging Flow: {dealer_flow.hedging_flow:.0f}")
    print(f"Delta Hedge Ratio: {dealer_flow.delta_hedge_ratio:.2%}")
    print(f"Dealer Selling: {dealer_flow.is_dealer_selling}")

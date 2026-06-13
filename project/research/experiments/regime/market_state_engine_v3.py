"""
Market State Engine (12 States)
Based on V3 Blueprint - Enhanced Market State Classification

Key findings from research:
- Regimes are too coarse; need more granular "states"
- Multi-dimensional market state classification
- Dimensions: trend, volatility, breadth, sentiment, liquidity
- States: bull_accumulation, bull_overextended, bull_distribution, panic_pullback, etc.

V3 Upgrade - Expected Sharpe increase: +0.2–0.3 (via better alpha weighting)
Priority: High
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from sklearn.cluster import KMeans
from scipy import stats


@dataclass
class MarketDimensions:
    """Market dimensions for state classification"""
    trend_strength: float  # -1 to 1 (bear to bull)
    trend_direction: str  # strong_up, weak_up, sideways, weak_down, strong_down
    volatility_level: float  # VIX level
    volatility_regime: str  # very_low, low, medium, high, very_high
    breadth: float  # Advance/decline ratio
    breadth_regime: str  # expanding, contracting, neutral
    sentiment: float  # -1 to 1 (fear to greed)
    sentiment_regime: str  # extreme_fear, fear, neutral, greed, extreme_greed
    liquidity: float  # Bid-ask spread (inverse)
    liquidity_regime: str  # high, normal, low


@dataclass
class MarketState:
    """Market state classification"""
    state_name: str
    state_probability: float
    dimensions: MarketDimensions
    alpha_weights: Dict[str, float]  # State-specific alpha weights
    position_multiplier: float  # Position sizing multiplier
    risk_multiplier: float  # Risk limit multiplier


class MarketStateEngine:
    """
    Market State Engine with 12 states.
    
    Dimensions:
    - Trend: strength and direction (using HMM on NIFTY returns + slope)
    - Volatility: VIX level and slope, realized vol ratio
    - Breadth: advance/decline, new highs/lows, sector dispersion
    - Sentiment: PCR, FII/DII flow, news sentiment
    - Liquidity: bid-ask spread, market depth
    
    States (12 total):
    - bull_accumulation: trend=strong_up + breadth=expanding + sentiment=neutral
    - bull_overextended: trend=strong_up + breadth=contracting + sentiment=greed
    - bull_distribution: trend=weak_up + breadth=contracting + sentiment=greed
    - bear_accumulation: trend=strong_down + breadth=expanding + sentiment=fear
    - bear_oversold: trend=strong_down + breadth=contracting + sentiment=extreme_fear
    - bear_distribution: trend=weak_down + breadth=contracting + sentiment=fear
    - sideways_low_vol: range, VIX < 15
    - sideways_high_vol: range, VIX > 20
    - panic: VIX spike > 30, large gap down, high volume
    - euphoria: VIX < 12, NIFTY up > 5% in a week, high breadth
    - transition: Mixed signals, unclear direction
    - normal: Balanced conditions
    """
    
    def __init__(self):
        self.state_history: List[MarketState] = []
        
        # State definitions (rule-based)
        self.state_rules = {
            "bull_accumulation": {
                "trend": "strong_up",
                "breadth": "expanding",
                "sentiment": "neutral"
            },
            "bull_overextended": {
                "trend": "strong_up",
                "breadth": "contracting",
                "sentiment": "greed"
            },
            "bull_distribution": {
                "trend": "weak_up",
                "breadth": "contracting",
                "sentiment": "greed"
            },
            "bear_accumulation": {
                "trend": "strong_down",
                "breadth": "expanding",
                "sentiment": "fear"
            },
            "bear_oversold": {
                "trend": "strong_down",
                "breadth": "contracting",
                "sentiment": "extreme_fear"
            },
            "bear_distribution": {
                "trend": "weak_down",
                "breadth": "contracting",
                "sentiment": "fear"
            },
            "sideways_low_vol": {
                "trend": "sideways",
                "volatility": "low"
            },
            "sideways_high_vol": {
                "trend": "sideways",
                "volatility": "high"
            },
            "panic": {
                "volatility": "very_high",
                "sentiment": "extreme_fear"
            },
            "euphoria": {
                "volatility": "very_low",
                "sentiment": "extreme_greed"
            }
        }
        
        # State-specific alpha weights
        self.state_alpha_weights = {
            "bull_accumulation": {"ORB": 0.35, "VWAP": 0.30, "PCP": 0.15, "VOL_CARRY": 0.10, "GAME_THEORETIC": 0.10},
            "bull_overextended": {"ORB": 0.20, "VWAP": 0.20, "PCP": 0.25, "VOL_CARRY": 0.20, "GAME_THEORETIC": 0.15},
            "bull_distribution": {"ORB": 0.15, "VWAP": 0.25, "PCP": 0.30, "VOL_CARRY": 0.20, "GAME_THEORETIC": 0.10},
            "bear_accumulation": {"ORB": 0.10, "VWAP": 0.15, "PCP": 0.35, "VOL_CARRY": 0.30, "GAME_THEORETIC": 0.10},
            "bear_oversold": {"ORB": 0.05, "VWAP": 0.10, "PCP": 0.40, "VOL_CARRY": 0.35, "GAME_THEORETIC": 0.10},
            "bear_distribution": {"ORB": 0.10, "VWAP": 0.15, "PCP": 0.35, "VOL_CARRY": 0.30, "GAME_THEORETIC": 0.10},
            "sideways_low_vol": {"ORB": 0.15, "VWAP": 0.20, "PCP": 0.25, "VOL_CARRY": 0.20, "GAME_THEORETIC": 0.20},
            "sideways_high_vol": {"ORB": 0.10, "VWAP": 0.15, "PCP": 0.35, "VOL_CARRY": 0.30, "GAME_THEORETIC": 0.10},
            "panic": {"ORB": 0.00, "VWAP": 0.00, "PCP": 0.40, "VOL_CARRY": 0.50, "GAME_THEORETIC": 0.10},
            "euphoria": {"ORB": 0.30, "VWAP": 0.30, "PCP": 0.10, "VOL_CARRY": 0.10, "GAME_THEORETIC": 0.20},
            "transition": {"ORB": 0.20, "VWAP": 0.20, "PCP": 0.20, "VOL_CARRY": 0.20, "GAME_THEORETIC": 0.20},
            "normal": {"ORB": 0.25, "VWAP": 0.25, "PCP": 0.20, "VOL_CARRY": 0.15, "GAME_THEORETIC": 0.15}
        }
        
        # State-specific position multipliers
        self.state_position_multipliers = {
            "bull_accumulation": 1.0,
            "bull_overextended": 0.8,
            "bull_distribution": 0.9,
            "bear_accumulation": 0.8,
            "bear_oversold": 0.5,
            "bear_distribution": 0.7,
            "sideways_low_vol": 0.9,
            "sideways_high_vol": 0.7,
            "panic": 0.3,
            "euphoria": 0.7,
            "transition": 0.8,
            "normal": 1.0
        }
        
        # State-specific risk multipliers
        self.state_risk_multipliers = {
            "bull_accumulation": 1.0,
            "bull_overextended": 1.2,
            "bull_distribution": 1.1,
            "bear_accumulation": 1.1,
            "bear_oversold": 0.8,
            "bear_distribution": 1.0,
            "sideways_low_vol": 0.9,
            "sideways_high_vol": 1.2,
            "panic": 2.0,
            "euphoria": 1.3,
            "transition": 1.1,
            "normal": 1.0
        }
    
    def classify_trend(self, returns: pd.Series, window: int = 20) -> Tuple[str, float]:
        """
        Classify trend direction and strength.
        
        Args:
            returns: Return series
            window: Rolling window
            
        Returns:
            (trend_direction, trend_strength)
        """
        if len(returns) < window:
            return "sideways", 0.0
        
        recent_returns = returns[-window:]
        mean_return = recent_returns.mean()
        std_return = recent_returns.std()
        
        if std_return == 0:
            return "sideways", 0.0
        
        # Calculate t-statistic
        t_stat = mean_return / std_return * np.sqrt(window)
        
        # Classify direction
        if t_stat > 2.0:
            direction = "strong_up"
        elif t_stat > 1.0:
            direction = "weak_up"
        elif t_stat < -2.0:
            direction = "strong_down"
        elif t_stat < -1.0:
            direction = "weak_down"
        else:
            direction = "sideways"
        
        # Strength (-1 to 1)
        strength = np.clip(t_stat / 3.0, -1, 1)
        
        return direction, strength
    
    def classify_volatility(self, vix: float) -> Tuple[str, float]:
        """
        Classify volatility regime.
        
        Args:
            vix: VIX level
            
        Returns:
            (volatility_regime, volatility_level)
        """
        if vix < 12:
            regime = "very_low"
        elif vix < 15:
            regime = "low"
        elif vix < 18:
            regime = "medium"
        elif vix < 25:
            regime = "high"
        else:
            regime = "very_high"
        
        return regime, vix
    
    def classify_breadth(self, adv_dec_ratio: float) -> Tuple[str, float]:
        """
        Classify breadth regime.
        
        Args:
            adv_dec_ratio: Advance/decline ratio
            
        Returns:
            (breadth_regime, breadth)
        """
        if adv_dec_ratio > 1.5:
            regime = "expanding"
        elif adv_dec_ratio < 0.67:
            regime = "contracting"
        else:
            regime = "neutral"
        
        return regime, adv_dec_ratio
    
    def classify_sentiment(self, pcr: float, fii_flow: float) -> Tuple[str, float]:
        """
        Classify sentiment regime.
        
        Args:
            pcr: Put-Call ratio
            fii_flow: FII net flow
            
        Returns:
            (sentiment_regime, sentiment_score)
        """
        # Combine PCR and FII flow
        # Low PCR + positive FII = bullish
        # High PCR + negative FII = bearish
        
        pcr_signal = (1.0 - pcr) / 1.0  # Normalize around 1.0
        fii_signal = fii_flow / 500.0  # Normalize around ₹500 crore
        
        sentiment_score = (pcr_signal + fii_signal) / 2
        sentiment_score = np.clip(sentiment_score, -1, 1)
        
        if sentiment_score > 0.5:
            regime = "extreme_greed"
        elif sentiment_score > 0.2:
            regime = "greed"
        elif sentiment_score < -0.5:
            regime = "extreme_fear"
        elif sentiment_score < -0.2:
            regime = "fear"
        else:
            regime = "neutral"
        
        return regime, sentiment_score
    
    def classify_liquidity(self, spread_bps: float) -> Tuple[str, float]:
        """
        Classify liquidity regime.
        
        Args:
            spread_bps: Bid-ask spread in bps
            
        Returns:
            (liquidity_regime, liquidity_score)
        """
        # Higher spread = lower liquidity
        liquidity_score = 1.0 - spread_bps / 10.0  # Normalize around 10 bps
        liquidity_score = np.clip(liquidity_score, 0, 1)
        
        if liquidity_score > 0.8:
            regime = "high"
        elif liquidity_score > 0.5:
            regime = "normal"
        else:
            regime = "low"
        
        return regime, liquidity_score
    
    def determine_state(self, dimensions: MarketDimensions) -> str:
        """
        Determine market state from dimensions.
        
        Args:
            dimensions: Market dimensions
            
        Returns:
            State name
        """
        # Check special states first
        if dimensions.volatility_regime == "very_high" and dimensions.sentiment_regime == "extreme_fear":
            return "panic"
        
        if dimensions.volatility_regime == "very_low" and dimensions.sentiment_regime == "extreme_greed":
            return "euphoria"
        
        # Check rule-based states
        for state_name, rules in self.state_rules.items():
            match = True
            
            if "trend" in rules and dimensions.trend_direction != rules["trend"]:
                match = False
            if "breadth" in rules and dimensions.breadth_regime != rules["breadth"]:
                match = False
            if "sentiment" in rules and dimensions.sentiment_regime != rules["sentiment"]:
                match = False
            if "volatility" in rules and dimensions.volatility_regime != rules["volatility"]:
                match = False
            
            if match:
                return state_name
        
        # Default to transition if no match
        return "transition"
    
    def compute_market_state(
        self,
        returns: pd.Series,
        vix: float,
        adv_dec_ratio: float,
        pcr: float,
        fii_flow: float,
        spread_bps: float
    ) -> MarketState:
        """
        Compute current market state.
        
        Args:
            returns: Return series
            vix: VIX level
            adv_dec_ratio: Advance/decline ratio
            pcr: Put-Call ratio
            fii_flow: FII net flow
            spread_bps: Bid-ask spread in bps
            
        Returns:
            MarketState
        """
        # Classify dimensions
        trend_direction, trend_strength = self.classify_trend(returns)
        volatility_regime, volatility_level = self.classify_volatility(vix)
        breadth_regime, breadth = self.classify_breadth(adv_dec_ratio)
        sentiment_regime, sentiment = self.classify_sentiment(pcr, fii_flow)
        liquidity_regime, liquidity = self.classify_liquidity(spread_bps)
        
        dimensions = MarketDimensions(
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            volatility_regime=volatility_regime,
            breadth=breadth,
            breadth_regime=breadth_regime,
            sentiment=sentiment,
            sentiment_regime=sentiment_regime,
            liquidity=liquidity,
            liquidity_regime=liquidity_regime
        )
        
        # Determine state
        state_name = self.determine_state(dimensions)
        
        # Get state-specific parameters
        alpha_weights = self.state_alpha_weights.get(state_name, self.state_alpha_weights["normal"])
        position_multiplier = self.state_position_multipliers.get(state_name, 1.0)
        risk_multiplier = self.state_risk_multipliers.get(state_name, 1.0)
        
        state = MarketState(
            state_name=state_name,
            state_probability=1.0,  # Rule-based, so probability is 1.0
            dimensions=dimensions,
            alpha_weights=alpha_weights,
            position_multiplier=position_multiplier,
            risk_multiplier=risk_multiplier
        )
        
        self.state_history.append(state)
        
        return state
    
    def print_state_report(self, state: MarketState) -> None:
        """Print market state report."""
        print("\n" + "="*60)
        print("MARKET STATE ENGINE REPORT")
        print("="*60)
        print(f"State: {state.state_name.upper()}")
        print(f"State Probability: {state.state_probability:.2%}")
        
        print("\nDimensions:")
        print(f"  Trend: {state.dimensions.trend_direction} (strength: {state.dimensions.trend_strength:.2f})")
        print(f"  Volatility: {state.dimensions.volatility_regime} (VIX: {state.dimensions.volatility_level:.2f})")
        print(f"  Breadth: {state.dimensions.breadth_regime} (A/D: {state.dimensions.breadth:.2f})")
        print(f"  Sentiment: {state.dimensions.sentiment_regime} (score: {state.dimensions.sentiment:.2f})")
        print(f"  Liquidity: {state.dimensions.liquidity_regime} (score: {state.dimensions.liquidity:.2f})")
        
        print("\nState-Specific Parameters:")
        print(f"  Position Multiplier: {state.position_multiplier:.2f}x")
        print(f"  Risk Multiplier: {state.risk_multiplier:.2f}x")
        
        print("\nAlpha Weights:")
        for alpha, weight in state.alpha_weights.items():
            print(f"  {alpha}: {weight:.2%}")
        
        print("="*60)


def run_sample_market_state_engine():
    """Run sample market state engine."""
    engine = MarketStateEngine()
    
    # Generate sample data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 100))
    
    # Sample scenarios
    scenarios = [
        {
            "vix": 14.0,
            "adv_dec_ratio": 1.8,
            "pcr": 0.8,
            "fii_flow": 200.0,
            "spread_bps": 2.0
        },
        {
            "vix": 28.0,
            "adv_dec_ratio": 0.5,
            "pcr": 1.3,
            "fii_flow": -300.0,
            "spread_bps": 8.0
        },
        {
            "vix": 11.0,
            "adv_dec_ratio": 2.0,
            "pcr": 0.7,
            "fii_flow": 400.0,
            "spread_bps": 1.5
        }
    ]
    
    for scenario in scenarios:
        state = engine.compute_market_state(
            returns=returns,
            vix=scenario["vix"],
            adv_dec_ratio=scenario["adv_dec_ratio"],
            pcr=scenario["pcr"],
            fii_flow=scenario["fii_flow"],
            spread_bps=scenario["spread_bps"]
        )
        engine.print_state_report(state)
    
    return engine


if __name__ == "__main__":
    run_sample_market_state_engine()

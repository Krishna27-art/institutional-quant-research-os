"""
Market State Engine
Enhanced market state classification with 12 states based on trend, volatility, breadth, and sentiment.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TrendState(Enum):
    """Trend states"""
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    SIDEWAYS = "sideways"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"


class VolatilityState(Enum):
    """Volatility states"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class BreadthState(Enum):
    """Breadth states"""
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    NEUTRAL = "neutral"


class SentimentState(Enum):
    """Sentiment states"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class StateDimensions:
    """Market state dimensions"""
    trend: TrendState
    volatility: VolatilityState
    breadth: BreadthState
    sentiment: SentimentState
    
    # Raw values
    trend_score: float = 0.0  # -1 to 1
    volatility_score: float = 0.0  # 0 to 1
    breadth_score: float = 0.0  # -1 to 1
    sentiment_score: float = 0.0  # 0 to 1
    
    def to_dict(self) -> Dict:
        return {
            "trend": self.trend.value,
            "volatility": self.volatility.value,
            "breadth": self.breadth.value,
            "sentiment": self.sentiment.value,
            "trend_score": self.trend_score,
            "volatility_score": self.volatility_score,
            "breadth_score": self.breadth_score,
            "sentiment_score": self.sentiment_score,
        }
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for clustering"""
        return np.array([
            self.trend_score,
            self.volatility_score,
            self.breadth_score,
            self.sentiment_score
        ])


@dataclass
class StateDefinition:
    """Definition of a market state"""
    state_name: str
    trend: TrendState
    volatility: VolatilityState
    breadth: BreadthState
    sentiment: SentimentState
    
    # State-specific parameters
    alpha_weight_multiplier: float = 1.0
    position_sizing_multiplier: float = 1.0
    risk_limit_multiplier: float = 1.0
    
    def matches(self, dimensions: StateDimensions) -> bool:
        """Check if dimensions match this state definition"""
        return (
            dimensions.trend == self.trend and
            dimensions.volatility == self.volatility and
            dimensions.breadth == self.breadth and
            dimensions.sentiment == self.sentiment
        )
    
    def to_dict(self) -> Dict:
        return {
            "state_name": self.state_name,
            "trend": self.trend.value,
            "volatility": self.volatility.value,
            "breadth": self.breadth.value,
            "sentiment": self.sentiment.value,
            "alpha_weight_multiplier": self.alpha_weight_multiplier,
            "position_sizing_multiplier": self.position_sizing_multiplier,
            "risk_limit_multiplier": self.risk_limit_multiplier,
        }


@dataclass
class MarketState:
    """Current market state"""
    state_name: str
    dimensions: StateDimensions
    timestamp: datetime
    
    # State parameters
    alpha_weight_multiplier: float = 1.0
    position_sizing_multiplier: float = 1.0
    risk_limit_multiplier: float = 1.0
    
    # Confidence
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "state_name": self.state_name,
            "dimensions": self.dimensions.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "alpha_weight_multiplier": self.alpha_weight_multiplier,
            "position_sizing_multiplier": self.position_sizing_multiplier,
            "risk_limit_multiplier": self.risk_limit_multiplier,
            "confidence": self.confidence,
        }


class MarketStateEngine:
    """
    Enhanced market state engine with 12 states.
    Uses clustering on 4 dimensions: trend, volatility, breadth, sentiment.
    """
    
    # Predefined state definitions (12 states)
    STATE_DEFINITIONS = [
        StateDefinition("bull_overextended", TrendState.STRONG_UP, VolatilityState.MEDIUM, BreadthState.EXPANDING, SentimentState.EXTREME_GREED, 0.8, 0.7, 1.2),
        StateDefinition("bull_accumulation", TrendState.WEAK_UP, VolatilityState.LOW, BreadthState.NEUTRAL, SentimentState.NEUTRAL, 1.2, 1.1, 1.0),
        StateDefinition("bull_distribution", TrendState.WEAK_UP, VolatilityState.MEDIUM, BreadthState.CONTRACTING, SentimentState.GREED, 0.9, 0.9, 1.1),
        StateDefinition("trend_following", TrendState.STRONG_UP, VolatilityState.LOW, BreadthState.EXPANDING, SentimentState.GREED, 1.1, 1.0, 1.0),
        StateDefinition("sideways_low_vol", TrendState.SIDEWAYS, VolatilityState.VERY_LOW, BreadthState.NEUTRAL, SentimentState.NEUTRAL, 1.0, 1.0, 1.0),
        StateDefinition("sideways_high_vol", TrendState.SIDEWAYS, VolatilityState.HIGH, BreadthState.NEUTRAL, SentimentState.FEAR, 0.8, 0.8, 1.2),
        StateDefinition("bear_weak", TrendState.WEAK_DOWN, VolatilityState.MEDIUM, BreadthState.NEUTRAL, SentimentState.FEAR, 0.9, 0.9, 1.1),
        StateDefinition("bear_strong", TrendState.STRONG_DOWN, VolatilityState.HIGH, BreadthState.CONTRACTING, SentimentState.EXTREME_FEAR, 0.5, 0.5, 1.5),
        StateDefinition("panic_pullback", TrendState.STRONG_DOWN, VolatilityState.VERY_HIGH, BreadthState.CONTRACTING, SentimentState.EXTREME_FEAR, 0.3, 0.3, 2.0),
        StateDefinition("reversal_bottom", TrendState.WEAK_DOWN, VolatilityState.HIGH, BreadthState.EXPANDING, SentimentState.EXTREME_FEAR, 1.2, 1.2, 1.3),
        StateDefinition("reversal_top", TrendState.WEAK_UP, VolatilityState.HIGH, BreadthState.EXPANDING, SentimentState.EXTREME_GREED, 0.7, 0.7, 1.3),
        StateDefinition("transition", TrendState.SIDEWAYS, VolatilityState.MEDIUM, BreadthState.NEUTRAL, SentimentState.NEUTRAL, 1.0, 1.0, 1.0),
    ]
    
    def __init__(self, n_clusters: int = 12):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.trained = False
        
        self.state_history: List[MarketState] = []
        self.cluster_labels: Dict[int, StateDefinition] = {}  # cluster_id -> state_definition
    
    def calculate_trend_score(self, returns: np.ndarray, window: int = 20) -> float:
        """
        Calculate trend score from returns.
        
        Args:
            returns: Array of returns
            window: Rolling window
        
        Returns:
            Trend score (-1 to 1)
        """
        if len(returns) < window:
            return 0.0
        
        recent_returns = returns[-window:]
        avg_return = np.mean(recent_returns)
        
        # Normalize to [-1, 1]
        # Assume reasonable range is -2% to +2% per day
        trend_score = np.clip(avg_return / 0.02, -1.0, 1.0)
        
        return trend_score
    
    def calculate_volatility_score(self, returns: np.ndarray, window: int = 20) -> float:
        """
        Calculate volatility score from returns.
        
        Args:
            returns: Array of returns
            window: Rolling window
        
        Returns:
            Volatility score (0 to 1)
        """
        if len(returns) < window:
            return 0.5
        
        recent_returns = returns[-window:]
        volatility = np.std(recent_returns)
        
        # Normalize to [0, 1]
        # Assume reasonable range is 0.5% to 3% daily vol
        vol_score = (volatility - 0.005) / (0.03 - 0.005)
        vol_score = np.clip(vol_score, 0.0, 1.0)
        
        return vol_score
    
    def calculate_breadth_score(self, advance_decline_ratio: float) -> float:
        """
        Calculate breadth score from advance/decline ratio.
        
        Args:
            advance_decline_ratio: Ratio of advancing to declining stocks
        
        Returns:
            Breadth score (-1 to 1)
        """
        # Normalize to [-1, 1]
        # Ratio of 1.0 = neutral, >1 = positive, <1 = negative
        if advance_decline_ratio == 1.0:
            return 0.0
        
        # Log transform for symmetry
        breadth_score = np.log(advance_decline_ratio) / np.log(2.0)
        breadth_score = np.clip(breadth_score, -1.0, 1.0)
        
        return breadth_score
    
    def calculate_sentiment_score(self, vix: float, put_call_ratio: float) -> float:
        """
        Calculate sentiment score from VIX and PCR.
        
        Args:
            vix: VIX value
            put_call_ratio: Put-Call ratio
        
        Returns:
            Sentiment score (0 to 1, 0 = extreme fear, 1 = extreme greed)
        """
        # VIX component (inverse relationship)
        # VIX < 15 = greedy, VIX > 25 = fearful
        vix_score = 1.0 - (vix - 15) / (25 - 15)
        vix_score = np.clip(vix_score, 0.0, 1.0)
        
        # PCR component (inverse relationship)
        # PCR < 0.8 = greedy, PCR > 1.2 = fearful
        pcr_score = 1.0 - (put_call_ratio - 0.8) / (1.2 - 0.8)
        pcr_score = np.clip(pcr_score, 0.0, 1.0)
        
        # Average both components
        sentiment_score = (vix_score + pcr_score) / 2.0
        
        return sentiment_score
    
    def classify_trend(self, score: float) -> TrendState:
        """Classify trend score to TrendState"""
        if score > 0.6:
            return TrendState.STRONG_UP
        elif score > 0.2:
            return TrendState.WEAK_UP
        elif score > -0.2:
            return TrendState.SIDEWAYS
        elif score > -0.6:
            return TrendState.WEAK_DOWN
        else:
            return TrendState.STRONG_DOWN
    
    def classify_volatility(self, score: float) -> VolatilityState:
        """Classify volatility score to VolatilityState"""
        if score < 0.2:
            return VolatilityState.VERY_LOW
        elif score < 0.4:
            return VolatilityState.LOW
        elif score < 0.6:
            return VolatilityState.MEDIUM
        elif score < 0.8:
            return VolatilityState.HIGH
        else:
            return VolatilityState.VERY_HIGH
    
    def classify_breadth(self, score: float) -> BreadthState:
        """Classify breadth score to BreadthState"""
        if score > 0.3:
            return BreadthState.EXPANDING
        elif score > -0.3:
            return BreadthState.NEUTRAL
        else:
            return BreadthState.CONTRACTING
    
    def classify_sentiment(self, score: float) -> SentimentState:
        """Classify sentiment score to SentimentState"""
        if score > 0.8:
            return SentimentState.EXTREME_GREED
        elif score > 0.6:
            return SentimentState.GREED
        elif score > 0.4:
            return SentimentState.NEUTRAL
        elif score > 0.2:
            return SentimentState.FEAR
        else:
            return SentimentState.EXTREME_FEAR
    
    def calculate_dimensions(
        self,
        returns: np.ndarray,
        advance_decline_ratio: float,
        vix: float,
        put_call_ratio: float
    ) -> StateDimensions:
        """
        Calculate all state dimensions.
        
        Args:
            returns: Array of market returns
            advance_decline_ratio: Advance/decline ratio
            vix: VIX value
            put_call_ratio: Put-Call ratio
        
        Returns:
            StateDimensions with all classifications
        """
        trend_score = self.calculate_trend_score(returns)
        volatility_score = self.calculate_volatility_score(returns)
        breadth_score = self.calculate_breadth_score(advance_decline_ratio)
        sentiment_score = self.calculate_sentiment_score(vix, put_call_ratio)
        
        dimensions = StateDimensions(
            trend=self.classify_trend(trend_score),
            volatility=self.classify_volatility(volatility_score),
            breadth=self.classify_breadth(breadth_score),
            sentiment=self.classify_sentiment(sentiment_score),
            trend_score=trend_score,
            volatility_score=volatility_score,
            breadth_score=breadth_score,
            sentiment_score=sentiment_score
        )
        
        return dimensions
    
    def detect_state(self, dimensions: StateDimensions) -> MarketState:
        """
        Detect market state from dimensions.
        
        Args:
            dimensions: State dimensions
        
        Returns:
            MarketState with detected state
        """
        # Try to match predefined state
        for state_def in self.STATE_DEFINITIONS:
            if state_def.matches(dimensions):
                return MarketState(
                    state_name=state_def.state_name,
                    dimensions=dimensions,
                    timestamp=datetime.now(),
                    alpha_weight_multiplier=state_def.alpha_weight_multiplier,
                    position_sizing_multiplier=state_def.position_sizing_multiplier,
                    risk_limit_multiplier=state_def.risk_limit_multiplier,
                    confidence=1.0
                )
        
        # If no match, use clustering (if trained)
        if self.trained:
            feature_vector = dimensions.to_feature_vector().reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            cluster_id = self.kmeans.predict(scaled_features)[0]
            
            if cluster_id in self.cluster_labels:
                state_def = self.cluster_labels[cluster_id]
                return MarketState(
                    state_name=state_def.state_name,
                    dimensions=dimensions,
                    timestamp=datetime.now(),
                    alpha_weight_multiplier=state_def.alpha_weight_multiplier,
                    position_sizing_multiplier=state_def.position_sizing_multiplier,
                    risk_limit_multiplier=state_def.risk_limit_multiplier,
                    confidence=0.8
                )
        
        # Fallback: transition state
        return MarketState(
            state_name="transition",
            dimensions=dimensions,
            timestamp=datetime.now(),
            confidence=0.5
        )
    
    def train_clustering(self, historical_dimensions: List[StateDimensions]) -> None:
        """
        Train clustering model on historical dimensions.
        
        Args:
            historical_dimensions: List of historical state dimensions
        """
        if len(historical_dimensions) < self.n_clusters:
            print(f"Warning: Not enough data points ({len(historical_dimensions)}) for {self.n_clusters} clusters")
            return
        
        # Prepare feature matrix
        feature_matrix = np.array([d.to_feature_vector() for d in historical_dimensions])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Fit K-means
        self.kmeans.fit(scaled_features)
        
        # Assign state definitions to clusters
        cluster_centers = self.kmeans.cluster_centers_
        for cluster_id in range(self.n_clusters):
            center = cluster_centers[cluster_id]
            
            # Find closest predefined state
            min_distance = float('inf')
            closest_state = None
            
            for state_def in self.STATE_DEFINITIONS:
                # Create dimensions for state definition
                state_dims = StateDimensions(
                    trend=state_def.trend,
                    volatility=state_def.volatility,
                    breadth=state_def.breadth,
                    sentiment=state_def.sentiment
                )
                state_vector = state_dims.to_feature_vector()
                state_vector = self.scaler.transform([state_vector])[0]
                
                distance = np.linalg.norm(center - state_vector)
                if distance < min_distance:
                    min_distance = distance
                    closest_state = state_def
            
            if closest_state:
                self.cluster_labels[cluster_id] = closest_state
        
        self.trained = True
    
    def update_state(
        self,
        returns: np.ndarray,
        advance_decline_ratio: float,
        vix: float,
        put_call_ratio: float
    ) -> MarketState:
        """
        Update market state with new data.
        
        Args:
            returns: Array of market returns
            advance_decline_ratio: Advance/decline ratio
            vix: VIX value
            put_call_ratio: Put-Call ratio
        
        Returns:
            Current market state
        """
        dimensions = self.calculate_dimensions(returns, advance_decline_ratio, vix, put_call_ratio)
        state = self.detect_state(dimensions)
        
        self.state_history.append(state)
        
        return state
    
    def get_current_state(self) -> Optional[MarketState]:
        """Get current market state"""
        if self.state_history:
            return self.state_history[-1]
        return None
    
    def get_state_history(self, days: int = 30) -> List[Dict]:
        """Get state history for last N days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_states = [
            s for s in self.state_history
            if s.timestamp >= cutoff_time
        ]
        return [s.to_dict() for s in recent_states]
    
    def get_state_frequency(self, days: int = 30) -> Dict[str, int]:
        """Get frequency of each state in recent history"""
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_states = [
            s for s in self.state_history
            if s.timestamp >= cutoff_time
        ]
        
        frequency = {}
        for state in recent_states:
            state_name = state.state_name
            frequency[state_name] = frequency.get(state_name, 0) + 1
        
        return frequency

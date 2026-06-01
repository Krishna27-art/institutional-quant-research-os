"""
Base Alpha Engine Interface
Architecture V2 - Quantitative Trading System for Indian Markets
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from enum import Enum


class SignalDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class AlphaSignal:
    """Alpha signal output"""
    symbol: str
    timestamp: datetime
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    expected_return: float  # Expected return in basis points
    holding_period_minutes: int
    features: Dict[str, float]
    regime: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class AlphaMetrics:
    """Alpha performance metrics"""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_period_minutes: float
    capacity_cr: float  # Capacity in Crores
    decay_months: int


class BaseAlphaEngine(ABC):
    """
    Base class for all alpha engines.
    
    All alpha engines must implement:
    - generate_signals(): Generate trading signals
    - get_required_features(): Return list of required features
    - get_metrics(): Return performance metrics
    """
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.is_enabled = config.get("enabled", True)
        self.last_update = None
        
    @abstractmethod
    def generate_signals(
        self, 
        market_data: Dict[str, np.ndarray],
        features: Dict[str, np.ndarray],
        timestamp: datetime
    ) -> List[AlphaSignal]:
        """
        Generate trading signals based on market data and features.
        
        Args:
            market_data: Dictionary of market data (OHLCV, etc.)
            features: Dictionary of computed features
            timestamp: Current timestamp
            
        Returns:
            List of AlphaSignal objects
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Return list of required feature names"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> AlphaMetrics:
        """Return performance metrics for this alpha"""
        pass
    
    def validate_signal(self, signal: AlphaSignal) -> bool:
        """
        Validate a signal before emission.
        
        Checks:
        - Confidence threshold
        - Market hours
        - Risk limits
        """
        # Check confidence threshold
        min_confidence = self.config.get("min_confidence", 0.5)
        if signal.confidence < min_confidence:
            return False
        
        # Check if market is open (9:15 AM to 3:30 PM IST)
        hour = signal.timestamp.hour
        minute = signal.timestamp.minute
        time_minutes = hour * 60 + minute
        
        # Market hours: 9:15 (555 min) to 15:30 (930 min)
        if not (555 <= time_minutes <= 930):
            return False
        
        return True
    
    def filter_signals(
        self, 
        signals: List[AlphaSignal]
    ) -> List[AlphaSignal]:
        """Filter signals based on validation rules"""
        return [s for s in signals if self.validate_signal(s)]
    
    def get_expected_capacity(self) -> float:
        """Return expected capacity in Crores"""
        return self.config.get("capacity_cr", 0)
    
    def get_decay_period(self) -> int:
        """Return expected decay period in months"""
        return self.config.get("decay_months", 12)
    
    def get_confidence_level(self) -> float:
        """Return confidence level (0.0 to 1.0)"""
        return self.config.get("confidence", 0.5)


class MicrostructureAlpha(BaseAlphaEngine):
    """Base class for microstructure-based alphas (ORB, VWAP, etc.)"""
    
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.slippage_model = config.get("slippage_model", "conservative")
        

class MLAlpha(BaseAlphaEngine):
    """Base class for ML-based alphas (LightGBM, etc.)"""
    
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model = None
        self.feature_importance = None
        self.last_retrain_date = None
        self.retrain_frequency_days = config.get("retrain_frequency_days", 5)
    
    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ML model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        pass
    
    def should_retrain(self, current_date: datetime) -> bool:
        """Check if model should be retrained"""
        if self.last_retrain_date is None:
            return True
        
        days_since_retrain = (current_date - self.last_retrain_date).days
        return days_since_retrain >= self.retrain_frequency_days


class RegimeAlpha(BaseAlphaEngine):
    """Base class for regime-dependent alphas"""
    
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.regime_weights = config.get("regime_weights", {})
        self.current_regime = None
    
    def set_regime(self, regime: str) -> None:
        """Set current market regime"""
        self.current_regime = regime
    
    def get_regime_weight(self) -> float:
        """Get weight for current regime"""
        if self.current_regime is None:
            return 1.0
        return self.regime_weights.get(self.current_regime, 1.0)

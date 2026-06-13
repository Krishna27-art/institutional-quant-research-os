"""
PEAD with Earnings Streak + ML Alpha Strategy

This module implements the Post-Earnings Announcement Drift (PEAD) strategy
enhanced with earnings streak effects and machine learning, capturing the
market's under-reaction to earnings surprises and the persistence of
historical earnings performance.

Based on Bernard & Thomas 1990; 2025 ML study (Beyond the last surprise).
Expected Sharpe: 0.4-0.7
Expected Capacity: Very High
Decay: Years
Difficulty: Medium

Priority: Medium (Research OS Phase 4)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available, PEAD ML will use fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurpriseDirection(Enum):
    """Earnings surprise direction."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class EarningsEvent:
    """Earnings announcement event."""
    symbol: str
    announcement_date: datetime
    eps_actual: float
    eps_estimate: float
    surprise: float  # Actual - Estimate
    surprise_pct: float  # Surprise as percentage of estimate
    direction: SurpriseDirection
    streak_length: int  # Consecutive surprises in same direction
    streak_direction: SurpriseDirection
    pre_announcement_price: float
    post_announcement_price: float
    post_drift_return: float  # Return over drift period


@dataclass
class PEADSignal:
    """PEAD trading signal."""
    timestamp: datetime
    symbol: str
    direction: SurpriseDirection
    surprise_pct: float
    streak_length: int
    ml_score: float  # ML model prediction
    combined_score: float  # Combined traditional + ML score
    signal: float  # -1 to 1, negative = short, positive = long
    confidence: float
    holding_period_days: int


class PEADEarningsStreakAlpha:
    """
    PEAD with earnings streak + ML alpha strategy.
    
    This class implements PEAD strategy enhanced with earnings streak
    tracking and machine learning for improved signal generation.
    """
    
    def __init__(
        self,
        drift_period_days: int = 60,
        surprise_threshold: float = 0.05,  # 5% surprise threshold
        streak_lookback: int = 4,  # Look at last 4 earnings
        min_streak_length: int = 2,
        use_ml: bool = True
    ):
        """
        Initialize PEAD alpha.
        
        Args:
            drift_period_days: Post-earnings drift period in days
            surprise_threshold: Minimum surprise percentage for signal
            streak_lookback: Number of past earnings to consider for streak
            min_streak_length: Minimum streak length for enhanced signal
            use_ml: Enable machine learning enhancement
        """
        self.drift_period_days = drift_period_days
        self.surprise_threshold = surprise_threshold
        self.streak_lookback = streak_lookback
        self.min_streak_length = min_streak_length
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        
        self.earnings_history: Dict[str, List[EarningsEvent]] = {}
        self.signals: List[PEADSignal] = []
        
        # ML components
        self.ml_model = None
        self.scaler = None
        self.features: List[Dict] = []
        self.targets: List[float] = []
        
        if self.use_ml:
            self._initialize_ml_model()
        
        logger.info(f"PEADEarningsStreakAlpha initialized: drift_period={drift_period_days}days, "
                   f"surprise_threshold={surprise_threshold}, use_ml={self.use_ml}")
    
    def _initialize_ml_model(self) -> None:
        """Initialize ML model for PEAD enhancement."""
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=10
        )
        self.scaler = StandardScaler()
        
        logger.info("ML model initialized for PEAD enhancement")
    
    def calculate_surprise(
        self,
        eps_actual: float,
        eps_estimate: float
    ) -> Tuple[float, float, SurpriseDirection]:
        """
        Calculate earnings surprise.
        
        Args:
            eps_actual: Actual EPS
            eps_estimate: Estimated EPS
            
        Returns:
            (surprise, surprise_pct, direction)
        """
        if eps_estimate == 0:
            return 0.0, 0.0, SurpriseDirection.NEUTRAL
        
        surprise = eps_actual - eps_estimate
        surprise_pct = surprise / abs(eps_estimate)
        
        if surprise_pct > self.surprise_threshold:
            direction = SurpriseDirection.POSITIVE
        elif surprise_pct < -self.surprise_threshold:
            direction = SurpriseDirection.NEGATIVE
        else:
            direction = SurpriseDirection.NEUTRAL
        
        return surprise, surprise_pct, direction
    
    def calculate_streak(
        self,
        symbol: str,
        current_direction: SurpriseDirection
    ) -> Tuple[int, SurpriseDirection]:
        """
        Calculate earnings streak length and direction.
        
        Args:
            symbol: Stock symbol
            current_direction: Current surprise direction
            
        Returns:
            (streak_length, streak_direction)
        """
        if symbol not in self.earnings_history:
            return 1, current_direction
        
        history = self.earnings_history[symbol]
        if not history:
            return 1, current_direction
        
        # Count consecutive surprises in same direction
        streak_length = 1
        streak_direction = current_direction
        
        for event in reversed(history[-self.streak_lookback:]):
            if event.direction == current_direction:
                streak_length += 1
            else:
                break
        
        return streak_length, streak_direction
    
    def extract_features(
        self,
        event: EarningsEvent
    ) -> Dict[str, float]:
        """
        Extract features for ML model.
        
        Args:
            event: Earnings event
            
        Returns:
            Feature dictionary
        """
        symbol = event.symbol
        history = self.earnings_history.get(symbol, [])
        
        features = {
            'surprise_pct': event.surprise_pct,
            'streak_length': event.streak_length,
            'streak_positive': 1.0 if event.streak_direction == SurpriseDirection.POSITIVE else 0.0,
            'is_positive': 1.0 if event.direction == SurpriseDirection.POSITIVE else 0.0,
        }
        
        # Historical features
        if len(history) >= 2:
            recent_surprises = [e.surprise_pct for e in history[-4:]]
            features['avg_recent_surprise'] = np.mean(recent_surprises)
            features['std_recent_surprise'] = np.std(recent_surprises)
            features['trend_surprise'] = recent_surprises[-1] - recent_surprises[0]
        else:
            features['avg_recent_surprise'] = event.surprise_pct
            features['std_recent_surprise'] = 0.0
            features['trend_surprise'] = 0.0
        
        return features
    
    def train_ml_model(self) -> None:
        """Train ML model on historical data."""
        if not self.use_ml or len(self.features) < 50:
            logger.warning("Insufficient data for ML training")
            return
        
        # Prepare data
        X = pd.DataFrame(self.features)
        y = np.array(self.targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.ml_model.fit(X_scaled, y)
        
        logger.info(f"ML model trained on {len(self.features)} samples")
    
    def predict_drift_return(
        self,
        event: EarningsEvent
    ) -> float:
        """
        Predict post-earnings drift return using ML.
        
        Args:
            event: Earnings event
            
        Returns:
            Predicted drift return
        """
        if not self.use_ml or self.ml_model is None:
            # Fallback: use simple linear model
            base_return = event.surprise_pct * 0.5
            streak_bonus = event.streak_length * 0.1 if event.streak_length >= self.min_streak_length else 0.0
            return base_return + streak_bonus
        
        # Extract features
        features = self.extract_features(event)
        feature_df = pd.DataFrame([features])
        
        # Scale features
        features_scaled = self.scaler.transform(feature_df)
        
        # Predict
        prediction = self.ml_model.predict(features_scaled)[0]
        
        return prediction
    
    def generate_signal(
        self,
        symbol: str,
        eps_actual: float,
        eps_estimate: float,
        pre_price: float,
        post_price: float,
        announcement_date: datetime
    ) -> Optional[PEADSignal]:
        """
        Generate PEAD trading signal.
        
        Args:
            symbol: Stock symbol
            eps_actual: Actual EPS
            eps_estimate: Estimated EPS
            pre_price: Pre-announcement price
            post_price: Post-announcement price
            announcement_date: Announcement date
            
        Returns:
            PEADSignal or None
        """
        # Calculate surprise
        surprise, surprise_pct, direction = self.calculate_surprise(eps_actual, eps_estimate)
        
        # Skip if surprise is too small
        if direction == SurpriseDirection.NEUTRAL:
            return None
        
        # Calculate streak
        streak_length, streak_direction = self.calculate_streak(symbol, direction)
        
        # Create earnings event
        event = EarningsEvent(
            symbol=symbol,
            announcement_date=announcement_date,
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            surprise=surprise,
            surprise_pct=surprise_pct,
            direction=direction,
            streak_length=streak_length,
            streak_direction=streak_direction,
            pre_announcement_price=pre_price,
            post_announcement_price=post_price,
            post_drift_return=0.0  # Will be calculated later
        )
        
        # Store event
        if symbol not in self.earnings_history:
            self.earnings_history[symbol] = []
        self.earnings_history[symbol].append(event)
        
        # Keep history manageable
        if len(self.earnings_history[symbol]) > 20:
            self.earnings_history[symbol] = self.earnings_history[symbol][-20:]
        
        # Predict drift return
        ml_score = self.predict_drift_return(event)
        
        # Calculate traditional PEAD score
        traditional_score = surprise_pct * 0.5
        
        # Add streak bonus
        if streak_length >= self.min_streak_length:
            streak_bonus = streak_length * 0.1 * (1.0 if direction == streak_direction else -0.5)
            traditional_score += streak_bonus
        
        # Combine scores
        if self.use_ml:
            combined_score = 0.6 * traditional_score + 0.4 * ml_score
        else:
            combined_score = traditional_score
        
        # Generate signal
        signal = np.sign(combined_score) * min(abs(combined_score), 1.0)
        
        # Calculate confidence
        confidence = min(abs(surprise_pct) / 0.1, 0.9)
        if streak_length >= self.min_streak_length:
            confidence = min(confidence + 0.1, 0.95)
        
        # Determine holding period based on streak
        if streak_length >= self.min_streak_length:
            holding_period = self.drift_period_days + streak_length * 5
        else:
            holding_period = self.drift_period_days
        
        pead_signal = PEADSignal(
            timestamp=announcement_date,
            symbol=symbol,
            direction=direction,
            surprise_pct=surprise_pct,
            streak_length=streak_length,
            ml_score=ml_score,
            combined_score=combined_score,
            signal=signal,
            confidence=confidence,
            holding_period_days=holding_period
        )
        
        self.signals.append(pead_signal)
        
        # Add to ML training data if we have actual drift return
        if len(self.earnings_history[symbol]) > 1:
            previous_event = self.earnings_history[symbol][-2]
            if previous_event.post_drift_return != 0.0:
                features = self.extract_features(previous_event)
                self.features.append(features)
                self.targets.append(previous_event.post_drift_return)
        
        return pead_signal
    
    def update_drift_return(
        self,
        symbol: str,
        announcement_date: datetime,
        drift_return: float
    ) -> None:
        """
        Update actual drift return for training.
        
        Args:
            symbol: Stock symbol
            announcement_date: Announcement date
            drift_return: Actual drift return
        """
        if symbol not in self.earnings_history:
            return
        
        for event in self.earnings_history[symbol]:
            if event.announcement_date == announcement_date:
                event.post_drift_return = drift_return
                break
    
    def train_on_historical_data(self) -> None:
        """Train ML model on accumulated historical data."""
        if self.use_ml:
            self.train_ml_model()
    
    def get_latest_signal(self, symbol: str) -> Optional[PEADSignal]:
        """Get the latest signal for a symbol."""
        for signal in reversed(self.signals):
            if signal.symbol == symbol:
                return signal
        return None
    
    def print_pead_report(self) -> None:
        """Print PEAD analysis report."""
        print("\n" + "="*60)
        print("PEAD EARNINGS STREAK + ML ALPHA REPORT")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Drift Period: {self.drift_period_days} days")
        print(f"  Surprise Threshold: {self.surprise_threshold:.2%}")
        print(f"  Streak Lookback: {self.streak_lookback}")
        print(f"  Min Streak Length: {self.min_streak_length}")
        print(f"  Use ML: {self.use_ml}")
        
        print(f"\nStatistics:")
        print(f"  Total Earnings Events: {sum(len(events) for events in self.earnings_history.values())}")
        print(f"  Total Signals: {len(self.signals)}")
        
        if self.signals:
            positive_signals = [s for s in self.signals if s.direction == SurpriseDirection.POSITIVE]
            negative_signals = [s for s in self.signals if s.direction == SurpriseDirection.NEGATIVE]
            
            print(f"\nSignal Distribution:")
            print(f"  Positive Surprises: {len(positive_signals)}")
            print(f"  Negative Surprises: {len(negative_signals)}")
            
            if self.signals:
                avg_surprise = np.mean([abs(s.surprise_pct) for s in self.signals])
                avg_streak = np.mean([s.streak_length for s in self.signals])
                avg_ml_score = np.mean([s.ml_score for s in self.signals]) if self.use_ml else 0.0
                
                print(f"\nSignal Quality:")
                print(f"  Average Surprise: {avg_surprise:.2%}")
                print(f"  Average Streak Length: {avg_streak:.2f}")
                if self.use_ml:
                    print(f"  Average ML Score: {avg_ml_score:.4f}")
            
            print(f"\nRecent Signals:")
            print(f"{'Timestamp':<20} {'Symbol':<10} {'Direction':<10} {'Surprise':<10} {'Streak':<8} {'ML Score':<10} {'Signal':<10} {'Confidence':<12}")
            print("-" * 105)
            
            for signal in self.signals[-5:]:
                print(f"{signal.timestamp.strftime('%Y-%m-%d'):<20} {signal.symbol:<10} "
                      f"{signal.direction.value:<10} {signal.surprise_pct:<10.2%} {signal.streak_length:<8} "
                      f"{signal.ml_score:<10.4f} {signal.signal:<10.3f} {signal.confidence:<12.2f}")
        
        if self.use_ml and len(self.features) > 0:
            print(f"\nML Training Data:")
            print(f"  Samples: {len(self.features)}")
            print(f"  Features: {len(self.features[0]) if self.features else 0}")
        
        print("\n" + "="*60)


def sample_pead_earnings_streak_alpha():
    """Demonstrate PEAD earnings streak alpha."""
    print("=== PEAD Earnings Streak + ML Alpha Demo ===\n")
    
    # Initialize alpha
    alpha = PEADEarningsStreakAlpha(
        drift_period_days=60,
        surprise_threshold=0.05,
        streak_lookback=4,
        min_streak_length=2,
        use_ml=True
    )
    
    # Generate sample earnings data
    np.random.seed(42)
    n_earnings = 20
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=n_earnings, freq='20D')
    base_eps = 10.0
    
    # Generate earnings with streak pattern
    eps_estimates = base_eps + np.random.randn(n_earnings) * 0.5
    eps_actuals = eps_estimates.copy()
    
    # Create streak pattern
    for i in range(5, n_earnings):
        if i < 10:
            eps_actuals[i] += 0.8  # Positive streak
        elif i < 15:
            eps_actuals[i] -= 0.8  # Negative streak
    
    pre_prices = 1000 + np.random.randn(n_earnings) * 50
    post_prices = pre_prices * (1 + np.random.randn(n_earnings) * 0.02)
    
    # Process earnings
    print("Processing earnings data...")
    for i in range(n_earnings):
        signal = alpha.generate_signal(
            'RELIANCE',
            eps_actuals[i],
            eps_estimates[i],
            pre_prices[i],
            post_prices[i],
            dates[i]
        )
        
        # Simulate drift return for training
        if signal and i > 0:
            drift_return = np.random.normal(signal.signal * 0.05, 0.02)
            alpha.update_drift_return('RELIANCE', dates[i], drift_return)
    
    # Train ML model
    alpha.train_on_historical_data()
    
    # Print report
    alpha.print_pead_report()
    
    print("\n=== PEAD Earnings Streak + ML Alpha Demo Complete ===")
    print("Key capabilities:")
    print("- Earnings surprise calculation")
    print("- Earnings streak tracking")
    print("- ML-enhanced drift prediction")
    print("- Combined traditional + ML signal generation")
    print("- Streak-based holding period adjustment")
    print("- Expected Sharpe: 0.4-0.7")
    print("- Expected Capacity: Very High")
    print("- Decay: Years")


if __name__ == "__main__":
    sample_pead_earnings_streak_alpha()

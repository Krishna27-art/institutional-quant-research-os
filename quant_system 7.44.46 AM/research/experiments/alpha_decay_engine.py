"""
Alpha Decay Engine, Crowding Detector, Capacity Model
Based on the critique: Build Market Efficiency tools to understand edge decay

Market Efficiency lesson: If everybody knows an edge, edge dies.

Need to measure:
- Alpha half-life
- Capacity limits
- Crowding levels
- Edge decay over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import curve_fit


@dataclass
class AlphaDecayMetrics:
    """Metrics for alpha decay analysis."""
    alpha_name: str
    start_date: datetime
    end_date: datetime
    initial_sharpe: float
    current_sharpe: float
    decay_rate: float
    half_life_days: float
    is_decayed: bool
    decay_stage: str  # "growth", "maturity", "decay", "death"


@dataclass
class CrowdingMetrics:
    """Metrics for crowding analysis."""
    alpha_name: str
    timestamp: datetime
    crowding_score: float  # 0 to 1
    estimated_participants: int
    capacity_utilization: float
    is_overcrowded: bool
    estimated_capacity_remaining: float


@dataclass
class CapacityMetrics:
    """Metrics for capacity analysis."""
    alpha_name: str
    max_capacity: float  # Maximum AUM
    current_aum: float  # Current AUM
    capacity_utilization: float
    slippage_at_current: float
    slippage_at_max: float
    is_at_capacity: bool


class AlphaDecayEngine:
    """
    Alpha Decay Engine for measuring edge decay over time.
    
    Features:
    - Alpha half-life calculation
    - Decay rate estimation
    - Decay stage classification
    - Crowding detection
    - Capacity modeling
    """
    
    def __init__(self):
        self.decay_history: Dict[str, List[AlphaDecayMetrics]] = {}
        self.crowding_history: Dict[str, List[CrowdingMetrics]] = {}
        self.capacity_history: Dict[str, List[CapacityMetrics]] = {}
        
        # Decay thresholds
        self.decay_threshold = 0.5  # 50% Sharpe decay
        self.death_threshold = 0.2  # 20% of original Sharpe
        self.crowding_threshold = 0.7  # 70% crowding score
        self.capacity_threshold = 0.8  # 80% capacity utilization
    
    def calculate_decay_rate(
        self,
        returns: pd.Series,
        window_days: int = 60
    ) -> float:
        """
        Calculate exponential decay rate of alpha.
        
        Uses rolling Sharpe to estimate decay.
        
        Args:
            returns: Strategy returns
            window_days: Window for rolling Sharpe
            
        Returns:
            Decay rate (negative if decaying)
        """
        if len(returns) < window_days * 2:
            return 0.0
        
        # Calculate rolling Sharpe
        rolling_sharpe = returns.rolling(window_days).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Fit exponential decay
        x = np.arange(len(rolling_sharpe))
        y = rolling_sharpe.dropna().values
        
        if len(y) < 10:
            return 0.0
        
        # Normalize y
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
        
        def exp_decay(x, a, b):
            return a * np.exp(b * x)
        
        try:
            popt, _ = curve_fit(exp_decay, x, y_norm, p0=[1.0, -0.01], maxfev=10000)
            decay_rate = popt[1]
            return decay_rate
        except:
            return 0.0
    
    def calculate_half_life(self, decay_rate: float) -> float:
        """
        Calculate half-life of alpha.
        
        Half-life = ln(0.5) / decay_rate
        
        Args:
            decay_rate: Exponential decay rate
            
        Returns:
            Half-life in days
        """
        if decay_rate >= 0:
            return float('inf')  # No decay
        
        half_life = np.log(0.5) / decay_rate
        return half_life
    
    def classify_decay_stage(
        self,
        initial_sharpe: float,
        current_sharpe: float,
        decay_rate: float
    ) -> str:
        """
        Classify decay stage of alpha.
        
        Stages:
        - Growth: Sharpe increasing
        - Maturity: Stable Sharpe
        - Decay: Sharpe declining but > 50% of initial
        - Death: Sharpe < 20% of initial
        
        Args:
            initial_sharpe: Initial Sharpe ratio
            current_sharpe: Current Sharpe ratio
            decay_rate: Decay rate
            
        Returns:
            Decay stage
        """
        sharpe_ratio = current_sharpe / initial_sharpe if initial_sharpe > 0 else 0
        
        if decay_rate > 0:
            return "growth"
        elif sharpe_ratio > 0.8:
            return "maturity"
        elif sharpe_ratio > 0.2:
            return "decay"
        else:
            return "death"
    
    def analyze_alpha_decay(
        self,
        alpha_name: str,
        returns: pd.Series,
        start_date: datetime,
        end_date: datetime
    ) -> AlphaDecayMetrics:
        """
        Analyze alpha decay over time.
        
        Args:
            alpha_name: Name of alpha
            returns: Strategy returns
            start_date: Start date
            end_date: End date
            
        Returns:
            AlphaDecayMetrics
        """
        # Calculate initial Sharpe (first 60 days)
        initial_returns = returns.iloc[:60]
        initial_sharpe = initial_returns.mean() / initial_returns.std() * np.sqrt(252) if initial_returns.std() > 0 else 0
        
        # Calculate current Sharpe (last 60 days)
        current_returns = returns.iloc[-60:]
        current_sharpe = current_returns.mean() / current_returns.std() * np.sqrt(252) if current_returns.std() > 0 else 0
        
        # Calculate decay rate
        decay_rate = self.calculate_decay_rate(returns)
        
        # Calculate half-life
        half_life = self.calculate_half_life(decay_rate)
        
        # Check if decayed
        sharpe_ratio = current_sharpe / initial_sharpe if initial_sharpe > 0 else 0
        is_decayed = sharpe_ratio < self.decay_threshold
        
        # Classify decay stage
        decay_stage = self.classify_decay_stage(initial_sharpe, current_sharpe, decay_rate)
        
        metrics = AlphaDecayMetrics(
            alpha_name=alpha_name,
            start_date=start_date,
            end_date=end_date,
            initial_sharpe=initial_sharpe,
            current_sharpe=current_sharpe,
            decay_rate=decay_rate,
            half_life_days=half_life,
            is_decayed=is_decayed,
            decay_stage=decay_stage
        )
        
        # Store in history
        if alpha_name not in self.decay_history:
            self.decay_history[alpha_name] = []
        self.decay_history[alpha_name].append(metrics)
        
        return metrics
    
    def detect_crowding(
        self,
        alpha_name: str,
        returns: pd.Series,
        volume: pd.Series,
        market_volume: pd.Series,
        estimated_participants: int = 100
    ) -> CrowdingMetrics:
        """
        Detect crowding in alpha.
        
        Crowding indicators:
        - Returns becoming more correlated with market
        - Volume increasing relative to market
        - Sharpe declining
        
        Args:
            alpha_name: Name of alpha
            returns: Strategy returns
            volume: Strategy volume
            market_volume: Market volume
            estimated_participants: Estimated number of participants
            
        Returns:
            CrowdingMetrics
        """
        # Calculate correlation with market
        correlation = returns.corr(market_volume.pct_change())
        
        # Calculate volume ratio
        volume_ratio = volume.mean() / market_volume.mean()
        
        # Calculate Sharpe trend
        sharpe_trend = self.calculate_decay_rate(returns)
        
        # Crowding score (0 to 1)
        # Higher correlation, higher volume ratio, negative Sharpe trend = more crowded
        crowding_score = (
            abs(correlation) * 0.4 +
            min(volume_ratio / 0.1, 1.0) * 0.3 +
            min(abs(sharpe_trend) / 0.01, 1.0) * 0.3
        )
        
        # Capacity utilization
        capacity_utilization = min(volume_ratio / 0.05, 1.0)  # Assume 5% of market volume is capacity
        
        # Check if overcrowded
        is_overcrowded = crowding_score > self.crowding_threshold
        
        # Estimate remaining capacity
        remaining_capacity = max(0, 1 - capacity_utilization)
        
        metrics = CrowdingMetrics(
            alpha_name=alpha_name,
            timestamp=datetime.now(),
            crowding_score=crowding_score,
            estimated_participants=estimated_participants,
            capacity_utilization=capacity_utilization,
            is_overcrowded=is_overcrowded,
            estimated_capacity_remaining=remaining_capacity
        )
        
        # Store in history
        if alpha_name not in self.crowding_history:
            self.crowding_history[alpha_name] = []
        self.crowding_history[alpha_name].append(metrics)
        
        return metrics
    
    def estimate_capacity(
        self,
        alpha_name: str,
        returns: pd.Series,
        volume: pd.Series,
        avg_trade_size: float = 100000
    ) -> CapacityMetrics:
        """
        Estimate capacity of alpha.
        
        Capacity is limited by:
        - Market depth
        - Slippage
        - Price impact
        
        Args:
            alpha_name: Name of alpha
            returns: Strategy returns
            volume: Trading volume
            avg_trade_size: Average trade size
            
        Returns:
            CapacityMetrics
        """
        # Calculate daily volume
        daily_volume = volume.mean()
        
        # Estimate capacity based on volume participation
        # Assume max 1% of daily volume without significant impact
        max_capacity = daily_volume * 0.01 * 252  # Annualized
        
        # Current AUM (simplified)
        current_aum = avg_trade_size * len(returns)
        
        # Capacity utilization
        capacity_utilization = min(current_aum / max_capacity, 1.0)
        
        # Estimate slippage at current size
        # Slippage increases with square root of size
        slippage_current = 0.001 * np.sqrt(capacity_utilization)
        
        # Estimate slippage at max capacity
        slippage_max = 0.001 * np.sqrt(1.0)
        
        # Check if at capacity
        is_at_capacity = capacity_utilization > self.capacity_threshold
        
        metrics = CapacityMetrics(
            alpha_name=alpha_name,
            max_capacity=max_capacity,
            current_aum=current_aum,
            capacity_utilization=capacity_utilization,
            slippage_at_current=slippage_current,
            slippage_at_max=slippage_max,
            is_at_capacity=is_at_capacity
        )
        
        # Store in history
        if alpha_name not in self.capacity_history:
            self.capacity_history[alpha_name] = []
        self.capacity_history[alpha_name].append(metrics)
        
        return metrics
    
    def get_decay_summary(self) -> pd.DataFrame:
        """Get summary of alpha decay for all alphas."""
        data = []
        
        for alpha_name, metrics_list in self.decay_history.items():
            latest = metrics_list[-1]
            data.append({
                'Alpha': alpha_name,
                'Initial Sharpe': latest.initial_sharpe,
                'Current Sharpe': latest.current_sharpe,
                'Decay Rate': latest.decay_rate,
                'Half-Life (days)': latest.half_life_days,
                'Stage': latest.decay_stage,
                'Decayed': latest.is_decayed
            })
        
        return pd.DataFrame(data)
    
    def get_crowding_summary(self) -> pd.DataFrame:
        """Get summary of crowding for all alphas."""
        data = []
        
        for alpha_name, metrics_list in self.crowding_history.items():
            latest = metrics_list[-1]
            data.append({
                'Alpha': alpha_name,
                'Crowding Score': latest.crowding_score,
                'Participants': latest.estimated_participants,
                'Capacity Utilization': latest.capacity_utilization,
                'Overcrowded': latest.is_overcrowded,
                'Remaining Capacity': latest.estimated_capacity_remaining
            })
        
        return pd.DataFrame(data)
    
    def get_capacity_summary(self) -> pd.DataFrame:
        """Get summary of capacity for all alphas."""
        data = []
        
        for alpha_name, metrics_list in self.capacity_history.items():
            latest = metrics_list[-1]
            data.append({
                'Alpha': alpha_name,
                'Max Capacity': latest.max_capacity,
                'Current AUM': latest.current_aum,
                'Utilization': latest.capacity_utilization,
                'Slippage Current': latest.slippage_at_current,
                'Slippage at Max': latest.slippage_at_max,
                'At Capacity': latest.is_at_capacity
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the Alpha Decay Engine
    print("Testing Alpha Decay Engine, Crowding Detector, Capacity Model...")
    
    engine = AlphaDecayEngine()
    
    # Generate sample returns with decay
    print("\nGenerating sample returns with decay...")
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    # Returns that decay over time
    decay_factor = np.exp(-np.linspace(0, 0.005, n))
    returns = pd.Series(np.random.normal(0.001, 0.02, n) * decay_factor, index=dates)
    
    volume = pd.Series(np.random.normal(1000000, 200000, n), index=dates)
    market_volume = pd.Series(np.random.normal(100000000, 10000000, n), index=dates)
    
    # Analyze alpha decay
    print("\nAnalyzing Alpha Decay...")
    decay_metrics = engine.analyze_alpha_decay("Sample Alpha", returns, dates[0], dates[-1])
    
    print(f"Initial Sharpe: {decay_metrics.initial_sharpe:.2f}")
    print(f"Current Sharpe: {decay_metrics.current_sharpe:.2f}")
    print(f"Decay Rate: {decay_metrics.decay_rate:.4f}")
    print(f"Half-Life: {decay_metrics.half_life_days:.0f} days")
    print(f"Decay Stage: {decay_metrics.decay_stage}")
    print(f"Is Decayed: {decay_metrics.is_decayed}")
    
    # Detect crowding
    print("\nDetecting Crowding...")
    crowding_metrics = engine.detect_crowding("Sample Alpha", returns, volume, market_volume, estimated_participants=150)
    
    print(f"Crowding Score: {crowding_metrics.crowding_score:.2%}")
    print(f"Estimated Participants: {crowding_metrics.estimated_participants}")
    print(f"Capacity Utilization: {crowding_metrics.capacity_utilization:.2%}")
    print(f"Overcrowded: {crowding_metrics.is_overcrowded}")
    print(f"Remaining Capacity: {crowding_metrics.estimated_capacity_remaining:.2%}")
    
    # Estimate capacity
    print("\nEstimating Capacity...")
    capacity_metrics = engine.estimate_capacity("Sample Alpha", returns, volume, avg_trade_size=1000000)
    
    print(f"Max Capacity: {capacity_metrics.max_capacity:.0f}")
    print(f"Current AUM: {capacity_metrics.current_aum:.0f}")
    print(f"Capacity Utilization: {capacity_metrics.capacity_utilization:.2%}")
    print(f"Slippage at Current: {capacity_metrics.slippage_at_current:.2%}")
    print(f"Slippage at Max: {capacity_metrics.slippage_at_max:.2%}")
    print(f"At Capacity: {capacity_metrics.is_at_capacity}")
    
    # Get summaries
    print("\nDecay Summary:")
    decay_summary = engine.get_decay_summary()
    print(decay_summary.to_string(index=False))
    
    print("\nCrowding Summary:")
    crowding_summary = engine.get_crowding_summary()
    print(crowding_summary.to_string(index=False))
    
    print("\nCapacity Summary:")
    capacity_summary = engine.get_capacity_summary()
    print(capacity_summary.to_string(index=False))

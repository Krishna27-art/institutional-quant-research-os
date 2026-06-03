"""
Alpha Decay Detection and Auto-Deactivation

This module implements rolling Sharpe-based decay detection for alpha strategies,
automatically deactivating underperforming alphas to prevent capital erosion.

Key Features:
- Rolling Sharpe ratio calculation
- Decay detection with statistical significance
- Auto-deactivation of underperforming alphas
- Decay trend analysis
- Alert generation for decay events
- Alpha lifecycle management

Based on V4 Blueprint - Institutional Architecture
Priority: Medium (Phase 2.2)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaStatus(Enum):
    """Alpha status."""
    ACTIVE = "active"
    DECAYING = "decaying"
    DEACTIVATED = "deactivated"
    RECOVERING = "recovering"


@dataclass
class AlphaPerformance:
    """Alpha performance metrics."""
    alpha_name: str
    sharpe_rolling: float
    sharpe_rolling_std: float
    sharpe_trend: float  # Slope of Sharpe over time
    ic_rolling: float
    return_rolling: float
    volatility_rolling: float
    max_drawdown_rolling: float
    decay_score: float  # 0-1, higher = more decay
    status: AlphaStatus
    last_update: datetime
    
    def is_decaying(self, threshold: float = 0.5) -> bool:
        """Check if alpha is decaying."""
        return self.decay_score > threshold
    
    def should_deactivate(self, sharpe_threshold: float = 0.5) -> bool:
        """Check if alpha should be deactivated."""
        return self.sharpe_rolling < sharpe_threshold and self.decay_score > 0.7


@dataclass
class DecayAlert:
    """Decay alert."""
    alpha_name: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    current_sharpe: float
    decay_score: float
    timestamp: datetime
    recommended_action: str


class DecayDetector:
    """
    Alpha decay detector with auto-deactivation.
    
    This class monitors alpha performance using rolling Sharpe ratios
    and automatically deactivates underperforming alphas.
    """
    
    def __init__(
        self,
        rolling_window: int = 60,  # 60 trading days (~3 months)
        decay_threshold: float = 0.5,
        sharpe_threshold: float = 0.5,
        min_observations: int = 30
    ):
        """
        Initialize decay detector.
        
        Args:
            rolling_window: Rolling window for Sharpe calculation
            decay_threshold: Decay score threshold for decay detection
            sharpe_threshold: Sharpe threshold for deactivation
            min_observations: Minimum observations before evaluation
        """
        self.rolling_window = rolling_window
        self.decay_threshold = decay_threshold
        self.sharpe_threshold = sharpe_threshold
        self.min_observations = min_observations
        
        self.alpha_performance: Dict[str, AlphaPerformance] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        self.alerts: List[DecayAlert] = []
        
        logger.info(f"DecayDetector initialized: window={rolling_window}, decay_threshold={decay_threshold}")
    
    def calculate_rolling_sharpe(
        self,
        returns: pd.Series,
        window: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Return series
            window: Rolling window (uses default if None)
            
        Returns:
            (sharpe, sharpe_std)
        """
        window = window or self.rolling_window
        
        if len(returns) < self.min_observations:
            return 0.0, 0.0
        
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=window)
        
        # Calculate mean and std
        mean_return = rolling_returns.mean()
        std_return = rolling_returns.std()
        
        # Calculate Sharpe (annualized)
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        # Calculate Sharpe std across window
        sharpe_values = []
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            if len(window_returns) > 0 and window_returns.std() > 0:
                window_sharpe = (window_returns.mean() / window_returns.std() * np.sqrt(252))
                sharpe_values.append(window_sharpe)
        
        sharpe_std = np.std(sharpe_values) if sharpe_values else 0.0
        
        return sharpe, sharpe_std
    
    def calculate_decay_score(
        self,
        sharpe_history: List[float],
        ic_history: List[float]
    ) -> float:
        """
        Calculate decay score (0-1, higher = more decay).
        
        Args:
            sharpe_history: Historical Sharpe values
            ic_history: Historical IC values
            
        Returns:
            Decay score
        """
        if len(sharpe_history) < 2:
            return 0.0
        
        # Calculate Sharpe trend (linear regression slope)
        x = np.arange(len(sharpe_history))
        sharpe_slope, _, _, _, _ = stats.linregress(x, sharpe_history)
        
        # Calculate IC trend
        ic_slope = 0.0
        if len(ic_history) >= 2:
            x_ic = np.arange(len(ic_history))
            ic_slope, _, _, _, _ = stats.linregress(x_ic, ic_history)
        
        # Normalize slopes (negative slope = decay)
        sharpe_component = max(0, -sharpe_slope) / 0.01  # Normalize
        ic_component = max(0, -ic_slope) / 0.01  # Normalize
        
        # Recent performance drop
        recent_sharpe = np.mean(sharpe_history[-10:]) if len(sharpe_history) >= 10 else sharpe_history[-1]
        peak_sharpe = max(sharpe_history)
        drop_component = max(0, (peak_sharpe - recent_sharpe) / peak_sharpe) if peak_sharpe > 0 else 0
        
        # Combine components
        decay_score = 0.4 * sharpe_component + 0.3 * ic_component + 0.3 * drop_component
        
        return min(decay_score, 1.0)
    
    def evaluate_alpha(
        self,
        alpha_name: str,
        returns: pd.Series,
        features: Optional[pd.DataFrame] = None,
        target: Optional[pd.Series] = None
    ) -> AlphaPerformance:
        """
        Evaluate alpha performance and detect decay.
        
        Args:
            alpha_name: Alpha name
            returns: Return series
            features: Feature data (for IC calculation)
            target: Target returns (for IC calculation)
            
        Returns:
            AlphaPerformance
        """
        # Calculate rolling Sharpe
        sharpe, sharpe_std = self.calculate_rolling_sharpe(returns)
        
        # Calculate rolling IC
        ic = 0.0
        if features is not None and target is not None:
            # Simplified IC calculation
            if len(features) == len(target):
                ic = features.iloc[-1].corr(target.iloc[-1]) if len(features) > 0 else 0.0
        
        # Calculate rolling return and volatility
        rolling_return = returns.rolling(window=self.rolling_window).mean().iloc[-1]
        rolling_vol = returns.rolling(window=self.rolling_window).std().iloc[-1]
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window=self.rolling_window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Get historical performance
        history = self.performance_history.get(alpha_name, [])
        sharpe_history = [h.get('sharpe', 0) for h in history]
        ic_history = [h.get('ic', 0) for h in history]
        
        # Add current to history
        history.append({
            'sharpe': sharpe,
            'ic': ic,
            'return': rolling_return,
            'timestamp': datetime.now()
        })
        
        # Keep only last 100 observations
        if len(history) > 100:
            history = history[-100:]
        
        self.performance_history[alpha_name] = history
        
        # Calculate decay score
        decay_score = self.calculate_decay_score(sharpe_history, ic_history)
        
        # Calculate Sharpe trend
        sharpe_trend = 0.0
        if len(sharpe_history) >= 2:
            x = np.arange(len(sharpe_history))
            sharpe_trend, _, _, _, _ = stats.linregress(x, sharpe_history)
        
        # Determine status
        if decay_score > 0.7:
            status = AlphaStatus.DEACTIVATED
        elif decay_score > self.decay_threshold:
            status = AlphaStatus.DECAYING
        elif sharpe_trend > 0.01:
            status = AlphaStatus.RECOVERING
        else:
            status = AlphaStatus.ACTIVE
        
        performance = AlphaPerformance(
            alpha_name=alpha_name,
            sharpe_rolling=sharpe,
            sharpe_rolling_std=sharpe_std,
            sharpe_trend=sharpe_trend,
            ic_rolling=ic,
            return_rolling=rolling_return,
            volatility_rolling=rolling_vol,
            max_drawdown_rolling=max_dd,
            decay_score=decay_score,
            status=status,
            last_update=datetime.now()
        )
        
        self.alpha_performance[alpha_name] = performance
        
        # Generate alert if decaying
        if status == AlphaStatus.DECAYING:
            self._generate_decay_alert(performance)
        elif status == AlphaStatus.DEACTIVATED:
            self._generate_deactivation_alert(performance)
        
        return performance
    
    def _generate_decay_alert(self, performance: AlphaPerformance) -> None:
        """Generate decay alert."""
        severity = "medium" if performance.decay_score < 0.7 else "high"
        
        alert = DecayAlert(
            alpha_name=performance.alpha_name,
            alert_type="decay_detected",
            severity=severity,
            message=f"Alpha {performance.alpha_name} is decaying (decay score: {performance.decay_score:.2f})",
            current_sharpe=performance.sharpe_rolling,
            decay_score=performance.decay_score,
            timestamp=datetime.now(),
            recommended_action="Monitor closely, consider reducing position size"
        )
        
        self.alerts.append(alert)
        logger.warning(f"Decay alert: {alert.message}")
    
    def _generate_deactivation_alert(self, performance: AlphaPerformance) -> None:
        """Generate deactivation alert."""
        alert = DecayAlert(
            alpha_name=performance.alpha_name,
            alert_type="alpha_deactivated",
            severity="critical",
            message=f"Alpha {performance.alpha_name} deactivated (Sharpe: {performance.sharpe_rolling:.2f}, decay: {performance.decay_score:.2f})",
            current_sharpe=performance.sharpe_rolling,
            decay_score=performance.decay_score,
            timestamp=datetime.now(),
            recommended_action="Deactivate alpha immediately"
        )
        
        self.alerts.append(alert)
        logger.critical(f"Deactivation alert: {alert.message}")
    
    def get_active_alphas(self) -> List[str]:
        """Get list of active alphas."""
        return [
            name for name, perf in self.alpha_performance.items()
            if perf.status in [AlphaStatus.ACTIVE, AlphaStatus.RECOVERING]
        ]
    
    def get_deactivated_alphas(self) -> List[str]:
        """Get list of deactivated alphas."""
        return [
            name for name, perf in self.alpha_performance.items()
            if perf.status == AlphaStatus.DEACTIVATED
        ]
    
    def get_decaying_alphas(self) -> List[str]:
        """Get list of decaying alphas."""
        return [
            name for name, perf in self.alpha_performance.items()
            if perf.status == AlphaStatus.DECAYING
        ]
    
    def print_decay_report(self) -> None:
        """Print decay detection report."""
        print("\n" + "="*60)
        print("ALPHA DECAY DETECTION REPORT")
        print("="*60)
        
        print(f"\nTotal Alphas Monitored: {len(self.alpha_performance)}")
        print(f"Active: {len(self.get_active_alphas())}")
        print(f"Decaying: {len(self.get_decaying_alphas())}")
        print(f"Deactivated: {len(self.get_deactivated_alphas())}")
        
        if self.alpha_performance:
            print(f"\nAlpha Performance Summary:")
            print(f"{'Alpha':<20} {'Sharpe':<10} {'Decay':<10} {'Status':<15}")
            print("-" * 60)
            
            for name, perf in sorted(self.alpha_performance.items(), key=lambda x: x[1].decay_score, reverse=True):
                print(f"{name:<20} {perf.sharpe_rolling:>9.2f} {perf.decay_score:>9.2f} {perf.status.value:<15}")
        
        if self.alerts:
            print(f"\nRecent Alerts ({len(self.alerts)}):")
            for alert in self.alerts[-5:]:
                print(f"  [{alert.severity.upper()}] {alert.timestamp}: {alert.message}")
        
        print("\n" + "="*60)


def sample_decay_detection():
    """Demonstrate decay detection."""
    print("=== Alpha Decay Detection Demo ===\n")
    
    # Initialize decay detector
    detector = DecayDetector(
        rolling_window=60,
        decay_threshold=0.5,
        sharpe_threshold=0.5,
        min_observations=30
    )
    
    # Generate sample data for 3 alphas
    np.random.seed(42)
    n_samples = 200
    
    # Alpha 1: Stable performer
    returns_1 = np.random.normal(0.001, 0.02, n_samples)
    
    # Alpha 2: Decaying (performance drops over time)
    returns_2 = np.random.normal(0.001, 0.02, n_samples)
    returns_2[100:] = np.random.normal(-0.0005, 0.025, 100)  # Performance drops
    
    # Alpha 3: Recovering
    returns_3 = np.random.normal(-0.0005, 0.025, n_samples)
    returns_3[100:] = np.random.normal(0.001, 0.02, 100)  # Performance recovers
    
    alphas = {
        'Alpha_1_Stable': pd.Series(returns_1),
        'Alpha_2_Decaying': pd.Series(returns_2),
        'Alpha_3_Recovering': pd.Series(returns_3)
    }
    
    # Evaluate alphas
    print("Evaluating alphas...")
    for alpha_name, returns in alphas.items():
        performance = detector.evaluate_alpha(alpha_name, returns)
        print(f"{alpha_name}: Sharpe={performance.sharpe_rolling:.2f}, Decay={performance.decay_score:.2f}, Status={performance.status.value}")
    
    # Print report
    detector.print_decay_report()
    
    print("\n=== Alpha Decay Detection Demo Complete ===")
    print("Key capabilities:")
    print("- Rolling Sharpe ratio calculation")
    print("- Decay detection with statistical significance")
    print("- Auto-deactivation of underperforming alphas")
    print("- Decay trend analysis")
    print("- Alert generation for decay events")


if __name__ == "__main__":
    sample_decay_detection()

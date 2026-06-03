"""
Crowding Monitor
Track factor z-scores to detect crowded trades.

Critical for institutional risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CrowdingLevel(Enum):
    """Crowding severity levels"""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class FactorExposure:
    """Factor exposure for a strategy"""
    factor_name: str
    exposure: float
    z_score: float
    percentile: float
    is_crowded: bool


@dataclass
class CrowdingAlert:
    """Crowding alert for a factor"""
    factor_name: str
    crowding_level: CrowdingLevel
    z_score: float
    threshold: float
    timestamp: datetime
    affected_strategies: List[str]


class CrowdingMonitor:
    """
    Crowding Monitor
    
    Tracks factor z-scores to detect crowded trades.
    
    Rules:
    - Z-score > 2: Moderate crowding
    - Z-score > 2.5: High crowding
    - Z-score > 3: Extreme crowding
    - Reduce weight on crowded factors
    """
    
    def __init__(self, moderate_threshold: float = 2.0,
                 high_threshold: float = 2.5,
                 extreme_threshold: float = 3.0):
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold
        self.extreme_threshold = extreme_threshold
        
        self.factor_history: Dict[str, List[float]] = {}
        self.alerts: List[CrowdingAlert] = []
        self.strategy_exposures: Dict[str, Dict[str, float]] = {}
    
    def add_factor_history(self, factor_name: str, exposure: float):
        """Add factor exposure to history"""
        if factor_name not in self.factor_history:
            self.factor_history[factor_name] = []
        
        self.factor_history[factor_name].append(exposure)
        
        # Keep last 252 observations (1 year)
        if len(self.factor_history[factor_name]) > 252:
            self.factor_history[factor_name] = self.factor_history[factor_name][-252:]
    
    def calculate_z_score(self, factor_name: str) -> Optional[float]:
        """Calculate z-score for factor"""
        if factor_name not in self.factor_history:
            return None
        
        history = self.factor_history[factor_name]
        if len(history) < 30:
            return None
        
        mean = np.mean(history)
        std = np.std(history)
        
        if std == 0:
            return 0.0
        
        current = history[-1]
        z_score = (current - mean) / std
        
        return z_score
    
    def calculate_percentile(self, factor_name: str) -> Optional[float]:
        """Calculate percentile for factor"""
        if factor_name not in self.factor_history:
            return None
        
        history = self.factor_history[factor_name]
        if len(history) < 30:
            return None
        
        current = history[-1]
        percentile = (sum(1 for h in history if h <= current) / len(history)) * 100
        
        return percentile
    
    def get_crowding_level(self, z_score: float) -> CrowdingLevel:
        """Get crowding level from z-score"""
        if abs(z_score) < self.moderate_threshold:
            return CrowdingLevel.NONE
        elif abs(z_score) < self.high_threshold:
            return CrowdingLevel.MODERATE
        elif abs(z_score) < self.extreme_threshold:
            return CrowdingLevel.HIGH
        else:
            return CrowdingLevel.EXTREME
    
    def update_strategy_exposure(self, strategy_id: str, factor_exposures: Dict[str, float]):
        """Update factor exposures for a strategy"""
        self.strategy_exposures[strategy_id] = factor_exposures
        
        # Add to factor history
        for factor_name, exposure in factor_exposures.items():
            self.add_factor_history(factor_name, exposure)
    
    def check_crowding(self) -> List[CrowdingAlert]:
        """Check for crowding across all factors"""
        alerts = []
        
        for factor_name in self.factor_history:
            z_score = self.calculate_z_score(factor_name)
            if z_score is None:
                continue
            
            crowding_level = self.get_crowding_level(z_score)
            
            if crowding_level != CrowdingLevel.NONE:
                # Find affected strategies
                affected_strategies = []
                for strategy_id, exposures in self.strategy_exposures.items():
                    if factor_name in exposures and abs(exposures[factor_name]) > 0.1:
                        affected_strategies.append(strategy_id)
                
                alert = CrowdingAlert(
                    factor_name=factor_name,
                    crowding_level=crowding_level,
                    z_score=z_score,
                    threshold=self.moderate_threshold,
                    timestamp=datetime.now(),
                    affected_strategies=affected_strategies
                )
                
                alerts.append(alert)
                self.alerts.append(alert)
        
        return alerts
    
    def get_crowding_adjustment(self, factor_name: str) -> float:
        """
        Get crowding adjustment factor for a factor.
        
        Returns:
            Adjustment factor (0-1), where 1 = no adjustment, 0 = full reduction
        """
        z_score = self.calculate_z_score(factor_name)
        if z_score is None:
            return 1.0
        
        crowding_level = self.get_crowding_level(z_score)
        
        if crowding_level == CrowdingLevel.NONE:
            return 1.0
        elif crowding_level == CrowdingLevel.MODERATE:
            return 0.8
        elif crowding_level == CrowdingLevel.HIGH:
            return 0.5
        elif crowding_level == CrowdingLevel.EXTREME:
            return 0.2
        
        return 1.0
    
    def get_all_factor_exposures(self) -> List[FactorExposure]:
        """Get all factor exposures with z-scores"""
        exposures = []
        
        for factor_name in self.factor_history:
            z_score = self.calculate_z_score(factor_name)
            percentile = self.calculate_percentile(factor_name)
            
            if z_score is not None:
                is_crowded = self.get_crowding_level(z_score) != CrowdingLevel.NONE
                
                exposure = FactorExposure(
                    factor_name=factor_name,
                    exposure=self.factor_history[factor_name][-1],
                    z_score=z_score,
                    percentile=percentile if percentile else 50.0,
                    is_crowded=is_crowded
                )
                
                exposures.append(exposure)
        
        return exposures
    
    def generate_report(self) -> str:
        """Generate crowding report"""
        all_exposures = self.get_all_factor_exposures()
        recent_alerts = self.alerts[-10:] if self.alerts else []
        
        crowded_factors = [e for e in all_exposures if e.is_crowded]
        
        report = f"""
Crowding Monitor Report
{'=' * 50}
Moderate Threshold: {self.moderate_threshold}
High Threshold: {self.high_threshold}
Extreme Threshold: {self.extreme_threshold}

Factor Exposures:
{'-' * 50}
"""
        
        for exposure in all_exposures:
            status = "CROWDED" if exposure.is_crowded else "OK"
            report += f"{exposure.factor_name}: Z={exposure.z_score:.2f}, "
            report += f"Pct={exposure.percentile:.1f}%, [{status}]\n"
        
        report += f"\nCrowded Factors: {len(crowded_factors)}\n"
        
        if recent_alerts:
            report += f"\nRecent Alerts:\n{'-' * 50}\n"
            for alert in recent_alerts:
                report += f"{alert.factor_name}: {alert.crowding_level.value.upper()}\n"
                report += f"  Z-score: {alert.z_score:.2f}\n"
                report += f"  Affected strategies: {', '.join(alert.affected_strategies)}\n\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    monitor = CrowdingMonitor()
    
    # Simulate factor exposures over time
    print("Simulating factor exposures...")
    for day in range(100):
        # Momentum factor becoming crowded
        momentum_exposure = np.random.normal(0.5, 0.1) + day * 0.01
        monitor.add_factor_history("momentum", momentum_exposure)
        
        # Value factor stable
        value_exposure = np.random.normal(0.0, 0.1)
        monitor.add_factor_history("value", value_exposure)
        
        # Size factor becoming crowded
        size_exposure = np.random.normal(-0.3, 0.1) + day * 0.008
        monitor.add_factor_history("size", size_exposure)
    
    # Update strategy exposures
    monitor.update_strategy_exposure("momentum_strategy", {"momentum": 0.8, "value": 0.1, "size": 0.1})
    monitor.update_strategy_exposure("value_strategy", {"momentum": 0.1, "value": 0.8, "size": 0.1})
    monitor.update_strategy_exposure("size_strategy", {"momentum": 0.1, "value": 0.1, "size": 0.8})
    
    # Check for crowding
    alerts = monitor.check_crowding()
    
    print(f"\nDetected {len(alerts)} crowding alerts")
    print(monitor.generate_report())

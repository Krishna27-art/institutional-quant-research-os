"""
Alpha Lifecycle Management: Birth, Growth, Maturity, Decay, Death tracking
Based on the critique: Every alpha should have lifecycle management

Every alpha should have:
- Birth
- Growth
- Maturity
- Decay
- Death

Most retail systems never measure this.
Top firms do.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class LifecycleStage(Enum):
    """Lifecycle stages of an alpha."""
    BIRTH = "birth"
    GROWTH = "growth"
    MATURITY = "maturity"
    DECAY = "decay"
    DEATH = "death"


@dataclass
class AlphaLifecycle:
    """Alpha lifecycle data."""
    alpha_id: str
    alpha_name: str
    birth_date: datetime
    current_stage: LifecycleStage
    stage_start_date: datetime
    days_in_stage: int
    total_days: int
    initial_sharpe: float
    current_sharpe: float
    peak_sharpe: float
    sharpe_decline_from_peak: float
    is_active: bool
    retirement_reason: Optional[str]


@dataclass
class LifecycleEvent:
    """Lifecycle event."""
    alpha_id: str
    timestamp: datetime
    event_type: str  # "birth", "stage_change", "retirement"
    from_stage: Optional[LifecycleStage]
    to_stage: Optional[LifecycleStage]
    reason: str


class AlphaLifecycleManager:
    """
    Alpha Lifecycle Manager for tracking alpha lifecycle.
    
    Features:
    - Lifecycle stage tracking
    - Stage transition detection
    - Retirement decision
    - Lifecycle event logging
    - Performance monitoring across lifecycle
    """
    
    def __init__(self):
        self.lifecycles: Dict[str, AlphaLifecycle] = {}
        self.lifecycle_events: Dict[str, List[LifecycleEvent]] = {}
        
        # Stage thresholds
        self.growth_threshold = 0.8  # Sharpe > 0.8
        self.maturity_threshold = 1.2  # Sharpe > 1.2
        self.decay_threshold = 0.6  # Sharpe < 0.6
        self.death_threshold = 0.3  # Sharpe < 0.3
        
        # Stage duration thresholds (days)
        self.min_growth_duration = 30
        self.min_maturity_duration = 90
        self.max_decay_duration = 60
    
    def birth_alpha(
        self,
        alpha_id: str,
        alpha_name: str,
        initial_sharpe: float
    ) -> AlphaLifecycle:
        """
        Birth a new alpha.
        
        Args:
            alpha_id: Unique alpha ID
            alpha_name: Name of alpha
            initial_sharpe: Initial Sharpe ratio
            
        Returns:
            AlphaLifecycle
        """
        lifecycle = AlphaLifecycle(
            alpha_id=alpha_id,
            alpha_name=alpha_name,
            birth_date=datetime.now(),
            current_stage=LifecycleStage.BIRTH,
            stage_start_date=datetime.now(),
            days_in_stage=0,
            total_days=0,
            initial_sharpe=initial_sharpe,
            current_sharpe=initial_sharpe,
            peak_sharpe=initial_sharpe,
            sharpe_decline_from_peak=0.0,
            is_active=True,
            retirement_reason=None
        )
        
        self.lifecycles[alpha_id] = lifecycle
        
        # Log birth event
        self._log_event(
            alpha_id=alpha_id,
            event_type="birth",
            from_stage=None,
            to_stage=LifecycleStage.BIRTH,
            reason=f"Alpha born with initial Sharpe: {initial_sharpe:.2f}"
        )
        
        return lifecycle
    
    def update_alpha_performance(
        self,
        alpha_id: str,
        current_sharpe: float
    ) -> Optional[LifecycleStage]:
        """
        Update alpha performance and check for stage transitions.
        
        Args:
            alpha_id: Alpha ID
            current_sharpe: Current Sharpe ratio
            
        Returns:
            New stage if transition occurred, None otherwise
        """
        if alpha_id not in self.lifecycles:
            return None
        
        lifecycle = self.lifecycles[alpha_id]
        
        if not lifecycle.is_active:
            return None
        
        # Update current Sharpe
        lifecycle.current_sharpe = current_sharpe
        
        # Update peak Sharpe
        if current_sharpe > lifecycle.peak_sharpe:
            lifecycle.peak_sharpe = current_sharpe
        
        # Calculate decline from peak
        lifecycle.sharpe_decline_from_peak = (lifecycle.peak_sharpe - current_sharpe) / lifecycle.peak_sharpe
        
        # Update days
        lifecycle.total_days = (datetime.now() - lifecycle.birth_date).days
        lifecycle.days_in_stage = (datetime.now() - lifecycle.stage_start_date).days
        
        # Check for stage transitions
        new_stage = self._check_stage_transition(lifecycle)
        
        if new_stage and new_stage != lifecycle.current_stage:
            old_stage = lifecycle.current_stage
            lifecycle.current_stage = new_stage
            lifecycle.stage_start_date = datetime.now()
            lifecycle.days_in_stage = 0
            
            # Log stage change
            self._log_event(
                alpha_id=alpha_id,
                event_type="stage_change",
                from_stage=old_stage,
                to_stage=new_stage,
                reason=f"Stage transition: {old_stage.value} -> {new_stage.value}"
            )
            
            # Check for retirement
            if new_stage == LifecycleStage.DEATH:
                self.retire_alpha(alpha_id, reason="Sharpe below death threshold")
            
            return new_stage
        
        return None
    
    def _check_stage_transition(self, lifecycle: AlphaLifecycle) -> Optional[LifecycleStage]:
        """
        Check if alpha should transition to a new stage.
        
        Args:
            lifecycle: Alpha lifecycle
            
        Returns:
            New stage if transition should occur, None otherwise
        """
        current_stage = lifecycle.current_stage
        sharpe = lifecycle.current_sharpe
        days_in_stage = lifecycle.days_in_stage
        
        # Birth -> Growth
        if current_stage == LifecycleStage.BIRTH:
            if sharpe > self.growth_threshold and days_in_stage >= 7:
                return LifecycleStage.GROWTH
        
        # Growth -> Maturity
        elif current_stage == LifecycleStage.GROWTH:
            if sharpe > self.maturity_threshold and days_in_stage >= self.min_growth_duration:
                return LifecycleStage.MATURITY
            elif sharpe < self.decay_threshold and days_in_stage >= 14:
                return LifecycleStage.DECAY
        
        # Maturity -> Decay
        elif current_stage == LifecycleStage.MATURITY:
            if sharpe < self.decay_threshold and days_in_stage >= self.min_maturity_duration:
                return LifecycleStage.DECAY
        
        # Decay -> Death
        elif current_stage == LifecycleStage.DECAY:
            if sharpe < self.death_threshold and days_in_stage >= 30:
                return LifecycleStage.DEATH
            elif days_in_stage >= self.max_decay_duration:
                return LifecycleStage.DEATH
        
        return None
    
    def retire_alpha(
        self,
        alpha_id: str,
        reason: str
    ) -> bool:
        """
        Retire an alpha.
        
        Args:
            alpha_id: Alpha ID
            reason: Reason for retirement
            
        Returns:
            True if retired successfully
        """
        if alpha_id not in self.lifecycles:
            return False
        
        lifecycle = self.lifecycles[alpha_id]
        lifecycle.is_active = False
        lifecycle.retirement_reason = reason
        
        # Log retirement event
        self._log_event(
            alpha_id=alpha_id,
            event_type="retirement",
            from_stage=lifecycle.current_stage,
            to_stage=LifecycleStage.DEATH,
            reason=reason
        )
        
        return True
    
    def _log_event(
        self,
        alpha_id: str,
        event_type: str,
        from_stage: Optional[LifecycleStage],
        to_stage: Optional[LifecycleStage],
        reason: str
    ):
        """Log a lifecycle event."""
        event = LifecycleEvent(
            alpha_id=alpha_id,
            timestamp=datetime.now(),
            event_type=event_type,
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason
        )
        
        if alpha_id not in self.lifecycle_events:
            self.lifecycle_events[alpha_id] = []
        self.lifecycle_events[alpha_id].append(event)
    
    def get_lifecycle_summary(self) -> pd.DataFrame:
        """Get summary of all alpha lifecycles."""
        data = []
        
        for alpha_id, lifecycle in self.lifecycles.items():
            data.append({
                'Alpha ID': alpha_id,
                'Alpha Name': lifecycle.alpha_name,
                'Birth Date': lifecycle.birth_date.strftime('%Y-%m-%d'),
                'Current Stage': lifecycle.current_stage.value,
                'Days in Stage': lifecycle.days_in_stage,
                'Total Days': lifecycle.total_days,
                'Initial Sharpe': lifecycle.initial_sharpe,
                'Current Sharpe': lifecycle.current_sharpe,
                'Peak Sharpe': lifecycle.peak_sharpe,
                'Decline from Peak': lifecycle.sharpe_decline_from_peak,
                'Active': lifecycle.is_active,
                'Retirement Reason': lifecycle.retirement_reason
            })
        
        return pd.DataFrame(data)
    
    def get_active_alphas(self) -> List[AlphaLifecycle]:
        """Get all active alphas."""
        return [
            lifecycle for lifecycle in self.lifecycles.values()
            if lifecycle.is_active
        ]
    
    def get_retired_alphas(self) -> List[AlphaLifecycle]:
        """Get all retired alphas."""
        return [
            lifecycle for lifecycle in self.lifecycles.values()
            if not lifecycle.is_active
        ]
    
    def get_alpha_events(self, alpha_id: str) -> List[LifecycleEvent]:
        """Get lifecycle events for an alpha."""
        if alpha_id not in self.lifecycle_events:
            return []
        return self.lifecycle_events[alpha_id]
    
    def get_stage_distribution(self) -> Dict[str, int]:
        """Get distribution of alphas across stages."""
        distribution = {
            'birth': 0,
            'growth': 0,
            'maturity': 0,
            'decay': 0,
            'death': 0
        }
        
        for lifecycle in self.lifecycles.values():
            stage = lifecycle.current_stage.value
            distribution[stage] += 1
        
        return distribution


if __name__ == "__main__":
    # Test the Alpha Lifecycle Manager
    print("Testing Alpha Lifecycle Management: Birth, Growth, Maturity, Decay, Death tracking...")
    
    manager = AlphaLifecycleManager()
    
    # Birth alphas
    print("\nBirthing alphas...")
    alpha1 = manager.birth_alpha("alpha_001", "ORB Strategy", 0.5)
    alpha2 = manager.birth_alpha("alpha_002", "VWAP Strategy", 1.0)
    alpha3 = manager.birth_alpha("alpha_003", "Momentum Strategy", 0.8)
    
    print(f"Birthed {len(manager.lifecycles)} alphas")
    
    # Simulate performance updates
    print("\nSimulating performance updates...")
    
    # Alpha 1: Growth -> Maturity -> Decay -> Death
    print("\nAlpha 1 lifecycle:")
    manager.update_alpha_performance("alpha_001", 0.9)  # Growth
    print(f"  Stage: {manager.lifecycles['alpha_001'].current_stage.value}")
    
    manager.update_alpha_performance("alpha_001", 1.5)  # Maturity
    print(f"  Stage: {manager.lifecycles['alpha_001'].current_stage.value}")
    
    manager.update_alpha_performance("alpha_001", 0.5)  # Decay
    print(f"  Stage: {manager.lifecycles['alpha_001'].current_stage.value}")
    
    manager.update_alpha_performance("alpha_001", 0.2)  # Death
    print(f"  Stage: {manager.lifecycles['alpha_001'].current_stage.value}")
    print(f"  Active: {manager.lifecycles['alpha_001'].is_active}")
    
    # Alpha 2: Stable in maturity
    print("\nAlpha 2 lifecycle:")
    manager.update_alpha_performance("alpha_002", 1.3)
    print(f"  Stage: {manager.lifecycles['alpha_002'].current_stage.value}")
    print(f"  Active: {manager.lifecycles['alpha_002'].is_active}")
    
    # Alpha 3: Decay
    print("\nAlpha 3 lifecycle:")
    manager.update_alpha_performance("alpha_003", 0.4)
    print(f"  Stage: {manager.lifecycles['alpha_003'].current_stage.value}")
    
    # Get lifecycle summary
    print("\nLifecycle Summary:")
    summary = manager.get_lifecycle_summary()
    print(summary.to_string(index=False))
    
    # Get stage distribution
    print("\nStage Distribution:")
    distribution = manager.get_stage_distribution()
    for stage, count in distribution.items():
        print(f"  {stage}: {count}")
    
    # Get active alphas
    print("\nActive Alphas:")
    active = manager.get_active_alphas()
    print(f"Number of active alphas: {len(active)}")
    for alpha in active:
        print(f"  {alpha.alpha_name}: {alpha.current_stage.value}")
    
    # Get retired alphas
    print("\nRetired Alphas:")
    retired = manager.get_retired_alphas()
    print(f"Number of retired alphas: {len(retired)}")
    for alpha in retired:
        print(f"  {alpha.alpha_name}: {alpha.retirement_reason}")
    
    # Get alpha events
    print("\nAlpha 1 Events:")
    events = manager.get_alpha_events("alpha_001")
    for event in events:
        print(f"  {event.event_type}: {event.from_stage.value if event.from_stage else 'None'} -> {event.to_stage.value if event.to_stage else 'None'}")
        print(f"    Reason: {event.reason}")
